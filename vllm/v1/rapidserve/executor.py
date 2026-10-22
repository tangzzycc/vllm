# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import suppress
from copy import deepcopy
from multiprocessing.connection import wait
from queue import Queue
from threading import Thread
from typing import cast
from weakref import ref

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.rapidserve.ipc import validate_shared_model
from vllm.v1.rapidserve.parallel import ParallelWorker
from vllm.v1.rapidserve.prefill import configure_prefill
from vllm.v1.rapidserve.process import run_prefill
from vllm.v1.rapidserve.resources import ResourcePolicyConfig
from vllm.v1.rapidserve.worker import SharedModelWorker


class PairedExecutor(UniProcExecutor):
    """Independent P/D worker groups sharing the same rank-local allocations."""

    decode_executor: MultiprocExecutor | None = None
    parallel_worker: ParallelWorker | None = None

    @property
    def paired_worker(self) -> ParallelWorker | SharedModelWorker:
        if self.parallel_worker is not None:
            return self.parallel_worker
        return cast(SharedModelWorker, self.driver_worker.worker)

    def _init_executor(self) -> None:
        self.shutting_down = False
        self.is_failed = False
        self.failure_callback = None
        self.outbound: Queue | None = None
        self.sender: Thread | None = None
        self.send_error: Exception | None = None
        validate_shared_model(self.vllm_config)
        if current_platform.is_rocm():
            # HIP's parallel graph queues do not inherit the launch stream CU mask.
            for name, value in (
                ("DEBUG_HIP_FORCE_GRAPH_QUEUES", "1"),
                ("DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING", "0"),
            ):
                if os.environ.get(name, value) != value:
                    raise ValueError(f"Paired ROCm execution requires {name}={value}")
                os.environ[name] = value
        options = self.vllm_config.additional_config
        assert isinstance(options, dict)
        self.auto_cu_config = ResourcePolicyConfig.from_options(options)
        self.constrained = options.get("rapidserve_constrained", False)
        if type(self.constrained) is not bool:
            raise ValueError("rapidserve_constrained must be a boolean")
        if self.constrained and (
            self.auto_cu_config is not None
            or options.get("rapidserve_cu_fraction", 0)
            or not current_platform.is_rocm()
        ):
            raise ValueError("Constrained CU selection requires ROCm and full-CU start")
        if self.auto_cu_config is not None and not current_platform.is_rocm():
            raise ValueError("Automatic CU selection requires ROCm")
        fraction = float(options.get("rapidserve_cu_fraction", 0))
        if fraction and (not current_platform.is_rocm() or not 0 < fraction < 1):
            raise ValueError(
                "CU fraction requires ROCm and must be between zero and one"
            )
        if not self.vllm_config.cache_config.kv_cache_memory_bytes:
            raise ValueError(
                "Paired execution currently requires a fixed KV byte budget"
            )
        context = get_mp_context()
        if context.get_start_method() != "spawn":
            raise ValueError("Paired model execution requires spawn")
        config = deepcopy(self.vllm_config)
        configure_prefill(config)
        config.compilation_config.static_forward_context = {}
        config.model_config.enforce_eager = True
        config.scheduler_config.async_scheduling = False
        if config.parallel_config.world_size > 1:
            # WorkerProc resolves completions in its copy thread; P still batches
            # via PrefillEngine rather than vLLM's decode scheduler.
            config.scheduler_config.async_scheduling = True
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        config.compilation_config.mode = CompilationMode.NONE
        config.kernel_config.enable_jit_warmup = False
        assert isinstance(config.additional_config, dict)
        config.additional_config = dict(
            config.additional_config, rapidserve_role="prefill"
        )
        config.parallel_config.worker_cls = (
            "vllm.v1.rapidserve.worker.SharedModelWorker"
        )
        # Both roles share one budget; the consumer doesn't reserve a second pool.
        config.cache_config.gpu_memory_utilization = min(
            config.cache_config.gpu_memory_utilization, 0.5
        )
        self.connection, child = context.Pipe()
        self.prefill_process = context.Process(
            target=run_prefill, args=(config, child), name="RapidServePrefill"
        )
        self.prefill_process.start()
        child.close()
        try:
            weights = self._receive("weights")
            options = self.vllm_config.additional_config
            if not isinstance(options, dict):
                raise ValueError(
                    "Paired execution requires dictionary additional_config"
                )
            self.vllm_config.additional_config = dict(
                options, rapidserve_weights=weights, rapidserve_role="decode"
            )
            self.vllm_config.parallel_config.worker_cls = (
                config.parallel_config.worker_cls
            )
            if self.parallel_config.world_size > 1:
                decode_config = deepcopy(self.vllm_config)
                # WorkerProc's output thread is independent of scheduler choice.
                decode_config.scheduler_config.async_scheduling = True
                self.decode_executor = MultiprocExecutor(decode_config)
                self.parallel_worker = ParallelWorker(self.decode_executor)
                self.vllm_config.additional_config.pop("rapidserve_weights", None)
            else:
                total_memory = current_platform.get_device_total_memory()
                shared_bytes = sum(storage[2] for storage in weights.storages)
                self.vllm_config.cache_config.gpu_memory_utilization = min(
                    self.vllm_config.cache_config.gpu_memory_utilization,
                    0.5 * (total_memory - shared_bytes) / total_memory,
                )
                super()._init_executor()
        except BaseException:
            self.shutdown()
            raise

    def _receive(self, expected: str):
        if not self.connection.poll(300):
            raise RuntimeError("Timed out waiting for the prefill process")
        kind, result = self.connection.recv()
        if kind == "error":
            raise RuntimeError(f"Prefill process failed:\n{result}")
        if kind != expected:
            raise RuntimeError(f"Expected {expected}, received {kind}")
        return result

    def prefill_rpc(self, method, *args):
        if self.outbound is not None:
            raise RuntimeError("Synchronous prefill RPC is only valid during bootstrap")
        self.connection.send((method, args))
        result = self._receive("result")
        if method == "start":
            self.outbound = Queue(maxsize=4)
            outbound = self.outbound
            connection = self.connection

            def send():
                try:
                    while True:
                        message = outbound.get()
                        connection.send(message)
                        if message[0] == "close":
                            return
                except (OSError, EOFError) as error:
                    self.send_error = error

            self.sender = Thread(target=send, name="PrefillSender", daemon=True)
            self.sender.start()
        return result

    def can_send(self) -> bool:
        return self.outbound is not None and not self.outbound.full()

    def determine_available_memory(self):
        return [
            self.vllm_config.cache_config.kv_cache_memory_bytes
        ] * self.parallel_config.world_size

    def initialize_from_config(self, kv_cache_configs):
        parallel = self.parallel_worker is not None
        config = kv_cache_configs if parallel else kv_cache_configs[0]
        self.prefill_rpc("initialize_from_config", config)
        manifest = self.prefill_rpc("export_kv")
        worker = self.paired_worker
        with set_current_vllm_config(self.vllm_config):
            worker.import_kv(config, manifest)

    def compile_or_warm_up_model(self):
        self.prefill_rpc("compile_or_warm_up_model")
        worker = self.paired_worker
        if isinstance(worker, ParallelWorker):
            times = worker.compile_or_warm_up_model()
            self.vllm_config.compilation_config.compilation_time = max(
                timing.language_model for timing in times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                timing.encoder for timing in times
            )
        else:
            super().compile_or_warm_up_model()
        if self.auto_cu_config is not None or getattr(self, "constrained", False):
            worker = self.paired_worker
            # Capture both policies before serving, rather than stalling live decode.
            candidate = (
                self.auto_cu_config.prefill_fraction if self.auto_cu_config else 0.5
            )
            for fraction in (candidate, 0.0):
                self.prefill_rpc("set_resource_policy", 0, fraction)
                worker.set_resource_policy(0, fraction)

    def send_prefills(self, works):
        assert self.outbound is not None
        self.outbound.put_nowait(("work", (works,)))

    def send_cancellations(self, works):
        assert self.outbound is not None
        self.outbound.put_nowait(("cancel", ([work.identity for work in works],)))

    def register_failure_callback(self, callback):
        self.failure_callback = callback
        if self.decode_executor is not None:
            self.decode_executor.register_failure_callback(callback)
        reference = ref(self)
        sentinel = self.prefill_process.sentinel

        def monitor():
            wait([sentinel])
            executor = reference()
            if executor is None or executor.shutting_down:
                return
            executor.is_failed = True
            callback()

        Thread(target=monitor, name="PrefillMonitor", daemon=True).start()

    def poll_prefills(self):
        results = []
        while self.connection.poll():
            kind, result = self.connection.recv()
            if kind == "error":
                raise RuntimeError(f"Prefill process failed:\n{result}")
            if kind == "policy_done":
                self.policy_ack = result
            elif kind == "quiesced":
                self.quiesce_ack = result
            elif kind == "prefill_done":
                results.append(result)
            else:
                raise RuntimeError(f"Unexpected prefill message: {kind}")
        return results

    def send_policy(self, epoch, fraction):
        assert self.outbound is not None
        self.policy_ack = None
        self.outbound.put_nowait(("policy", (epoch, fraction)))

    def advance_prefill_policy(self, epoch, fraction):
        assert self.outbound is not None
        self.policy_ack = None
        self.outbound.put_nowait(("advance_policy", (epoch, fraction)))

    def quiesce_prefill(self, epoch):
        assert self.outbound is not None
        self.quiesce_ack = None
        self.outbound.put_nowait(("quiesce", (epoch,)))

    def resume_prefill(self):
        assert self.outbound is not None
        self.outbound.put_nowait(("resume", ()))

    def check_health(self):
        if self.send_error is not None:
            raise RuntimeError("Prefill communication failed") from self.send_error
        if self.prefill_process is None or not self.prefill_process.is_alive():
            raise RuntimeError("Prefill process exited")
        if self.decode_executor is not None and self.decode_executor.is_failed:
            raise RuntimeError("Decode worker group failed")
        if self.decode_executor is not None:
            self.paired_worker.check_health()

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        single_value=False,
    ):
        if method in (
            "sleep",
            "wake_up",
            "release_kv_cache_memory",
            "discard",
            "reload_weights",
            "execute_dummy_batch",
            "update_weights",
            "start_weight_update",
            "finish_weight_update",
        ):
            raise ValueError(f"Paired execution does not support {method}")
        if method == "synchronize_device" and getattr(self, "outbound", None) is None:
            self.prefill_rpc(method)
        if self.decode_executor is not None:
            if single_value and non_block:
                raise ValueError("Paired TP does not support single-value async RPC")
            worker = self.paired_worker
            assert isinstance(worker, ParallelWorker)
            result = worker.rpc(
                method,
                *args,
                non_block=non_block,
                timeout=timeout,
                kwargs=kwargs,
            )
            if single_value:
                return result[0]
            return result
        return super().collective_rpc(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            non_block=non_block,
            single_value=single_value,
        )

    def shutdown(self):
        self.shutting_down = True
        try:
            if (decode := self.decode_executor) is not None:
                self.decode_executor = None
                worker = self.parallel_worker
                self.parallel_worker = None
                try:
                    decode.shutdown()
                finally:
                    if worker is not None:
                        worker.close()
            elif getattr(self, "driver_worker", None) is not None:
                try:
                    super().shutdown()
                finally:
                    del self.driver_worker
        finally:
            connection = getattr(self, "connection", None)
            process = getattr(self, "prefill_process", None)
            if connection is not None:
                if getattr(self, "outbound", None) is not None:
                    assert self.outbound is not None
                    with suppress(Exception):
                        self.outbound.put(("close", ()), timeout=1)
                    if self.sender is not None:
                        self.sender.join(1)
                else:
                    with suppress(EOFError, BrokenPipeError, OSError):
                        connection.send(("close", ()))
                connection.close()
                self.connection = None
            if process is not None:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                self.prefill_process = None
