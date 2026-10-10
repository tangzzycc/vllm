# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from copy import deepcopy

from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.rapidserve.ipc import validate_shared_model
from vllm.v1.rapidserve.process import run_prefill
from vllm.v1.rapidserve.worker import SharedModelWorker


class PairedExecutor(UniProcExecutor):
    """Decode's local worker with a separately scheduled prefill process."""

    def _init_executor(self) -> None:
        validate_shared_model(self.vllm_config)
        if not self.vllm_config.model_config.enforce_eager:
            raise ValueError("Paired execution initially requires eager mode")
        if not self.vllm_config.cache_config.kv_cache_memory_bytes:
            raise ValueError(
                "Paired execution currently requires a fixed KV byte budget"
            )
        context = get_mp_context()
        if context.get_start_method() != "spawn":
            raise ValueError("Paired model execution requires spawn")
        config = deepcopy(self.vllm_config)
        config.compilation_config.static_forward_context = {}
        config.model_config.enforce_eager = True
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        config.compilation_config.mode = CompilationMode.NONE
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
                options, rapidserve_weights=weights
            )
            self.vllm_config.parallel_config.worker_cls = (
                config.parallel_config.worker_cls
            )
            self.vllm_config.cache_config.gpu_memory_utilization = min(
                self.vllm_config.cache_config.gpu_memory_utilization, 0.5
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
        self.connection.send((method, args))
        return self._receive("result")

    def determine_available_memory(self):
        return [self.vllm_config.cache_config.kv_cache_memory_bytes]

    def initialize_from_config(self, kv_cache_configs):
        config = kv_cache_configs[0]
        self.prefill_rpc("initialize_from_config", config)
        manifest = self.prefill_rpc("export_kv")
        worker = self.driver_worker.worker
        assert isinstance(worker, SharedModelWorker)
        with set_current_vllm_config(self.vllm_config):
            worker.import_kv(config, manifest)

    def compile_or_warm_up_model(self):
        self.prefill_rpc("compile_or_warm_up_model")
        super().compile_or_warm_up_model()

    def send_prefills(self, works):
        self.connection.send(("work", (works,)))

    def poll_prefills(self):
        results = []
        while self.connection.poll():
            results.append(self._receive("prefill_done"))
        return results

    def check_health(self):
        if self.prefill_process is None or not self.prefill_process.is_alive():
            raise RuntimeError("Prefill process exited")

    def shutdown(self):
        if getattr(self, "driver_worker", None) is not None:
            super().shutdown()
            del self.driver_worker
        connection = getattr(self, "connection", None)
        process = getattr(self, "prefill_process", None)
        if connection is not None:
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
