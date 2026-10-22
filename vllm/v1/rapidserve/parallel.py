# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from queue import Queue
from threading import Thread
from types import SimpleNamespace
from typing import Any

import torch

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.gpu.async_utils import AsyncOutput


class RemoteEvent:
    def __init__(self, future: Future):
        self.future = future

    def query(self) -> bool:
        if self.future.done():
            self.future.result()
            return True
        return False

    def elapsed_time(self, other: "RemoteEvent") -> float:
        return max(duration for _, duration in other.future.result())


class RemoteOutput(AsyncOutput):
    """Completion includes GPU work on every rank, not only rank zero's sample."""

    def __init__(self, future: Future):
        self.future = future
        self.copy_event = RemoteEvent(future)

    def get_output(self):
        return self.future.result()[0][0]


class ParallelWorker:
    """Use the existing worker RPC queues with a single response reader."""

    def __init__(self, executor: MultiprocExecutor):
        self.executor = executor
        config = executor.vllm_config
        self.model_runner = SimpleNamespace(
            scheduler_config=config.scheduler_config,
            max_num_tokens=config.scheduler_config.max_num_batched_tokens,
            max_num_reqs=config.scheduler_config.max_num_seqs,
            finish_requests=self.finish_requests,
        )
        self.responses: Queue = Queue()
        self.error: Exception | None = None
        self.reader = Thread(target=self._read, name="PairedTPResponses", daemon=True)
        self.reader.start()

    def _read(self):
        error = None
        while (item := self.responses.get()) is not None:
            rpc, result = item
            try:
                if error is not None:
                    raise error
                result.set_result(rpc.result())
            except Exception as exc:
                error = exc
                self.error = exc
                result.set_exception(exc)

    def rpc(self, method, *args, non_block=False, timeout=300, kwargs=None):
        self.check_health()
        result: Future[Any] = Future()
        result.set_running_or_notify_cancel()
        rpc = self.executor.collective_rpc(
            method, args=args, kwargs=kwargs, non_block=True, timeout=timeout
        )
        self.responses.put((rpc, result))
        return result if non_block else result.result()

    def execute_model(self, batch, timed=False):
        self.output = RemoteOutput(
            self.rpc("execute_paired_batch", batch, timed, non_block=True)
        )

    def sample_tokens(self, grammar):
        assert grammar is None
        return self.output

    def finish_requests(self, batch):
        # All subsequent submissions use the same ordered broadcast queue.
        self.rpc("finish_paired_requests", batch, non_block=True)

    def export_weights(self):
        return self.rpc("export_weights")

    def export_kv(self):
        return self.rpc("export_kv")

    def initialize_from_config(self, configs):
        self.rpc("initialize_from_config", configs)
        self.model_runner.kv_cache_config = configs[0]

    def import_kv(self, configs, manifests):
        self.rpc("import_rank_kv", configs, manifests)

    def compile_or_warm_up_model(self):
        return self.rpc("compile_or_warm_up_model")

    def set_resource_policy(self, *args):
        return self.rpc("set_resource_policy", *args)[0]

    def advance_resource_policy(self, *args):
        return self.rpc("advance_resource_policy", *args, non_block=True)

    def synchronize_device(self):
        self.rpc("synchronize_device")

    def check_health(self):
        if self.error is not None:
            raise RuntimeError("Paired TP worker RPC failed") from self.error
        if self.executor.is_failed:
            raise RuntimeError("Paired TP worker group failed")

    def close(self):
        self.responses.put(None)
        self.reader.join(5)


def submit_batch(worker, batch, timed=False):
    if isinstance(worker, ParallelWorker):
        worker.execute_model(batch, timed)
        output = worker.sample_tokens(None)
        event = output.copy_event
        return output, event, event if timed else None
    start = None
    if timed:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
    worker.execute_model(batch)
    output = worker.sample_tokens(None)
    if output is not None and not isinstance(output, AsyncOutput):
        raise RuntimeError("Paired execution requires MRV2 asynchronous output")
    event = torch.cuda.Event(enable_timing=timed)
    event.record()
    return output, event, start
