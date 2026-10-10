# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import replace

import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.rapidserve.scheduler import PrefillWork
from vllm.v1.worker.gpu.async_utils import AsyncOutput


class PrefillEngine:
    """Batch externally assigned work without owning a KV block manager."""

    def __init__(self, worker, connection):
        self.worker = worker
        self.connection = connection
        self.waiting: deque[tuple[PrefillWork, int]] = deque()
        self.pending = None
        self.last_sequence = 0
        self.session: str | None = None
        self.cancelled: set[int] = set()

    def add(self, works):
        for work in works:
            if self.session is None:
                self.session = work.session
            if work.session != self.session:
                raise ValueError("Prefill work belongs to a different session")
            if work.sequence <= self.last_sequence:
                continue
            self.last_sequence = work.sequence
            self.waiting.append((work, 0))

    def cancel(self, works):
        for work in works:
            if work.session == self.session:
                self.cancelled.add(work.sequence)

    def finish(self, work, result):
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids = {work.request.req_id}
        self.worker.model_runner.finish_requests(cleanup)
        self.cancelled.discard(work.sequence)
        self.connection.send(("prefill_done", (work.identity, result)))

    @torch.inference_mode()
    def step(self):
        progressed = False
        if self.pending:
            selected, output, event = self.pending
            if not event.query() or (
                output is not None and not output.copy_event.query()
            ):
                return False
            result = (
                output.get_output() if output is not None else EMPTY_MODEL_RUNNER_OUTPUT
            )
            self.pending = None
            for work, count in selected:
                req_id = work.request.req_id
                if work.sequence in self.cancelled:
                    self.finish(work, EMPTY_MODEL_RUNNER_OUTPUT)
                    continue
                if count < work.num_tokens:
                    self.waiting.appendleft((work, count))
                    continue
                index = result.req_id_to_index[req_id]
                sub = ModelRunnerOutput(
                    req_ids=[req_id],
                    req_id_to_index={req_id: 0},
                    sampled_token_ids=[result.sampled_token_ids[index]],
                )
                self.finish(work, sub)
            progressed = True
        if not self.waiting:
            return progressed
        batch = SchedulerOutput.make_empty()
        batch.num_common_prefix_blocks = [0] * len(
            self.worker.model_runner.kv_cache_config.kv_cache_groups
        )
        budget = self.worker.model_runner.max_num_tokens
        selected = []
        while (
            self.waiting
            and budget
            and len(selected) < self.worker.model_runner.max_num_reqs
        ):
            work, computed = self.waiting.popleft()
            if work.sequence in self.cancelled:
                self.finish(work, EMPTY_MODEL_RUNNER_OUTPUT)
                progressed = True
                continue
            count = min(budget, work.num_tokens - computed)
            data = replace(
                work.request,
                num_computed_tokens=work.request.num_computed_tokens + computed,
            )
            batch.scheduled_new_reqs.append(data)
            batch.num_scheduled_tokens[data.req_id] = count
            batch.total_num_scheduled_tokens += count
            selected.append((work, computed + count))
            budget -= count
        if not selected:
            return progressed
        self.worker.execute_model(batch)
        output = self.worker.sample_tokens(None)
        if output is not None and not isinstance(output, AsyncOutput):
            raise RuntimeError("Prefill requires MRV2 asynchronous output")
        event = torch.cuda.Event()
        event.record()
        self.pending = (selected, output, event)
        return True

    def has_work(self):
        return bool(self.waiting or self.pending)
