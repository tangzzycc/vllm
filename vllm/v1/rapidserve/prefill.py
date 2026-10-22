# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import replace

import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.rapidserve.parallel import submit_batch
from vllm.v1.rapidserve.scheduler import PrefillWork


def configure_prefill(config):
    """Size the P runner independently of decode's token budget."""
    options = config.additional_config
    chunked = options.get("rapidserve_chunked_prefill", False)
    if type(chunked) is not bool:
        raise ValueError("rapidserve_chunked_prefill must be a boolean")
    scheduler = config.scheduler_config
    default_budget = scheduler.max_num_batched_tokens
    if not chunked:
        default_budget = max(default_budget, config.model_config.max_model_len)
    budget = options.get("rapidserve_prefill_max_num_batched_tokens", default_budget)
    if type(budget) is not int or budget < 1:
        raise ValueError("rapidserve_prefill_max_num_batched_tokens must be positive")
    scheduler.max_num_batched_tokens = budget
    scheduler.max_num_scheduled_tokens = budget
    scheduler.enable_chunked_prefill = chunked
    max_seqs = options.get("rapidserve_prefill_max_num_seqs", scheduler.max_num_seqs)
    if type(max_seqs) is not int or max_seqs < 1:
        raise ValueError("rapidserve_prefill_max_num_seqs must be positive")
    scheduler.max_num_seqs = max_seqs
    scheduler.verify_max_model_len(config.model_config.max_model_len)


class PrefillEngine:
    """Batch externally assigned work without owning a KV block manager."""

    def __init__(self, worker, connection):
        self.worker = worker
        self.connection = connection
        self.waiting: deque[tuple[PrefillWork, int]] = deque()
        self.pending: deque = deque()
        self.last_sequence = 0
        self.session: str | None = None
        self.cancelled: set[int] = set()
        self.resident: set[int] = set()
        self.paused = False
        self.chunked = worker.model_runner.scheduler_config.enable_chunked_prefill

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
        active = {work.sequence for work, _ in self.waiting}
        active.update(
            work.sequence for selected, _, _ in self.pending for work, _ in selected
        )
        for work in works:
            if work.session == self.session and work.sequence in active:
                self.cancelled.add(work.sequence)

    def finish(self, work, result):
        cleanup = SchedulerOutput.make_empty()
        cleanup.finished_req_ids = {work.request.req_id}
        self.worker.model_runner.finish_requests(cleanup)
        self.cancelled.discard(work.sequence)
        self.resident.discard(work.sequence)
        self.connection.send(("prefill_done", (work.identity, result)))

    @torch.inference_mode()
    def step(self):
        if hasattr(self.worker, "check_health"):
            self.worker.check_health()
        progressed = False
        if self.pending:
            selected, output, event = self.pending[0]
            if not event.query() or (
                output is not None and not output.copy_event.query()
            ):
                selected = []
            else:
                self.pending.popleft()
                progressed = True
            result = (
                output.get_output()
                if selected and output is not None
                else EMPTY_MODEL_RUNNER_OUTPUT
            )
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
        if self.paused or not self.waiting or len(self.pending) >= 2:
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
            if (
                work.sequence not in self.resident
                and len(self.resident) >= self.worker.model_runner.max_num_reqs
            ):
                self.waiting.appendleft((work, computed))
                break
            remaining = work.num_tokens - computed
            if not self.chunked and remaining > budget:
                self.waiting.appendleft((work, computed))
                break
            count = min(budget, remaining)
            data = replace(
                work.request,
                num_computed_tokens=work.request.num_computed_tokens + computed,
            )
            batch.scheduled_new_reqs.append(data)
            batch.num_scheduled_tokens[data.req_id] = count
            batch.total_num_scheduled_tokens += count
            selected.append((work, computed + count))
            self.resident.add(work.sequence)
            budget -= count
        if not selected:
            return progressed
        output, event, _ = submit_batch(self.worker, batch)
        self.pending.append((selected, output, event))
        return True

    def has_work(self):
        return bool(self.waiting or self.pending)
