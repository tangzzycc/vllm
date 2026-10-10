# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from uuid import uuid4

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


@dataclass(frozen=True)
class PrefillWork:
    session: str
    sequence: int
    request: NewRequestData
    num_tokens: int


class DecodeScheduler(Scheduler):
    """Own the block pool and lease prompt work to an external prefill engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = uuid4().hex
        self.sequence = 0
        self.leases: dict[str, tuple[PrefillWork, SchedulerOutput]] = {}
        self.retired: set[str] = set()
        self.decode_in_flight: set[str] = set()
        self.prefill_selection = False

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError("Paired execution requires unique active request IDs")
        params = request.sampling_params
        if (
            request.resumable
            or request.prompt_embeds is not None
            or request.use_structured_output
            or params is None
            or params.prompt_logprobs is not None
            or params.temperature != 0
        ):
            raise ValueError("Paired execution supports greedy text requests")
        super().add_request(request)

    @staticmethod
    def needs_prefill(request: Request) -> bool:
        return request.num_computed_tokens < max(
            request.num_prompt_tokens, request.num_tokens - 1
        )

    def _is_request_eligible(self, request: Request) -> bool:
        return (
            request.request_id not in self.leases
            and request.request_id not in self.decode_in_flight
            and not request.is_finished()
            and self.needs_prefill(request) == self.prefill_selection
        )

    def _can_schedule_waiting(self) -> bool:
        return self.prefill_selection

    def _request_blocks_can_be_freed(self, request: Request) -> bool:
        return (
            request.request_id not in self.leases
            and request.request_id not in self.decode_in_flight
            and super()._request_blocks_can_be_freed(request)
        )

    def _free_blocks(self, request: Request):
        if (
            request.request_id in self.leases
            or request.request_id in self.decode_in_flight
        ):
            self.retired.add(request.request_id)
        else:
            super()._free_blocks(request)

    def _free_request_blocks(self, request: Request):
        if (
            request.request_id in self.leases
            or request.request_id in self.decode_in_flight
        ):
            return
        super()._free_request_blocks(request)

    def grant_prefills(self) -> list[PrefillWork]:
        """Reserve complete contexts; the consumer chooses actual execution chunks."""
        self.prefill_selection = True
        token_budget = self.max_num_scheduled_tokens
        input_budget = self.scheduler_config.max_num_batched_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        full_budget = self.max_model_len * self.max_num_running_reqs
        self.max_num_scheduled_tokens = full_budget
        self.scheduler_config.max_num_batched_tokens = full_budget
        self.scheduler_config.long_prefill_token_threshold = 0
        try:
            output = super().schedule()
        finally:
            self.prefill_selection = False
            self.max_num_scheduled_tokens = token_budget
            self.scheduler_config.max_num_batched_tokens = input_budget
            self.scheduler_config.long_prefill_token_threshold = threshold
        works = []
        for req_id, count in output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            data = NewRequestData.from_request(
                request,
                self.kv_cache_manager.get_block_ids(req_id),
                list(request._all_token_ids),
            )
            data.num_computed_tokens -= count
            self.sequence += 1
            work = PrefillWork(self.session, self.sequence, data, count)
            sub = replace(
                output,
                scheduled_new_reqs=[data],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={req_id: count},
                total_num_scheduled_tokens=count,
                finished_req_ids=set(),
                preempted_req_ids=set(),
            )
            self.leases[req_id] = (work, sub)
            works.append(work)
        self.finished_req_ids.update(output.finished_req_ids)
        self.reset_preempted_req_ids.update(output.preempted_req_ids or set())
        return works

    def complete_prefill(self, work: PrefillWork, result: ModelRunnerOutput):
        req_id = work.request.req_id
        lease = self.leases.get(req_id)
        if lease is None or (lease[0].session, lease[0].sequence) != (
            work.session,
            work.sequence,
        ):
            return {}
        del self.leases[req_id]
        outputs = self.update_from_output(lease[1], result)
        self._release_retired({req_id})
        return outputs

    def _release_retired(self, req_ids: set[str]):
        for req_id in self.retired & req_ids:
            request = self.requests.get(req_id)
            if request is not None:
                super()._free_blocks(request)
            self.retired.remove(req_id)

    def complete_decode(self, batch, result):
        ids = set(batch.num_scheduled_tokens)
        self.decode_in_flight.difference_update(ids)
        output = self.update_from_output(batch, result)
        self._release_retired(ids)
        return output

    def has_requests(self) -> bool:
        return bool(self.leases or self.decode_in_flight) or super().has_requests()
