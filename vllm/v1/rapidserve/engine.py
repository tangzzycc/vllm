# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.rapidserve.executor import PairedExecutor
from vllm.v1.rapidserve.resources import ResourceController
from vllm.v1.rapidserve.scheduler import DecodeScheduler
from vllm.v1.worker.gpu.async_utils import AsyncOutput

logger = init_logger(__name__)


def configure(config):
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Paired execution does not yet support prefix caching")
    name = (
        "AsyncDecodeScheduler"
        if config.scheduler_config.async_scheduling
        else "DecodeScheduler"
    )
    config.scheduler_config.scheduler_cls = f"vllm.v1.rapidserve.scheduler.{name}"
    return PairedExecutor


class PairedCoordinator:
    def __init__(self, core):
        self.core = core
        self.pending: deque = deque()
        self.depth = core.vllm_config.max_concurrent_batches
        self.registered = set()
        self.cancel_sent: set[int] = set()
        self.buffered: dict[int, EngineCoreOutputs] = {}
        self.policy_epoch = 0
        self.policy_target: float | None = None
        self.policy_dispatched = False
        self.auto_policy_epoch: int | None = None
        config = getattr(core.model_executor, "auto_cu_config", None)
        self.resource_controller = ResourceController(config) if config else None
        self.draining = False
        self.policy_started_at = 0.0
        self.policy_switch_seconds = 0.0
        self.next_policy_at = 0.0
        self.decode_timings: dict[int, tuple[int, torch.cuda.Event]] = {}
        self.core.model_executor.prefill_rpc("start")

    @staticmethod
    def merge(outputs, incoming):
        for client, result in incoming.items():
            previous = outputs.get(client)
            if previous is None:
                outputs[client] = result
                continue
            previous.outputs.extend(result.outputs)
            if result.finished_requests:
                previous.finished_requests = (
                    previous.finished_requests or set()
                ) | result.finished_requests
            previous.scheduler_stats = result.scheduler_stats
            previous.timestamp = result.timestamp

    @torch.inference_mode()
    def step(self):
        core = self.core
        executor = core.model_executor
        scheduler = core.scheduler
        assert isinstance(executor, PairedExecutor)
        assert isinstance(scheduler, DecodeScheduler)
        executor.check_health()
        core._process_aborts_queue()
        outputs, self.buffered = self.buffered, {}
        progressed = False
        cancelled = [
            work
            for req_id, (work, _) in scheduler.leases.items()
            if scheduler.requests[req_id].is_finished()
            and work.sequence not in self.cancel_sent
        ]
        if cancelled and executor.can_send():
            executor.send_cancellations(cancelled)
            self.cancel_sent.update(work.sequence for work in cancelled)
        for work, result in executor.poll_prefills():
            self.cancel_sent.discard(work.sequence)
            self.merge(outputs, scheduler.complete_prefill(work, result))
            progressed = True
        if (
            self.auto_policy_epoch is not None
            and executor.policy_ack == self.auto_policy_epoch
        ):
            self.auto_policy_epoch = None
            scheduler.resource_policy_pending = False
            self.policy_switch_seconds += time.monotonic() - self.policy_started_at
        if self.pending:
            batch, result, event = self.pending[0]
            if event.query() and (result is None or result.copy_event.query()):
                self.pending.popleft()
                model_output = (
                    result.get_output()
                    if result is not None
                    else EMPTY_MODEL_RUNNER_OUTPUT
                )
                self.merge(outputs, scheduler.complete_decode(batch, model_output))
                timing = self.decode_timings.pop(id(batch), None)
                if (
                    self.resource_controller is not None
                    and self.policy_target is None
                    and self.auto_policy_epoch is None
                    and timing is not None
                    and timing[0] == self.policy_epoch
                ):
                    self.resource_controller.observe_step(
                        len(batch.num_scheduled_tokens),
                        timing[1].elapsed_time(event) / 1000,
                    )
                progressed = True
        controller = self.resource_controller
        if (
            self.policy_target is not None or self.auto_policy_epoch is not None
        ) and time.monotonic() - self.policy_started_at > 30:
            raise RuntimeError("Timed out switching paired resource policy")
        now = time.monotonic()
        if (
            controller is not None
            and self.policy_target is None
            and self.auto_policy_epoch is None
            and not self.draining
            and now >= self.next_policy_at
            and executor.can_send()
        ):
            self.next_policy_at = now + 0.05
            prefill, decode = scheduler.resource_pressure()
            fraction = controller.select(prefill, decode, now)
            if fraction != controller.fraction:
                logger.info(
                    "RapidServe automatic CU policy %.3f -> %.3f: %s",
                    controller.fraction,
                    fraction,
                    controller.reason,
                )
                self.policy_epoch += 1
                self.auto_policy_epoch = self.policy_epoch
                scheduler.resource_policy_pending = True
                self.policy_started_at = now
                executor.advance_prefill_policy(self.policy_epoch, fraction)
                executor.driver_worker.worker.advance_resource_policy(
                    self.policy_epoch, fraction
                )
                controller.applied(fraction, now)
        if self.policy_target is not None and not scheduler.leases and not self.pending:
            if not self.policy_dispatched and executor.can_send():
                executor.send_policy(self.policy_epoch, self.policy_target)
                self.policy_dispatched = True
            elif (
                self.policy_dispatched
                and executor.policy_ack == self.policy_epoch
                and executor.can_send()
            ):
                executor.driver_worker.worker.set_resource_policy(
                    self.policy_epoch, self.policy_target
                )
                now = time.monotonic()
                self.policy_switch_seconds += now - self.policy_started_at
                if controller is not None:
                    controller.applied(self.policy_target, now)
                self.policy_target = None
                self.policy_dispatched = False
                scheduler.resource_policy_pending = False
                progressed = True
        works = (
            scheduler.grant_prefills()
            if self.policy_target is None and executor.can_send()
            else []
        )
        if works:
            executor.send_prefills(works)
            progressed = True
        if self.policy_target is None and len(self.pending) < self.depth:
            batch = scheduler.schedule()
            self.registered.difference_update(batch.finished_req_ids)
            self.registered.difference_update(batch.preempted_req_ids or set())
            if batch.num_scheduled_tokens:
                additions = []
                for req_id in batch.num_scheduled_tokens:
                    if req_id in self.registered:
                        continue
                    request = scheduler.requests[req_id]
                    data = NewRequestData.from_request(
                        request,
                        scheduler.kv_cache_manager.get_block_ids(req_id),
                        list(request._all_token_ids),
                    )
                    data.num_computed_tokens -= batch.num_scheduled_tokens[req_id]
                    additions.append(data)
                ids = {data.req_id for data in additions}
                cached = batch.scheduled_cached_reqs
                indices = [i for i, rid in enumerate(cached.req_ids) if rid not in ids]
                batch = replace(
                    batch,
                    scheduled_new_reqs=additions,
                    scheduled_cached_reqs=replace(
                        cached,
                        req_ids=[cached.req_ids[i] for i in indices],
                        new_block_ids=[cached.new_block_ids[i] for i in indices],
                        num_computed_tokens=[
                            cached.num_computed_tokens[i] for i in indices
                        ],
                        num_output_tokens=[
                            cached.num_output_tokens[i] for i in indices
                        ],
                        new_token_ids=[],
                        all_token_ids={},
                        resumed_req_ids=set(),
                    ),
                )
                self.registered.update(ids)
                scheduler.decode_in_flight.update(batch.num_scheduled_tokens)
            if batch.num_scheduled_tokens:
                timed = self.resource_controller is not None
                if timed:
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self.decode_timings[id(batch)] = (self.policy_epoch, start)
                executor.driver_worker.worker.execute_model(batch)
                result = executor.driver_worker.worker.sample_tokens(None)
                if result is not None and not isinstance(result, AsyncOutput):
                    raise RuntimeError(
                        "Paired decode requires MRV2 asynchronous output"
                    )
                event = torch.cuda.Event(enable_timing=timed)
                event.record()
                self.pending.append((batch, result, event))
                progressed = True
            else:
                if batch.finished_req_ids or batch.preempted_req_ids:
                    executor.driver_worker.worker.model_runner.finish_requests(batch)
                    self.merge(
                        outputs,
                        scheduler.update_from_output(batch, EMPTY_MODEL_RUNNER_OUTPUT),
                    )
                    progressed = True
        if controller is not None and not scheduler.has_requests() and not self.pending:
            controller.idle()
        if not progressed and scheduler.has_requests():
            time.sleep(0.0001)
        return outputs, progressed

    def request_resource_policy(self, fraction: float):
        """Apply a manual policy and relinquish automatic resource control."""
        self._start_resource_policy(fraction)
        self.resource_controller = None

    def _start_resource_policy(self, fraction: float):
        if self.policy_target is not None or self.auto_policy_epoch is not None:
            raise ValueError("A resource policy change is already pending")
        if fraction and not 0 < fraction < 1:
            raise ValueError("CU fraction must be zero or between zero and one")
        self.policy_epoch += 1
        self.policy_target = fraction
        self.policy_started_at = time.monotonic()
        self.core.scheduler.resource_policy_pending = True

    def drain(self):
        """Wait for submitted GPU work before an in-process pause completes."""
        deadline = time.monotonic() + 30
        outputs: dict[int, EngineCoreOutputs] = {}
        self.draining = True
        try:
            while (
                self.core.scheduler.leases
                or self.pending
                or self.policy_target is not None
                or self.auto_policy_epoch is not None
            ):
                incoming, _ = self.step()
                self.merge(outputs, incoming)
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out draining paired execution")
        finally:
            self.draining = False
        self.merge(self.buffered, outputs)
