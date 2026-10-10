# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import replace

import torch

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.rapidserve.executor import PairedExecutor
from vllm.v1.rapidserve.scheduler import DecodeScheduler
from vllm.v1.worker.gpu.async_utils import AsyncOutput


def configure(config):
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Paired execution does not yet support prefix caching")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Paired eager execution initially requires sync scheduling")
    config.scheduler_config.scheduler_cls = (
        "vllm.v1.rapidserve.scheduler.DecodeScheduler"
    )
    return PairedExecutor


class PairedCoordinator:
    def __init__(self, core):
        self.core = core
        self.pending = None
        self.registered = set()
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
        outputs: dict[int, EngineCoreOutputs] = {}
        progressed = False
        for work, result in executor.poll_prefills():
            self.merge(outputs, scheduler.complete_prefill(work, result))
            progressed = True
        if self.pending:
            batch, result, event = self.pending
            if event.query() and (result is None or result.copy_event.query()):
                self.pending = None
                model_output = (
                    result.get_output()
                    if result is not None
                    else EMPTY_MODEL_RUNNER_OUTPUT
                )
                self.merge(outputs, scheduler.complete_decode(batch, model_output))
                progressed = True
        # Bound the P queue by the global admission cap owned by D.
        works = scheduler.grant_prefills()
        if works:
            executor.send_prefills(works)
            progressed = True
        if self.pending is None:
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
            executor.driver_worker.worker.execute_model(batch)
            result = executor.driver_worker.worker.sample_tokens(None)
            if result is not None and not isinstance(result, AsyncOutput):
                raise RuntimeError("Paired decode requires MRV2 asynchronous output")
            if batch.num_scheduled_tokens:
                event = torch.cuda.Event()
                event.record()
                self.pending = (batch, result, event)
                progressed = True
            elif batch.finished_req_ids:
                self.merge(
                    outputs,
                    scheduler.update_from_output(batch, EMPTY_MODEL_RUNNER_OUTPUT),
                )
                progressed = True
        if not progressed and scheduler.has_requests():
            time.sleep(0.0001)
        return outputs, progressed
