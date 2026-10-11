# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest

from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


class Mock: ...


def test_paired_quiesce_preserves_prefill_work_and_resumes(monkeypatch):
    """A policy barrier retires GPU work while retaining KV leases and queued work."""
    from unittest.mock import Mock

    import torch

    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.rapidserve.prefill import PrefillEngine
    from vllm.v1.rapidserve.scheduler import PrefillWork

    worker = Mock()
    worker.model_runner.max_num_tokens = 8
    worker.model_runner.max_num_reqs = 1
    worker.model_runner.kv_cache_config.kv_cache_groups = []
    worker.sample_tokens.return_value = None
    engine = PrefillEngine(worker, Mock())
    work = PrefillWork(
        "session",
        1,
        NewRequestData(
            req_id="r",
            prompt_token_ids=list(range(16)),
            mm_features=[],
            sampling_params=SamplingParams(temperature=0),
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=0,
            lora_request=None,
        ),
        16,
    )
    engine.waiting.append((work, 8))
    engine.resident.add(1)
    engine.paused = True
    assert not engine.step()
    assert engine.has_work()
    assert engine.resident == {1}
    worker.execute_model.assert_not_called()
    engine.paused = False
    monkeypatch.setattr(torch.cuda, "Event", Mock())
    assert engine.step()
    assert worker.execute_model.call_args.args[0].num_scheduled_tokens == {"r": 8}


def test_paired_auto_cu_prewarms_partition_before_serving(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from vllm.v1.rapidserve.executor import PairedExecutor
    from vllm.v1.rapidserve.resources import ResourcePolicyConfig
    from vllm.v1.rapidserve.worker import SharedModelWorker

    executor = PairedExecutor.__new__(PairedExecutor)
    executor.auto_cu_config = ResourcePolicyConfig(prefill_fraction=0.4)
    executor.prefill_rpc = Mock()
    worker = SharedModelWorker.__new__(SharedModelWorker)
    set_policy = Mock()
    monkeypatch.setattr(worker, "set_resource_policy", set_policy)
    executor.driver_worker = SimpleNamespace(worker=worker)
    with patch.object(UniProcExecutor, "compile_or_warm_up_model"):
        executor.compile_or_warm_up_model()
    assert [call.args for call in set_policy.call_args_list] == [
        (0, 0.4),
        (0, 0.0),
    ]


def test_paired_prefill_policy_barrier_waits_for_gpu_then_resumes(monkeypatch):
    """The owner ACK follows GPU retirement, while queued work survives the switch."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.v1.rapidserve import process
    from vllm.v1.rapidserve.worker import SharedModelWorker

    worker = SharedModelWorker.__new__(SharedModelWorker)
    monkeypatch.setattr(worker, "export_weights", lambda: "weights")
    set_policy = Mock(return_value=1)
    monkeypatch.setattr(worker, "set_resource_policy", set_policy)
    executor = Mock()
    executor.driver_worker.worker = worker
    engine = SimpleNamespace(pending=[1], paused=False, has_work=lambda: True)
    connection = Mock()
    # Skip receiving once to retire the simulated GPU batch.
    connection.poll.side_effect = [True, False, True, True, True]

    def step():
        assert engine.paused
        engine.pending.clear()
        return True

    engine.step = step

    def receive():
        messages = [
            ("start", ()),
            ("quiesce", (1,)),
            ("policy", (1, 0.5)),
            ("resume", ()),
            ("close", ()),
        ]
        yield from messages

    # Bootstrap does not call poll; the first poll receives quiesce.
    connection.recv.side_effect = receive()
    monkeypatch.setattr(process, "UniProcExecutor", lambda _: executor)
    monkeypatch.setattr(process, "PrefillEngine", lambda *_: engine)
    monkeypatch.setattr(process, "destroy_model_parallel", lambda: None)
    monkeypatch.setattr(process, "destroy_distributed_environment", lambda: None)
    process.run_prefill(Mock(), connection)
    kinds = [call.args[0][0] for call in connection.send.call_args_list]
    assert kinds == ["weights", "result", "quiesced", "policy_done"]
    assert not engine.paused
    set_policy.assert_called_once_with(1, 0.5)


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeProcess:
    def __init__(self, clock: _FakeClock, exits_at: float) -> None:
        self.clock = clock
        self.exits_at = exits_at
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self.clock.time() < self.exits_at

    def terminate(self) -> None:
        self.terminate_called = True


@pytest.mark.parametrize(
    ("timeout", "exits_at", "expected_terminate"),
    [
        pytest.param(6, 5, False, id="worker-exits-before-timeout"),
        pytest.param(6, 7, True, id="worker-exceeds-timeout"),
    ],
)
def test_multiproc_executor_worker_termination_timeout(
    monkeypatch, timeout, exits_at, expected_terminate
):
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    clock = _FakeClock()
    monkeypatch.setattr(multiproc_executor_module.time, "time", clock.time)
    monkeypatch.setattr(multiproc_executor_module.time, "sleep", clock.sleep)
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    proc = _FakeProcess(clock, exits_at=exits_at)
    executor._ensure_worker_termination([proc])  # type: ignore[list-item]
    assert proc.terminate_called is expected_terminate


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
        ec_output_aggregator: ECOutputAggregator | None = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
            ec_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


def test_paired_executor_surfaces_prefill_bootstrap_error():
    """A failed allocation owner cannot leave decode waiting for READY."""
    from unittest.mock import Mock

    from vllm.v1.rapidserve.executor import PairedExecutor

    executor = PairedExecutor.__new__(PairedExecutor)
    executor.connection = Mock()
    executor.connection.poll.return_value = True
    executor.connection.recv.return_value = ("error", "weight import failed")
    with pytest.raises(RuntimeError, match="weight import failed"):
        executor._receive("weights")


def test_paired_executor_closes_consumer_before_allocation_owner(monkeypatch):
    """The owner remains alive until decode releases its IPC mappings."""
    from unittest.mock import Mock

    from vllm.v1.rapidserve.executor import PairedExecutor

    order = []
    executor = PairedExecutor.__new__(PairedExecutor)
    executor.driver_worker = Mock()
    executor.connection = Mock()
    executor.connection.send.side_effect = lambda _: order.append("owner-close")
    executor.prefill_process = Mock()
    executor.prefill_process.is_alive.return_value = False
    monkeypatch.setattr(
        UniProcExecutor, "shutdown", lambda _: order.append("consumer-close")
    )
    executor.shutdown()
    executor.shutdown()
    assert order == ["consumer-close", "owner-close"]


def test_paired_late_cancel_does_not_retain_a_completed_identity():
    """A cancel sent just after P completion cannot grow the cancelled work set."""
    from unittest.mock import Mock

    from vllm.v1.rapidserve.prefill import PrefillEngine
    from vllm.v1.rapidserve.scheduler import PrefillIdentity

    engine = PrefillEngine(Mock(), Mock())
    engine.session = "session"
    engine.last_sequence = 1
    engine.cancel([PrefillIdentity("session", 1, "request")])
    assert not engine.cancelled


def test_paired_owner_error_waits_for_consumer_before_teardown(monkeypatch):
    """A P bootstrap error retains exported storage until D closes its channel."""
    from unittest.mock import Mock

    from vllm.v1.rapidserve import process
    from vllm.v1.rapidserve.worker import SharedModelWorker

    order = []
    worker = SharedModelWorker.__new__(SharedModelWorker)
    monkeypatch.setattr(worker, "export_weights", Mock(return_value="weights"))
    monkeypatch.setattr(
        worker, "export_kv", Mock(side_effect=RuntimeError("KV export failed"))
    )
    executor = Mock()
    executor.driver_worker.worker = worker
    executor.shutdown.side_effect = lambda: order.append("owner-shutdown")
    monkeypatch.setattr(process, "UniProcExecutor", lambda _: executor)
    monkeypatch.setattr(process, "destroy_model_parallel", lambda: None)
    monkeypatch.setattr(process, "destroy_distributed_environment", lambda: None)
    connection = Mock()
    connection.poll.return_value = True

    def receive():
        if order:
            order.append("consumer-close")
            return "close", ()
        order.append("export")
        return "export_kv", ()

    connection.recv.side_effect = receive
    process.run_prefill(Mock(), connection)
    assert order == ["export", "consumer-close", "owner-shutdown"]
    error = connection.send.call_args_list[-1].args[0]
    assert error[0] == "error" and "KV export failed" in error[1]


def test_paired_consumer_shutdown_failure_still_closes_owner(monkeypatch):
    from unittest.mock import Mock

    from vllm.v1.rapidserve.executor import PairedExecutor

    executor = PairedExecutor.__new__(PairedExecutor)
    executor.driver_worker = Mock()
    executor.connection = Mock()
    executor.prefill_process = Mock()
    executor.prefill_process.is_alive.return_value = False
    monkeypatch.setattr(
        UniProcExecutor, "shutdown", Mock(side_effect=RuntimeError("GPU failed"))
    )
    connection, process = executor.connection, executor.prefill_process
    with pytest.raises(RuntimeError, match="GPU failed"):
        executor.shutdown()
    connection.send.assert_called_once_with(("close", ()))
    process.join.assert_called_once_with(10)
    executor.shutdown()
    connection.send.assert_called_once_with(("close", ()))


def test_paired_prefill_waits_for_resident_slot_after_cancel(monkeypatch):
    """P retires its old request before executing work admitted after cancellation."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.rapidserve.prefill import PrefillEngine
    from vllm.v1.rapidserve.scheduler import PrefillWork

    runner = SimpleNamespace(
        max_num_reqs=1,
        max_num_tokens=8,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
        finish_requests=Mock(),
    )
    worker = SimpleNamespace(
        model_runner=runner, execute_model=Mock(), sample_tokens=lambda _: None
    )
    connection = Mock()
    engine = PrefillEngine(worker, connection)
    old = PrefillWork("session", 1, SimpleNamespace(req_id="old"), 8)
    from vllm.v1.core.sched.output import NewRequestData

    new = PrefillWork(
        "session",
        2,
        NewRequestData("new", [1] * 8, [], None, None, ([1],), 0, None),
        8,
    )
    event = Mock()
    event.query.return_value = False
    engine.session = "session"
    engine.resident.add(old.sequence)
    engine.pending.append(([(old, 8)], None, event))
    engine.cancel([old.identity])
    engine.add([new])
    assert not engine.step()
    worker.execute_model.assert_not_called()

    event.query.return_value = True
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    assert engine.step()
    connection.send.assert_called_once_with(
        ("prefill_done", (old.identity, EMPTY_MODEL_RUNNER_OUTPUT))
    )
    batch = worker.execute_model.call_args.args[0]
    assert batch.num_scheduled_tokens == {"new": 8}


@pytest.mark.parametrize("stage", ["manager", "scheduler", "ready"])
def test_paired_initialization_failure_closes_executor(monkeypatch, stage):
    """All startup stages after KV allocation must release the paired owner."""
    from unittest.mock import Mock

    from tests.v1.core.utils import create_scheduler
    from vllm.v1.engine.core import EngineCore

    scheduler = create_scheduler()
    config = scheduler.vllm_config
    config.additional_config = {"rapidserve": True}
    executor = Mock()
    monkeypatch.setattr(
        "vllm.v1.rapidserve.engine.configure", lambda _: lambda _: executor
    )
    monkeypatch.setattr(
        EngineCore,
        "_initialize_kv_caches",
        lambda *_: scheduler.kv_cache_manager.kv_cache_config,
    )
    error = RuntimeError("post-KV failure")
    monkeypatch.setattr(
        "vllm.v1.engine.core.StructuredOutputManager",
        Mock(side_effect=error) if stage == "manager" else Mock(),
    )
    monkeypatch.setattr(
        config.scheduler_config,
        "get_scheduler_cls",
        lambda: (
            Mock(side_effect=error) if stage == "scheduler" else lambda **_: scheduler
        ),
    )
    monkeypatch.setattr(
        EngineCore, "_initialize_effective_attention_block_size", lambda _: None
    )
    monkeypatch.setattr(
        "vllm.v1.rapidserve.engine.PairedCoordinator", Mock(side_effect=error)
    )
    cleanup = Mock()
    monkeypatch.setattr("vllm.v1.engine.core.cleanup_dist_env_and_memory", cleanup)
    with pytest.raises(RuntimeError, match="post-KV failure"):
        EngineCore(config, Mock(), log_stats=False)
    executor.shutdown.assert_called_once()
    cleanup.assert_called_once()
