# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.kv_offload.adaptive_policy import (
    AdaptiveAction,
    AdaptiveOffloadPolicy,
    AdaptivePolicyConfig,
)
from vllm.v1.kv_offload.transfer_estimator import TransferRateEstimator


def _policy(**kwargs) -> AdaptiveOffloadPolicy:
    values = {
        "mode": "enforce",
        "prefill_tokens_per_second": 1_000.0,
        "h2d_bandwidth_bytes_per_second": 1_000.0,
        "min_savings_ms": 0.0,
        "min_recompute_tokens": 1,
    }
    values.update(kwargs)
    return AdaptiveOffloadPolicy(AdaptivePolicyConfig(**values))


def test_config_requires_prefill_rate_when_enforced():
    with pytest.raises(ValueError, match="prefill_tokens_per_second"):
        AdaptivePolicyConfig.from_extra_config({"adaptive_load": {"mode": "enforce"}})


def test_policy_loads_cpu_ready_prefix_when_h2d_is_cheaper():
    decision = _policy().decide(
        ready_tokens=100,
        candidate_tokens=100,
        ready_h2d_bytes=10,
        candidate_h2d_bytes=10,
    )
    assert decision.action is AdaptiveAction.LOAD_FULL


def test_policy_recomputes_cpu_ready_prefix_when_h2d_is_slower():
    decision = _policy().decide(
        ready_tokens=100,
        candidate_tokens=100,
        ready_h2d_bytes=10_000,
        candidate_h2d_bytes=10_000,
    )
    assert decision.action is AdaptiveAction.RECOMPUTE
    assert decision.recompute_tokens == 100


def test_policy_loads_ready_prefix_and_recomputes_pending_tail():
    decision = _policy().decide(
        ready_tokens=80,
        candidate_tokens=100,
        ready_h2d_bytes=10,
        candidate_h2d_bytes=10_000,
        promotion_seconds=10.0,
    )
    assert decision.action is AdaptiveAction.LOAD_READY
    assert decision.recompute_tokens == 20


def test_observe_mode_reports_decision_without_enforcement():
    policy = _policy(mode="observe")
    decision = policy.decide(
        ready_tokens=100,
        candidate_tokens=100,
        ready_h2d_bytes=10_000,
        candidate_h2d_bytes=10_000,
    )
    assert decision.action is AdaptiveAction.RECOMPUTE
    assert not policy.enforce


def test_policy_requires_material_savings():
    decision = _policy(safety_factor=2.0, min_savings_ms=10.0).decide(
        ready_tokens=10,
        candidate_tokens=10,
        ready_h2d_bytes=15,
        candidate_h2d_bytes=15,
    )
    assert decision.action is AdaptiveAction.LOAD_FULL
    assert decision.reason == "insufficient_savings"


def test_recompute_admission_uses_request_and_token_budgets():
    policy = _policy(
        max_active_recompute_requests=1,
        max_active_recompute_tokens=100,
    )
    assert policy.reserve(80)
    assert policy.capacity_reason(1) == "max_active_requests"
    policy.release(80)
    assert policy.reserve(100)
    policy.release(100)
    assert policy.active_requests == 0
    assert policy.active_tokens == 0


def test_transfer_estimator_uses_seed_until_enough_samples():
    estimator = TransferRateEstimator(
        min_samples=2,
        seed_bytes_per_second=100.0,
    )
    estimator.record(100, 0.5)
    assert estimator.bytes_per_second == 100.0
    estimator.record(100, 0.5)
    assert estimator.bytes_per_second == 200.0
