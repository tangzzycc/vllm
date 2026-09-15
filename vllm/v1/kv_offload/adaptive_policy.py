# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from vllm.v1.kv_offload.transfer_estimator import TransferRateEstimator


class AdaptiveAction(Enum):
    LOAD_FULL = "load_full"
    WAIT_FULL = "wait_full"
    LOAD_READY = "load_ready"
    RECOMPUTE = "recompute"


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    action: AdaptiveAction
    baseline_action: AdaptiveAction
    wait_seconds: float | None = None
    selected_seconds: float | None = None
    recompute_tokens: int = 0
    reason: str = "baseline"


@dataclass(frozen=True, slots=True)
class AdaptivePolicyConfig:
    mode: str = "off"
    prefill_tokens_per_second: float | None = None
    prefill_fixed_ms: float = 0.0
    h2d_bandwidth_bytes_per_second: float | None = None
    secondary_bandwidth_bytes_per_second: Mapping[str, float] | None = None
    min_transfer_samples: int = 8
    estimator_window: int = 64
    safety_factor: float = 1.15
    min_savings_ms: float = 2.0
    min_recompute_tokens: int = 256
    max_recompute_tokens_per_request: int | None = 8192
    max_active_recompute_requests: int = 2
    max_active_recompute_tokens: int = 8192
    recompute_window_seconds: float = 10.0
    max_recompute_tokens_per_window: int | None = None

    VALID_MODES = frozenset({"off", "observe", "enforce"})

    @classmethod
    def from_extra_config(
        cls, extra_config: Mapping[str, Any] | None
    ) -> "AdaptivePolicyConfig":
        if extra_config is None:
            extra_config = {}
        raw = extra_config.get("adaptive_load", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("adaptive_load must be an object")

        mode = str(raw.get("mode", "off"))
        if mode not in cls.VALID_MODES:
            raise ValueError("adaptive_load.mode must be off, observe, or enforce")

        def positive_float(name: str) -> float | None:
            value = raw.get(name)
            if value is None:
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"adaptive_load.{name} must be a number") from exc
            if not isfinite(parsed) or parsed <= 0:
                raise ValueError(f"adaptive_load.{name} must be positive")
            return parsed

        def nonnegative_float(name: str, default: float) -> float:
            value = raw.get(name, default)
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"adaptive_load.{name} must be a number") from exc
            if not isfinite(parsed) or parsed < 0:
                raise ValueError(f"adaptive_load.{name} must be non-negative")
            return parsed

        def positive_int(name: str, default: int) -> int:
            try:
                parsed = int(raw.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"adaptive_load.{name} must be an integer") from exc
            if parsed <= 0:
                raise ValueError(f"adaptive_load.{name} must be positive")
            return parsed

        def optional_positive_int(name: str, default: int | None) -> int | None:
            value = raw.get(name, default)
            if value is None:
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"adaptive_load.{name} must be an integer") from exc
            if parsed <= 0:
                raise ValueError(f"adaptive_load.{name} must be positive")
            return parsed

        prefill_tps = positive_float("prefill_tokens_per_second")
        if mode == "enforce" and prefill_tps is None:
            raise ValueError(
                "adaptive_load.prefill_tokens_per_second is required in enforce mode"
            )

        safety_factor = nonnegative_float("safety_factor", 1.15)
        if safety_factor < 1:
            raise ValueError("adaptive_load.safety_factor must be at least 1")

        raw_secondary = raw.get("secondary_bandwidth_bytes_per_second", {})
        if not isinstance(raw_secondary, Mapping):
            raise ValueError(
                "adaptive_load.secondary_bandwidth_bytes_per_second must be an object"
            )
        secondary_bandwidth: dict[str, float] = {}
        for tier_type, value in raw_secondary.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "adaptive_load.secondary_bandwidth_bytes_per_second values "
                    "must be numbers"
                ) from exc
            if not isfinite(parsed) or parsed <= 0:
                raise ValueError(
                    "adaptive_load.secondary_bandwidth_bytes_per_second values "
                    "must be positive"
                )
            secondary_bandwidth[str(tier_type)] = parsed

        recompute_window_seconds = nonnegative_float("recompute_window_seconds", 10.0)
        max_recompute_tokens_per_window = optional_positive_int(
            "max_recompute_tokens_per_window", None
        )
        if max_recompute_tokens_per_window is not None and not (
            recompute_window_seconds > 0
        ):
            raise ValueError(
                "adaptive_load.recompute_window_seconds must be positive when "
                "max_recompute_tokens_per_window is set"
            )

        return cls(
            mode=mode,
            prefill_tokens_per_second=prefill_tps,
            prefill_fixed_ms=nonnegative_float("prefill_fixed_ms", 0.0),
            h2d_bandwidth_bytes_per_second=positive_float(
                "h2d_bandwidth_bytes_per_second"
            ),
            secondary_bandwidth_bytes_per_second=secondary_bandwidth,
            min_transfer_samples=positive_int("min_transfer_samples", 8),
            estimator_window=positive_int("estimator_window", 64),
            safety_factor=safety_factor,
            min_savings_ms=nonnegative_float("min_savings_ms", 2.0),
            min_recompute_tokens=positive_int("min_recompute_tokens", 256),
            max_recompute_tokens_per_request=optional_positive_int(
                "max_recompute_tokens_per_request", 8192
            ),
            max_active_recompute_requests=positive_int(
                "max_active_recompute_requests", 2
            ),
            max_active_recompute_tokens=positive_int(
                "max_active_recompute_tokens", 8192
            ),
            recompute_window_seconds=recompute_window_seconds,
            max_recompute_tokens_per_window=max_recompute_tokens_per_window,
        )


class AdaptiveOffloadPolicy:
    """Choose between waiting, loading a ready prefix, and recomputation."""

    def __init__(self, config: AdaptivePolicyConfig) -> None:
        self.config = config
        self.h2d = TransferRateEstimator(
            window_size=config.estimator_window,
            min_samples=config.min_transfer_samples,
            seed_bytes_per_second=config.h2d_bandwidth_bytes_per_second,
        )
        self._active_requests = 0
        self._active_tokens = 0
        self._recent_recomputes: deque[tuple[float, int]] = deque()
        self._recent_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    @property
    def enforce(self) -> bool:
        return self.config.mode == "enforce"

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def active_tokens(self) -> int:
        return self._active_tokens

    def _expire_recompute_history(self, now: float) -> None:
        cutoff = now - self.config.recompute_window_seconds
        while self._recent_recomputes and self._recent_recomputes[0][0] <= cutoff:
            _, tokens = self._recent_recomputes.popleft()
            self._recent_tokens -= tokens

    def observe_h2d(self, num_bytes: int, elapsed_seconds: float) -> None:
        self.h2d.record(num_bytes, elapsed_seconds)

    def _prefill_seconds(self, tokens: int) -> float | None:
        throughput = self.config.prefill_tokens_per_second
        if tokens < 0 or throughput is None:
            return None
        return self.config.prefill_fixed_ms / 1000 + tokens / throughput

    def _h2d_seconds(self, num_bytes: int, queued_bytes: int) -> float | None:
        return self.h2d.estimate_seconds(num_bytes + queued_bytes)

    def decide(
        self,
        *,
        ready_tokens: int,
        candidate_tokens: int,
        ready_h2d_bytes: int,
        candidate_h2d_bytes: int,
        queued_h2d_bytes: int = 0,
        promotion_seconds: float = 0.0,
    ) -> AdaptiveDecision:
        baseline = (
            AdaptiveAction.WAIT_FULL
            if promotion_seconds > 0 or ready_tokens < candidate_tokens
            else AdaptiveAction.LOAD_FULL
        )
        if not self.enabled or candidate_tokens <= 0:
            return AdaptiveDecision(baseline, baseline)

        wait_h2d = self._h2d_seconds(candidate_h2d_bytes, queued_h2d_bytes)
        recompute_all = self._prefill_seconds(candidate_tokens)
        if wait_h2d is None or recompute_all is None:
            return AdaptiveDecision(baseline, baseline, reason="insufficient_samples")
        wait_seconds = promotion_seconds + wait_h2d

        alternatives = [(AdaptiveAction.RECOMPUTE, recompute_all, candidate_tokens)]
        if 0 < ready_tokens < candidate_tokens:
            ready_h2d = self._h2d_seconds(ready_h2d_bytes, queued_h2d_bytes)
            recompute_tail = self._prefill_seconds(candidate_tokens - ready_tokens)
            if ready_h2d is not None and recompute_tail is not None:
                alternatives.append(
                    (
                        AdaptiveAction.LOAD_READY,
                        ready_h2d + recompute_tail,
                        candidate_tokens - ready_tokens,
                    )
                )

        action, selected_seconds, recompute_tokens = min(
            alternatives, key=lambda item: item[1]
        )
        threshold = (
            selected_seconds * self.config.safety_factor
            + self.config.min_savings_ms / 1000
        )
        if wait_seconds <= threshold:
            return AdaptiveDecision(
                baseline,
                baseline,
                wait_seconds,
                selected_seconds,
                reason="insufficient_savings",
            )
        if recompute_tokens < self.config.min_recompute_tokens:
            return AdaptiveDecision(
                baseline,
                baseline,
                wait_seconds,
                selected_seconds,
                reason="below_min_tokens",
            )
        max_tokens = self.config.max_recompute_tokens_per_request
        if max_tokens is not None and recompute_tokens > max_tokens:
            return AdaptiveDecision(
                baseline,
                baseline,
                wait_seconds,
                selected_seconds,
                reason="above_max_tokens",
            )
        return AdaptiveDecision(
            action,
            baseline,
            wait_seconds,
            selected_seconds,
            recompute_tokens,
            reason="lower_cost",
        )

    def capacity_reason(self, tokens: int) -> str | None:
        now = time.monotonic()
        self._expire_recompute_history(now)
        if self._active_requests >= self.config.max_active_recompute_requests:
            return "max_active_requests"
        if self._active_tokens + tokens > self.config.max_active_recompute_tokens:
            return "max_active_tokens"
        window_tokens = self.config.max_recompute_tokens_per_window
        if window_tokens is not None and self._recent_tokens + tokens > window_tokens:
            return "max_window_tokens"
        return None

    def reserve(self, tokens: int) -> bool:
        if tokens <= 0 or self.capacity_reason(tokens) is not None:
            return False
        self._active_requests += 1
        self._active_tokens += tokens
        if self.config.max_recompute_tokens_per_window is not None:
            self._recent_recomputes.append((time.monotonic(), tokens))
            self._recent_tokens += tokens
        return True

    def release(self, tokens: int) -> None:
        if tokens <= 0:
            return
        self._active_requests = max(0, self._active_requests - 1)
        self._active_tokens = max(0, self._active_tokens - tokens)

    def rollback_reservation(self, tokens: int) -> None:
        """Undo a reservation when recomputation cannot start."""
        self.release(tokens)
        if self.config.max_recompute_tokens_per_window is None:
            return
        _, reserved_tokens = self._recent_recomputes.pop()
        assert reserved_tokens == tokens
        self._recent_tokens -= reserved_tokens

    def reset(self) -> None:
        self.h2d.reset()
        self._active_requests = 0
        self._active_tokens = 0
        self._recent_recomputes.clear()
        self._recent_tokens = 0
