# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourcePolicyConfig:
    """Controls a full-CU/partition trial using observed decode step times."""

    prefill_fraction: float = 0.5
    min_decode_batch: int = 64
    enter_seconds: float = 0.5
    hold_seconds: float = 2.0
    retry_seconds: float = 30.0
    max_decode_slowdown: float = 0.1
    target_itl_ms: float = 0.0

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "ResourcePolicyConfig | None":
        value = options.get("rapidserve_auto_cu", False)
        if value is False:
            return None
        if value is True:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("rapidserve_auto_cu must be a boolean or dictionary")
        if options.get("rapidserve_cu_fraction", 0):
            raise ValueError("Automatic CU selection must start with full-CU")
        try:
            config = cls(**value)
        except TypeError as error:
            raise ValueError("Invalid rapidserve_auto_cu options") from error
        for name in (
            "prefill_fraction",
            "enter_seconds",
            "hold_seconds",
            "retry_seconds",
            "max_decode_slowdown",
            "target_itl_ms",
        ):
            number = getattr(config, name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (float, int))
                or not math.isfinite(number)
                or number < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if not 0 < config.prefill_fraction < 1:
            raise ValueError("prefill_fraction must be between zero and one")
        if type(config.min_decode_batch) is not int or config.min_decode_batch < 1:
            raise ValueError("min_decode_batch must be a positive integer")
        return config


class ResourceController:
    """Try one partition during sustained P/D overlap; retreat on D slowdown."""

    def __init__(self, config: ResourcePolicyConfig):
        self.config = config
        self.fraction = 0.0
        self.changed_at = float("-inf")
        self.retry_at = float("-inf")
        self.overlap_since: float | None = None
        self.full_gaps: dict[int, tuple[float, int]] = {}
        self.split_gaps: dict[int, tuple[float, int]] = {}
        self.rejected: set[int] = set()
        self.reason = "full-CU"
        self.reject_bucket: int | None = None
        self.stats = {"switches": 0, "fallbacks": 0, "decode_samples": 0}

    def observe_step(self, batch_size: int, gap: float):
        """Observe one completed GPU step under the current resource policy."""
        if gap <= 0:
            return
        bucket = batch_size.bit_length()
        table = self.split_gaps if self.fraction else self.full_gaps
        average, count = table.get(bucket, (gap, 0))
        table[bucket] = (average * 0.75 + gap * 0.25, count + 1)
        self.stats["decode_samples"] += 1

    def select(self, prefill_pending: bool, decode_batch: int, now: float) -> float:
        config = self.config
        bucket = decode_batch.bit_length()
        overlap = prefill_pending and decode_batch >= config.min_decode_batch
        if not overlap:
            self.overlap_since = None
        elif self.overlap_since is None:
            self.overlap_since = now
        if self.fraction:
            average, count = self.split_gaps.get(bucket, (0.0, 0))
            baseline, _ = self.full_gaps.get(bucket, (0.0, 0))
            if count >= 4 and (
                (baseline and average > baseline * (1 + config.max_decode_slowdown))
                or (config.target_itl_ms and average * 1000 > config.target_itl_ms)
            ):
                self.reject_bucket = bucket
                self.reason = "decode latency fallback"
                return 0.0
            if not prefill_pending or not decode_batch:
                self.reason = "single role active"
                return 0.0
            if not overlap and now - self.changed_at >= config.hold_seconds:
                self.reason = "decode batch below threshold"
                return 0.0
            return self.fraction
        if (
            overlap
            and bucket not in self.rejected
            and now >= self.retry_at
            and now - self.changed_at >= config.hold_seconds
            and self.overlap_since is not None
            and now - self.overlap_since >= config.enter_seconds
            and self.full_gaps.get(bucket, (0.0, 0))[1] >= 4
        ):
            self.reason = "sustained P/D overlap"
            return config.prefill_fraction
        return 0.0

    def applied(self, fraction: float, now: float):
        if not fraction and self.reject_bucket is not None:
            self.rejected.add(self.reject_bucket)
            self.retry_at = now + self.config.retry_seconds
            self.stats["fallbacks"] += 1
        self.reject_bucket = None
        self.fraction = fraction
        self.changed_at = now
        self.split_gaps.clear()
        self.stats["switches"] += 1

    def idle(self):
        self.overlap_since = None
        self.full_gaps.clear()
        self.rejected.clear()
        self.reject_bucket = None


def partition_cus(total: int, fraction: float) -> tuple[list[int], list[int]]:
    """Partition CU pairs into disjoint prefill and decode masks."""
    if total < 4 or total % 2 or not 0 < fraction < 1:
        raise ValueError(
            "CU partition requires even CU count >= 4 and 0 < fraction < 1"
        )
    split = max(2, min(total - 2, round(total * fraction / 2) * 2))
    masks = ([0] * ((total + 31) // 32), [0] * ((total + 31) // 32))
    for cu in range(total):
        masks[int(cu >= split)][cu // 32] |= 1 << (cu % 32)
    return masks
