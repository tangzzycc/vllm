# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from math import isfinite


class TransferRateEstimator:
    """Estimate transfer throughput from a bounded observation window."""

    def __init__(
        self,
        *,
        window_size: int = 64,
        min_samples: int = 8,
        seed_bytes_per_second: float | None = None,
    ) -> None:
        self._observations: deque[tuple[int, float]] = deque(maxlen=window_size)
        self.min_samples = min_samples
        self.seed_bytes_per_second = seed_bytes_per_second

    def record(self, num_bytes: int, elapsed_seconds: float) -> None:
        if num_bytes <= 0 or elapsed_seconds <= 0 or not isfinite(elapsed_seconds):
            return
        self._observations.append((num_bytes, elapsed_seconds))

    @property
    def bytes_per_second(self) -> float | None:
        if len(self._observations) >= self.min_samples:
            total_bytes = sum(size for size, _ in self._observations)
            total_seconds = sum(duration for _, duration in self._observations)
            if total_bytes > 0 and total_seconds > 0:
                return total_bytes / total_seconds
        return self.seed_bytes_per_second

    def estimate_seconds(self, num_bytes: int) -> float | None:
        bandwidth = self.bytes_per_second
        if num_bytes < 0 or bandwidth is None or bandwidth <= 0:
            return None
        return num_bytes / bandwidth

    def reset(self) -> None:
        self._observations.clear()

    def __len__(self) -> int:
        return len(self._observations)
