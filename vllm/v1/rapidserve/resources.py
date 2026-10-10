# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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
