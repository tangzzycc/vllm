# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA LoRA shrink kernels tuned for gfx1151 decode and prefill."""

from collections.abc import Sequence

import torch

from vllm.lora.ops.triton_ops.lora_shrink_op import (
    lora_shrink as native_lora_shrink,
)
from vllm.lora.ops.triton_ops.utils import get_lora_kernel_grid_dim
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

MAX_RDNA_LORA_DECODE_TOKENS = 16
MIN_RDNA_LORA_PREFILL_TOKENS = 128
MIN_RDNA_LORA_SPECIALIZED_PREFILL_TOKENS = 64
MAX_RDNA_LORA_PREFILL_TOKENS = 1024
MAX_RDNA_LORA_RANK = 128
MAX_RDNA_LORA_SLICES = 4
_RDNA_LORA_DECODE_RANKS = (16, 32, 64, 128)
_RDNA_LORA_PREFILL_RANKS = (16, 32, 64, 128)
_RDNA_LORA_BASELINE_HIDDEN_SIZES = (2048, 6144)
_RDNA_LORA_SPECIALIZED_HIDDEN_SIZES = (2560, 4096, 9728)
_RDNA_LORA_SPECIALIZED_RANKS = (16, 32, 64)
_PREFILL_NUM_STAGES = 1
_LARGE_PREFILL_MIN_TOKENS = 768

_RDNA_LORA_SHRINK_DECODE_EXACT_CONFIGS = {
    (64, 2560, 3): (4, 512, 8, 1),
}
_RDNA_LORA_SHRINK_DECODE_SHAPE_CONFIGS = {
    (64, 4096): (8, 2048, 4, 1),
    (32, 6144): (4, 2048, 4, 2),
    (128, 6144): (4, 2048, 4, 1),
}
_RDNA_LORA_SHRINK_SPECIALIZED_DECODE_DEFAULT = (4, 2048, 4, 1)
_RDNA_LORA_SHRINK_DECODE_RANK_DEFAULTS = {
    16: (4, 2048, 4, 1),
    32: (8, 2048, 4, 1),
    128: (4, 2048, 8, 1),
}
_RDNA_LORA_SHRINK_DECODE_DEFAULT = (8, 2048, 4, 2)

# Config order: num_tokens < 128, 128 <= num_tokens < 768, num_tokens >= 768.
_RDNA_LORA_SHRINK_PREFILL_EXACT_CONFIGS = {
    (16, 4096, 1): (
        (16, 16, 128, 8, 4),
        None,
        None,
    ),
    (16, 9728, 1): (
        (16, 16, 64, 16, 4),
        None,
        None,
    ),
    (64, 9728, 1): (
        (16, 64, 64, 16, 4),
        None,
        None,
    ),
    (16, 2560, 2): (
        None,
        None,
        (16, 16, 64, 8, 2),
    ),
    (32, 4096, 1): (
        None,
        None,
        (16, 32, 64, 8, 2),
    ),
}
_RDNA_LORA_SHRINK_PREFILL_SHAPE_CONFIGS = {
    (64, 2560): (
        None,
        (16, 64, 64, 4, 4),
        (16, 64, 64, 4, 4),
    ),
    (64, 4096): (
        None,
        None,
        (16, 64, 64, 8, 4),
    ),
}
_RDNA_LORA_SHRINK_SPECIALIZED_PREFILL_RANK_DEFAULTS = {
    16: (32, 16, 64, 8, 4),
}
_RDNA_LORA_SHRINK_SPECIALIZED_PREFILL_DEFAULT = (32, 32, 64, 8, 4)


def _get_prefill_config_index(num_tokens: int) -> int:
    if num_tokens < MIN_RDNA_LORA_PREFILL_TOKENS:
        return 0
    if num_tokens >= _LARGE_PREFILL_MIN_TOKENS:
        return 2
    return 1


def _select_rdna_lora_shrink_decode_config(
    rank: int, hidden_size: int, num_slices: int
) -> tuple[int, int, int, int]:
    """Select (BLOCK_RANK, BLOCK_K, num_warps, num_stages) for gfx1151."""
    config = _RDNA_LORA_SHRINK_DECODE_EXACT_CONFIGS.get((rank, hidden_size, num_slices))
    if config is not None:
        return config

    config = _RDNA_LORA_SHRINK_DECODE_SHAPE_CONFIGS.get((rank, hidden_size))
    if config is not None:
        return config

    if hidden_size in _RDNA_LORA_SPECIALIZED_HIDDEN_SIZES:
        return _RDNA_LORA_SHRINK_SPECIALIZED_DECODE_DEFAULT
    return _RDNA_LORA_SHRINK_DECODE_RANK_DEFAULTS.get(
        rank, _RDNA_LORA_SHRINK_DECODE_DEFAULT
    )


def _select_rdna_lora_shrink_prefill_fallback(
    num_tokens: int, rank: int, hidden_size: int, num_slices: int
) -> tuple[int, int, int, int, int]:
    """Select a heuristic prefill config for shapes absent from the table."""
    if rank == 16:
        if hidden_size == 6144:
            return 32, 16, 64, 8, 4
        if num_slices == 3 or num_tokens < _LARGE_PREFILL_MIN_TOKENS:
            return 64, 16, 64, 4, 4
        return 32, 16, 64, 8, 4

    if num_tokens >= 1024:
        if hidden_size == 6144:
            if rank == 32:
                return 16, 32, 64, 8, 2
            if rank == 64:
                return 16, 64, 64, 8, 4
            return 32, 64, 64, 8, 4
        if rank == 32:
            return 16, 32, 64, 4, 2
        if rank == 128:
            return 16, 64, 64, 4, 4

    if hidden_size == 6144:
        return 32, 32, 64, 8, 4

    if num_tokens <= 192 and rank == 128:
        return 32, 32, 64, 8, 4
    if rank == 32 and num_slices == 1:
        return 32, 32, 64, 8, 4
    return 32, 32, 64, 4, 4


def _select_rdna_lora_shrink_prefill_config(
    num_tokens: int, rank: int, hidden_size: int, num_slices: int
) -> tuple[int, int, int, int, int]:
    """Select (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps) for gfx1151.

    The split-K choices keep enough workgroups in flight on the 40 physical
    CUs when LoRA rank is small. They were measured with rotating per-layer
    weights for the supported projection shapes.
    """
    if hidden_size in _RDNA_LORA_SPECIALIZED_HIDDEN_SIZES:
        config_index = _get_prefill_config_index(num_tokens)
        configs = _RDNA_LORA_SHRINK_PREFILL_EXACT_CONFIGS.get(
            (rank, hidden_size, num_slices)
        )
        if configs is not None:
            config = configs[config_index]
            if config is not None:
                return config

        configs = _RDNA_LORA_SHRINK_PREFILL_SHAPE_CONFIGS.get((rank, hidden_size))
        if configs is not None:
            config = configs[config_index]
            if config is not None:
                return config

        return _RDNA_LORA_SHRINK_SPECIALIZED_PREFILL_RANK_DEFAULTS.get(
            rank, _RDNA_LORA_SHRINK_SPECIALIZED_PREFILL_DEFAULT
        )

    return _select_rdna_lora_shrink_prefill_fallback(
        num_tokens, rank, hidden_size, num_slices
    )


def _can_run_rdna_lora_shrink_kernel(
    num_tokens: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    num_slices: int,
) -> bool:
    is_specialized_shape = (
        rank in _RDNA_LORA_SPECIALIZED_RANKS
        and hidden_size in _RDNA_LORA_SPECIALIZED_HIDDEN_SIZES
    )
    supported_hidden_size = (
        hidden_size in _RDNA_LORA_BASELINE_HIDDEN_SIZES or is_specialized_shape
    )
    if num_tokens <= MAX_RDNA_LORA_DECODE_TOKENS:
        return rank in _RDNA_LORA_DECODE_RANKS and supported_hidden_size

    min_prefill_tokens = (
        MIN_RDNA_LORA_SPECIALIZED_PREFILL_TOKENS
        if is_specialized_shape
        else MIN_RDNA_LORA_PREFILL_TOKENS
    )
    if (
        rank not in _RDNA_LORA_PREFILL_RANKS
        or dtype != torch.bfloat16
        or not supported_hidden_size
        or num_tokens < min_prefill_tokens
        or num_tokens > MAX_RDNA_LORA_PREFILL_TOKENS
    ):
        return False
    if rank == 16 and num_tokens < 256 and not is_specialized_shape:
        return False
    if rank == 16 and hidden_size == 2048 and num_slices == 1:
        return False
    if rank == 32 and num_slices == 1 and num_tokens < 256 and not is_specialized_shape:
        return False
    return not (
        is_specialized_shape
        and rank == 16
        and hidden_size == 9728
        and num_tokens >= 128
    )


def get_rdna_lora_shrink_config(
    lora_a_weights: Sequence[torch.Tensor],
) -> tuple[int, int, int] | None:
    """Return (num_slices, rank, hidden_size) for supported weights."""
    num_slices = len(lora_a_weights)
    if not (1 <= num_slices <= MAX_RDNA_LORA_SLICES):
        return None

    first_weight = lora_a_weights[0]
    if first_weight.ndim not in (3, 4) or (
        first_weight.ndim == 4 and first_weight.size(1) != 1
    ):
        return None
    rank, hidden_size = first_weight.shape[-2:]
    if not (1 <= rank <= MAX_RDNA_LORA_RANK) or hidden_size < 1:
        return None

    num_loras = first_weight.size(0)
    if num_loras < 1:
        return None
    device = first_weight.device
    for weight in lora_a_weights:
        if not (
            weight.ndim in (3, 4)
            and (weight.ndim == 3 or weight.size(1) == 1)
            and weight.size(0) == num_loras
            and weight.shape[-2:] == (rank, hidden_size)
            and weight.dtype in (torch.float16, torch.bfloat16)
            and weight.dtype == first_weight.dtype
            and weight.device == device
            and weight.is_contiguous()
        ):
            return None
    return num_slices, rank, hidden_size


def can_use_rdna_lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: Sequence[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    config: tuple[int, int, int] | None = None,
) -> bool:
    """Return whether tensors satisfy the RDNA shrink custom-op contract."""
    if config is None:
        config = get_rdna_lora_shrink_config(lora_a_weights)
    if config is None:
        return False
    num_slices, rank, hidden_size = config
    if inputs.ndim != 2 or output_tensor.ndim != 3:
        return False
    if inputs.size(0) < 1:
        return False
    if inputs.dtype not in (torch.float16, torch.bfloat16):
        return False
    if inputs.dtype != lora_a_weights[0].dtype:
        return False
    if output_tensor.dtype != torch.float32:
        return False
    if token_lora_mapping.dtype != torch.int32:
        return False
    if not (
        inputs.is_contiguous()
        and output_tensor.is_contiguous()
        and token_lora_mapping.is_contiguous()
    ):
        return False

    if inputs.shape[1] != hidden_size:
        return False
    if output_tensor.shape != (num_slices, inputs.size(0), rank):
        return False
    if token_lora_mapping.shape != (inputs.size(0),):
        return False

    return (
        lora_a_weights[0].device == inputs.device
        and output_tensor.device == inputs.device
        and token_lora_mapping.device == inputs.device
    )


@triton.jit
def _rdna_lora_shrink_decode_kernel(
    input_ptr,
    weight_0_ptr,
    weight_1_ptr,
    weight_2_ptr,
    weight_3_ptr,
    token_lora_mapping_ptr,
    output_ptr,
    input_d0_stride,
    input_d1_stride,
    weight_d0_stride,
    weight_d1_stride,
    weight_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    RANK: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_LORAS: tl.constexpr,
):
    rank_tiles = tl.cdiv(RANK, BLOCK_RANK)
    pid = tl.program_id(axis=0)
    token_id = pid // rank_tiles
    rank_tile = pid % rank_tiles
    slice_id = tl.program_id(axis=1)

    if slice_id == 0:
        weight_ptr = weight_0_ptr
    elif slice_id == 1:
        weight_ptr = weight_1_ptr
    elif slice_id == 2:
        weight_ptr = weight_2_ptr
    else:
        weight_ptr = weight_3_ptr

    lora_id = tl.load(token_lora_mapping_ptr + token_id)
    active = (lora_id >= 0) & (lora_id < NUM_LORAS)
    safe_lora_id = tl.where(active, lora_id, 0)
    offsets_rank = rank_tile * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
    accumulator = tl.zeros((BLOCK_RANK,), dtype=tl.float32)

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        input_values = tl.load(
            input_ptr + token_id * input_d0_stride + offsets_k * input_d1_stride,
            mask=active & (offsets_k < HIDDEN_SIZE),
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weight_ptr
            + safe_lora_id * weight_d0_stride
            + offsets_rank[:, None] * weight_d1_stride
            + offsets_k[None, :] * weight_d2_stride,
            mask=active
            & (offsets_rank[:, None] < RANK)
            & (offsets_k[None, :] < HIDDEN_SIZE),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(weight_values * input_values[None, :], axis=1)

    output_ptrs = (
        output_ptr
        + slice_id * output_d0_stride
        + token_id * output_d1_stride
        + offsets_rank * output_d2_stride
    )
    output_values = tl.where(active, accumulator * scaling, 0.0)
    tl.store(output_ptrs, output_values, mask=offsets_rank < RANK)


@triton.jit
def _rdna_lora_shrink_prefill_kernel(
    input_ptr,
    weight_0_ptr,
    weight_1_ptr,
    weight_2_ptr,
    weight_3_ptr,
    token_indices_sorted_by_lora_ids_ptr,
    num_tokens_per_lora_ptr,
    lora_token_start_loc_ptr,
    lora_ids_ptr,
    output_ptr,
    input_d0_stride,
    input_d1_stride,
    weight_d0_stride,
    weight_d1_stride,
    weight_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    RANK: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_LORAS: tl.constexpr,
):
    split_m = tl.program_id(axis=0)
    split_id = split_m % SPLIT_K
    pid_m = split_m // SPLIT_K
    pid_n = tl.program_id(axis=1)
    slice_lora_id = tl.program_id(axis=2)

    num_lora_groups = tl.num_programs(axis=2) // NUM_SLICES
    slice_id = slice_lora_id // num_lora_groups
    lora_idx = slice_lora_id % num_lora_groups
    lora_id = tl.load(lora_ids_ptr + lora_idx)
    if lora_id < 0:
        return
    if lora_id >= NUM_LORAS:
        return

    if slice_id == 0:
        weight_ptr = weight_0_ptr
    elif slice_id == 1:
        weight_ptr = weight_1_ptr
    elif slice_id == 2:
        weight_ptr = weight_2_ptr
    else:
        weight_ptr = weight_3_ptr

    lora_m_size = tl.load(num_tokens_per_lora_ptr + lora_idx)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < lora_m_size
    lora_m_start = tl.load(lora_token_start_loc_ptr + lora_idx)
    rows = tl.load(
        token_indices_sorted_by_lora_ids_ptr + lora_m_start + offsets_m,
        mask=mask_m,
        other=0,
    )
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_step = BLOCK_K * SPLIT_K
    for k_start in range(0, HIDDEN_SIZE, k_step):
        offsets_k = k_start + split_id * BLOCK_K + tl.arange(0, BLOCK_K)
        input_values = tl.load(
            input_ptr
            + rows[:, None] * input_d0_stride
            + offsets_k[None, :] * input_d1_stride,
            mask=mask_m[:, None] & (offsets_k[None, :] < HIDDEN_SIZE),
            other=0.0,
        )
        weight_values = tl.load(
            weight_ptr
            + lora_id * weight_d0_stride
            + offsets_k[:, None] * weight_d2_stride
            + offsets_n[None, :] * weight_d1_stride,
            mask=(offsets_k[:, None] < HIDDEN_SIZE) & (offsets_n[None, :] < RANK),
            other=0.0,
        )
        accumulator += tl.dot(input_values, weight_values, out_dtype=tl.float32)

    output_ptrs = (
        output_ptr
        + slice_id * output_d0_stride
        + rows[:, None] * output_d1_stride
        + offsets_n[None, :] * output_d2_stride
    )
    output_mask = mask_m[:, None] & (offsets_n[None, :] < RANK)
    value = accumulator * scaling
    if SPLIT_K == 1:
        tl.store(output_ptrs, value, mask=output_mask)
    else:
        tl.atomic_add(output_ptrs, value, mask=output_mask, sem="relaxed")


@torch.inference_mode()
def _rdna_lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    scaling: float,
) -> None:
    """Run an RDNA shrink kernel or fall back to the default LoRA op."""
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return

    weights = tuple(lora_a_weights)
    num_tokens = inputs.size(0)
    num_slices = len(weights)
    num_loras = weights[0].size(0)
    padded_weights = weights + (weights[-1],) * (MAX_RDNA_LORA_SLICES - num_slices)
    rank, hidden_size = weights[0].shape[-2:]
    if not _can_run_rdna_lora_shrink_kernel(
        num_tokens, rank, hidden_size, weights[0].dtype, num_slices
    ):
        native_lora_shrink(
            inputs,
            lora_a_weights,
            output_tensor,
            token_lora_mapping,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            no_lora_flag_cpu,
            num_active_loras,
            scaling,
        )
        return

    # Keep dispatch inside the opaque custom op so one dynamic compiled graph
    # can serve prefill and decode. HIP Graph still records concrete launches.
    if num_tokens > MAX_RDNA_LORA_DECODE_TOKENS:
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NUM_WARPS = (
            _select_rdna_lora_shrink_prefill_config(
                num_tokens, rank, hidden_size, num_slices
            )
        )
        output_tensor.zero_()
        grid = (
            triton.cdiv(num_tokens, BLOCK_M) * SPLIT_K,
            triton.cdiv(rank, BLOCK_N),
            num_slices * get_lora_kernel_grid_dim(num_active_loras, lora_ids),
        )
        _rdna_lora_shrink_prefill_kernel[grid](
            inputs,
            *padded_weights,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            output_tensor,
            inputs.stride(0),
            inputs.stride(1),
            weights[0].stride(0),
            weights[0].stride(-2),
            weights[0].stride(-1),
            output_tensor.stride(0),
            output_tensor.stride(1),
            output_tensor.stride(2),
            scaling,
            rank,
            hidden_size,
            num_slices,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            SPLIT_K,
            NUM_LORAS=num_loras,
            num_warps=NUM_WARPS,
            num_stages=_PREFILL_NUM_STAGES,
        )
        return

    BLOCK_RANK, BLOCK_K, NUM_WARPS, NUM_STAGES = _select_rdna_lora_shrink_decode_config(
        rank, hidden_size, num_slices
    )
    grid = (
        num_tokens * triton.cdiv(rank, BLOCK_RANK),
        num_slices,
    )

    _rdna_lora_shrink_decode_kernel[grid](
        inputs,
        *padded_weights,
        token_lora_mapping,
        output_tensor,
        inputs.stride(0),
        inputs.stride(1),
        weights[0].stride(0),
        weights[0].stride(-2),
        weights[0].stride(-1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        scaling,
        rank,
        hidden_size,
        BLOCK_RANK,
        BLOCK_K,
        NUM_LORAS=num_loras,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def _rdna_lora_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    scaling: float,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="rdna_lora_shrink",
        op_func=_rdna_lora_shrink,
        mutates_args=["output_tensor"],
        fake_impl=_rdna_lora_shrink_fake,
    )
    rdna_lora_shrink = torch.ops.vllm.rdna_lora_shrink
except AttributeError:
    rdna_lora_shrink = _rdna_lora_shrink
