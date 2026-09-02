# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA LoRA expand kernels tuned for gfx1151 decode and prefill."""

from collections.abc import Sequence

import torch

from vllm.lora.ops.triton_ops.lora_expand_op import (
    lora_expand as native_lora_expand,
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
_OUTPUT_SHAPE_2560 = (2560,)
_OUTPUT_SHAPE_4096_1024_1024 = (4096, 1024, 1024)
_OUTPUT_SHAPE_9728_9728 = (9728, 9728)
_LARGE_PREFILL_MIN_TOKENS = 768

_RDNA_LORA_EXPAND_DECODE_CONFIGS: dict[
    tuple[int, tuple[int, ...]], tuple[int, int, int]
] = {
    (16, _OUTPUT_SHAPE_2560): (32, 2, 1),
    (16, _OUTPUT_SHAPE_4096_1024_1024): (32, 2, 1),
    (16, _OUTPUT_SHAPE_9728_9728): (32, 2, 1),
    (32, _OUTPUT_SHAPE_2560): (32, 2, 1),
    (32, _OUTPUT_SHAPE_4096_1024_1024): (64, 4, 1),
    (32, _OUTPUT_SHAPE_9728_9728): (128, 8, 1),
    (64, _OUTPUT_SHAPE_2560): (64, 8, 1),
    (64, _OUTPUT_SHAPE_4096_1024_1024): (64, 8, 1),
    (64, _OUTPUT_SHAPE_9728_9728): (128, 8, 1),
}

# Config order: num_tokens < 128, 128 <= num_tokens < 768, num_tokens >= 768.
_RDNA_LORA_EXPAND_PREFILL_CONFIGS = {
    (16, _OUTPUT_SHAPE_2560): (
        (16, 32, 32, 2, 1),
        (64, 32, 16, 2, 1),
        (64, 32, 16, 2, 1),
    ),
    (16, _OUTPUT_SHAPE_4096_1024_1024): (
        (32, 256, 16, 4, 1),
        (32, 256, 16, 4, 1),
        (16, 256, 16, 2, 1),
    ),
    (16, _OUTPUT_SHAPE_9728_9728): (
        (16, 256, 16, 2, 1),
        (16, 256, 16, 2, 1),
        (16, 256, 16, 2, 1),
    ),
    (32, _OUTPUT_SHAPE_2560): (
        (16, 32, 32, 2, 1),
        (16, 32, 32, 2, 1),
        (64, 32, 16, 2, 1),
    ),
    (32, _OUTPUT_SHAPE_4096_1024_1024): (
        (16, 256, 32, 4, 1),
        (32, 128, 32, 2, 1),
        (32, 128, 32, 2, 1),
    ),
    (32, _OUTPUT_SHAPE_9728_9728): (
        (32, 128, 32, 2, 1),
        (32, 128, 32, 2, 1),
        (32, 256, 32, 4, 2),
    ),
    (64, _OUTPUT_SHAPE_2560): (
        (16, 32, 32, 2, 1),
        (64, 32, 16, 2, 1),
        (32, 256, 32, 4, 1),
    ),
    (64, _OUTPUT_SHAPE_4096_1024_1024): (
        (32, 128, 32, 2, 1),
        (32, 128, 32, 2, 1),
        (32, 256, 32, 4, 2),
    ),
    (64, _OUTPUT_SHAPE_9728_9728): (
        (32, 256, 32, 4, 2),
        (32, 256, 32, 4, 2),
        (32, 256, 32, 4, 2),
    ),
}


def _get_prefill_config_index(num_tokens: int) -> int:
    if num_tokens < MIN_RDNA_LORA_PREFILL_TOKENS:
        return 0
    if num_tokens >= _LARGE_PREFILL_MIN_TOKENS:
        return 2
    return 1


def _is_specialized_shape(rank: int, output_sizes: Sequence[int]) -> bool:
    return (rank, tuple(output_sizes)) in _RDNA_LORA_EXPAND_DECODE_CONFIGS


def _select_rdna_lora_expand_decode_fallback(
    num_tokens: int, rank: int, output_sizes: Sequence[int]
) -> tuple[int, int, int]:
    """Select a heuristic decode config for shapes absent from the table."""
    num_slices = len(output_sizes)
    if rank == 16:
        if num_tokens <= 4:
            return 32, 2, 1
        if num_tokens <= 8 and max(output_sizes) >= 6144:
            return 256, 2, 1
        if num_tokens <= 8 and num_slices > 1:
            return 128, 4, 1
        return 128, 2, 1
    if rank == 32:
        if num_slices == 1:
            return 32, 2, 1
        return 128, 8, 1
    if rank == 128:
        if num_slices == 1:
            return 32, 8, 1
        if max(output_sizes) >= 6144:
            return 128, 8, 1
        return 64, 8, 1
    return 128, 8, 1


def _select_rdna_lora_expand_decode_config(
    num_tokens: int, rank: int, output_sizes: Sequence[int]
) -> tuple[int, int, int]:
    """Select (BLOCK_N, num_warps, num_stages) for gfx1151."""
    config = _RDNA_LORA_EXPAND_DECODE_CONFIGS.get((rank, tuple(output_sizes)))
    if config is not None:
        return config
    return _select_rdna_lora_expand_decode_fallback(num_tokens, rank, output_sizes)


def _select_rdna_lora_expand_prefill_fallback(
    num_tokens: int, rank: int, output_sizes: Sequence[int]
) -> tuple[int, int, int, int, int]:
    """Select a heuristic prefill config for shapes absent from the table."""
    num_slices = len(output_sizes)
    if rank == 16:
        if num_slices == 1:
            return 64, 32, 16, 2, 1
        if num_tokens <= 128:
            return 32, 256, 16, 4, 1
        return 16, 256, 16, 2, 1

    if num_tokens >= 1024:
        if rank == 32:
            if num_slices == 1:
                return 32, 64, 32, 2, 1
            if max(output_sizes) >= 6144:
                return 32, 256, 32, 4, 1
            return 64, 64, 32, 2, 1
        return 32, 256, 32, 4, 1

    if rank == 32:
        if num_slices == 1:
            return 64, 32, 16, 2, 1
        if max(output_sizes) >= 6144:
            return 32, 128, 32, 2, 1
        return 64, 64, 32, 2, 1
    if rank == 128:
        return 32, 256, 32, 4, 1
    if num_slices == 1:
        return 64, 32, 16, 2, 1
    return 32, 128, 32, 2, 1


def _select_rdna_lora_expand_prefill_config(
    num_tokens: int, rank: int, output_sizes: Sequence[int]
) -> tuple[int, int, int, int, int]:
    """Select (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for gfx1151."""
    configs = _RDNA_LORA_EXPAND_PREFILL_CONFIGS.get((rank, tuple(output_sizes)))
    if configs is not None:
        return configs[_get_prefill_config_index(num_tokens)]
    return _select_rdna_lora_expand_prefill_fallback(num_tokens, rank, output_sizes)


def get_rdna_lora_expand_config(
    lora_b_weights: Sequence[torch.Tensor],
) -> tuple[int, int, tuple[int, ...]] | None:
    """Return (num_slices, rank, output_sizes) for supported weights."""
    num_slices = len(lora_b_weights)
    if not (1 <= num_slices <= MAX_RDNA_LORA_SLICES):
        return None

    first_weight = lora_b_weights[0]
    if first_weight.ndim not in (3, 4) or (
        first_weight.ndim == 4 and first_weight.size(1) != 1
    ):
        return None
    rank = first_weight.size(-1)
    if not (1 <= rank <= MAX_RDNA_LORA_RANK):
        return None

    num_loras = first_weight.size(0)
    if num_loras < 1:
        return None
    device = first_weight.device
    output_sizes: list[int] = []
    for weight in lora_b_weights:
        if not (
            weight.ndim in (3, 4)
            and (weight.ndim == 3 or weight.size(1) == 1)
            and weight.size(0) == num_loras
            and weight.size(-1) == rank
            and weight.size(-2) > 0
            and weight.dtype in (torch.float16, torch.bfloat16)
            and weight.dtype == first_weight.dtype
            and weight.device == device
            and weight.is_contiguous()
        ):
            return None
        output_sizes.append(weight.size(-2))
    return num_slices, rank, tuple(output_sizes)


def can_use_rdna_lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: Sequence[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    offset_start: int = 0,
    output_slices: tuple[int, ...] | None = None,
    config: tuple[int, int, tuple[int, ...]] | None = None,
) -> bool:
    """Return whether tensors satisfy the RDNA expand custom-op contract."""
    if config is None:
        config = get_rdna_lora_expand_config(lora_b_weights)
    if config is None:
        return False
    num_slices, rank, output_sizes = config
    if inputs.ndim != 3 or output_tensor.ndim != 2:
        return False
    input_slices, num_tokens, input_rank = inputs.shape
    if num_tokens < 1:
        return False
    if input_slices != num_slices or input_rank != rank:
        return False
    if inputs.dtype != torch.float32:
        return False
    if output_tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    if output_tensor.dtype != lora_b_weights[0].dtype:
        return False
    if token_lora_mapping.dtype != torch.int32:
        return False
    if not (
        inputs.is_contiguous()
        and output_tensor.is_contiguous()
        and token_lora_mapping.is_contiguous()
    ):
        return False
    if output_tensor.size(0) != num_tokens:
        return False
    if token_lora_mapping.shape != (num_tokens,):
        return False
    if offset_start < 0:
        return False
    if output_slices is not None and output_slices != output_sizes:
        return False

    return (
        lora_b_weights[0].device == inputs.device
        and output_tensor.device == inputs.device
        and token_lora_mapping.device == inputs.device
        and offset_start + sum(output_sizes) <= output_tensor.size(1)
    )


@triton.jit
def _rdna_lora_expand_decode_kernel(
    input_ptr,
    weight_0_ptr,
    weight_1_ptr,
    weight_2_ptr,
    weight_3_ptr,
    token_lora_mapping_ptr,
    output_ptr,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    weight_d1_stride,
    weight_d2_stride,
    output_d0_stride,
    output_d1_stride,
    RANK: tl.constexpr,
    NUM_LORAS: tl.constexpr,
    OUTPUT_0: tl.constexpr,
    OUTPUT_1: tl.constexpr,
    OUTPUT_2: tl.constexpr,
    OUTPUT_3: tl.constexpr,
    TILES_0: tl.constexpr,
    TILES_1: tl.constexpr,
    TILES_2: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    OFFSET_START: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(axis=0)
    output_tile = tl.program_id(axis=1)

    slice_id = 0
    slice_tile = output_tile
    slice_start = 0
    slice_size = OUTPUT_0
    if NUM_SLICES >= 2 and output_tile >= TILES_0:
        slice_id = 1
        slice_tile = output_tile - TILES_0
        slice_start = OUTPUT_0
        slice_size = OUTPUT_1
    if NUM_SLICES >= 3 and output_tile >= TILES_0 + TILES_1:
        slice_id = 2
        slice_tile = output_tile - TILES_0 - TILES_1
        slice_start = OUTPUT_0 + OUTPUT_1
        slice_size = OUTPUT_2
    if NUM_SLICES >= 4 and output_tile >= TILES_0 + TILES_1 + TILES_2:
        slice_id = 3
        slice_tile = output_tile - TILES_0 - TILES_1 - TILES_2
        slice_start = OUTPUT_0 + OUTPUT_1 + OUTPUT_2
        slice_size = OUTPUT_3

    lora_id = tl.load(token_lora_mapping_ptr + token_id)
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

    offsets_n = slice_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    input_values = tl.load(
        input_ptr
        + slice_id * input_d0_stride
        + token_id * input_d1_stride
        + offsets_k * input_d2_stride,
        mask=offsets_k < RANK,
        other=0.0,
    )
    input_values = input_values.to(weight_ptr.dtype.element_ty).to(tl.float32)
    weight_values = tl.load(
        weight_ptr
        + lora_id * slice_size * RANK
        + offsets_n[:, None] * weight_d1_stride
        + offsets_k[None, :] * weight_d2_stride,
        mask=(offsets_n[:, None] < slice_size) & (offsets_k[None, :] < RANK),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(weight_values * input_values[None, :], axis=1).to(
        output_ptr.dtype.element_ty
    )

    output_offsets = OFFSET_START + slice_start + offsets_n
    output_ptrs = (
        output_ptr + token_id * output_d0_stride + output_offsets * output_d1_stride
    )
    output_mask = offsets_n < slice_size
    if ADD_INPUTS:
        delta += tl.load(output_ptrs, mask=output_mask)
    tl.store(output_ptrs, delta, mask=output_mask)


@triton.jit
def _rdna_lora_expand_prefill_kernel(
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
    input_d2_stride,
    weight_d1_stride,
    weight_d2_stride,
    output_d0_stride,
    output_d1_stride,
    RANK: tl.constexpr,
    NUM_LORAS: tl.constexpr,
    OUTPUT_0: tl.constexpr,
    OUTPUT_1: tl.constexpr,
    OUTPUT_2: tl.constexpr,
    OUTPUT_3: tl.constexpr,
    TILES_0: tl.constexpr,
    TILES_1: tl.constexpr,
    TILES_2: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    OFFSET_START: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    output_tile = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids_ptr + lora_idx)
    if lora_id < 0:
        return
    if lora_id >= NUM_LORAS:
        return

    slice_id = 0
    slice_tile = output_tile
    slice_start = 0
    slice_size = OUTPUT_0
    if NUM_SLICES >= 2 and output_tile >= TILES_0:
        slice_id = 1
        slice_tile = output_tile - TILES_0
        slice_start = OUTPUT_0
        slice_size = OUTPUT_1
    if NUM_SLICES >= 3 and output_tile >= TILES_0 + TILES_1:
        slice_id = 2
        slice_tile = output_tile - TILES_0 - TILES_1
        slice_start = OUTPUT_0 + OUTPUT_1
        slice_size = OUTPUT_2
    if NUM_SLICES >= 4 and output_tile >= TILES_0 + TILES_1 + TILES_2:
        slice_id = 3
        slice_tile = output_tile - TILES_0 - TILES_1 - TILES_2
        slice_start = OUTPUT_0 + OUTPUT_1 + OUTPUT_2
        slice_size = OUTPUT_3

    lora_m_size = tl.load(num_tokens_per_lora_ptr + lora_idx)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < lora_m_size
    lora_m_start = tl.load(lora_token_start_loc_ptr + lora_idx)
    rows = tl.load(
        token_indices_sorted_by_lora_ids_ptr + lora_m_start + offsets_m,
        mask=mask_m,
        other=0,
    )

    offsets_n = slice_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    if slice_id == 0:
        weight_ptr = weight_0_ptr
    elif slice_id == 1:
        weight_ptr = weight_1_ptr
    elif slice_id == 2:
        weight_ptr = weight_2_ptr
    else:
        weight_ptr = weight_3_ptr

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, RANK, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        input_values = tl.load(
            input_ptr
            + slice_id * input_d0_stride
            + rows[:, None] * input_d1_stride
            + offsets_k[None, :] * input_d2_stride,
            mask=mask_m[:, None] & (offsets_k[None, :] < RANK),
            other=0.0,
        )
        weight_values = tl.load(
            weight_ptr
            + lora_id * slice_size * RANK
            + offsets_k[:, None] * weight_d2_stride
            + offsets_n[None, :] * weight_d1_stride,
            mask=(offsets_k[:, None] < RANK) & (offsets_n[None, :] < slice_size),
            other=0.0,
        )
        accumulator += tl.dot(
            input_values.to(weight_ptr.dtype.element_ty),
            weight_values,
            out_dtype=tl.float32,
        )

    output_offsets = OFFSET_START + slice_start + offsets_n
    output_ptrs = (
        output_ptr
        + rows[:, None] * output_d0_stride
        + output_offsets[None, :] * output_d1_stride
    )
    output_mask = mask_m[:, None] & (offsets_n[None, :] < slice_size)
    result = accumulator.to(output_ptr.dtype.element_ty)
    if ADD_INPUTS:
        result += tl.load(output_ptrs, mask=output_mask)
    tl.store(output_ptrs, result, mask=output_mask)


@torch.inference_mode()
def _rdna_lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """Run an RDNA expand kernel or fall back to the default LoRA op."""
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return

    weights = tuple(lora_b_weights)
    num_slices, num_tokens, rank = inputs.shape
    num_loras = weights[0].size(0)
    output_sizes = [weight.size(-2) for weight in weights]
    is_specialized_shape = _is_specialized_shape(rank, output_sizes)
    min_prefill_tokens = (
        MIN_RDNA_LORA_SPECIALIZED_PREFILL_TOKENS
        if is_specialized_shape
        else MIN_RDNA_LORA_PREFILL_TOKENS
    )
    padded_weights = weights + (weights[-1],) * (MAX_RDNA_LORA_SLICES - num_slices)
    padded_sizes = output_sizes + [output_sizes[-1]] * (
        MAX_RDNA_LORA_SLICES - num_slices
    )

    if num_tokens > MAX_RDNA_LORA_DECODE_TOKENS:
        if (
            rank not in _RDNA_LORA_PREFILL_RANKS
            or weights[0].dtype != torch.bfloat16
            or num_tokens < min_prefill_tokens
            or num_tokens > MAX_RDNA_LORA_PREFILL_TOKENS
        ):
            native_lora_expand(
                inputs,
                lora_b_weights,
                output_tensor,
                token_lora_mapping,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                no_lora_flag_cpu,
                num_active_loras,
                offset_start,
                add_inputs,
            )
            return

        BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = (
            _select_rdna_lora_expand_prefill_config(num_tokens, rank, output_sizes)
        )
        output_tiles = [triton.cdiv(size, BLOCK_N) for size in output_sizes]
        padded_tiles = output_tiles + [0] * (MAX_RDNA_LORA_SLICES - num_slices)
        grid = (
            triton.cdiv(num_tokens, BLOCK_M),
            sum(output_tiles),
            get_lora_kernel_grid_dim(num_active_loras, lora_ids),
        )
        _rdna_lora_expand_prefill_kernel[grid](
            inputs,
            *padded_weights,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            output_tensor,
            inputs.stride(0),
            inputs.stride(1),
            inputs.stride(2),
            weights[0].stride(-2),
            weights[0].stride(-1),
            output_tensor.stride(0),
            output_tensor.stride(1),
            rank,
            num_loras,
            *padded_sizes,
            *padded_tiles[:3],
            num_slices,
            offset_start,
            add_inputs,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return

    if rank not in _RDNA_LORA_DECODE_RANKS:
        native_lora_expand(
            inputs,
            lora_b_weights,
            output_tensor,
            token_lora_mapping,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            no_lora_flag_cpu,
            num_active_loras,
            offset_start,
            add_inputs,
        )
        return

    BLOCK_N, NUM_WARPS, NUM_STAGES = _select_rdna_lora_expand_decode_config(
        num_tokens, rank, output_sizes
    )
    output_tiles = [triton.cdiv(size, BLOCK_N) for size in output_sizes]
    padded_tiles = output_tiles + [0] * (MAX_RDNA_LORA_SLICES - num_slices)
    BLOCK_K = triton.next_power_of_2(rank)
    grid = (num_tokens, sum(output_tiles))

    _rdna_lora_expand_decode_kernel[grid](
        inputs,
        *padded_weights,
        token_lora_mapping,
        output_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        weights[0].stride(-2),
        weights[0].stride(-1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        rank,
        num_loras,
        *padded_sizes,
        *padded_tiles[:3],
        num_slices,
        offset_start,
        add_inputs,
        BLOCK_N,
        BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def _rdna_lora_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="rdna_lora_expand",
        op_func=_rdna_lora_expand,
        mutates_args=["output_tensor"],
        fake_impl=_rdna_lora_expand_fake,
    )
    rdna_lora_expand = torch.ops.vllm.rdna_lora_expand
except AttributeError:
    rdna_lora_expand = _rdna_lora_expand
