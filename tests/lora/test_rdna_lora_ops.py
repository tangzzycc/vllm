# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.lora.punica_wrapper.punica_gpu as punica_gpu
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.lora.ops.triton_ops.rdna_lora_expand import (
    can_use_rdna_lora_expand,
    rdna_lora_expand,
)
from vllm.lora.ops.triton_ops.rdna_lora_shrink import (
    can_use_rdna_lora_shrink,
    rdna_lora_shrink,
)
from vllm.lora.ops.triton_ops.utils import get_lora_kernel_grid_dim
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skip_global_cleanup

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx1151
else:

    def on_gfx1151() -> bool:
        return False


class _Metadata:
    def __init__(self, num_tokens: int) -> None:
        self.token_lora_mapping = torch.zeros(num_tokens, dtype=torch.int32)

    def meta_args(self, token_nums: int, specialize_active_lora: bool):
        tensor = torch.empty(0)
        return (tensor, tensor, tensor, tensor, tensor, tensor, tensor)


def _wrapper(num_tokens: int):
    wrapper = object.__new__(punica_gpu.PunicaWrapperGPU)
    wrapper.token_mapping_meta = _Metadata(num_tokens)
    wrapper.lora_config = SimpleNamespace(specialize_active_lora=False)
    return wrapper


def test_rdna_lora_shape_gates() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        inputs = torch.empty((16, 2048), dtype=dtype)
        lora_a = (torch.empty((2, 1, 64, 2048), dtype=dtype),)
        scratch = torch.empty((1, 16, 64), dtype=torch.float32)
        mapping = torch.zeros(16, dtype=torch.int32)
        assert can_use_rdna_lora_shrink(inputs, lora_a, scratch, mapping)

        lora_b = (torch.empty((2, 1, 2048, 64), dtype=dtype),)
        output = torch.empty((16, 2048), dtype=dtype)
        assert can_use_rdna_lora_expand(scratch, lora_b, output, mapping)

    # The custom op accepts larger dynamic-graph batches and dispatches them
    # to the native implementation at runtime.
    assert can_use_rdna_lora_shrink(
        torch.empty((17, 2048), dtype=torch.bfloat16),
        (torch.empty((2, 1, 64, 2048), dtype=torch.bfloat16),),
        torch.empty((1, 17, 64), dtype=torch.float32),
        torch.zeros(17, dtype=torch.int32),
    )
    assert can_use_rdna_lora_shrink(
        torch.empty((366, 6144), dtype=torch.bfloat16),
        (torch.empty((2, 1, 128, 6144), dtype=torch.bfloat16),),
        torch.empty((1, 366, 128), dtype=torch.float32),
        torch.zeros(366, dtype=torch.int32),
    )
    assert not can_use_rdna_lora_shrink(
        torch.empty((366, 2048), dtype=torch.bfloat16),
        (torch.empty((2, 1, 256, 2048), dtype=torch.bfloat16),),
        torch.empty((1, 366, 256), dtype=torch.float32),
        torch.zeros(366, dtype=torch.int32),
    )


@pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")
def test_rdna_lora_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        punica_gpu, "lora_shrink", lambda *args, **kwargs: calls.append("shrink")
    )
    monkeypatch.setattr(
        punica_gpu,
        "rdna_lora_shrink",
        lambda *args, **kwargs: calls.append("rdna_shrink"),
    )
    monkeypatch.setattr(
        punica_gpu,
        "get_rdna_lora_shrink_config",
        lambda weights: (len(weights), 64, 2048),
    )
    monkeypatch.setattr(
        punica_gpu, "can_use_rdna_lora_shrink", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        punica_gpu, "lora_expand", lambda *args, **kwargs: calls.append("expand")
    )
    monkeypatch.setattr(
        punica_gpu,
        "rdna_lora_expand",
        lambda *args, **kwargs: calls.append("rdna_expand"),
    )
    monkeypatch.setattr(
        punica_gpu,
        "get_rdna_lora_expand_config",
        lambda weights: (len(weights), 64, (2048,)),
    )
    monkeypatch.setattr(
        punica_gpu, "can_use_rdna_lora_expand", lambda *args, **kwargs: True
    )

    wrapper = _wrapper(1)
    lora_a = (torch.empty((1, 1, 64, 2048), dtype=torch.bfloat16),)
    lora_b = (torch.empty((1, 1, 2048, 64), dtype=torch.bfloat16),)
    wrapper.add_shrink(
        torch.empty((1, 1, 64), dtype=torch.float32),
        torch.empty((1, 2048), dtype=torch.bfloat16),
        lora_a,
        1.0,
    )
    wrapper.add_expand(
        torch.empty((1, 2048), dtype=torch.bfloat16),
        torch.empty((1, 1, 64), dtype=torch.float32),
        lora_b,
        (2048,),
    )

    assert calls == ["rdna_shrink", "rdna_expand"]


def _metadata(num_tokens: int, max_loras: int, device: torch.device):
    mapping = torch.arange(num_tokens, dtype=torch.int32, device=device) % max_loras
    metadata = LoRAKernelMeta.make(
        max_loras=max_loras, max_num_tokens=num_tokens, device=device
    )
    metadata.prepare_tensors(mapping)
    return mapping, metadata.meta_args(num_tokens, specialize_active_lora=False)


def test_lora_metadata_mapping_is_initialized_to_no_lora() -> None:
    metadata = LoRAKernelMeta.make(max_loras=2, max_num_tokens=8, device="cpu")

    torch.testing.assert_close(
        metadata.token_lora_mapping,
        torch.full((8,), -1, dtype=torch.int32),
    )


def test_generic_lora_metadata_clears_mapping_tail_after_batch_shrink() -> None:
    metadata = LoRAKernelMeta.make(max_loras=2, max_num_tokens=8, device="cpu")
    metadata.prepare_tensors(torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32))

    metadata.prepare_tensors(torch.tensor([1, 0], dtype=torch.int32))

    torch.testing.assert_close(
        metadata.token_lora_mapping,
        torch.tensor([1, 0, -1, -1, -1, -1, -1, -1], dtype=torch.int32),
    )


def test_generic_no_lora_metadata_clears_previous_mapping() -> None:
    metadata = LoRAKernelMeta.make(max_loras=2, max_num_tokens=8, device="cpu")
    metadata.prepare_tensors(torch.tensor([0, 1, 0, 1], dtype=torch.int32))

    metadata.prepare_tensors(torch.full((4,), -1, dtype=torch.int32))

    assert metadata.no_lora_flag_cpu.item()
    assert metadata.num_active_loras_cpu.item() == 0
    torch.testing.assert_close(
        metadata.token_lora_mapping,
        torch.full((8,), -1, dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.active_lora_ids,
        torch.full_like(metadata.active_lora_ids, -1),
    )
    torch.testing.assert_close(
        metadata.num_tokens_per_lora,
        torch.zeros_like(metadata.num_tokens_per_lora),
    )
    torch.testing.assert_close(
        metadata.lora_token_start_loc,
        torch.zeros_like(metadata.lora_token_start_loc),
    )


def test_generic_lora_metadata_clears_inactive_groups() -> None:
    metadata = LoRAKernelMeta.make(max_loras=3, max_num_tokens=8, device="cpu")
    metadata.prepare_tensors(torch.tensor([0, 1, 0, 1], dtype=torch.int32))

    metadata.prepare_tensors(torch.tensor([1, 1, 1], dtype=torch.int32))

    torch.testing.assert_close(
        metadata.active_lora_ids,
        torch.tensor([1, -1, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.num_tokens_per_lora,
        torch.tensor([3, 0, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.lora_token_start_loc,
        torch.tensor([0, 3, 0, 0, 0], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("num_active_loras", "num_lora_slots", "expected"),
    [(0, 3, 0), (1, 3, 2), (2, 3, 3), (3, 3, 3)],
)
def test_lora_kernel_grid_dim_reserves_base_group(
    num_active_loras: int, num_lora_slots: int, expected: int
) -> None:
    count = torch.tensor([num_active_loras], dtype=torch.int32)
    lora_ids = torch.empty(num_lora_slots, dtype=torch.int32)

    assert get_lora_kernel_grid_dim(count, lora_ids) == expected


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize("lora_id", [0, 1])
def test_uniform_lora_metadata_matches_generic(lora_id: int) -> None:
    num_tokens = 366
    max_loras = 2
    device = torch.device(current_platform.device_type)
    mapping = torch.full((num_tokens,), lora_id, dtype=torch.int32, device=device)
    generic = LoRAKernelMeta.make(max_loras, num_tokens, device)
    uniform = LoRAKernelMeta.make(max_loras, num_tokens, device)

    generic.prepare_tensors(mapping)
    uniform.prepare_tensors_uniform(num_tokens, lora_id)
    torch.cuda.synchronize()

    torch.testing.assert_close(uniform.token_lora_mapping, generic.token_lora_mapping)
    torch.testing.assert_close(
        uniform.token_indices_sorted_by_lora_ids,
        generic.token_indices_sorted_by_lora_ids,
    )
    torch.testing.assert_close(uniform.active_lora_ids, generic.active_lora_ids)
    torch.testing.assert_close(uniform.num_tokens_per_lora, generic.num_tokens_per_lora)
    torch.testing.assert_close(
        uniform.lora_token_start_loc, generic.lora_token_start_loc
    )
    assert not uniform.no_lora_flag_cpu.item()
    assert uniform.num_active_loras_cpu.item() == 1


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
def test_uniform_lora_metadata_clears_mapping_tail_after_batch_shrink() -> None:
    device = torch.device(current_platform.device_type)
    metadata = LoRAKernelMeta.make(3, 8, device)
    metadata.prepare_tensors_uniform(8, 2)

    metadata.prepare_tensors_uniform(3, 1)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        metadata.token_lora_mapping,
        torch.tensor(
            [1, 1, 1, -1, -1, -1, -1, -1],
            dtype=torch.int32,
            device=device,
        ),
    )


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
def test_uniform_no_lora_metadata_resets_previous_active_state() -> None:
    device = torch.device(current_platform.device_type)
    metadata = LoRAKernelMeta.make(2, 16, device)
    metadata.prepare_tensors_uniform(8, 1)

    metadata.prepare_tensors_uniform(4, -1)
    torch.cuda.synchronize()

    assert metadata.no_lora_flag_cpu.item()
    assert metadata.num_active_loras_cpu.item() == 0
    torch.testing.assert_close(
        metadata.token_lora_mapping,
        torch.full((16,), -1, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        metadata.active_lora_ids,
        torch.full_like(metadata.active_lora_ids, -1),
    )
    torch.testing.assert_close(
        metadata.num_tokens_per_lora,
        torch.zeros_like(metadata.num_tokens_per_lora),
    )
    torch.testing.assert_close(
        metadata.lora_token_start_loc,
        torch.zeros_like(metadata.lora_token_start_loc),
    )


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize("inactive_lora_id", [-1, 1])
def test_rdna_lora_shrink_decode_zeros_inactive_nonfinite_row(
    inactive_lora_id: int,
) -> None:
    device = torch.device(current_platform.device_type)
    num_tokens, hidden_size, rank = 2, 2560, 16
    inputs = torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    inputs[0, 0] = float("nan")
    inputs[0, 1] = float("inf")
    inputs[0, 2] = float("-inf")
    weights = (
        torch.ones((1, 1, rank, hidden_size), dtype=torch.bfloat16, device=device),
    )
    mapping = torch.tensor([inactive_lora_id, 0], dtype=torch.int32, device=device)
    metadata = LoRAKernelMeta.make(1, num_tokens, device)
    metadata.prepare_tensors(mapping)
    output = torch.full(
        (1, num_tokens, rank), float("nan"), dtype=torch.float32, device=device
    )

    rdna_lora_shrink(
        inputs,
        weights,
        output,
        *metadata.meta_args(num_tokens, specialize_active_lora=False),
        1.0,
    )

    inactive_output = output[0, 0]
    assert torch.isfinite(inactive_output).all()
    torch.testing.assert_close(
        inactive_output,
        torch.zeros_like(inactive_output),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
def test_rdna_lora_expand_decode_ignores_out_of_range_lora_id() -> None:
    device = torch.device(current_platform.device_type)
    num_tokens, output_size, rank = 2, 2560, 16
    inputs = torch.ones((1, num_tokens, rank), dtype=torch.float32, device=device)
    weights = (
        torch.ones((1, 1, output_size, rank), dtype=torch.bfloat16, device=device),
    )
    mapping = torch.tensor([1, 0], dtype=torch.int32, device=device)
    metadata = LoRAKernelMeta.make(1, num_tokens, device)
    metadata.prepare_tensors(mapping)
    output = torch.randn((num_tokens, output_size), dtype=torch.bfloat16, device=device)
    expected_invalid = output[0].clone()
    expected_active = output[1].clone() + rank

    rdna_lora_expand(
        inputs,
        weights,
        output,
        *metadata.meta_args(num_tokens, specialize_active_lora=False),
        add_inputs=True,
    )

    torch.testing.assert_close(output[0], expected_invalid, rtol=0, atol=0)
    torch.testing.assert_close(output[1], expected_active)


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
def test_rdna_lora_prefill_ignores_out_of_range_lora_id() -> None:
    device = torch.device(current_platform.device_type)
    num_tokens, hidden_size, rank = 64, 2560, 16
    mapping = torch.arange(num_tokens, dtype=torch.int32, device=device) % 2
    metadata = LoRAKernelMeta.make(1, num_tokens, device)
    metadata.prepare_tensors(mapping)
    meta_args = metadata.meta_args(num_tokens, specialize_active_lora=False)

    inputs = torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    lora_a = (
        torch.ones((1, 1, rank, hidden_size), dtype=torch.bfloat16, device=device),
    )
    scratch = torch.full(
        (1, num_tokens, rank), float("nan"), dtype=torch.float32, device=device
    )
    rdna_lora_shrink(inputs, lora_a, scratch, *meta_args, 1.0)

    invalid_rows = mapping == 1
    torch.testing.assert_close(
        scratch[0, invalid_rows],
        torch.zeros_like(scratch[0, invalid_rows]),
        rtol=0,
        atol=0,
    )

    lora_b = (
        torch.ones((1, 1, hidden_size, rank), dtype=torch.bfloat16, device=device),
    )
    output = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    expected_invalid = output[invalid_rows].clone()
    rdna_lora_expand(
        scratch,
        lora_b,
        output,
        *meta_args,
        add_inputs=True,
    )

    torch.testing.assert_close(output[invalid_rows], expected_invalid, rtol=0, atol=0)


def test_uniform_lora_index_detection() -> None:
    detect = punica_gpu.PunicaWrapperGPU._uniform_lora_index

    assert detect((), [7, None]) == -1
    assert detect((-1, -1), [7, None]) == -1
    assert detect((7, 7, 7), [7, None]) == 0
    assert detect((9, 9), [7, 9]) == 1
    assert detect((7, 9), [7, 9]) is None


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "rank", "num_slices"),
    [
        (1, 2048, 16, 3),
        (1, 2048, 32, 3),
        (1, 2048, 64, 3),
        (1, 2048, 128, 3),
        (1, 6144, 32, 1),
        (1, 6144, 64, 2),
        (1, 6144, 128, 1),
        (16, 2048, 16, 1),
        (16, 2048, 64, 1),
        (17, 2048, 64, 1),
        (1, 2560, 16, 3),
        (1, 2560, 32, 2),
        (1, 2560, 64, 3),
        (1, 4096, 16, 1),
        (1, 4096, 32, 1),
        (1, 4096, 64, 1),
        (1, 9728, 16, 1),
        (1, 9728, 32, 1),
        (1, 9728, 64, 1),
    ],
)
def test_rdna_lora_shrink_parity(
    dtype: torch.dtype,
    num_tokens: int,
    hidden_size: int,
    rank: int,
    num_slices: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    inputs = torch.empty(
        (num_tokens, hidden_size), dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    weights = tuple(
        torch.empty((2, 1, rank, hidden_size), dtype=dtype, device=device).uniform_(
            -0.1, 0.1
        )
        for _ in range(num_slices)
    )
    mapping, meta_args = _metadata(num_tokens, 2, device)
    output = torch.empty(
        (num_slices, num_tokens, rank), dtype=torch.float32, device=device
    )

    rdna_lora_shrink(inputs, weights, output, *meta_args, 0.5)

    expected = torch.zeros_like(output)
    for token_id, lora_id in enumerate(mapping.tolist()):
        for slice_id, weight in enumerate(weights):
            expected[slice_id, token_id] = (
                inputs[token_id].float() @ weight[lora_id, 0].float().T
            ) * 0.5
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("num_tokens", "rank", "output_sizes"),
    [
        (1, 16, (2048, 1024, 1024)),
        (1, 32, (2048, 1024, 1024)),
        (1, 64, (2048, 1024, 1024)),
        (1, 128, (2048, 1024, 1024)),
        (1, 32, (6144, 6144)),
        (1, 128, (6144, 6144)),
        (16, 16, (2048,)),
        (16, 64, (2048,)),
        (17, 64, (2048,)),
        (1, 16, (4096, 1024, 1024)),
        (1, 32, (4096, 1024, 1024)),
        (1, 64, (4096, 1024, 1024)),
        (1, 16, (2560,)),
        (1, 32, (2560,)),
        (1, 64, (2560,)),
        (1, 16, (9728, 9728)),
        (1, 32, (9728, 9728)),
        (1, 64, (9728, 9728)),
    ],
)
def test_rdna_lora_expand_parity(
    dtype: torch.dtype,
    num_tokens: int,
    rank: int,
    output_sizes: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    inputs = torch.empty(
        (len(output_sizes), num_tokens, rank), dtype=torch.float32, device=device
    ).uniform_(-0.1, 0.1)
    weights = tuple(
        torch.empty((2, 1, size, rank), dtype=dtype, device=device).uniform_(-0.1, 0.1)
        for size in output_sizes
    )
    mapping, meta_args = _metadata(num_tokens, 2, device)
    output = torch.empty(
        (num_tokens, sum(output_sizes)), dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    expected = output.clone()

    rdna_lora_expand(
        inputs,
        weights,
        output,
        *meta_args,
        offset_start=0,
        add_inputs=True,
    )

    offset = 0
    for slice_id, (weight, size) in enumerate(zip(weights, output_sizes)):
        for token_id, lora_id in enumerate(mapping.tolist()):
            rank_buffer = inputs[slice_id, token_id].to(dtype).float()
            delta = rank_buffer @ weight[lora_id, 0].float().T
            expected[token_id, offset : offset + size] += delta.to(dtype)
        offset += size
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize(
    ("num_tokens", "rank", "hidden_size", "output_sizes"),
    [
        (128, 16, 2048, (2048, 1024, 1024)),
        (128, 32, 2048, (2048, 1024, 1024)),
        (128, 64, 2048, (2048, 1024, 1024)),
        (128, 128, 2048, (2048, 1024, 1024)),
        (366, 16, 2048, (2048,)),
        (366, 16, 6144, (2048,)),
        (366, 32, 6144, (2048,)),
        (366, 64, 6144, (2048,)),
        (366, 128, 6144, (2048,)),
        (1024, 16, 2048, (6144, 6144)),
        (1024, 32, 2048, (2048, 1024, 1024)),
        (1024, 64, 6144, (2048,)),
        (1024, 128, 2048, (6144, 6144)),
        (85, 16, 2560, (4096, 1024, 1024)),
        (85, 32, 2560, (4096, 1024, 1024)),
        (85, 64, 2560, (4096, 1024, 1024)),
        (85, 16, 4096, (2560,)),
        (85, 32, 4096, (2560,)),
        (85, 64, 4096, (2560,)),
        (85, 16, 2560, (9728, 9728)),
        (85, 32, 2560, (9728, 9728)),
        (85, 64, 2560, (9728, 9728)),
        (85, 16, 9728, (2560,)),
        (85, 32, 9728, (2560,)),
        (85, 64, 9728, (2560,)),
        (366, 16, 9728, (2560,)),
    ],
)
def test_rdna_lora_bf16_prefill_parity(
    num_tokens: int,
    rank: int,
    hidden_size: int,
    output_sizes: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    dtype = torch.bfloat16
    mapping = torch.arange(num_tokens, dtype=torch.int32, device=device) % 2
    metadata = LoRAKernelMeta.make(2, num_tokens, device)
    metadata.prepare_tensors(mapping)
    meta_args = metadata.meta_args(num_tokens, specialize_active_lora=False)

    inputs = torch.empty(
        (num_tokens, hidden_size), dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    lora_a = tuple(
        torch.empty((2, 1, rank, hidden_size), dtype=dtype, device=device).uniform_(
            -0.1, 0.1
        )
        for _ in output_sizes
    )
    scratch = torch.empty(
        (len(output_sizes), num_tokens, rank),
        dtype=torch.float32,
        device=device,
    )
    rdna_lora_shrink(inputs, lora_a, scratch, *meta_args, 0.5)

    expected_scratch = torch.empty_like(scratch)
    for lora_id in range(2):
        rows = mapping == lora_id
        for slice_id, weight in enumerate(lora_a):
            expected_scratch[slice_id, rows] = (
                inputs[rows].float() @ weight[lora_id, 0].float().T
            ) * 0.5
    torch.testing.assert_close(scratch, expected_scratch, rtol=1e-3, atol=1e-3)

    lora_b = tuple(
        torch.empty((2, 1, size, rank), dtype=dtype, device=device).uniform_(-0.1, 0.1)
        for size in output_sizes
    )
    output = torch.empty(
        (num_tokens, sum(output_sizes)), dtype=dtype, device=device
    ).uniform_(-0.1, 0.1)
    expected_output = output.clone()
    rdna_lora_expand(
        scratch,
        lora_b,
        output,
        *meta_args,
        offset_start=0,
        add_inputs=True,
    )

    offset = 0
    for slice_id, (weight, size) in enumerate(zip(lora_b, output_sizes)):
        for lora_id in range(2):
            rows = mapping == lora_id
            expected_output[rows, offset : offset + size] += (
                scratch[slice_id, rows].to(dtype).float() @ weight[lora_id, 0].float().T
            ).to(dtype)
        offset += size
    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize("rank", [16, 32, 64, 128])
@pytest.mark.parametrize("num_tokens", [1, 128, 366, 1024])
def test_rdna_lora_bf16_hip_graph_parity(num_tokens: int, rank: int) -> None:
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    dtype = torch.bfloat16
    hidden_size, output_size = 2048, 2048
    inputs = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    lora_a = (torch.randn((2, 1, rank, hidden_size), dtype=dtype, device=device),)
    lora_b = (torch.randn((2, 1, output_size, rank), dtype=dtype, device=device),)
    _, meta_args = _metadata(num_tokens, 2, device)

    eager_scratch = torch.empty(
        (1, num_tokens, rank), dtype=torch.float32, device=device
    )
    eager_output = torch.empty((num_tokens, output_size), dtype=dtype, device=device)
    rdna_lora_shrink(inputs, lora_a, eager_scratch, *meta_args, 1.0)
    rdna_lora_expand(eager_scratch, lora_b, eager_output, *meta_args, add_inputs=False)

    graph_scratch = torch.empty_like(eager_scratch)
    graph_output = torch.empty_like(eager_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rdna_lora_shrink(inputs, lora_a, graph_scratch, *meta_args, 1.0)
        rdna_lora_expand(
            graph_scratch, lora_b, graph_output, *meta_args, add_inputs=False
        )

    graph.replay()
    torch.cuda.synchronize()
    # Split-K atomics can change the FP32 accumulation order between eager and
    # graph replay. Random unit-scale inputs amplify the difference in the
    # following BF16 expand GEMM, but it remains within BF16 precision.
    split_k_prefill = num_tokens > 16 and not (rank == 32 and num_tokens < 256)
    rtol = 1e-2 if split_k_prefill else 1e-3
    atol = 4.0 if split_k_prefill else 2e-2
    torch.testing.assert_close(graph_output, eager_output, rtol=rtol, atol=atol)


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
def test_rdna_lora_hip_graph_replay_adds_base_metadata_group() -> None:
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    num_tokens, hidden_size, rank = 64, 2560, 16
    inputs = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).uniform_(-0.1, 0.1)
    weights = (
        torch.empty(
            (1, 1, rank, hidden_size), dtype=torch.bfloat16, device=device
        ).uniform_(-0.1, 0.1),
    )
    metadata = LoRAKernelMeta.make(
        max_loras=1,
        max_num_tokens=num_tokens,
        device=device,
        captured_lora_counts=[1, 2],
    )
    metadata.prepare_tensors_uniform(num_tokens, 0)
    meta_args = metadata.meta_args(num_tokens, specialize_active_lora=True)

    warmup_output = torch.empty(
        (1, num_tokens, rank), dtype=torch.float32, device=device
    )
    rdna_lora_shrink(inputs, weights, warmup_output, *meta_args, 1.0)
    torch.cuda.synchronize()

    graph_output = torch.empty_like(warmup_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rdna_lora_shrink(inputs, weights, graph_output, *meta_args, 1.0)

    mixed_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device) % 2 - 1
    metadata.prepare_tensors(mixed_mapping)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.zeros_like(graph_output)
    active_rows = mixed_mapping == 0
    expected[0, active_rows] = inputs[active_rows].float() @ weights[0][0, 0].float().T
    torch.testing.assert_close(graph_output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not on_gfx1151(), reason="requires gfx1151")
@pytest.mark.parametrize("rank", [16, 32, 64])
@pytest.mark.parametrize("num_tokens", [1, 85])
def test_rdna_lora_qwen3_vl_4b_hip_graph_parity(num_tokens: int, rank: int) -> None:
    torch.manual_seed(0)
    device = torch.device(current_platform.device_type)
    dtype = torch.bfloat16
    hidden_size = 2560
    output_sizes = (4096, 1024, 1024)
    inputs = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    lora_a = tuple(
        torch.randn((2, 1, rank, hidden_size), dtype=dtype, device=device)
        for _ in output_sizes
    )
    lora_b = tuple(
        torch.randn((2, 1, size, rank), dtype=dtype, device=device)
        for size in output_sizes
    )
    _, meta_args = _metadata(num_tokens, 2, device)

    eager_scratch = torch.empty(
        (len(output_sizes), num_tokens, rank), dtype=torch.float32, device=device
    )
    eager_output = torch.empty(
        (num_tokens, sum(output_sizes)), dtype=dtype, device=device
    )
    rdna_lora_shrink(inputs, lora_a, eager_scratch, *meta_args, 1.0)
    rdna_lora_expand(eager_scratch, lora_b, eager_output, *meta_args, add_inputs=False)

    graph_scratch = torch.empty_like(eager_scratch)
    graph_output = torch.empty_like(eager_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rdna_lora_shrink(inputs, lora_a, graph_scratch, *meta_args, 1.0)
        rdna_lora_expand(
            graph_scratch, lora_b, graph_output, *meta_args, add_inputs=False
        )

    graph.replay()
    torch.cuda.synchronize()
    rtol = 1e-2 if num_tokens == 85 else 1e-3
    atol = 4.0 if num_tokens == 85 else 2e-2
    torch.testing.assert_close(graph_output, eager_output, rtol=rtol, atol=atol)
