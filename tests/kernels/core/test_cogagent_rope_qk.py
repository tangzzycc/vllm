# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.triton_utils import HAS_TRITON

IS_SUPPORTED = HAS_TRITON and torch.version.hip is not None

if IS_SUPPORTED:
    from vllm.model_executor.models.cogagent_vision_encoder import (
        _apply_eva_rope_qk_native,
        _apply_eva_rope_qk_triton,
    )


@pytest.mark.skipif(not IS_SUPPORTED, reason="Requires Triton on ROCm")
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "qkv_chunk"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eva_rope_qk_triton_matches_native(
    strided: bool,
    dtype: torch.dtype,
) -> None:
    batch_size = 2
    num_tokens = 17
    num_heads = 4
    head_dim = 8
    hidden_size = num_heads * head_dim
    torch.manual_seed(0)

    if strided:
        qkv = torch.randn(
            batch_size,
            num_tokens,
            3 * hidden_size,
            device="cuda",
            dtype=dtype,
        )
        q, k, _ = qkv.chunk(3, dim=-1)
    else:
        q = torch.randn(
            batch_size,
            num_tokens,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        k = torch.randn_like(q)

    freqs_cos = torch.randn(
        num_tokens - 1,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    freqs_sin = torch.randn_like(freqs_cos)
    expected_q, expected_k = _apply_eva_rope_qk_native(
        q,
        k,
        freqs_cos,
        freqs_sin,
        num_heads,
        head_dim,
        dtype,
    )
    actual_q, actual_k = _apply_eva_rope_qk_triton(
        q,
        k,
        freqs_cos,
        freqs_sin,
        num_heads,
        head_dim,
        dtype,
    )

    assert actual_q.stride() == expected_q.stride()
    assert actual_k.stride() == expected_k.stride()
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
