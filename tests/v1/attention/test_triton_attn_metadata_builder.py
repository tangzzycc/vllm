# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends import triton_attn
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_metadata_builder_uses_attention_spec_head_shape(monkeypatch, tmp_path):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 1,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    vllm_config = create_vllm_config(model_name=str(tmp_path))

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        triton_attn,
        "get_num_attention_heads_from_layers",
        lambda *args, **kwargs: 32,
    )

    builder = triton_attn.TritonAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["model.layers.0.cross_attn"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    assert builder.num_heads_kv == kv_cache_spec.num_kv_heads
    assert builder.headdim == kv_cache_spec.head_size
    assert builder.softmax_segm_output.shape[-1] == kv_cache_spec.head_size
