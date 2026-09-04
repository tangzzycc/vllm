# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from transformers.utils.chat_template_utils import _compile_jinja_template

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.cogagent import (
    CogAgentForCausalLM,
    CogAgentRMSNorm,
    _compute_cogagent_rope_cache,
)
from vllm.model_executor.models.cogagent_vision_encoder import (
    EVAVisionRotaryEmbeddingFast,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.chat_templates.registry import (
    get_chat_template_fallback_path,
)
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.cogagent import (
    CogAgentConfig,
    EVACLIPVisionConfig,
)
from vllm.transformers_utils.processors.cogagent import CogAgentProcessor, compose

from ...utils import build_model_context, dummy_hf_overrides


@pytest.mark.cpu_test
def test_processor_preserves_encoder_decoder_placeholder_contract():
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    prompt = "Describe the image."

    assert ctx.model_config._model_info.supports_rocm_cudagraph

    outputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"image": Image.new("RGB", (32, 32))}),
    )

    bos_token_id = tokenizer.bos_token_id
    assert bos_token_id is not None
    image_token_id = processor.info.get_hf_config().image_token_id
    text_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert outputs["prompt_token_ids"] == [
        bos_token_id,
        *([image_token_id] * 258),
        *text_token_ids,
    ]

    placeholder = outputs["mm_placeholders"]["image"][0]
    assert placeholder.offset == 0
    assert placeholder.length == 259
    assert placeholder.get_num_embeds() == 258
    assert placeholder.is_embed is not None
    assert not placeholder.is_embed[0].item()
    assert placeholder.is_embed[1:].all().item()

    assert len(outputs["encoder_prompt_token_ids"]) == 6400
    assert processor.get_encoder_output_seq_len("image", placeholder) == 6658
    assert processor.get_cross_attention_seq_len("image", placeholder) == 6400

    image_inputs = outputs["mm_kwargs"]["image"][0].get_data()
    assert image_inputs["pixel_values"].shape == (3, 224, 224)
    assert image_inputs["cross_pixel_values"].shape == (3, 1120, 1120)
    assert image_inputs["pixel_values"].dtype == torch.bfloat16
    assert image_inputs["cross_pixel_values"].dtype == torch.bfloat16


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("num_images", "error_type", "match"),
    [
        (0, ValueError, "exactly one image"),
        (2, VLLMValidationError, "At most 1 image"),
    ],
)
def test_processor_requires_exactly_one_image(
    num_images: int, error_type: type[Exception], match: str
):
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": max(num_images, 1)},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", (32, 32)) for _ in range(num_images)]
    mm_data = {} if not images else {"image": images}

    with pytest.raises(error_type, match=match):
        processor(
            "Describe the image.",
            mm_items=processor.info.parse_mm_data(mm_data),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "image",
    [
        np.zeros((19, 31, 3), dtype=np.uint8),
        np.zeros((3, 19, 31), dtype=np.uint8),
        torch.zeros((19, 31, 3), dtype=torch.uint8),
        torch.zeros((3, 19, 31), dtype=torch.uint8),
        torch.zeros((3, 19, 31), dtype=torch.bfloat16),
    ],
)
def test_processor_accepts_supported_image_array_layouts(image):
    output = CogAgentProcessor._normalize_images(image)

    assert len(output) == 1
    assert output[0].mode == "RGB"
    assert output[0].size == (31, 19)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "images",
    [
        np.zeros((2, 19, 31, 3), dtype=np.uint8),
        np.zeros((2, 3, 19, 31), dtype=np.uint8),
        torch.zeros((2, 19, 31, 3), dtype=torch.uint8),
        torch.zeros((2, 3, 19, 31), dtype=torch.uint8),
    ],
)
def test_processor_accepts_batched_image_arrays(images):
    output = CogAgentProcessor._normalize_images(images)

    assert len(output) == 2
    assert all(image.mode == "RGB" for image in output)
    assert all(image.size == (31, 19) for image in output)


@pytest.mark.cpu_test
def test_processor_detaches_tensor_images():
    image = torch.zeros((3, 19, 31), dtype=torch.float32, requires_grad=True)

    output = CogAgentProcessor._normalize_images(image)

    assert len(output) == 1
    assert output[0].size == (31, 19)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 5, 4), dtype=np.uint8),
        torch.zeros((3, 5, 4), dtype=torch.uint8),
    ],
)
def test_processor_rejects_ambiguous_image_layout(image):
    with pytest.raises(ValueError, match="cannot infer whether an image is HWC or CHW"):
        CogAgentProcessor._normalize_images(image)


@pytest.mark.cpu_test
def test_processor_rejects_shape_overrides():
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    with pytest.raises(ValueError, match="fixed by the model configuration"):
        processor.info.get_hf_processor(image_size=128)


@pytest.mark.cpu_test
@pytest.mark.parametrize("size", [32, 80])
def test_image_transform_matches_reference(size: int):
    pixels = np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3)
    image = Image.fromarray(pixels, mode="RGB")
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    reference = transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )(image).to(torch.bfloat16)

    output = compose(torch.bfloat16, size=(size, size))(image)

    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_vision_rope_matches_reference_bf16_initialization():
    dim = 32
    pt_seq_len = 16
    ft_seq_len = 80
    freqs = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
    )
    positions = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
    freqs = torch.einsum("..., f -> ... f", positions, freqs)
    freqs = freqs.repeat_interleave(2, dim=-1)
    freqs = torch.cat(
        (
            freqs[:, None, :].expand(ft_seq_len, ft_seq_len, -1),
            freqs[None, :, :].expand(ft_seq_len, ft_seq_len, -1),
        ),
        dim=-1,
    )
    expected_cos = freqs.cos().to(torch.bfloat16).float().flatten(0, 1)
    expected_sin = freqs.sin().to(torch.bfloat16).float().flatten(0, 1)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = EVAVisionRotaryEmbeddingFast(dim, pt_seq_len, ft_seq_len)
    finally:
        torch.set_default_dtype(previous_dtype)

    torch.testing.assert_close(rope.freqs_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(rope.freqs_sin, expected_sin, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_rms_norm_matches_reference_bf16():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    weight = torch.randn(32, dtype=torch.bfloat16)
    eps = 1e-6

    norm = CogAgentRMSNorm(hidden_states.shape[-1], eps).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)

    hidden_states_fp32 = hidden_states.float()
    variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
    expected = (weight * hidden_states_fp32 * torch.rsqrt(variance + eps)).to(
        hidden_states.dtype
    )

    output = norm(hidden_states)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_decoder_rope_cache_matches_reference_bf16():
    head_size = 32
    max_position_embeddings = 80
    base = 10000
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    )
    inv_freq = inv_freq.to(torch.bfloat16)
    positions = torch.arange(
        max_position_embeddings,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    output = _compute_cogagent_rope_cache(
        head_size,
        max_position_embeddings,
        base,
        torch.bfloat16,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.cpu_test
def test_config_round_trip():
    config = CogAgentConfig()

    restored = CogAgentConfig.from_dict(config.to_dict())

    assert restored.image_token_id == restored.vocab_size
    assert restored.vision_config.layers == config.vision_config.layers
    assert (
        restored.vision_config.mlp_intermediate_size
        == config.vision_config.mlp_intermediate_size
    )
    assert (
        restored.vision_config.intermediate_size
        == config.vision_config.intermediate_size
    )
    assert restored.max_source_positions == 6400
    assert restored.rms_norm_eps == 1e-6
    assert restored.use_cache
    assert not restored.tie_encoder_decoder


@pytest.mark.cpu_test
def test_config_rejects_unsupported_template(tmp_path):
    config_dict = {
        "architectures": ["CogAgentForCausalLM"],
        "template_version": "chat_old",
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))

    with pytest.raises(ValueError, match="only template_version='chat'"):
        get_config(tmp_path, trust_remote_code=False)


@pytest.mark.cpu_test
def test_vision_config_round_trip():
    config = EVACLIPVisionConfig()

    restored = EVACLIPVisionConfig.from_dict(config.to_dict())

    assert restored.layers == config.layers


@pytest.mark.cpu_test
def test_token_only_processing_normalizes_bos():
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    bos_token_id = processor.info.get_tokenizer().bos_token_id
    assert bos_token_id is not None
    assert processor._apply_hf_processor_tokens_only([42]) == [bos_token_id, 42]
    assert processor._apply_hf_processor_tokens_only([bos_token_id, 42]) == [
        bos_token_id,
        42,
    ]


@pytest.mark.cpu_test
def test_token_prompt_processes_images_with_composite_processor():
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    bos_token_id = processor.info.get_tokenizer().bos_token_id
    assert bos_token_id is not None

    output = processor(
        [bos_token_id, 42],
        mm_items=processor.info.parse_mm_data(
            {"image": Image.new("RGB", (32, 32))}
        ),
    )

    image_inputs = output["mm_kwargs"]["image"][0].get_data()
    assert image_inputs["pixel_values"].shape == (3, 224, 224)
    assert image_inputs["cross_pixel_values"].shape == (3, 1120, 1120)


@pytest.mark.cpu_test
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_hf_processor_owns_required_bos(add_special_tokens: bool):
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()
    bos_token_id = processor.info.get_tokenizer().bos_token_id
    assert bos_token_id is not None

    output = hf_processor(
        text="Describe the image.",
        add_special_tokens=add_special_tokens,
        return_tensors=None,
    )

    (token_ids,) = output["input_ids"]
    assert token_ids[0] == bos_token_id
    assert token_ids[1] != bos_token_id


@pytest.mark.cpu_test
def test_hf_processor_allows_inactive_truncation():
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    output = processor.info.get_hf_processor()(
        text="Describe the image.",
        truncation=False,
        max_length=1,
        return_tensors=None,
    )

    assert output["input_ids"]


@pytest.mark.cpu_test
def test_hf_processor_rejects_active_truncation():
    ctx = build_model_context(
        "zai-org/cogagent-chat-hf",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    with pytest.raises(ValueError, match="does not currently support"):
        processor.info.get_hf_processor()(
            text="Describe the image.",
            truncation=True,
            max_length=1,
            return_tensors=None,
        )


@pytest.mark.cpu_test
def test_config_preserves_disabled_cache():
    config = CogAgentConfig(use_cache=False)

    restored = CogAgentConfig.from_dict(config.to_dict())

    assert not restored.use_cache


@pytest.mark.cpu_test
def test_model_rejects_tied_word_embeddings_before_initialization():
    vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        quant_config=None,
        model_config=SimpleNamespace(
            hf_config=CogAgentConfig(tie_word_embeddings=True),
            dtype=torch.bfloat16,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )

    with pytest.raises(ValueError, match="does not support tied word embeddings"):
        CogAgentForCausalLM(vllm_config=vllm_config)


@pytest.mark.cpu_test
def test_model_requires_v2_runner():
    with pytest.raises(NotImplementedError, match="requires the V2 model runner"):
        CogAgentForCausalLM(
            vllm_config=SimpleNamespace(use_v2_model_runner=False)
        )


@pytest.mark.cpu_test
def test_dummy_overrides_reduce_vision_layers():
    config = dummy_hf_overrides(CogAgentConfig(), model_arch="CogAgentForCausalLM")

    assert config.vision_config.layers == 1
    assert config.cross_vision_config.layers == 1


@pytest.mark.cpu_test
def test_chat_template_matches_reference_format():
    template_path = get_chat_template_fallback_path("cogagent", "tokenizer")
    assert template_path is not None
    template = _compile_jinja_template(template_path.read_text())
    messages = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Second?"},
    ]

    rendered = template.render(
        messages=messages,
        add_generation_prompt=True,
    )

    assert rendered == " [INST] First? [/INST] One. [INST] Second? [/INST] "
