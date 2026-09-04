# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/zai-org/cogagent-chat-hf/blob/main/modeling_cogagent.py

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.cogagent_processing import (
    CogAgentDummyInputsBuilder,
    CogAgentImagePixelInputs,
    CogAgentMultiModalProcessor,
    CogAgentProcessingInfo,
    get_max_image_tokens,
)
from vllm.model_executor.models.cogagent_vision_encoder import (
    CrossVisionModel,
    EVA2CLIPModel,
    sharded_weight_loader,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsQuant
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

if TYPE_CHECKING:
    from vllm.transformers_utils.configs.cogagent import CogAgentConfig


class CogAgentRMSNorm(nn.Module):
    """RMSNorm matching CogAgent's Hugging Face reference implementation."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def _compute_cogagent_rope_cache(
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    inv_freq = inv_freq.to(dtype)
    positions = torch.arange(
        max_position_embeddings,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


class CogAgentRotaryEmbedding(RotaryEmbedding):
    """RoPE cache matching CogAgent's Hugging Face BF16 construction."""

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # CogAgent constructs inv_freq in FP32, casts the model buffers to the
        # model dtype, and only then creates the frequency table. Reproducing
        # that order matters for BF16 because casting only the final generic
        # vLLM cache produces measurably different frequencies.
        return _compute_cogagent_rope_cache(
            self.rotary_dim,
            self.max_position_embeddings,
            self.base,
            self.dtype,
        )


def build_positions(
    max_model_len: int,
    num_image_tokens: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Build CogAgent position IDs for the configured context length."""

    # we make two assumptions here.
    # 1. There is no text before the image.
    #   - This is enforced via our processor.
    # 2. All prompts had an image during prefill.
    #   - The model errors out on CrossAttention otherwise.

    position_ids = [0, 1]
    position_ids += [2] * num_image_tokens
    position_ids += list(range(3, (max_model_len + 3) - len(position_ids)))
    position_ids = torch.tensor(position_ids, dtype=torch.int64, device=device)

    return position_ids


class MLP(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        self.act_and_mul = get_act_and_mul_fn(config.hidden_act)  # SiLU default

    def forward(self, x):  # HD, HD
        x, _ = self.gate_up_proj(x)
        x = self.act_and_mul(x)
        x, _ = self.down_proj(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        weights_mapper = {
            f"{source}.{suffix}": (source, "gate_up_proj", shard_id)
            for source, shard_id in (("gate_proj", 0), ("up_proj", 1))
            for suffix in ("weight", "qweight", "qzeros", "scales")
        }

        loaded_params = sharded_weight_loader(
            params_dict=params_dict, weights=weights, weights_mapper=weights_mapper
        )

        return loaded_params


class VisionExpertMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.language_mlp = MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.language_mlp",
        )
        self.vision_mlp = MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        language_token_ids: torch.Tensor | None,
        vision_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if vision_token_ids is None:
            output = self.language_mlp(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            output[vision_token_ids] = self.vision_mlp(hidden_states[vision_token_ids])
            output[language_token_ids] = self.language_mlp(
                hidden_states[language_token_ids]
            )

        return output


class VisionExpertAttention(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = CogAgentRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=10000,
            is_neox_style=True,
            dtype=config.dtype,
        )

        self.vision_expert_query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_expert_query_key_value",
        )

        self.vision_expert_dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.vision_expert_dense",
        )

        self.language_expert_query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.language_expert_query_key_value",
        )

        self.language_expert_dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.language_expert_dense",
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.LongTensor,
        hidden_states: torch.Tensor,
        language_token_ids: torch.BoolTensor | None,  # 1D
        vision_token_ids: torch.BoolTensor | None,  # 1D
    ) -> torch.Tensor:
        # we don't expect only image tokens.
        # the bos token, the boi, and the eoi will always be text.
        if language_token_ids is None or vision_token_ids is None:
            mixed_raw_layer, _ = self.language_expert_query_key_value(hidden_states)
        else:
            # expects num_tokens, hidden_size
            mixed_raw_layer = hidden_states.new_zeros(
                hidden_states.shape[-2], hidden_states.shape[-1] * 3
            )
            mixed_raw_layer[vision_token_ids], _ = self.vision_expert_query_key_value(
                hidden_states[vision_token_ids]
            )
            mixed_raw_layer[language_token_ids], _ = (
                self.language_expert_query_key_value(hidden_states[language_token_ids])
            )
        query_states, key_states, value_states = torch.split(
            mixed_raw_layer, self.hidden_size, dim=-1
        )

        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        # context_layer -> [num_tokens, head * head_dim]
        context_layer = self.attn(query_states, key_states, value_states)

        if language_token_ids is None or vision_token_ids is None:
            attn_output, _ = self.language_expert_dense(context_layer)
        else:
            attn_output = torch.zeros_like(hidden_states)
            attn_output[vision_token_ids], _ = self.vision_expert_dense(
                context_layer[vision_token_ids]
            )
            attn_output[language_token_ids], _ = self.language_expert_dense(
                context_layer[language_token_ids]
            )
        return attn_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        loaded_params = set()
        for name, loaded_weight in weights:
            if "inv_freq" in name:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


class CogAgentCrossAttention(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size  # 4096
        self.cross_hidden_size = config.cross_hidden_size
        self.cross_compute_hidden_size = config.cross_compute_hidden_size

        self.num_heads = config.num_attention_heads
        self.cross_head_dim = (
            self.cross_compute_hidden_size // self.num_heads
        )  # default is 32
        self.max_position_embeddings = config.max_position_embeddings

        # query and key_value can have different head sizes,
        # so init them separately.
        self.query = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.cross_compute_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.query",
        )

        self.key_value = QKVParallelLinear(
            hidden_size=self.cross_hidden_size,
            head_size=self.cross_head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_value",
        )

        self.dense = RowParallelLinear(
            input_size=self.cross_compute_hidden_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        self.cross_attn = CrossAttention(
            self.num_heads,
            self.cross_head_dim,
            scale=self.cross_head_dim**-0.5,
            cache_config=cache_config,
            prefix=f"{prefix}.cross_attn",
        )

    def forward(
        self, hidden_states: torch.Tensor, encoder_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        query_states, _ = self.query(hidden_states)

        if encoder_embeds is None:
            key_states = None
            value_states = None
        else:
            encoder_states, _ = self.key_value(encoder_embeds.contiguous())
            key_states, value_states = encoder_states.chunk(2, dim=-1)

        context_layer = self.cross_attn(
            query_states,
            key_states,
            value_states,
        )

        attn_output, _ = self.dense(context_layer)
        return attn_output


class CogAgentDecoderLayer(nn.Module):
    def __init__(
        self,
        config: "CogAgentConfig",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.hidden_size = config.hidden_size

        # NOTE: VisionExpertAttention and CrossAttention can
        # have different hidden sizes. The current implementation
        # relies on padding KV cache block sizes to match.

        self.self_attn = VisionExpertAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.cross_attn = CogAgentCrossAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )
        self.mlp = VisionExpertMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = CogAgentRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = CogAgentRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_cross_attention_layernorm = CogAgentRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.LongTensor,
        hidden_states: torch.Tensor,
        vision_token_ids: torch.Tensor | None,
        language_token_ids: torch.Tensor | None,
        encoder_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            vision_token_ids=vision_token_ids,
            language_token_ids=language_token_ids,
        )

        hidden_states = residual + hidden_states
        cross_input = self.post_cross_attention_layernorm(hidden_states)

        attention_output = self.cross_attn(
            hidden_states=cross_input, encoder_embeds=encoder_embeds
        )

        hidden_states = hidden_states + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(
            mlp_input,
            vision_token_ids=vision_token_ids,
            language_token_ids=language_token_ids,
        )

        hidden_states = mlp_output + hidden_states

        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,
        "inputs_embeds": 0,
        "encoder_embeds": 0,
        "vision_token_ids": 0,
        "language_token_ids": 0,
    }
)
class CogAgentModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hf_config = vllm_config.model_config.hf_config  # type: CogAgentConfig
        self.vocab_size = self.hf_config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hf_config.hidden_size,
            org_num_embeddings=self.vocab_size,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList(
            [
                CogAgentDecoderLayer(
                    self.hf_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(self.hf_config.num_hidden_layers)
            ]
        )

        self.norm = CogAgentRMSNorm(
            self.hf_config.hidden_size,
            eps=self.hf_config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        positions: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        encoder_embeds: torch.Tensor | None,
        vision_token_ids: torch.BoolTensor | torch.LongTensor | None,
        language_token_ids: torch.BoolTensor | torch.LongTensor | None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,  # L, D
                positions=positions,
                vision_token_ids=vision_token_ids,
                language_token_ids=language_token_ids,
                encoder_embeds=encoder_embeds,  # None or B, L, D
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    CogAgentMultiModalProcessor,
    info=CogAgentProcessingInfo,
    dummy_inputs=CogAgentDummyInputsBuilder,
)
class CogAgentForCausalLM(nn.Module, SupportsMultiModal, SupportsQuant):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_suffix={
            "linear_proj.gate_proj.weight": "linear_proj.glu_gate_proj.weight"
        },
        orig_to_new_prefix={
            "model.vision": "vision",
            "model.cross_vision": "cross_vision",
        },
    )
    requires_raw_input_tokens = True
    supports_rocm_cudagraph = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        if not vllm_config.use_v2_model_runner:
            raise NotImplementedError(
                "CogAgentForCausalLM requires the V2 model runner; it is selected "
                "automatically unless VLLM_USE_V2_MODEL_RUNNER=0 is set."
            )

        model_config = vllm_config.model_config
        self.hf_config = model_config.hf_config  # type: CogAgentConfig
        parallel_config = vllm_config.parallel_config

        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("CogAgent currently supports only TP=1 and PP=1")
        if model_config.dtype != torch.bfloat16:
            raise ValueError("CogAgent currently supports only bfloat16 inference")
        if self.hf_config.tie_word_embeddings:
            raise ValueError("CogAgent does not support tied word embeddings")
        if vllm_config.ec_transfer_config is not None:
            raise ValueError("CogAgent does not support encoder cache transfer")
        if model_config.enable_prompt_embeds:
            raise ValueError("CogAgent does not currently support prompt embeddings")
        multimodal_config = model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.enable_mm_embeds:
            raise ValueError(
                "CogAgent does not currently support multimodal embeddings"
            )

        self.dtype = model_config.dtype
        self.image_token_id = self.hf_config.image_token_id

        with self._mark_language_model(vllm_config):
            self.model = CogAgentModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision = EVA2CLIPModel(
                self.hf_config.vision_config,
                prefix=maybe_prefix(prefix, "model.vision"),
            )

            self.cross_vision = CrossVisionModel(
                self.hf_config.cross_vision_config,
                prefix=maybe_prefix(prefix, "model.cross_vision"),
            )

        self.lm_head = ParallelLMHead(
            self.hf_config.vocab_size,
            self.hf_config.hidden_size,
            org_num_embeddings=self.hf_config.vocab_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.hf_config.vocab_size, self.hf_config.vocab_size
        )
        self.configure_mm_token_handling(
            self.hf_config.vocab_size,
            [self.image_token_id],
        )

        self.cross_hidden_size = self.hf_config.cross_hidden_size
        self.num_vision_tokens = get_max_image_tokens(self.hf_config.vision_config)
        self.num_cross_vision_tokens = self.cross_vision.num_tokens

        self.register_buffer(
            "position_ids",
            build_positions(
                model_config.max_model_len,
                self.num_vision_tokens - 2,  # -2 is for BOI/EOI
                device=vllm_config.device_config.device_type,
            ),
            persistent=False,
        )

    def build_token_masks(self, input_ids: torch.LongTensor):
        vision_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        vision_token_mask[:-1] = (input_ids[:-1] == self.image_token_id) & (
            input_ids[1:] == self.image_token_id
        )
        language_token_mask = ~vision_token_mask

        return vision_token_mask, language_token_mask

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            raise ValueError("CogAgent does not currently support inputs_embeds")
        if input_ids is None:
            raise ValueError("CogAgent requires input_ids")
        if intermediate_tensors is not None:
            raise ValueError("CogAgent does not support pipeline parallelism")

        image_embeds = None
        cross_embeds = None
        vision_token_ids = None
        language_token_ids = None

        positions = self.position_ids.index_select(0, positions)

        if encoder_outputs:
            stacked_encoder_outputs = torch.stack(encoder_outputs)
            vision_token_ids, language_token_ids = self.build_token_masks(input_ids)
            image_embeds = stacked_encoder_outputs[:, : self.num_vision_tokens, :]
            cross_embeds = stacked_encoder_outputs[
                :, self.num_vision_tokens :, : self.cross_hidden_size
            ]

        inputs_embeds = self.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=image_embeds,
            is_multimodal=input_ids == self.image_token_id,
        )

        hidden_states = self.model(
            positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_embeds=cross_embeds,
            vision_token_ids=vision_token_ids,
            language_token_ids=language_token_ids,
        )
        return hidden_states

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> CogAgentImagePixelInputs | None:
        images = kwargs.pop("pixel_values", None)
        cross_images = kwargs.pop("cross_pixel_values", None)
        if images is None and cross_images is None:
            return None
        if images is None or cross_images is None:
            raise ValueError(
                "pixel_values and cross_pixel_values must be provided together"
            )

        assert isinstance(images, torch.Tensor)
        assert isinstance(cross_images, torch.Tensor)

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim == 5:
            images = images.squeeze(1)

        if cross_images.ndim == 3:
            cross_images = cross_images.unsqueeze(0)
        if cross_images.ndim == 5:
            cross_images = cross_images.squeeze(1)

        images = images.to(dtype=self.dtype)
        cross_images = cross_images.to(dtype=self.dtype)

        inputs = CogAgentImagePixelInputs(
            type="pixel_values",
            pixel_values=images,
            cross_pixel_values=cross_images,
            resolve_bindings={
                "side": self.hf_config.image_size,
                "cross_side": self.hf_config.cross_image_size,
            },
        )
        return inputs

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> torch.Tensor | None:
        image_inputs = self._parse_and_validate_image_input(**kwargs)

        hidden_size: int = self.hf_config.hidden_size
        sequence_length: int = self.num_vision_tokens + self.num_cross_vision_tokens

        if image_inputs is None:
            return None

        # B, C, H, W -> L, D
        cross_embeds: torch.Tensor = self.cross_vision(image_inputs.cross_pixel_values)
        image_embeds: torch.Tensor = self.vision(image_inputs.pixel_values)

        # Pad both towers to the decoder hidden size so one cache entry can
        # carry the decoder embeddings and cross-attention states together.
        batch = image_embeds.shape[0] if image_embeds.ndim >= 3 else 1
        embed_shape = [batch, sequence_length, hidden_size]

        multimodal_embeddings = image_embeds.new_zeros(*embed_shape)
        multimodal_embeddings[:, : self.num_vision_tokens, :] = image_embeds
        multimodal_embeddings[:, self.num_vision_tokens :, : self.cross_hidden_size] = (
            cross_embeds
        )

        return multimodal_embeddings

    def get_language_model(self):
        return self.model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        skip_prefixes = ["cross_vision.vit.model.rope"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)

        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
