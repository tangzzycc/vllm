# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

import torch

if TYPE_CHECKING:
    from transformers import BatchFeature

    from vllm.config.multimodal import BaseDummyOptions

from vllm.inputs import MultiModalDataDict, MultiModalEncDecInput
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdateDetails,
    TimingContext,
)
from vllm.transformers_utils.configs.cogagent import (
    CogAgentConfig,
    EVACLIPVisionConfig,
    EVALargeVisionConfig,
)
from vllm.transformers_utils.processors.cogagent import CogAgentProcessor
from vllm.utils.tensor_schema import TensorSchema, TensorShape


def get_max_image_tokens(hf_config: EVACLIPVisionConfig | EVALargeVisionConfig) -> int:
    image_size = hf_config.image_size
    patch_size = hf_config.patch_size

    return (image_size // patch_size) ** 2 + 2


class CogAgentImagePixelInputs(TensorSchema):
    """
    images: bn, C, H, W. Images
    cross_images: bn, C, H, W. Resized Images passed to the Large Encoder.
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "side", "side")]
    cross_pixel_values: Annotated[
        torch.Tensor, TensorShape("bn", 3, "cross_side", "cross_side")
    ]


class CogAgentProcessorConfig(TypedDict, total=True):
    image_size: int
    cross_image_size: int
    patch_size: int
    dtype: torch.dtype


class CogAgentProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> CogAgentConfig:
        return self.ctx.get_hf_config(CogAgentConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        # Cache one combined output from the large and small vision encoders.
        # Only the small-encoder tokens are inserted into the decoder prompt.

        num_tokens = get_max_image_tokens(self.get_hf_config().vision_config)
        num_tokens += get_max_image_tokens(self.get_hf_config().cross_vision_config) - 2

        return {"image": num_tokens}

    def get_hf_processor(self, **kwargs):
        hf_config = self.get_hf_config()

        defaults = CogAgentProcessorConfig(
            image_size=hf_config.image_size,
            cross_image_size=hf_config.cross_image_size,
            patch_size=hf_config.vision_config.patch_size,
            dtype=self.ctx.model_config.dtype,
        )
        for name, expected in defaults.items():
            if name in kwargs and kwargs[name] != expected:
                raise ValueError(
                    f"CogAgent processor argument {name!r} is fixed by the model "
                    f"configuration and must be {expected!r}."
                )
        kwargs = {**defaults, **kwargs}

        return self.ctx.init_processor(
            CogAgentProcessor,
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


class CogAgentDummyInputsBuilder(BaseDummyInputsBuilder[CogAgentProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        num_images = mm_counts.get("image", 0)

        return " " * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        cfg = self.info.get_hf_config()
        image_size = cfg.image_size

        mm_data = {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        }

        return mm_data


class CogAgentMultiModalProcessor(EncDecMultiModalProcessor[CogAgentProcessingInfo]):
    """Build CogAgent's cross-attention encoder and multimodal decoder inputs."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        # Mark CogAgentProcessor as a composite processor so the base class
        # processes text and images together instead of calling its internal
        # torchvision image transform through the MM-only fast path.
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def get_encoder_output_seq_len(
        self,
        modality: str,
        mm_position: PlaceholderRange,
    ) -> int:
        config = self.info.get_hf_config()
        return get_max_image_tokens(
            config.vision_config
        ) + self.get_cross_attention_seq_len(
            modality,
            mm_position,
        )

    def get_cross_attention_seq_len(
        self,
        modality: str,
        mm_position: PlaceholderRange,
    ) -> int:
        return get_max_image_tokens(self.info.get_hf_config().cross_vision_config) - 2

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        bos_token_id = self.info.get_tokenizer().bos_token_id
        if bos_token_id is None:
            raise ValueError("CogAgent requires a tokenizer with a BOS token")
        if prompt_tokens and prompt_tokens[0] == bos_token_id:
            return prompt_tokens
        return [bos_token_id, *prompt_tokens]

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> list[int]:
        config = self.info.get_hf_config()
        num_encoder_tokens = get_max_image_tokens(config.cross_vision_config) - 2

        # For similar reasons as Whisper, we ignore this prompt.
        return [0] * num_encoder_tokens

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()  # type: LlamaTokenizer

        image_token_id = self.info.get_hf_config().image_token_id
        bos_token_id = tokenizer.bos_token_id
        assert bos_token_id is not None

        num_image_tokens = get_max_image_tokens(self.info.get_hf_config().vision_config)
        image_tokens = [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[bos_token_id],
                replacement=PromptUpdateDetails.select_token_id(
                    [bos_token_id, *image_tokens],
                    embed_token_id=image_token_id,
                ),
            )
        ]

    def _get_mm_fields_config(
        self, hf_inputs: "BatchFeature", hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "cross_pixel_values": MultiModalFieldConfig.batched("image"),
        }

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalEncDecInput:
        """Build the fixed encoder prompt and processed decoder prompt."""
        image_count = inputs.mm_data_items.get_all_counts().get("image", 0)
        if image_count != 1:
            raise ValueError(
                "CogAgent currently requires exactly one image per request, "
                f"but received {image_count}."
            )

        encoder_prompt = self.create_encoder_prompt(
            inputs.prompt,
            inputs.mm_data_items,
        )

        # CogAgent applies multimodal replacements to the decoder prompt.
        mm_inputs = BaseMultiModalProcessor.apply(self, inputs, timing_ctx)

        # Skip over _get_enc_dec_inputs as result is fixed
        mm_inputs = MultiModalEncDecInput(
            encoder_prompt_token_ids=encoder_prompt,
            **mm_inputs,
        )

        return mm_inputs
