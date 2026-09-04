# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/zai-org/cogagent-chat-hf/blob/main/modeling_cogagent.py

from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import (
    BatchFeature,
    LlamaTokenizer,
    LlamaTokenizerFast,
    ProcessorMixin,
)

ImageData: TypeAlias = (
    list[Image.Image | np.ndarray | torch.Tensor]
    | np.ndarray
    | torch.Tensor
    | Image.Image
)


def compose(
    dtype: torch.dtype,
    size: int | tuple[int, int],
) -> transforms.Compose:
    if isinstance(size, int):
        size = (size, size)

    return transforms.Compose(
        [
            transforms.Resize(
                size=size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
            transforms.ConvertImageDtype(dtype),
        ]
    )


class CogAgentProcessor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]
    valid_kwargs = [
        "cross_image_size",
        "dtype",
        "image_size",
        "patch_size",
    ]

    def __init__(
        self,
        tokenizer: LlamaTokenizer | LlamaTokenizerFast,
        cross_image_size: int = 1120,
        dtype: torch.dtype = torch.bfloat16,
        image_size: int = 224,
        patch_size: int = 14,
    ) -> None:
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size "
                f"({patch_size})"
            )

        self.cross_image_size = cross_image_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokenizer = tokenizer
        self.image_processor = compose(dtype, image_size)
        self.cross_image_processor = compose(dtype, cross_image_size)

        super().__init__(tokenizer)

    def _process_token_ids(
        self,
        token_ids: Sequence[int] | Sequence[Sequence[int]],
        *,
        return_tensors: str | None,
    ) -> list[list[int]] | torch.Tensor | np.ndarray:
        if not token_ids:
            raise ValueError("text input must not be empty")

        if isinstance(token_ids[0], int):
            rows = [list(cast(Sequence[int], token_ids))]
        else:
            rows = [list(row) for row in cast(Sequence[Sequence[int]], token_ids)]

        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("CogAgent requires a tokenizer with a BOS token")

        rows = [
            row if row and row[0] == bos_token_id else [bos_token_id, *row]
            for row in rows
        ]
        if return_tensors is None:
            return rows

        max_len = max(map(len, rows))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("CogAgent requires a tokenizer with a padding token")
        input_ids = torch.full(
            (len(rows), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        for index, row in enumerate(rows):
            input_ids[index, -len(row) :] = torch.tensor(row, dtype=torch.long)

        if return_tensors == "pt":
            return input_ids
        if return_tensors == "np":
            return input_ids.numpy()
        raise ValueError(f"Unsupported tensor type: {return_tensors}")

    def _process_text(
        self,
        text: str | Sequence[str] | Sequence[int] | Sequence[Sequence[int]],
        *,
        return_tensors: str | None,
        tokenizer_kwargs: dict[str, object],
    ) -> list[list[int]] | torch.Tensor | np.ndarray:
        if isinstance(text, str):
            token_ids: Sequence[Sequence[int]] = [
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )
            ]
        elif text and isinstance(text[0], str):
            token_ids = [
                self.tokenizer.encode(
                    item,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )
                for item in cast(Sequence[str], text)
            ]
        else:
            token_ids = cast(Sequence[int] | Sequence[Sequence[int]], text)

        return self._process_token_ids(
            token_ids,
            return_tensors=return_tensors,
        )

    @staticmethod
    def _to_pil_image(image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()

        if image.ndim != 3:
            raise ValueError(
                "CogAgent image arrays must have 3 dimensions, "
                f"but received shape {tuple(image.shape)}."
            )

        channel_dims = (1, 3, 4)
        channel_first = image.shape[0] in channel_dims
        channel_last = image.shape[-1] in channel_dims
        if channel_first and channel_last:
            raise ValueError(
                "CogAgent cannot infer whether an image is HWC or CHW when "
                f"both edge dimensions look like channels: {tuple(image.shape)}."
            )
        if channel_last:
            if isinstance(image, torch.Tensor):
                image = image.permute(2, 0, 1)
        elif not channel_first:
            raise ValueError(
                "CogAgent image arrays must be HWC or CHW with 1, 3, "
                f"or 4 channels, but received shape {tuple(image.shape)}."
            )
        elif isinstance(image, np.ndarray):
            image = np.transpose(image, (1, 2, 0))

        if isinstance(image, torch.Tensor) and image.dtype == torch.bfloat16:
            image = image.float()

        return transforms.functional.to_pil_image(image).convert("RGB")

    @classmethod
    def _normalize_images(cls, images: ImageData) -> list[Image.Image]:
        if isinstance(images, Image.Image):
            image_items: Sequence[Image.Image | np.ndarray | torch.Tensor] = [images]
        elif isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                image_items = [images]
            elif images.ndim == 4:
                image_items = list(images)
            else:
                raise ValueError(
                    "CogAgent images must have 3 or 4 dimensions, "
                    f"but received shape {tuple(images.shape)}."
                )
        else:
            image_items = cast(
                Sequence[Image.Image | np.ndarray | torch.Tensor], images
            )

        if len(image_items) == 0:
            raise ValueError("images must not be empty")

        return [cls._to_pil_image(image) for image in image_items]

    def _process_images(
        self,
        images: ImageData,
    ) -> dict[str, torch.Tensor]:
        image_items = self._normalize_images(images)
        return {
            "pixel_values": torch.stack(
                [self.image_processor(image) for image in image_items]
            ),
            "cross_pixel_values": torch.stack(
                [self.cross_image_processor(image) for image in image_items]
            ),
        }

    def __call__(
        self,
        text: str
        | Sequence[str]
        | Sequence[int]
        | Sequence[Sequence[int]]
        | None = None,
        images: ImageData | None = None,
        **kwargs,
    ) -> BatchFeature:
        return_tensors = kwargs.pop("return_tensors", "pt")
        for key in self.valid_kwargs:
            kwargs.pop(key, None)

        # CogAgent always needs a leading BOS token. The tokenizer option is
        # consumed for API compatibility; _process_token_ids inserts BOS when
        # it is missing without duplicating an existing leading BOS.
        kwargs.pop("add_special_tokens", None)

        truncation = kwargs.pop("truncation", None)
        kwargs.pop("max_length", None)
        if truncation not in (None, False, "do_not_truncate"):
            raise ValueError(
                "CogAgent does not currently support prompt truncation because "
                "its image placeholders are inserted after tokenization"
            )
        if kwargs:
            raise TypeError(f"Unexpected processor arguments: {sorted(kwargs)}")

        inputs = {}
        if text is not None:
            inputs["input_ids"] = self._process_text(
                text,
                return_tensors=return_tensors,
                tokenizer_kwargs={},
            )
        if images is not None:
            inputs.update(self._process_images(images))
        if not inputs:
            raise ValueError("At least one of text or images must be provided")

        return BatchFeature(inputs, tensor_type=return_tensors)
