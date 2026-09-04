# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange


@pytest.mark.parametrize(
    "is_embed,expected",
    [
        (None, 5),
        (torch.tensor([True, True, True, True, True]), 5),
        (torch.tensor([False, False, False, False, False]), 0),
        (torch.tensor([True, False, True, False, True]), 3),
        (torch.tensor([True]), 1),
    ],
)
def test_placeholder_range_get_num_embeds(is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)
    assert pr.get_num_embeds() == expected


@pytest.mark.parametrize(
    "is_embed,expected",
    [
        (None, None),
        (torch.tensor([False, True, False, True, True]), [0, 1, 1, 2, 3]),
        (torch.tensor([True, True, True]), [1, 2, 3]),
    ],
)
def test_placeholder_range_embeds_cumsum(is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)

    if expected is None:
        assert pr.embeds_cumsum is None
        return

    assert pr.embeds_cumsum == expected
    # cached_property should return the same object on repeated access
    assert pr.embeds_cumsum is pr.embeds_cumsum


@pytest.mark.parametrize(
    (
        "encoder_output_seq_len",
        "cross_attention_seq_len",
        "expected_encoder",
        "expected_cross_attention",
    ),
    [
        (None, None, 3, 3),
        (7, None, 7, 7),
        (7, 5, 7, 5),
        (None, 2, 3, 2),
    ],
)
def test_feature_encoder_seq_lens(
    encoder_output_seq_len,
    cross_attention_seq_len,
    expected_encoder,
    expected_cross_attention,
):
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="test",
        mm_position=PlaceholderRange(offset=0, length=3),
        encoder_output_seq_len=encoder_output_seq_len,
        cross_attention_seq_len=cross_attention_seq_len,
    )

    assert feature.get_num_encoder_output_tokens() == expected_encoder
    assert feature.get_num_cross_attention_tokens() == expected_cross_attention
