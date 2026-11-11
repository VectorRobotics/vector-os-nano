# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The shared zh/en object vocabulary (perception/vocab.py, from config/worlds/*.yaml)."""
from __future__ import annotations

import pytest

from vector_os_nano.perception.front_object import parse_color
from vector_os_nano.perception.vocab import vocab
from vector_os_nano.skills.perception_grasp import _names_object


@pytest.mark.parametrize("q, phrase", [
    ("绿色的瓶子", "green bottle"), ("green bottle", "green bottle"), ("绿瓶", "green bottle"),
    ("拿起那个罐子", "can"), ("红色的", "red object"), ("香蕉", "banana"),
    ("把红色的杯子拿来", "red cup"), ("抓前面的东西", ""), ("all objects", ""), ("", ""),
    ("拿那个苹果", None),
])
def test_english_phrase(q, phrase) -> None:
    assert vocab().english(q) == phrase


def test_colour_words_match_on_word_boundaries() -> None:
    """perception:parse-color-substring — 'shredded'/'infrared'/'greenhouse' routed to HSV."""
    for q in ("the shredded paper", "infrared camera", "greenhouse", "bored"):
        assert parse_color(q) is None, q
    assert parse_color("grab the red can") == "red"
    assert parse_color("抓红色的东西") == "red"
    assert parse_color("黄色的杯子") == "yellow"   # every manifest colour has a hue band


def test_grasp_routing_and_prompt_share_one_noun_table() -> None:
    """The routing nouns and the prompt nouns can no longer disagree (盘子 was routed
    to the detector, which had no English for it)."""
    for q in ("罐子", "瓶子", "the can", "red bottle", "拿起那个杯子", "banana", "香蕉"):
        assert _names_object(q) and vocab().english(q), q
    for q in ("前面的东西", "抓红色的", "蓝色的", "东西", "object", ""):
        assert not _names_object(q), q
