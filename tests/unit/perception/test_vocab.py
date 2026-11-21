# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The shared zh/en object vocabulary (perception/vocab.py, from worlds/*/world.yaml)."""
from __future__ import annotations

import pytest

from vector_os_nano.perception.front_object import parse_color
from vector_os_nano.perception.vocab import is_reference, vocab
from vector_os_nano.schemas.world import WORLDS_DIR, load_world


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


def test_every_name_of_every_world_object_has_an_english_phrase() -> None:
    """A query the grasp hands the detector can always be put into English (盘子 was
    routed to the detector, which had no English for it)."""
    for w in sorted(p.parent.name for p in WORLDS_DIR.glob("*/world.yaml")):
        for pk in load_world(w).pickables.values():
            for name in (*pk.aliases, pk.english):
                assert vocab().english(name), (w, name)


def test_specific_package_names_do_not_collapse_to_generic_categories(monkeypatch):
    from pathlib import Path
    monkeypatch.setenv('VECTOR_HOME', str(Path(__file__).resolve().parents[3]/'examples/household'))
    for query, expected in [('tissue refill', 'tissue refill'), ('纸巾补充装', 'tissue refill'),
                            ('zip bag box', 'zip bag box'), ('一盒密封袋', 'zip bag box'),
                            ('pick up the cardboard box on the table', 'cardboard box'),
                            ('green sponge pack', 'green sponge pack')]:
        assert vocab().english(query) == expected


@pytest.mark.parametrize("text, ok", [
    ("那个罐子", True), ("香蕉", True), ("红色的", True), ("红色的苹果", False), ("白色的瓶子", False),
    ("抓红色的罐子", False), ("", False),
])
def test_a_reference_is_one_in_the_words_of_some_world(text, ok) -> None:
    assert is_reference(text) is ok
