# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundingDinoDetector — query->prompt mapping (pure) and label filtering.

The real weights on a rendered simulator frame are exercised end to end by
tests/e2e_repl/test_e11_perception_memory.py (look localizes every object within 5 cm).
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.perception.grounding_dino import (
    GroundingDinoDetector,
    UngroundableQuery,
    get_shared_detector,
    query_to_prompt,
)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


# --- pure prompt mapping (no torch) -----------------------------------------


def test_zh_noun_maps_to_english_phrase():
    assert query_to_prompt("拿起那个罐子") == "a can."
    assert query_to_prompt("抓瓶子") == "a bottle."
    assert query_to_prompt("杯子") == "a cup."


def test_zh_colour_plus_noun_preserves_colour():
    """D48 caveat 1: grounding-dino is colour-conditioned, so the zh colour
    adjective is PRESERVED into the prompt (not dropped to a bare noun)."""
    assert query_to_prompt("绿色的瓶子") == "a green bottle."
    assert query_to_prompt("红色的罐子") == "a red can."
    assert query_to_prompt("蓝色的瓶子") == "a blue bottle."
    # bare zh colour char + noun also maps
    assert query_to_prompt("红罐子") == "a red can."
    assert query_to_prompt("蓝瓶") == "a blue bottle."


def test_zh_colour_without_noun_builds_colour_object():
    """A colour-only zh query (skill routes these to HSV, not here) is colour-aware
    if ever handed to the detector directly — colour never dropped."""
    assert query_to_prompt("红色的") == "a red object."
    assert query_to_prompt("绿色") == "a green object."


def test_english_query_becomes_lowercase_period_phrase():
    assert query_to_prompt("RED CUP") == "a red cup."
    # punctuation stripped, trailing period guaranteed
    assert query_to_prompt("bottle!") == "a bottle."


def test_articles_and_request_verbs_are_dropped():
    """perception:prompt-articles-verbs — 'a can' was 'a a can.', a request kept its verb."""
    assert query_to_prompt("the can") == "a can."
    assert query_to_prompt("a can") == "a can."
    assert query_to_prompt("pick up the bottle") == "a bottle."
    assert query_to_prompt("pick up the red can on the table") == "a red can."
    assert query_to_prompt("apple") == "a apple."  # an unknown English noun is kept


def test_untranslatable_cjk_fails_loud_instead_of_unk_tokens():
    """perception:zh-query-unk-prompt — '拿那个苹果' became 'a 拿那个苹果.' ([UNK] boxes)."""
    with pytest.raises(UngroundableQuery, match="English"):
        query_to_prompt("拿那个苹果")
    with pytest.raises(UngroundableQuery):
        query_to_prompt("拿那个盘子")
    # every prompt is ASCII: no CJK ever reaches the English-only text encoder
    for q in ("绿色的瓶子", "抓前面的东西", "红色的", "香蕉", "把红色的杯子拿来"):
        assert query_to_prompt(q).isascii(), q


def test_unk_labels_are_dropped():
    """A box the model labels with [UNK] text tokens is never returned."""
    import types

    det = GroundingDinoDetector()
    det._model = lambda **_kw: None
    det._device = "cpu"
    det._torch = types.SimpleNamespace(no_grad=__import__("contextlib").nullcontext)

    class _Proc:
        def __call__(self, **_kw):
            return types.SimpleNamespace(to=lambda _d: {"input_ids": None})

        def post_process_grounded_object_detection(self, *_a, **_kw):
            import torch

            return [{"boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0], [2.0, 2.0, 8.0, 8.0]]),
                     "scores": torch.tensor([0.6, 0.5]),
                     "text_labels": ["a [UNK] [UNK]", "a green bottle"]}]

    det._processor = _Proc()
    if not _torch_available():
        pytest.skip("torch absent")
    out = det.detect(np.zeros((10, 10, 3), np.uint8), "green bottle")
    assert [d.label for d in out] == ["a green bottle"]


def test_empty_or_generic_query_falls_back_to_generic_prompt():
    assert query_to_prompt("") == "a can. a bottle. a cylinder."
    assert query_to_prompt(None) == "a can. a bottle. a cylinder."


def test_shared_singleton_is_same_instance():
    a = get_shared_detector()
    b = get_shared_detector()
    assert a is b
    assert isinstance(a, GroundingDinoDetector)


def test_detect_rejects_non_image_without_loading_model():
    # ValueError is raised BEFORE _ensure_loaded (no torch needed).
    det = GroundingDinoDetector()
    with pytest.raises(ValueError):
        det.detect(None, "a can.")  # type: ignore[arg-type]
    assert det._model is None  # model was never loaded
