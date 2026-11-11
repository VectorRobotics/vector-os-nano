# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The scene VLM's reply parsing and request shape (no network)."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from vector_os_nano.perception import vlm_go2
from vector_os_nano.perception.vlm_go2 import Go2VLMPerception, _parse_json_response, _parse_objects


def test_reply_parsing_survives_prose_nulls_words_and_junk():
    """perception:vlm-parse-fragile — 'Here is the JSON: {...}' gave {}; null or
    'high' confidence raised; a non-dict entry or a list root raised."""
    assert _parse_json_response('Here is the JSON: {"room": "kitchen", "x": {"a": 1}} done')["room"] == "kitchen"
    assert _parse_json_response('[{"room": "kitchen"}]') == {}
    assert _parse_json_response("no json at all") == {}
    objs = _parse_objects([{"name": "cup", "confidence": None}, {"name": "can", "confidence": "high"},
                           "a lamp", {"name": "box", "confidence": 1.7}])
    assert [(o.name, o.confidence) for o in objs] == [("cup", 0.0), ("can", 0.0), ("box", 1.0)]
    assert _parse_objects({"name": "x"}) == []


def test_frames_go_out_at_full_640px_not_160(monkeypatch):
    """perception:vlm-160px-gates-discovery — the remote VLM saw 160 px images."""
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)
    vlm = Go2VLMPerception(config={"api_key": "k"})
    jpeg = base64.b64decode(vlm._encode_frame(np.zeros((480, 640, 3), np.uint8)))
    assert Image.open(io.BytesIO(jpeg)).size == (640, 480)


def test_rooms_and_model_come_from_config(monkeypatch):
    """perception:vlm-cost-prompt-drift — the prompt hard-coded one house's rooms and a
    30 cm camera; the cost model priced gemma as GPT-4o."""
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)
    vlm = Go2VLMPerception(config={"api_key": "k", "rooms": ["kitchen", "study"], "model": "m/x"})
    assert "kitchen, study" in vlm._room_prompt and "blue sofa" not in vlm._room_prompt
    assert vlm._model == "m/x" and vlm._price == (0.0, 0.0)
    assert "30cm" not in vlm_go2._DESCRIBE_SCENE_PROMPT


def test_no_key_and_no_local_url_fails_loud(monkeypatch):
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="key"):
        Go2VLMPerception()
