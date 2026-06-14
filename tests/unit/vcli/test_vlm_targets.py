# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R1 — VLM semantic target detection (pure, offline, mocked VLM).

VlmTargetDetector wraps a real VLM (Qwen-VL via OpenRouter in production) and
adapts its grounding output to the SAME detection contract as
color_targets.detect_targets — ``[{label, x_norm, y_norm, area_frac}]`` — so the
vision-seek state machine drives toward a SEMANTIC class (chair/plant/table),
not a colour. These tests inject a fake vlm_call (no network, deterministic):
the real call is exercised only in the GUI acceptance (rule 5 — VLM really
called, never peeking ground truth).
"""
from __future__ import annotations

import numpy as np

from vector_os_nano.perception.vlm_targets import VlmTargetDetector


def _frame(h=64, w=128):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _caller(payload: str):
    """A fake vlm_call(frame, prompt) -> str returning canned JSON."""
    def call(frame, prompt):       # noqa: ARG001 — signature match
        return payload
    return call


class TestContract:
    def test_box_centre_and_area_to_contract(self):
        # a chair box spanning normalised [0.4,0.5]..[0.6,0.9] (x0,y0,x1,y1)
        payload = (
            '{"objects": [{"label": "chair", '
            '"box": [0.4, 0.5, 0.6, 0.9]}]}'
        )
        det = VlmTargetDetector(vlm_call=_caller(payload))
        dets = det.detect_targets(_frame(), query="chair")
        assert len(dets) == 1
        d = dets[0]
        assert d["label"] == "chair"
        # centre x = 0.5 -> x_norm 0.0; centre y = 0.7 -> y_norm 0.4
        assert abs(d["x_norm"] - 0.0) < 1e-6
        assert abs(d["y_norm"] - 0.4) < 1e-6
        # area = 0.2 * 0.4 = 0.08
        assert abs(d["area_frac"] - 0.08) < 1e-6

    def test_left_object_negative_x(self):
        payload = '{"objects": [{"label": "plant", "box": [0.0, 0.2, 0.2, 0.6]}]}'
        det = VlmTargetDetector(vlm_call=_caller(payload))
        d = det.detect_targets(_frame(), query="plant")[0]
        assert d["x_norm"] < 0      # left half of the frame

    def test_no_objects_returns_empty(self):
        det = VlmTargetDetector(vlm_call=_caller('{"objects": []}'))
        assert det.detect_targets(_frame(), query="chair") == []

    def test_malformed_json_returns_empty_not_crash(self):
        det = VlmTargetDetector(vlm_call=_caller("the model said hi, no json"))
        assert det.detect_targets(_frame(), query="chair") == []

    def test_thousand_scale_autodetected(self):
        # Qwen often emits [0,1000] pixel-ish coords — must be normalised.
        payload = '{"objects": [{"label": "table", "box": [400, 500, 600, 900]}]}'
        det = VlmTargetDetector(vlm_call=_caller(payload))
        d = det.detect_targets(_frame(), query="table")[0]
        assert abs(d["x_norm"] - 0.0) < 1e-6
        assert abs(d["area_frac"] - 0.08) < 1e-6

    def test_query_filters_to_matching_label(self):
        payload = (
            '{"objects": [{"label": "sofa", "box": [0.1, 0.1, 0.2, 0.2]}, '
            '{"label": "chair", "box": [0.4, 0.5, 0.6, 0.9]}]}'
        )
        det = VlmTargetDetector(vlm_call=_caller(payload))
        dets = det.detect_targets(_frame(), query="chair")
        assert [d["label"] for d in dets] == ["chair"]

    def test_min_area_filters_tiny_box(self):
        payload = '{"objects": [{"label": "cup", "box": [0.50, 0.50, 0.51, 0.51]}]}'
        det = VlmTargetDetector(vlm_call=_caller(payload))
        # default min area rejects a sub-promille speck
        assert det.detect_targets(_frame(), query="cup") == []
        assert det.detect_targets(_frame(), query="cup", min_area_frac=0.0)


class TestRobustParsing:
    """Real Qwen-VL responses observed in the campaign #9 R1 GUI run: ```json
    fences (some unclosed) and trailing garbage after a valid object. The detector
    must still recover the box, not silently drop the frame."""

    def test_json_fence_recovered(self):
        payload = '```json\n{"objects": [{"label": "chair", "box": [0.4,0.5,0.6,0.9]}]}\n```'
        det = VlmTargetDetector(vlm_call=_caller(payload))
        assert det.detect_targets(_frame(), query="chair")

    def test_unclosed_fence_recovered(self):
        payload = '```json\n{"objects": [{"label": "chair", "box": [0.4,0.5,0.6,0.9]}]}'
        det = VlmTargetDetector(vlm_call=_caller(payload))
        assert det.detect_targets(_frame(), query="chair")

    def test_trailing_garbage_recovered(self):
        # the exact malformed shape Qwen emitted in the GUI run
        payload = '{"objects": [{"label": "chair", "box": [0.59,0.61,1.0,1.0]}]}\']'
        det = VlmTargetDetector(vlm_call=_caller(payload))
        assert det.detect_targets(_frame(), query="chair")


class TestHonesty:
    def test_call_is_actually_invoked(self):
        seen = {}

        def call(frame, prompt):
            seen["called"] = True
            seen["query_in_prompt"] = "chair" in prompt
            return '{"objects": []}'

        det = VlmTargetDetector(vlm_call=call)
        det.detect_targets(_frame(), query="chair")
        assert seen.get("called") is True
        assert seen.get("query_in_prompt") is True
