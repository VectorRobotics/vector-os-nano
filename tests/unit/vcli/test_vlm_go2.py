# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #11 track-debt R10 — _call_vlm HTTP-status classification.

A 429 (rate limited) must surface as a TYPED ``VlmRateLimitError`` on the FIRST
attempt (never retried, never swallowed into the generic "after N attempts"
RuntimeError) so recognize_navigate can give up honestly instead of mistaking a
throttle for "object absent". Other 4xx stay plain RuntimeError. Deterministic:
httpx.Client.post is monkeypatched — no network.
"""
from __future__ import annotations

import httpx
import numpy as np
import pytest

# Resolve Go2VLMPerception / VlmRateLimitError LIVE inside each test (not at
# module import): an earlier test (test_qwen_vl_wiring) importlib.reloads
# vlm_go2, rebinding the class — a collection-time binding would go stale and
# pytest.raises would miss the (live) exception _call_vlm raises.
def _live():
    import vector_os_nano.perception.vlm_go2 as v
    return v


def _frame():
    return np.zeros((32, 32, 3), dtype=np.uint8)


class _Resp:
    def __init__(self, status_code, text="err"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):           # only reached for 2xx/non-4xx
        raise AssertionError("raise_for_status should not run for a 4xx test")

    def json(self):
        raise AssertionError("json() should not run for a 4xx test")


def _patch_post(monkeypatch, status_code):
    calls = {"n": 0}

    def fake_post(self, url, json=None, headers=None):  # noqa: ARG001
        calls["n"] += 1
        return _Resp(status_code, text="rate limited" if status_code == 429 else "bad")

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return calls


def test_call_vlm_429_raises_typed_not_generic(monkeypatch):
    v = _live()
    calls = _patch_post(monkeypatch, 429)
    vlm = v.Go2VLMPerception(config={"api_key": "x"})
    with pytest.raises(v.VlmRateLimitError):
        vlm._call_vlm(_frame(), "prompt")
    # propagates on the FIRST attempt — NOT retried, NOT swallowed into the
    # "VLM API failed after N attempts" wrapper.
    assert calls["n"] == 1


def test_call_vlm_400_raises_plain_runtimeerror_not_rate_limit(monkeypatch):
    v = _live()
    calls = _patch_post(monkeypatch, 400)
    vlm = v.Go2VLMPerception(config={"api_key": "x"})
    with pytest.raises(RuntimeError) as ei:
        vlm._call_vlm(_frame(), "prompt")
    assert not isinstance(ei.value, v.VlmRateLimitError)   # non-429 4xx unchanged
    assert calls["n"] == 1
