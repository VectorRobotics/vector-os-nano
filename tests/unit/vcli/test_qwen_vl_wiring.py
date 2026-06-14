# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""DQ-6 — Qwen3-VL vision backbone via OpenRouter (owner directive) +
visited() world-model-object fallback (R8 GUI-test regression).

The VLM backbone must be a REAL vision model (Qwen3-VL on OpenRouter, the
owner's key) — never a weak-vision text model; the habitat agent gets the
same visual-verification capability as go2 (camera frame + _vlm); and
visited('<label>') falls back to world-model object proximity so the LLM's
natural phrasing ('到过沙发') is a valid predicate instead of fail-safe
False on non-room labels.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.core.world_model import ObjectState, WorldModel


class TestVlmBackbone:
    def test_default_model_is_qwen3_vl(self, monkeypatch):
        monkeypatch.delenv("VECTOR_VLM_MODEL_OPENROUTER", raising=False)
        import importlib

        import vector_os_nano.perception.vlm_go2 as v

        importlib.reload(v)
        assert "vl-72b" in v._MODEL

    def test_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("VECTOR_VLM_MODEL_OPENROUTER", "qwen/qwen3-vl-30b-a3b-instruct")
        import importlib

        import vector_os_nano.perception.vlm_go2 as v

        importlib.reload(v)
        assert v._MODEL == "qwen/qwen3-vl-30b-a3b-instruct"
        monkeypatch.delenv("VECTOR_VLM_MODEL_OPENROUTER", raising=False)
        importlib.reload(v)


class TestVisitedObjectFallback:
    def _visited(self, agent):
        from vector_os_nano.vcli.worlds.go2_sim_oracle import make_visited

        rooms = {"kitchen": (-3.6, 1.0, 0.2, 2.8)}
        return make_visited(agent, rooms)

    def _agent(self, pos, objs=()):
        wm = WorldModel()
        for o in objs:
            wm.add_object(o)
        base = SimpleNamespace(get_position=lambda: list(pos))
        return SimpleNamespace(_base=base, _world_model=wm)

    def test_room_semantics_unchanged(self):
        v = self._visited(self._agent([-1.7, 1.9, 0.0]))
        assert v("kitchen") is True

    def test_object_label_near_is_visited(self):
        sofa = ObjectState(object_id="sysnav_1", label="sofa", x=1.0, y=2.0)
        v = self._visited(self._agent([1.5, 2.0, 0.0], [sofa]))
        assert v("sofa") is True

    def test_object_label_far_is_not(self):
        sofa = ObjectState(object_id="sysnav_1", label="sofa", x=1.0, y=2.0)
        v = self._visited(self._agent([9.0, 9.0, 0.0], [sofa]))
        assert v("sofa") is False

    def test_unknown_label_fails_safe(self):
        v = self._visited(self._agent([0.0, 0.0, 0.0]))
        assert v("swimming_pool") is False


class TestHabitatVisualVerifyWiring:
    def test_habitat_base_exposes_camera_frame(self):
        from vector_os_nano.playground.habitat.base import HabitatBase

        assert callable(getattr(HabitatBase, "get_camera_frame", None))
