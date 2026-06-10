# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Stage 3 — referring-expression grounding via the live object list.

The decomposer can only resolve a referring expression ("拿红色的杯子" /
"the red cup") to a REAL scene name if `world_context` carries the live
object set. The kernel stays deterministic: it surfaces names (+ attribute
hints) from the world model / sim oracle; the LLM — the language component —
binds the user's words to one of the listed names at plan/replan time, and
the fail-loud + replan loop re-binds with a fresh list when a binding misses.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.vcli.engine import VectorEngine


def _engine_with_agent(agent: Any) -> VectorEngine:
    eng = VectorEngine.__new__(VectorEngine)  # context builder needs no backend
    eng._vgg_agent = agent
    eng._world_context_cache = None
    eng._world_context_ts = 0.0
    eng._world_context_ttl = 0.0  # no caching in tests
    return eng


def _wm(*objs: ObjectState) -> WorldModel:
    wm = WorldModel()
    for o in objs:
        wm.add_object(o)
    return wm


class TestLiveObjectsInWorldContext:
    def test_world_model_objects_listed_with_attributes(self) -> None:
        agent = SimpleNamespace(
            _base=None,
            _spatial_memory=None,
            _world_model=_wm(
                ObjectState("cup_red_0", "cup_red", properties={"color": "red"}),
                ObjectState("cup_blue_0", "cup_blue"),
            ),
        )
        ctx = _engine_with_agent(agent)._build_world_context(force=True)
        assert "Objects (live):" in ctx
        assert "cup_red" in ctx and "cup_blue" in ctx
        assert "color=red" in ctx  # attribute hint surfaces for resolution

    def test_sim_oracle_fallback_when_world_model_empty(self) -> None:
        arm = SimpleNamespace(
            get_object_positions=lambda: {"banana_0": (0.2, 0.1, 0.03)}
        )
        agent = SimpleNamespace(
            _base=None, _spatial_memory=None, _world_model=_wm(), _arm=arm
        )
        ctx = _engine_with_agent(agent)._build_world_context(force=True)
        assert "Objects (live):" in ctx
        assert "banana_0" in ctx

    def test_no_objects_no_line(self) -> None:
        agent = SimpleNamespace(_base=None, _spatial_memory=None, _world_model=_wm())
        ctx = _engine_with_agent(agent)._build_world_context(force=True)
        assert "Objects (live):" not in ctx

    def test_object_list_is_capped(self) -> None:
        objs = [
            ObjectState(f"obj_{i}", f"obj_{i}") for i in range(40)
        ]
        agent = SimpleNamespace(
            _base=None, _spatial_memory=None, _world_model=_wm(*objs)
        )
        ctx = _engine_with_agent(agent)._build_world_context(force=True)
        line = next(l for l in ctx.splitlines() if l.startswith("Objects (live):"))
        assert line.count("obj_") <= 20
        assert "(+20 more)" in line

    def test_oracle_failure_is_silent(self) -> None:
        def _boom() -> dict:
            raise RuntimeError("sim down")

        agent = SimpleNamespace(
            _base=None, _spatial_memory=None, _world_model=_wm(),
            _arm=SimpleNamespace(get_object_positions=_boom),
        )
        ctx = _engine_with_agent(agent)._build_world_context(force=True)
        assert "Objects (live):" not in ctx  # fail-safe, no raise


class TestBindingGuidanceTeachesResolution:
    def test_guidance_mentions_live_object_resolution(self) -> None:
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            _TARGET_BINDING_GUIDANCE,
        )

        assert "Objects (live)" in _TARGET_BINDING_GUIDANCE
        assert "EXACT" in _TARGET_BINDING_GUIDANCE

    def test_guidance_dropped_stale_detect_example(self) -> None:
        """The old text taught len(detect_objects()) > 0 — superseded by the
        step_output() form (backlog #3)."""
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            _TARGET_BINDING_GUIDANCE,
        )

        assert "len(detect_objects()) > 0" not in _TARGET_BINDING_GUIDANCE
