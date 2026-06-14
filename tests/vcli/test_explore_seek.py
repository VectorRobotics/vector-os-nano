# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R11 — ExploreAndSeekSkill: the grand VLN capstone.

One command ('探索房间找到红色物体并走过去') = autonomously explore to discover
the area, THEN visually find & approach the named colour target — composing the
proven ExploreSkill + VisionSeekSkill deterministically (no reliance on the LLM
splitting the goal correctly). Tested by stubbing the two sub-skills.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.explore_seek import ExploreAndSeekSkill


def _ctx():
    return SimpleNamespace(base=object(), world_model=None, agent=None)


class TestExploreAndSeek:
    def test_explores_then_seeks(self, monkeypatch):
        calls = []
        import vector_os_nano.skills.explore_seek as es

        def fake_explore(self, params, ctx):
            calls.append(("explore", params))
            return SkillResult(success=True, result_data={"coverage": 0.7})

        def fake_seek(self, params, ctx):
            calls.append(("seek", params))
            return SkillResult(success=True, result_data={"reason": "arrived"})

        monkeypatch.setattr(es.ExploreSkill, "execute", fake_explore)
        monkeypatch.setattr(es.VisionSeekSkill, "execute", fake_seek)
        res = ExploreAndSeekSkill().execute({"label": "red"}, _ctx())
        assert res.success is True
        # explore ran before seek, and the colour propagated to seek
        assert [c[0] for c in calls] == ["explore", "seek"]
        assert calls[1][1]["label"] == "red"

    def test_seek_failure_fails_overall(self, monkeypatch):
        import vector_os_nano.skills.explore_seek as es
        monkeypatch.setattr(es.ExploreSkill, "execute",
                            lambda self, p, c: SkillResult(success=True, result_data={}))
        monkeypatch.setattr(es.VisionSeekSkill, "execute",
                            lambda self, p, c: SkillResult(
                                success=False, error_message="could not find the red object by camera",
                                result_data={"seen": False}))
        res = ExploreAndSeekSkill().execute({"label": "red"}, _ctx())
        assert res.success is False
        assert "red" in (res.error_message or "")
