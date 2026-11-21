# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Honesty attacks on the skills (the R2 verification round's extras 3 and 4, moved
into the suite by REF1): nothing opens a holding gripper, a failing pick keeps the held
object, coordinate navigation succeeds only on arrival, a named grasp without its
detector grasps nothing, and an empty target picks nothing."""
from __future__ import annotations

import threading

import pytest

from tests.unit.test_execute_skill import MockGripper, _Perception, _Seeing, _agent
from tests.unit.skills.test_grasp_model_routing import RoutingPerception, _ctx

from vector_os_nano.skills.grasp import GraspSkill


# --- 3: nothing opens a holding gripper after detect / describe / pick -----------------




def test_the_tool_result_carries_the_skills_own_diagnosis():
    from vector_os_nano.vcli.tools.base import ToolContext
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

    a = _agent()
    tool = SkillWrapperTool(a._skill_registry.get("diagnosing"), a)
    ctx = ToolContext(agent=a, cwd=None, session=None, permissions=None, abort=threading.Event())
    out = tool.execute({}, ctx)
    assert out.is_error and out.metadata.get("mine") == 1 and "my_code" in out.content, (out.content, out.metadata)


# --- 4: coordinate navigation inside the old 1.5 m radius --------------------------------
@pytest.mark.parametrize("reason", ["aborted", "fell", "stalled", "unreachable", "timeout"])
def test_coordinate_nav_that_did_not_arrive_fails_even_close(reason):
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.sim.nav.follow import NavResult
    from vector_os_nano.skills.navigate import NavigateSkill

    class _Base:
        def navigate_to(self, x, y, **kw):
            return NavResult(reached=False, reason=reason, target=[x, y])

        def stop(self):
            pass

        def get_position(self):
            return [1.0, 0.5, 0.3]  # 0.5 m from the goal

        def get_heading(self):
            return 0.0

    ctx = SkillContext(base=_Base(), config={})
    r = NavigateSkill().execute({"x": 1.0, "y": 1.0}, ctx)
    assert not r.success and reason in r.error_message


# --- 4: a named grasp fails loud and grasps nothing -------------------------------------
class _Raising(RoutingPerception):
    def detect(self, query):
        self.calls.append(f"detect:{query}")
        raise RuntimeError("weights missing")


class _NoDetector(RoutingPerception):
    detect = None  # attribute present but unusable
    detect_in = None


@pytest.mark.parametrize("perc,query", [
    (_Raising, "grab the banana"),
    (lambda: RoutingPerception(detections=[]), "grab the banana"),
    (RoutingPerception, "抓黑色的罐子"),
    (RoutingPerception, "grab the white bottle"),
])
def test_a_named_grasp_without_its_detector_grasps_nothing(perc, query):
    p = perc()
    ctx = _ctx(p)
    r = GraspSkill().execute({"query": query}, ctx)
    assert not r.success and not ctx.gripper.is_holding()
    assert not any(c.startswith("front") for c in p.calls)


def test_a_named_grasp_with_no_detect_method_fails():
    class _Bare(RoutingPerception):
        pass
    del_attrs = ("detect", "detect_in")
    p = _Bare()
    for a in del_attrs:
        setattr(_Bare, a, property(lambda s: (_ for _ in ()).throw(AttributeError("no detector"))))
    ctx = _ctx(p)
    r = GraspSkill().execute({"query": "grab the banana"}, ctx)
    assert not r.success and not ctx.gripper.is_holding()
    assert not any(c.startswith("front") for c in p.calls)


# --- 4: pick with no target; empty label ----------------------------------------------


