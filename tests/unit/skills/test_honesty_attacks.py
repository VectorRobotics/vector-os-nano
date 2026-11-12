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
from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill


# --- 3: nothing opens a holding gripper after detect / describe / pick -----------------
@pytest.mark.parametrize("skill,params,perc", [
    ("detect", {"query": "banana"}, _Seeing),
    ("describe", {}, _Seeing),
    ("scan", {}, _Seeing),
])
def test_no_arm_skill_opens_a_holding_gripper(skill, params, perc):
    g = MockGripper(holding=True)
    r = _agent(g, perception=perc()).execute_skill(skill, params)
    assert g.opened == 0 and g.holding, (skill, r.result_data)


def test_a_failing_pick_while_holding_keeps_the_held_object():
    """pick of an object that is not there, while holding one: is the held one dropped?"""
    g = MockGripper(holding=True)
    r = _agent(g, perception=_Perception()).execute_skill("pick", {"object_label": "banana"})
    assert not r.success
    assert g.opened == 0 and g.holding, "pick opened a holding gripper before it found a target"


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

    ctx = SkillContext(base=_Base(), world_model=WorldModel(), config={})
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
    r = PerceptionGraspSkill().execute({"query": query}, ctx)
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
    r = PerceptionGraspSkill().execute({"query": "grab the banana"}, ctx)
    assert not r.success and not ctx.gripper.is_holding()
    assert not any(c.startswith("front") for c in p.calls)


# --- 4: pick with no target; empty label ----------------------------------------------
@pytest.mark.parametrize("params", [{}, {"object_label": ""}, {"object_label": "  "}, {"object_id": ""}])
def test_pick_with_no_target_picks_nothing(params):
    g = MockGripper()
    a = _agent(g, perception=_Seeing())
    a._world_model.add_object(ObjectState(object_id="can_1", label="can", x=0.2, y=0.0, z=0.05))
    r = a.execute_skill("pick", params)
    assert not r.success and r.result_data["diagnosis"] == "no_target"
    assert a._world_model.get_object("can_1") is not None


def test_an_empty_label_matches_and_removes_nothing():
    wm = WorldModel()
    wm.add_object(ObjectState(object_id="can_1", label="can", x=0.2, y=0.0, z=0.05))
    for q in ("", "  ", None):
        assert wm.get_objects_by_label(q) == []
    assert wm.get_object("can_1") is not None
