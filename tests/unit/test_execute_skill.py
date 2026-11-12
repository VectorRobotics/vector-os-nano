# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Agent.execute_skill and the skill tool run ONE skill and return ITS result.

There is no hidden mini-planner: no plan expansion, no appended ``home`` (it opened
the gripper and dropped a carried object), no payload swapped for another step's,
no lost ``diagnosis_code``. A skill that needs a prerequisite (the SO-101 scan pose)
calls it itself and still returns its own result.
"""
from __future__ import annotations

import threading

import pytest

from vector_os_nano.core.agent import Agent
from vector_os_nano.core.skill import skill
from vector_os_nano.core.types import Detection, SkillResult
from vector_os_nano.perception.calibration import Calibration
from vector_os_nano.skills import get_so101_skills


class MockArm:
    name = "mock"
    joint_names = ["j1", "j2", "j3", "j4", "j5"]
    dof = 5
    _bus = None  # no auto-created SO101Gripper

    def __init__(self) -> None:
        self.moves: list[list[float]] = []

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 5

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        self.moves.append(list(positions))
        return True

    def fk(self, joints):
        return ([0.2, 0.0, 0.1], None)

    def ik(self, xyz, current=None):
        return [0.0] * 5

    def stop(self) -> None: ...


class MockGripper:
    def __init__(self, holding: bool = False) -> None:
        self.holding = holding
        self.opened = 0

    def open(self) -> bool:
        self.opened += 1
        self.holding = False
        return True

    def close(self) -> bool:
        return True

    def is_holding(self) -> bool:
        return self.holding

    def get_position(self) -> float:
        return 0.0


class _Perception:
    """Sees nothing: detect / pick fail on the perception step."""

    calibration = Calibration()  # it reports arm-base points, as the sim arm's does

    def detect(self, query: str):
        return []

    def track(self, detections):
        return []


class _Seeing(_Perception):
    """Sees one 2D box (no depth): detect succeeds with a 2D-only object."""

    def detect(self, query: str):
        return [Detection(label=query, bbox=(1.0, 1.0, 9.0, 9.0), confidence=0.9)]


@skill(actuates=False)
class _Diagnosing:
    name = "diagnosing"
    description = "returns a diagnosis_code"
    parameters: dict = {}
    preconditions: list = []
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = []

    def execute(self, params, context):
        return SkillResult(success=False, error_message="nope", diagnosis_code="my_code",
                           result_data={"mine": 1})


def _agent(gripper: MockGripper | None = None, **kw) -> Agent:
    return Agent(arm=MockArm(), gripper=gripper or MockGripper(), config={},
                 skills=[*get_so101_skills(), _Diagnosing()], **kw)


def test_unknown_skill_fails_with_a_diagnosis() -> None:
    r = _agent().execute_skill("fly", {"height": 10})
    assert isinstance(r, SkillResult) and not r.success
    assert "Unknown skill" in r.error_message and r.result_data["diagnosis"] == "unknown_skill"


def test_home_runs_once_and_returns_its_own_result() -> None:
    a = _agent()
    r = a.execute_skill("home", None)
    assert isinstance(r, SkillResult) and r.success
    assert r.result_data["diagnosis"] == "ok" and len(a._arm.moves) == 1


def test_the_skill_result_is_returned_unchanged() -> None:
    r = _agent().execute_skill("diagnosing")
    assert r.diagnosis_code == "my_code" and r.result_data == {"mine": 1}


def test_detect_while_holding_never_opens_the_gripper() -> None:
    """Base: detect expanded to scan -> detect -> home, and home opened the gripper."""
    g = MockGripper(holding=True)
    r = _agent(g, perception=_Seeing()).execute_skill("detect", {"query": "banana"})
    assert r.success and g.opened == 0 and g.holding


def test_home_never_opens_a_holding_gripper() -> None:
    """skills-manip:home-opens-gripper-drops-object: home opened the gripper always."""
    g = MockGripper(holding=True)
    r = _agent(g).execute_skill("home")
    assert r.success and r.result_data["kept_holding"] is True
    assert g.opened == 0 and g.holding
    empty = MockGripper()
    assert _agent(empty).execute_skill("home").result_data["kept_holding"] is False
    assert empty.opened == 1  # an empty gripper is still opened at home


def test_pick_runs_its_scan_prerequisite_itself_and_returns_its_own_failure() -> None:
    a = _agent(perception=_Perception())
    r = a.execute_skill("pick", {"object_label": "banana", "mode": "hold"})
    assert not r.success and r.result_data["diagnosis"] == "object_not_found"
    assert a._arm.moves, "pick moved to the scan pose before it looked"


def test_the_skill_tool_payload_is_the_skills_own() -> None:
    """Base: a detect tool call returned the appended home step's data."""
    from vector_os_nano.vcli.tools.base import ToolContext
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

    a = _agent(perception=_Seeing())
    tool = SkillWrapperTool(a._skill_registry.get("detect"), a)
    ctx = ToolContext(agent=a, cwd=None, session=None, permissions=None, abort=threading.Event())
    out = tool.execute({"query": "banana"}, ctx)
    assert not out.is_error and out.metadata["count"] == 1
    assert out.metadata["objects"][0]["label"] == "banana" and "joint_values" not in out.metadata


@pytest.mark.parametrize("attr", ["__skill_auto_steps__", "__skill_direct__"])
def test_the_decorator_carries_no_planner_metadata(attr: str) -> None:
    assert not any(hasattr(type(s), attr) for s in get_so101_skills())
