"""Unit tests for PickTopDownSkill using mocked hardware.

The skill's wiring (hardware preconditions, world-model lookup, motion
sequencing, return-data structure) is verified here with a mock arm and
gripper. The actual physics / MuJoCo behaviour is exercised by the E2E
stability runs in scripts/verify_pick_top_down.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.pick_top_down import PickTopDownSkill


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockArm:
    """ArmProtocol-compatible mock. Records calls; ik_top_down returns a
    pre-canned solution unless force_ik_fail is set for a specific phase.
    """

    name = "mock_arm"
    dof = 6
    joint_names = [f"piper_joint{i}" for i in range(1, 7)]

    def __init__(self, ik_fail_phase: str | None = None):
        self.calls: list[tuple[str, object]] = []
        self._joints = [0.0] * 6
        self._ik_call_count = 0
        self._ik_fail_phase = ik_fail_phase  # "pre" or "grasp" or None

    def get_joint_positions(self):
        return list(self._joints)

    def move_joints(self, positions, duration=3.0):
        self.calls.append(("move_joints", (list(positions), duration)))
        self._joints = list(positions)
        return True

    def ik_top_down(self, target, current_joints=None):
        self.calls.append(("ik_top_down", (tuple(target), current_joints)))
        self._ik_call_count += 1
        if self._ik_fail_phase == "pre" and self._ik_call_count == 1:
            return None
        if self._ik_fail_phase == "grasp" and self._ik_call_count == 2:
            return None
        return [0.1] * 6

    def stop(self):
        self.calls.append(("stop", None))

    def fk(self, joint_positions):
        return [0.0, 0.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class _MockGripper:
    def __init__(self, holding_after_close: bool = True):
        self.calls: list[str] = []
        self._holding_after_close = holding_after_close
        self._closed = False

    def open(self):
        self.calls.append("open")
        self._closed = False
        return True

    def close(self):
        self.calls.append("close")
        self._closed = True
        return True

    def is_holding(self):
        return self._closed and self._holding_after_close

    def get_position(self):
        return 0.0 if self._closed else 1.0

    def get_force(self):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mkctx(arm=None, gripper=None, world_model=None, config=None):
    return SkillContext(
        arm=arm, gripper=gripper, base=SimpleNamespace(),
        world_model=world_model, config=config or {},
    )


def _mkworld(*objs: ObjectState) -> WorldModel:
    wm = WorldModel()
    for o in objs:
        wm.add_object(o)
    return wm


# ---------------------------------------------------------------------------
# Preconditions / failure modes
# ---------------------------------------------------------------------------


def test_no_arm_returns_no_arm_diag():
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=None, gripper=_MockGripper(), world_model=_mkworld())
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.success is False
    assert r.result_data["diagnosis"] == "no_arm"


def test_no_gripper_returns_no_gripper_diag():
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=_MockArm(), gripper=None, world_model=_mkworld())
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.result_data["diagnosis"] == "no_gripper"


def test_no_world_model_returns_no_world_model_diag():
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=_MockArm(), gripper=_MockGripper(), world_model=None)
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.result_data["diagnosis"] == "no_world_model"


def test_arm_without_ik_top_down_rejected():
    """An arm missing ``ik_top_down`` must fail with arm_unsupported."""
    class _BareArm:
        name = "bare"
        dof = 5
    arm = _BareArm()
    skill = PickTopDownSkill()
    ctx = _mkctx(arm=arm, gripper=_MockGripper(), world_model=_mkworld())
    r = skill.execute({"object_id": "x"}, ctx)
    assert r.result_data["diagnosis"] == "arm_unsupported"


def test_object_not_found_lists_known():
    skill = PickTopDownSkill()
    wm = _mkworld(ObjectState(object_id="pickable_a", label="a cube", x=0.5, y=0, z=0.1))
    ctx = _mkctx(arm=_MockArm(), gripper=_MockGripper(), world_model=wm)
    r = skill.execute({"object_id": "pickable_nonexistent"}, ctx)
    assert r.result_data["diagnosis"] == "object_not_found"
    assert "a cube" in r.result_data["known_objects"]


def test_ik_unreachable_pregrasp():
    skill = PickTopDownSkill()
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=1.0, y=0, z=0.2))
    ctx = _mkctx(arm=_MockArm(ik_fail_phase="pre"), gripper=_MockGripper(), world_model=wm)
    r = skill.execute({"object_id": "o1"}, ctx)
    assert r.success is False
    assert r.result_data["diagnosis"] == "ik_unreachable"
    assert r.result_data["phase"] == "pre_grasp"


def test_ik_unreachable_grasp():
    skill = PickTopDownSkill()
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=1.0, y=0, z=0.2))
    ctx = _mkctx(arm=_MockArm(ik_fail_phase="grasp"), gripper=_MockGripper(), world_model=wm)
    r = skill.execute({"object_id": "o1"}, ctx)
    assert r.result_data["diagnosis"] == "ik_unreachable"
    assert r.result_data["phase"] == "grasp"


# ---------------------------------------------------------------------------
# Happy path — motion sequencing
# ---------------------------------------------------------------------------


def test_pick_sequence_opens_then_moves_then_closes():
    """Verify the call order: gripper.open → move(pre) → move(grasp) →
    gripper.close → move(pre lifted). No final "home" move."""
    arm = _MockArm()
    gripper = _MockGripper(holding_after_close=True)
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=0.5, y=0, z=0.2))
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"object_id": "o1"}, ctx)

    assert r.success is True
    assert r.result_data["grasped_heuristic"] is True
    assert r.result_data["object_id"] == "o1"

    # Gripper: open before motion, close after descent — exactly once each
    assert gripper.calls == ["open", "close"]

    # Arm: two IK solves (pre + grasp), three moves (pre, grasp, lift)
    ik_calls = [c for c in arm.calls if c[0] == "ik_top_down"]
    move_calls = [c for c in arm.calls if c[0] == "move_joints"]
    assert len(ik_calls) == 2
    assert len(move_calls) == 3, f"expected 3 moves (pre/grasp/lift), got {len(move_calls)}"


def test_pick_uses_object_label_when_id_missing():
    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld(ObjectState(object_id="pickable_bottle_blue",
                              label="blue bottle", x=0.5, y=0, z=0.2))
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"object_label": "blue bottle"}, ctx)
    assert r.success is True
    assert r.result_data["object_id"] == "pickable_bottle_blue"


def test_pick_target_xyz_override():
    """Explicit target_xyz bypasses world_model lookup."""
    arm = _MockArm()
    gripper = _MockGripper()
    wm = _mkworld()  # empty
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"target_xyz": [0.5, 0.0, 0.2]}, ctx)
    assert r.success is True


def test_grasp_heuristic_false_reports_possibly_missed():
    arm = _MockArm()
    gripper = _MockGripper(holding_after_close=False)
    wm = _mkworld(ObjectState(object_id="o1", label="thing", x=0.5, y=0, z=0.2))
    ctx = _mkctx(arm=arm, gripper=gripper, world_model=wm)

    r = PickTopDownSkill().execute({"object_id": "o1"}, ctx)
    assert r.success is True  # motion sequence completed
    assert r.result_data["diagnosis"] == "possibly_missed"
    assert r.result_data["grasped_heuristic"] is False


# ---------------------------------------------------------------------------
# Metadata / descriptor
# ---------------------------------------------------------------------------


def test_skill_has_canonical_fields():
    skill = PickTopDownSkill()
    assert skill.name == "pick_top_down"
    assert "object_id" in skill.parameters
    assert "object_label" in skill.parameters
    assert "ik_unreachable" in skill.failure_modes
    assert "arm_unsupported" in skill.failure_modes


def test_skill_aliases_registered_for_chinese_and_english():
    aliases = PickTopDownSkill.__skill_aliases__
    assert "抓" in aliases
    assert "grab" in aliases
    assert "pick" in aliases
