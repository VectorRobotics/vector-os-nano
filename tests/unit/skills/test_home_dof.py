# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""HomeSkill DoF-awareness regression (the Piper 6-vs-5 fix).

The default home pose is the 5-DoF SO-101 pose; sending it to a 6-DoF arm (Piper)
must NOT crash ('expected 6 positions, got 5') — home adapts to the arm's DoF by
falling back to the URDF-zero neutral pose (`[0.0]*dof`, what MuJoCoPiper.home()
itself uses). SO-101 (5-DoF) behavior is byte-identical.

This is backlog #2: the bug blocked the bare-`vector-cli` + NL fetch path because
the planner/executor calls `home` on the go2+Piper arm, which raised.
"""
from __future__ import annotations

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills.home import HomeSkill, _DEFAULT_HOME_JOINTS


class _ArmWithDof:  # Piper-like: 6-DoF, exposes .dof + rejects a wrong-length move
    dof = 6

    def __init__(self) -> None:
        self.moved = None

    def get_joint_positions(self):
        return [0.1] * 6

    def move_joints(self, positions, duration=3.0):
        assert len(positions) == self.dof, f"expected {self.dof}, got {len(positions)}"
        self.moved = list(positions)
        return True


class _ArmNoDof:  # SO-101-like: 5-DoF, no .dof attr (DoF inferred from joint count)
    def __init__(self) -> None:
        self.moved = None

    def get_joint_positions(self):
        return [0.0] * 5

    def move_joints(self, positions, duration=3.0):
        assert len(positions) == 5, f"expected 5, got {len(positions)}"
        self.moved = list(positions)
        return True


class _Gripper:
    def __init__(self) -> None:
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def is_holding(self) -> bool:
        return False


def test_home_adapts_to_6dof_piper():
    arm = _ArmWithDof()
    gripper = _Gripper()
    res = HomeSkill().execute({}, SkillContext(arm=arm, gripper=gripper, config={}))
    assert res.success, res.error_message
    assert arm.moved is not None and len(arm.moved) == 6  # adapted to 6-DoF, no crash
    assert arm.moved == [0.0] * 6  # URDF-zero neutral home (matches MuJoCoPiper.home())
    assert gripper.opened


def test_home_so101_5dof_unchanged():
    arm = _ArmNoDof()
    res = HomeSkill().execute({}, SkillContext(arm=arm, config={}))
    assert res.success, res.error_message
    assert arm.moved == _DEFAULT_HOME_JOINTS  # 5==5: byte-identical SO-101 behavior


def test_home_config_override_respected_when_dof_matches():
    arm = _ArmWithDof()
    custom = [0.2, -0.5, 0.1, 0.3, -0.2, 0.4]  # 6-DoF override
    cfg = {"skills": {"home": {"joint_values": custom}}}
    res = HomeSkill().execute({}, SkillContext(arm=arm, config=cfg))
    assert res.success, res.error_message
    assert arm.moved == custom  # matching length → used verbatim


def test_home_no_arm_still_fails_cleanly():
    res = HomeSkill().execute({}, SkillContext(arm=None, config={}))
    assert not res.success and res.result_data.get("diagnosis") == "no_arm"
