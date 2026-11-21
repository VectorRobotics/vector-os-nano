# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The Agent uses each arm's OWN kinematics; the SO-101 IKSolver only for an arm
that takes an external solver (it used to build it for any arm and write SO-101
FK of Piper joints into the world model)."""
from __future__ import annotations

from vector_os_nano.core.agent import Agent
from vector_os_nano.hardware.so101 import ik_solver


class _Arm:
    name, dof, joint_names, _bus = "piper_like", 6, [f"j{i}" for i in range(6)], None

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 6

    def fk(self, joints):
        return [10.4, 3.0, 0.6], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class _ExternalSolverArm(_Arm):
    def set_ik_solver(self, solver) -> None:
        self.solver = solver


def test_an_arm_with_its_own_kinematics_never_gets_the_so101_solver(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(ik_solver, "IKSolver", lambda: built.append(1))
    agent = Agent(arm=_Arm())
    agent._build_context()
    assert built == [] and agent._ik_solver is None
    assert agent._build_context().arm.fk([0.0] * 6)[0] == [10.4, 3.0, 0.6]


def test_a_solver_failure_is_cached(monkeypatch) -> None:
    calls = []

    def _broken():
        calls.append(1)
        raise RuntimeError("no pinocchio")

    monkeypatch.setattr(ik_solver, "IKSolver", _broken)
    agent = Agent(arm=_ExternalSolverArm())
    agent._build_context()
    agent._build_context()
    assert calls == [1] and agent._ik_solver is False


def test_hardware_driver_refuses_an_unsupported_orientation():
    import pytest
    from vector_os_nano.hardware.so101.arm import SO101Arm

    arm = SO101Arm.__new__(SO101Arm)
    with pytest.raises(NotImplementedError, match='oriented IK'):
        arm.ik((0.,0.,0.), target_rotation=[[1,0,0],[0,1,0],[0,0,1]])
