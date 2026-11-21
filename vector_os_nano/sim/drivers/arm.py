# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Arm — the ONE arm driver (``interfaces.ArmProtocol``) for every body and backend.

Built from the manifest's ``arm`` block and a ``SimRuntime``:

* joint state from ``runtime.snapshot()`` (one instant, a copy);
* servo set-points through the model's own position actuators
  (``world.set_targets`` / ``targets``) at a step boundary;
* FK / IK on the canonical model (``sim/scene/kinematics.py``) against the mount
  of the live snapshot (the arm may ride a moving base);
* motions timed in SIM seconds (``runtime.wait_sim``), settle-checked against the
  manifest's ``track_tol``, preempted by :meth:`stop` and by the injected *abort*.

It never steps physics and never reads anything but its own robot.
"""
from __future__ import annotations

from typing import Any, Callable

from vector_os_nano.schemas.robot import ArmSpec
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.drivers.joints import JointGroup
from vector_os_nano.sim.scene.kinematics import ArmKinematics


class Arm(JointGroup):
    def __init__(self, spec: ArmSpec, runtime: SimRuntime, *,
                 abort: Callable[[], bool] = lambda: False,
                 wait_mount: Callable[[], Any] | None = None) -> None:
        super().__init__(spec, runtime, abort=abort)
        self._kin = ArmKinematics(runtime.world.scene, spec.joints, spec.ee_site, spec.ik_tolerance)
        self._wait_mount = wait_mount

    def move_cartesian(self, target_xyz: tuple[float, float, float], duration: float = 3.0) -> bool:
        """Top-down IK + move_joints; False when IK has no solution."""
        q = self.ik_top_down(target_xyz)
        return False if q is None else self.move_joints(q, duration=duration)

    def home(self, duration: float = 3.0) -> bool:
        """Move to the manifest home pose."""
        return self.move_joints(list(self._spec.home), duration=duration)

    # -- kinematics (the canonical model, the mount of the live snapshot) ------

    def fk(self, joint_positions: list[float]) -> tuple[list[float], list[list[float]]]:
        """(EE position, 3x3 rotation row-major) in the world, the live mount."""
        if len(joint_positions) != self.dof:
            raise ValueError(f"Arm.fk: expected {self.dof} joints, got {len(joint_positions)}")
        return self._kin.fk(self._rt.snapshot().q, joint_positions)

    def _live(self) -> Any:
        if self._wait_mount is not None:
            self._wait_mount()  # never solve against a body still braking from a stop
        return self._rt.snapshot().q

    def ik(self, target_xyz: tuple[float, float, float],
           current_joints: list[float] | None = None, *,
           target_rotation: list[list[float]] | None = None) -> list[float] | None:
        """IK, optionally constraining the tool's world rotation (e.g. a handle)."""
        q = self._live()
        seed = current_joints or self._kin.arm_q(q)
        if target_rotation is None:
            return self._kin.ik(q, target_xyz, seed)
        for initial in (seed, *self._spec.ik_seeds):
            result = self._kin.ik(q, target_xyz, initial, target_rotation=target_rotation)
            if result is not None:
                return result
        return None

    def ik_top_down(self, target_xyz: tuple[float, float, float],
                    current_joints: list[float] | None = None) -> list[float] | None:
        """Tool z = world -Z. Seeds: *current_joints*, the live arm, then the manifest's
        ``ik_seeds`` (each clipped to the joint limits), for each jaw yaw."""
        q = self._live()
        seeds = [] if current_joints is None else [[float(v) for v in current_joints]]
        seeds += [self._kin.arm_q(q), *([float(v) for v in s] for s in self._spec.ik_seeds)]
        return self._kin.ik_top_down(q, target_xyz, seeds)
