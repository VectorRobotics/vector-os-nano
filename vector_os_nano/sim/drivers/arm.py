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

import logging
import math
import threading
from typing import Any, Callable

from vector_os_nano.schemas.robot import ArmSpec
from vector_os_nano.sim.runtime import ACTOR_EPOCH, SimRuntime
from vector_os_nano.sim.scene.kinematics import ArmKinematics

logger = logging.getLogger(__name__)

# move_joints writes the servo set-points at this SIM rate; after the interpolation
# the joints get up to _SETTLE_S sim seconds to come within the manifest's track_tol.
_MOVE_UPDATE_HZ = 50.0
_SETTLE_S = 1.0
_SETTLE_POLL_S = 0.02


class Arm:
    """Args:
        spec: the manifest ``arm`` block (names of the compiled scene).
        runtime: the SimRuntime that owns the world the arm is in.
        abort: polled during a motion; True preempts it like :meth:`stop`.
        wait_mount: called before every IK solve: returns once the body the arm is
            mounted on stands still (a legged base's ``wait_settled``); None = fixed.
    """

    def __init__(
        self, spec: ArmSpec, runtime: SimRuntime, *,
        abort: Callable[[], bool] = lambda: False, wait_mount: Callable[[], Any] | None = None,
    ) -> None:
        scene = runtime.world.scene
        self._spec = spec
        self._rt = runtime
        self._joints = [scene.joint(n) for n in spec.joints]
        self._qadr = [scene.joint_q(j) for j in self._joints]
        self._kin = ArmKinematics(scene, spec.joints, spec.ee_site)
        self._abort = abort
        self._wait_mount = wait_mount
        # stop() sets it: an in-flight move returns False within one set-point period
        # and writes nothing after the hold.
        self._stop_evt = threading.Event()
        # The actor's commanded set-point displacement (causation grading).
        self._ctrl_motion = 0.0

    @property
    def joint_names(self) -> list[str]:
        return list(self._spec.joints)

    @property
    def dof(self) -> int:
        return self._spec.dof

    def get_joint_positions(self) -> list[float]:
        q = self._rt.snapshot().q
        return [float(q[a]) for a in self._qadr]

    # -- motion --------------------------------------------------------------

    def _preempted(self) -> bool:
        return self._stop_evt.is_set() or bool(self._abort())

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Interpolate the servo set-points to *positions* over *duration* SIM seconds.

        True iff, after the interpolation, every joint settles within the manifest's
        ``track_tol`` of its target (within ``_SETTLE_S`` sim seconds). False when the
        arm is blocked (a collision, a limit) or :meth:`stop` / the abort preempts the
        move; the arm then holds where it is.
        """
        if len(positions) != self.dof:
            raise ValueError(f"Arm.move_joints: expected {self.dof} positions, got {len(positions)}")
        goal = [float(g) for g in positions]
        if not all(math.isfinite(g) for g in goal) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Arm.move_joints requires finite positions and positive duration")
        rt, joints, stop = self._rt, list(self._joints), self._stop_evt
        stop.clear()
        start = [float(v) for v in rt.call(lambda w: w.targets(joints))]
        commanded = sum(abs(g - s) for s, g in zip(start, goal))
        spec = self._spec
        if spec.velocity:
            # Exact derivative maxima of the quintic rest-to-rest trajectory.
            duration = max(duration, *(max(1.875 * abs(g-s) / v,
                                         (10 / math.sqrt(3) * abs(g-s) / a) ** .5,
                                         (60 * abs(g-s) / j) ** (1/3))
                                       for s, g, v, a, j in zip(start, goal, spec.velocity,
                                                               spec.acceleration, spec.jerk)))
        counted = ACTOR_EPOCH.active

        def _write(w: Any, v: list[float]) -> None:
            if not stop.is_set():  # checked at the step boundary: never after stop()'s hold
                w.set_targets(joints, v)

        steps = max(1, int(duration * _MOVE_UPDATE_HZ))
        dt = duration / steps
        done = 0.0
        try:
            for i in range(1, steps + 1):
                if self._preempted():
                    break
                t = i / steps
                if spec.velocity:
                    t = t**3 * (10 + t * (-15 + 6 * t))
                targets = [s + t * (g - s) for s, g in zip(start, goal)]
                rt.submit(lambda w, v=targets: _write(w, v))
                done = t
                rt.wait_sim(dt, cancel=stop)
        finally:
            # also when the loop raises: a call that wrote set-points commanded motion
            # (the verdict tells a refused call from a failed attempt by this counter)
            if counted:  # the actor's command only inside a producer-dispatched tool
                self._ctrl_motion += commanded * done
        if self._preempted():
            logger.info("Arm.move_joints: preempted at %.0f%% of the move", 100 * done)
            return False
        return self._settle(goal)

    def _settle(self, goal: list[float]) -> bool:
        """Wait (sim time) until max|q - goal| < track_tol; False if it never gets there."""
        rt, tol = self._rt, float(self._spec.track_tol)
        deadline = rt.sim_time + _SETTLE_S
        while True:
            q = rt.snapshot().q
            err = max(abs(float(q[a]) - g) for a, g in zip(self._qadr, goal))
            if err < tol:
                return True
            if (rt.sim_time >= deadline or self._preempted()
                    or not rt.wait_sim(_SETTLE_POLL_S, cancel=self._stop_evt)):
                logger.warning("Arm.move_joints: did not reach the target: max|q-target|=%.3f rad "
                               "(tol %.3f)", err, tol)
                return False

    def ctrl_motion(self) -> float:
        """Cumulative ``sum(|target - start|)`` of the moves started while a
        producer-dispatched tool runs (``ACTOR_EPOCH``); a preempted move counts the
        part it wrote, a move to the current pose adds zero. Never decreases."""
        return float(self._ctrl_motion)

    def move_cartesian(self, target_xyz: tuple[float, float, float], duration: float = 3.0) -> bool:
        """Top-down IK + move_joints; False when IK has no solution."""
        q = self.ik_top_down(target_xyz)
        return False if q is None else self.move_joints(q, duration=duration)

    def stop(self) -> None:
        """Preempt any move and hold the joint positions of this step boundary."""
        self._stop_evt.set()  # BEFORE the hold, so no queued set-point lands after it
        joints, qadr = list(self._joints), list(self._qadr)
        self._rt.call(lambda w: w.set_targets(joints, w.state()[0][qadr]))

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
