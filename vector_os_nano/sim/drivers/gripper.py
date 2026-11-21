# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Gripper — the ONE jaw driver (``interfaces.GripperProtocol``) for every body and backend.

Built from the manifest's ``arm.gripper`` block and a ``SimRuntime``. A grasp is
physical first: :meth:`close` servoes the jaws shut, waits (sim time) for them to
settle and asks the runtime for the world-side grasp decision (``sim/grasp.py``): an
object is attached ONLY when every finger body touches it while the jaws are held open
by it. The attachment keeps the soft position-controlled grip from slipping; it never
pulls in an object that is merely near. With ``assist=False`` no attachment is created:
holding means current bilateral contact, not proof of a successful lift or transport.
``speed`` bounds jaw set-point travel in joint units per second.
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.schemas.robot import GripperSpec
from vector_os_nano.interfaces.physics import GraspSpec
from vector_os_nano.sim.runtime import SimRuntime

logger = logging.getLogger(__name__)

# The jaws get up to _SETTLE_S sim seconds to close; they are settled once they move
# less than _SETTLE_EPS (joint units) over one _SETTLE_POLL_S poll.
_SETTLE_S = 1.0
_SETTLE_POLL_S = 0.02
_SETTLE_EPS = 2e-4


class Gripper:
    def __init__(self, spec: GripperSpec, runtime: SimRuntime) -> None:
        scene = runtime.world.scene
        self._spec = spec
        self._rt = runtime
        self._names = {b: n for n, b in scene.body_ids.items()}
        self._jaw = scene.joint(spec.joint)
        self._jaw_q = scene.joint_q(self._jaw)
        self._grasp = GraspSpec(
            weld_body=scene.body(spec.weld_body),
            finger_bodies=tuple(scene.body(b) for b in spec.finger_bodies),
            jaw_joint=self._jaw, closed=float(spec.closed), min_opening=float(spec.min_opening),
            assist=spec.assist,
        )
        self._held: int | None = None  # the body the last close() gripped
        if not spec.assist:
            runtime.watch_contact_grip(self._grasp)

    def _servo(self, q: float) -> None:
        jaw = [self._jaw]
        rt, speed = self._rt, self._spec.speed
        start = float(rt.snapshot().q[self._jaw_q])
        steps = 1 if speed is None else max(1, math.ceil(abs(q - start) / (speed * _SETTLE_POLL_S)))
        for i in range(1, steps + 1):
            target = q if i == steps else start + (q - start) * i / steps
            rt.call(lambda w, v=target: w.set_targets(jaw, [v]))
            if speed is not None and not rt.wait_sim(_SETTLE_POLL_S):
                raise RuntimeError("gripper motion interrupted")

    def open(self) -> bool:
        """Open the jaws and release whatever the gripper holds, verified: False (the
        held state kept) when something is still attached or the sim refused."""
        try:
            self._servo(self._spec.open)
            still = self._rt.release(self._grasp.weld_body) if self._spec.assist else []
        except Exception as exc:  # noqa: BLE001 — reported, never a silent success
            logger.error("Gripper: open/release failed: %s", exc)
            return False
        if still:
            logger.error("Gripper: bodies %s still attached after release", still)
            return False
        self._held = None
        return True

    def close(self) -> bool:
        """Close the jaws; the runtime attaches what they actually grip. True once the
        jaws were commanded and settled, gripping or not (:meth:`is_holding` says
        which); False when the sim refused."""
        try:
            self._servo(self._spec.closed)
            held = self._settle_and_grasp()
        except Exception as exc:  # noqa: BLE001 — a failed grasp is a no-hold, never a crash
            logger.warning("Gripper: close failed: %s", exc)
            return False
        self._held = held
        return True

    def is_holding(self, name: str | None = None) -> bool:
        """The last gripped body remains attached (assisted) or in bilateral contact.
        With *name* (a scene body name, any case), only that body counts."""
        held = self._held
        if held is None or (name is not None and str(name).strip().lower() != self._names[held].lower()):
            return False
        if self._spec.assist:
            return self._rt.attachments().get(held) == self._grasp.weld_body
        return self._rt.grasp(self._grasp) == held

    def get_position(self) -> float:
        """Normalized jaw position: 0.0 closed, 1.0 fully open."""
        s = self._spec
        q = float(self._rt.snapshot().q[self._jaw_q])
        return max(0.0, min(1.0, (q - s.closed) / (s.open - s.closed)))

    def _settle_and_grasp(self) -> int | None:
        """Wait (sim time) for the jaws to stop, then ask for the grasp decision. A stop
        with the jaws still held open but not every finger on the object (one finger
        sliding it into the other) waits on and asks again, up to _SETTLE_S; asking
        changes nothing in the world unless it grips."""
        rt, s = self._rt, self._spec
        deadline = rt.sim_time + _SETTLE_S
        last = float(rt.snapshot().q[self._jaw_q])
        rt.wait_sim(5 * _SETTLE_POLL_S)  # let the servo start moving
        while rt.sim_time < deadline:
            now = float(rt.snapshot().q[self._jaw_q])
            if abs(now - last) < _SETTLE_EPS:
                held = rt.grasp(self._grasp)
                if held is not None or now - s.closed < s.min_opening:
                    return held
            last = now
            if not rt.wait_sim(_SETTLE_POLL_S):
                break
        return rt.grasp(self._grasp)
