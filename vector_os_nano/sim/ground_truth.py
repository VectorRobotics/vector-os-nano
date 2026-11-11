# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundTruth — the world-side truth of one :class:`SimRuntime`: ONE generic class
over (runtime, scene), the same on every backend.

For oracles and graders: free-body poses and speeds, joint positions, attachments and
their log, contacts, the support under an object and what the robot's camera sees —
everything the actor cannot author. Poses and speeds slice the runtime's published
snapshot (one instant) by the scene's canonical addresses; clearance is today's AABB
rule over canonical shape AABBs of that instant; contacts and pixels ask the world.
Skills, perception and the actor-side drivers never import this module
(tests/unit/vcli/test_gt_firewall.py).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vector_os_nano.sim.scene import canonical

# A shape counts as the support under an object when its top is at most this far
# above the object's lowest point (resting contact interpenetrates by ~1 mm).
_SUPPORT_EPS = 0.01


class GroundTruth:
    """Ground truth of one runtime (ids resolved once, from its compiled scene).

    The robot is the scene's ``robot_root`` and every body below it; it is never one of
    the scene's objects.
    """

    def __init__(self, runtime: Any) -> None:
        self._rt = runtime
        sc = self._scene = runtime.world.scene
        self.robot_bodies: frozenset[int] = sc.subtree(sc.robot_root) if sc.robot_root >= 0 else frozenset()
        free = sc.free_bodies()
        # The scene's objects: every free body that is not the robot, name -> body id.
        self.bid: dict[str, int] = {n: b for n, b in free.items() if b not in self.robot_bodies}
        self.has_base: bool = sc.robot_root in free.values()  # the robot root moves freely
        self._names = {b: n for n, b in sc.body_ids.items()}

    @property
    def sim_time(self) -> float:
        return self._rt.sim_time

    # -- poses -------------------------------------------------------------

    def object_positions(self) -> dict[str, list[float]]:
        """``{object: [x, y, z]}`` of every object, one instant."""
        q, sc = self._rt.snapshot().q, self._scene
        return {n: q[sc.free_q(b)][:3].tolist() for n, b in self.bid.items()}

    def object_speed(self, name: str) -> float:
        """Linear speed (m/s) of object *name*."""
        return float(np.linalg.norm(self._rt.snapshot().qd[self._scene.free_qd(self.bid[name])][:3]))

    def base_pose(self) -> tuple[float, float, float, float] | None:
        """``(x, y, z, yaw)`` of the robot root; None for a fixed base."""
        if not self.has_base:
            return None
        x, y, z, w, qx, qy, qz = (float(v) for v in self._rt.snapshot().q[self._scene.free_q(self._scene.robot_root)])
        return x, y, z, math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def joint_positions(self, joints: tuple[str, ...]) -> list[float]:
        """q of the named hinge / slide joints (fails loud on an unknown name)."""
        sc = self._scene
        adr = [sc.joint_q(sc.joint(name)) for name in joints]
        q = self._rt.snapshot().q
        return [float(q[a]) for a in adr]

    # -- attachments -------------------------------------------------------

    def attachments(self) -> dict[str, str]:
        """``{object: the body it is attached to}`` right now."""
        return {self._names[c]: self._names[p] for c, p in self._rt.attachments().items()}

    def attached_to(self, name: str) -> str | None:
        """The body *name* is attached to, or None."""
        return self.attachments().get(name)

    def attach_events(self, since: float = -math.inf) -> list[tuple[float, str, str | None]]:
        """``(sim_time, object, parent or None for a detach)`` after *since*, oldest first.

        From the runtime's attachment log (every grasp / release logs at call time, any
        other change at the next publish; a read publishes first). Fails closed (raises)
        when the log no longer reaches back to *since*: an event may have been dropped.
        """
        self._rt.call(lambda _w: None)
        log = self._rt.attach_log
        events = list(log)
        if len(events) == log.maxlen and since < events[0][0]:
            raise RuntimeError(f"attach log truncated: it starts at {events[0][0]:.3f}, after {since:.3f}")
        return [(t, self._names[c], None if p is None else self._names[p]) for t, c, p in events if t > since]

    # -- contacts / support ------------------------------------------------

    def contacts(self, name: str) -> frozenset[int]:
        """Bodies in contact with object *name* right now (the world's contact sentence)."""
        bid = self.bid[name]
        return self._rt.call(lambda w: w.contacts(bid))

    def body_id(self, name: str) -> int:
        return self._scene.body(name)

    def clearance(self, name: str) -> float:
        """Gap (m) between the object's lowest point and the highest colliding,
        non-robot shape top under it — about 0 when it rests on a support, large
        while it is carried; inf over nothing at all."""
        sc, bid = self._scene, self.bid[name]
        q = self._rt.snapshot().q
        lo, hi, owner, colliding = canonical.shape_aabbs(sc, q)
        own = owner == bid
        bottom = float(lo[own, 2].min())
        x, y = (float(v) for v in q[sc.free_q(bid)][:2])
        under = (
            colliding & ~own & ~np.isin(owner, sorted(self.robot_bodies))
            & (lo[:, 0] <= x) & (x <= hi[:, 0]) & (lo[:, 1] <= y) & (y <= hi[:, 1])
            & (hi[:, 2] <= bottom + _SUPPORT_EPS)
        )
        return bottom - float(hi[under, 2].max()) if under.any() else math.inf

    # -- camera ------------------------------------------------------------

    def visible_pixels(self, name: str, camera: str, width: int, height: int) -> int:
        """Pixels of object *name* in a segmentation render of *camera* (occlusion-exact;
        ``seg`` holds canonical body ids)."""
        seg = self._rt.world.render(camera, width, height, ("seg",)).seg
        return int((seg == self.bid[name]).sum())
