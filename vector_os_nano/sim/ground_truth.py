# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundTruth — world-side truth of a :class:`SimRuntime`, read at a step boundary.

For oracles and graders: free-body poses, weld states, contacts, the support
under an object, what the robot's camera sees (segmentation) and the weld
transition log — everything the actor cannot author. Skills and perception
never import this module (tests/unit/vcli/test_gt_firewall.py).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def in_subtree(model: Any, bid: int, root_bid: int) -> bool:
    """True iff body *bid* is *root_bid* or one of its descendants."""
    while bid > 0:
        if bid == root_bid:
            return True
        bid = int(model.body_parentid[bid])
    return False


# A geom counts as the support under an object when its top is at most this far
# above the object's lowest point (resting contact interpenetrates by ~1 mm).
_SUPPORT_EPS = 0.01


class GroundTruth:
    """Ground truth of one runtime (names resolved once at construction).

    ``robot_root_bid``: the robot's root body; it and every body in its kinematic
    tree are left out, so the robot is never one of the scene's objects.
    """

    def __init__(self, runtime: Any, robot_root_bid: int = -1) -> None:
        import mujoco as mj

        self._rt = runtime
        m = runtime.model
        self.robot_root_bid = int(robot_root_bid)
        self.robot_bodies: frozenset[int] = frozenset(
            b for b in range(m.nbody) if robot_root_bid > 0 and in_subtree(m, b, robot_root_bid)
        )
        # A free joint's qpos IS its body's world pose (x y z qw qx qy qz).
        self._free: dict[str, int] = {}
        self.bid: dict[str, int] = {}
        for b in range(m.nbody):
            j = int(m.body_jntadr[b])
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
            if b in self.robot_bodies:
                continue
            if j >= 0 and name and m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                self._free[name] = int(m.jnt_qposadr[j])
                self.bid[name] = b
        self._welds: dict[str, int] = {}
        for i in range(m.neq):
            if m.eq_type[i] == mj.mjtEq.mjEQ_WELD:
                name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.eq_obj2id[i]))
                if name:
                    self._welds[name] = i
        self._weld_body = {i: n for n, i in self._welds.items()}
        root_j = int(m.body_jntadr[robot_root_bid]) if robot_root_bid > 0 else -1
        self._root_qadr = (
            int(m.jnt_qposadr[root_j])
            if root_j >= 0 and m.jnt_type[root_j] == mj.mjtJoint.mjJNT_FREE else -1
        )
        self._colliding = np.asarray((m.geom_contype != 0) | (m.geom_conaffinity != 0))

    @property
    def runtime(self) -> Any:
        return self._rt

    @property
    def sim_time(self) -> float:
        return float(self._rt.call(lambda _m, d: float(d.time)))

    # -- poses -------------------------------------------------------------

    def free_body_poses(self) -> dict[str, tuple[list[float], list[float]]]:
        """``{body: ([x, y, z], [qw, qx, qy, qz])}`` for every free body, one instant."""
        q = self._rt.snapshot().qpos
        return {n: (q[a:a + 3].tolist(), q[a + 3:a + 7].tolist()) for n, a in self._free.items()}

    def free_body_positions(self) -> dict[str, list[float]]:
        return {n: xyz for n, (xyz, _quat) in self.free_body_poses().items()}

    def object_positions(self) -> dict[str, list[float]]:
        """Like :meth:`free_body_positions`, read from the live data at a step boundary
        (also right for a caller-stepped scene that steps MjData directly)."""
        free = self._free
        return self._rt.call(lambda _m, d: {n: d.qpos[a:a + 3].tolist() for n, a in free.items()})

    def object_speed(self, name: str) -> float:
        """Linear speed (m/s) of free body *name*."""
        j = self.bid[name]

        def read(m: Any, d: Any) -> float:
            a = int(m.jnt_dofadr[int(m.body_jntadr[j])])
            return float(np.linalg.norm(d.qvel[a:a + 3]))

        return self._rt.call(read)

    def base_pose(self) -> tuple[float, float, float, float] | None:
        """``(x, y, z, yaw)`` of the robot root's free joint; None for a fixed base."""
        a = self._root_qadr
        if a < 0:
            return None
        q = self._rt.call(lambda _m, d: d.qpos[a:a + 7].copy())
        w, x, y, z = (float(v) for v in q[3:7])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(q[0]), float(q[1]), float(q[2]), yaw

    def joint_positions(self, joints: tuple[str, ...]) -> list[float]:
        """qpos of the named hinge/slide joints (fails loud on an unknown name)."""
        import mujoco as mj

        m = self._rt.model
        adr = []
        for name in joints:
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name!r} not in the model")
            adr.append(int(m.jnt_qposadr[jid]))
        return self._rt.call(lambda _m, d: [float(d.qpos[a]) for a in adr])

    # -- welds -------------------------------------------------------------

    def weld_active(self) -> dict[str, bool]:
        """``{welded body2 name: active}`` over every weld equality."""
        welds = self._welds
        return self._rt.call(
            lambda _m, d: {n: bool(int(np.asarray(d.eq_active)[i]) != 0) for n, i in welds.items()}
        )

    def weld_events(self, since: float = -math.inf) -> list[tuple[float, str, bool]]:
        """``(sim_time, welded body, active)`` weld transitions after *since*, oldest first.

        From the runtime's transition log (every step-boundary publish compares
        ``eq_active``); a read publishes first, so a write made outside the runtime
        is logged by the next read at the latest.
        """
        self._rt.call(lambda _m, _d: None)
        return [
            (t, self._weld_body[i], on) for t, i, on in list(self._rt.eq_log)
            if i in self._weld_body and t > since
        ]

    # -- contacts / support ------------------------------------------------

    def contacts(self, name: str) -> set[int]:
        """Body ids in contact with free body *name* right now."""
        bid = self.bid[name]

        def read(m: Any, d: Any) -> set[int]:
            n = int(d.ncon)
            b1 = m.geom_bodyid[np.asarray(d.contact.geom1[:n])]
            b2 = m.geom_bodyid[np.asarray(d.contact.geom2[:n])]
            return {int(o) for a, o in zip(b1, b2) if a == bid} | {
                int(a) for a, o in zip(b1, b2) if o == bid
            }

        return self._rt.call(read)

    def body_id(self, name: str) -> int:
        import mujoco as mj

        return int(mj.mj_name2id(self._rt.model, mj.mjtObj.mjOBJ_BODY, name))

    def clearance(self, name: str) -> float:
        """Gap (m) between the object's lowest point and the highest colliding,
        non-robot geom top under it — about 0 when it rests on a support, large
        while it is carried; inf over nothing at all."""
        bid = self.bid[name]
        robot = np.array(sorted(self.robot_bodies), dtype=int)
        colliding = self._colliding

        def read(m: Any, d: Any) -> float:
            # World AABB of every geom (its local aabb rotated by the geom frame).
            R = d.geom_xmat.reshape(-1, 3, 3)
            c = d.geom_xpos + np.einsum("gij,gj->gi", R, m.geom_aabb[:, :3])
            h = np.einsum("gij,gj->gi", np.abs(R), m.geom_aabb[:, 3:])
            lo, hi = c - h, c + h
            own = m.geom_bodyid == bid
            bottom = float(lo[own, 2].min())
            x, y = float(d.xpos[bid][0]), float(d.xpos[bid][1])
            under = (
                colliding & ~own & ~np.isin(m.geom_bodyid, robot)
                & (lo[:, 0] <= x) & (x <= hi[:, 0]) & (lo[:, 1] <= y) & (y <= hi[:, 1])
                & (hi[:, 2] <= bottom + _SUPPORT_EPS)
            )
            return bottom - float(hi[under, 2].max()) if under.any() else math.inf

        return self._rt.call(read)

    # -- camera ------------------------------------------------------------

    def visible_pixels(self, name: str, camera: str, width: int, height: int) -> int:
        """Pixels of free body *name* in a segmentation render of *camera* (occlusion-exact)."""
        m = self._rt.model
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.bid[name]]
        # ponytail: RGBDFrame.seg keeps the object id only (not its type), so a
        # rendered site whose id equals one of these geom ids would count too; the
        # scenes' few sites sit at ids far below the pickables' geoms.
        seg = self._rt.render(camera, width, height, ("seg",)).seg
        return int(np.isin(seg, geoms).sum()) if seg is not None else 0
