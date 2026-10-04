# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundTruth — world-side truth of a :class:`SimRuntime`, read at a step boundary.

For oracles and graders: the free-body poses and weld states the actor cannot
author. Skills and perception must never import this module (WP10 adds the
firewall test and the full oracle set on top of it).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def in_subtree(model: Any, bid: int, root_bid: int) -> bool:
    """True iff body *bid* is *root_bid* or one of its descendants."""
    while bid > 0:
        if bid == root_bid:
            return True
        bid = int(model.body_parentid[bid])
    return False


class GroundTruth:
    """Free-body poses and weld states of one runtime (names resolved once).

    ``robot_root_bid``: the robot's root body; it and every body in its kinematic
    tree are left out, so the robot is never one of the scene's objects.
    """

    def __init__(self, runtime: Any, robot_root_bid: int = -1) -> None:
        import mujoco as mj

        self._rt = runtime
        m = runtime.model
        # A free joint's qpos IS its body's world pose (x y z qw qx qy qz).
        self._free: dict[str, int] = {}
        for b in range(m.nbody):
            j = int(m.body_jntadr[b])
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b)
            if robot_root_bid > 0 and in_subtree(m, b, robot_root_bid):
                continue
            if j >= 0 and name and m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                self._free[name] = int(m.jnt_qposadr[j])
        self._welds: dict[str, int] = {}
        for i in range(m.neq):
            if m.eq_type[i] == mj.mjtEq.mjEQ_WELD:
                name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.eq_obj2id[i]))
                if name:
                    self._welds[name] = i

    def free_body_poses(self) -> dict[str, tuple[list[float], list[float]]]:
        """``{body: ([x, y, z], [qw, qx, qy, qz])}`` for every free body, one instant."""
        q = self._rt.snapshot().qpos
        return {n: (q[a:a + 3].tolist(), q[a + 3:a + 7].tolist()) for n, a in self._free.items()}

    def free_body_positions(self) -> dict[str, list[float]]:
        return {n: xyz for n, (xyz, _quat) in self.free_body_poses().items()}

    def weld_active(self) -> dict[str, bool]:
        """``{welded body2 name: active}`` over every weld equality."""
        welds = self._welds
        return self._rt.call(
            lambda _m, d: {n: bool(int(np.asarray(d.eq_active)[i]) != 0) for n, i in welds.items()}
        )
