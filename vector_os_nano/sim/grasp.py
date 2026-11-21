# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The world-side grasp decision (charter 3.2), the same on every backend.

A gripper driver only servoes its jaws; whether that grips something is decided
here, from the world's own contacts: :func:`grasp` attaches the free body that EVERY
finger body touches while the jaws are held open by it (``opening >= min_opening``),
and :func:`release` detaches whatever the gripper holds. Both run at a step boundary
through ``SimRuntime.grasp`` / ``SimRuntime.release``, which log every attach and
detach at call time. No driver, skill or perception module calls
``PhysicsWorld.attach`` / ``detach``.
"""
from __future__ import annotations

import logging

from vector_os_nano.interfaces.physics import GraspSpec, PhysicsWorld

logger = logging.getLogger(__name__)


def grasp(world: PhysicsWorld, spec: GraspSpec) -> int | None:
    """Attach and return the free body every finger touches while the jaws are held
    open by it; None (nothing attached) otherwise. Never pulls in an object that is
    merely near the jaws. Without assistance, returns only current contact evidence
    for an external body (including an articulated handle), without changing physics."""
    scene = world.scene
    q, _qd = world.state()
    opening = float(q[scene.joint_q(spec.jaw_joint)]) - spec.closed
    touching = [world.contacts(f) for f in spec.finger_bodies]
    graspable = ({child for parent, child in scene.grasp_pairs if parent == spec.weld_body}
                 if spec.assist else set(scene.body_ids.values()) - scene.subtree(scene.robot_root) - {0})
    gripped = sorted(frozenset.intersection(*touching) & graspable)
    if not gripped or opening < spec.min_opening:
        logger.info("grasp: none — jaw opening %.1f mm, finger contacts %s", opening * 1000,
                    [sorted(scene.body_name(b) for b in t) for t in touching])
        return None
    if spec.assist:
        world.attach(spec.weld_body, gripped[0])
    logger.info("grasp: %r (jaw opening %.1f mm)", scene.body_name(gripped[0]), opening * 1000)
    return gripped[0]


def release(world: PhysicsWorld, parent: int) -> list[int]:
    """Detach everything attached to *parent*; return what is still attached to it."""
    for child, p in world.attached().items():
        if p == parent:
            world.detach(child)
    return [child for child, p in world.attached().items() if p == parent]
