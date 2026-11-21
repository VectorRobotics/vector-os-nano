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
import math

import numpy as np

from vector_os_nano.interfaces.physics import GraspSpec, PhysicsWorld

logger = logging.getLogger(__name__)


class ContactHistory:
    """World-side, sustained opposing finger loads; remembers every qualified loss.

    This proves contact possession only. Lift, support and task geometry are
    checked separately by GroundTruth and the offline physical grader.
    """

    dwell = .2

    def __init__(self):
        self.specs = {}
        self.started = {}
        self.lost = {}
        self.since = {}
        self.time = -math.inf

    def register(self, spec, world):
        if spec.assist:
            return
        self.specs[spec.weld_body] = spec
        self.since[spec.weld_body] = world.time()

    def observe(self, world):
        if not self.specs:
            return
        now = world.time()
        if now < self.time:
            self.started.clear()
            self.lost.clear()
            self.since = dict.fromkeys(self.specs, now)
        self.time = now
        scene = world.scene
        objects = set(scene.free_bodies().values()) - scene.subtree(scene.robot_root)
        objects.difference_update(world.attached())
        q, _ = world.state()
        loads = world.contact_loads()
        active = set()
        for parent, spec in self.specs.items():
            if len(spec.finger_bodies) < 2 or q[scene.joint_q(spec.jaw_joint)]-spec.closed < spec.min_opening:
                continue
            forces = {}
            for c in loads:
                normal = np.asarray(c.normal_on_b)
                compression = float(np.dot(c.force_on_b, normal))
                if compression <= 0:
                    continue
                for finger, body, sign in ((c.body_a, c.body_b, 1), (c.body_b, c.body_a, -1)):
                    if finger in spec.finger_bodies and body in objects:
                        key = (body, finger)
                        forces[key] = forces.get(key, np.zeros(3)) + sign*compression*normal
            for body in {b for b, _ in forces}:
                vectors = [forces.get((body, f), np.zeros(3)) for f in spec.finger_bodies]
                norms = [float(np.linalg.norm(v)) for v in vectors]
                if min(norms) > .05 and any(np.dot(a, b) < -.5*na*nb
                        for i, (a, na) in enumerate(zip(vectors, norms))
                        for b, nb in zip(vectors[i+1:], norms[i+1:])):
                    active.add((parent, body))
        for key in self.started.keys() - active:
            if now-self.started[key] >= self.dwell:
                self.lost[key] = now
        self.started = {key: self.started.get(key, now) for key in active}

    def held(self, parent, body, since=None):
        key = (parent, body)
        return (key in self.started and self.time-self.started[key] >= self.dwell and
                (since is None or (since >= self.since[parent] and self.lost.get(key, -math.inf) <= since)))


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
