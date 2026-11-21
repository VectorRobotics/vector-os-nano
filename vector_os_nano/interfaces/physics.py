# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The physics contract — the ONLY thing a simulator implements.

Conventions, everywhere: world frame, z up, SI units, quaternions ``w x y z``.

**Canonical model.** MuJoCo compiles every scene (world ⊕ robot ⊕ arm); the result
is a :class:`CompiledScene`. A body / joint id at this interface is the index in
that compiled model, on every backend. State is ``(q, qd)`` in the compiled
model's generalized coordinates: a hinge or slide joint is one scalar; a free
body is 7 in ``q`` (position of the body FRAME, then the quaternion) and 6 in
``qd`` (linear velocity of the frame origin in WORLD coordinates, then angular
velocity in the BODY frame). Body poses, site poses and shape AABBs are computed
from a state on the compiled model (``sim/scene/canonical.py``); a backend
integrates dynamics and contact and renders its own sensors, nothing else.

Every sentence in the method docstrings below is binding and has a test in
``tests/conformance`` that runs on every backend. A backend also provides two
conformance seams the suite reads (never a runtime caller): ``_link_frames()``
-> the engine's own body-frame poses ``(pos (nbody, 3), quat (nbody, 4))``, NaN
for a body it does not move, and ``_inertials()`` -> its own ``(mass (nbody,),
com (nbody, 3), inertia (nbody, 3, 3))`` in the body frame, NaN likewise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from vector_os_nano.schemas.robot import EmbodimentConfig
from vector_os_nano.schemas.world import WorldManifest

# Two bodies are "in contact" when a contact point is at most this far apart (m).
CONTACT_DISTANCE_M: float = 1e-3

_JNT_FREE = 0  # mujoco.mjtJoint.mjJNT_FREE (pinned by tests/conformance)


@dataclass(frozen=True)
class RGBDFrame:
    """One camera frame; every field comes from ONE sim instant. World side only:
    ``seg`` is ground truth and never reaches a skill or a perception module.

    ``rgb`` (H, W, 3) uint8. ``depth`` (H, W) float32, metres along the optical
    axis (z-depth), 0 = invalid. ``seg`` (H, W) int32, canonical BODY id of the
    visible geom, -1 = none. ``K`` (3, 3) from the vertical fovy and the ACTUAL
    shape: fy = (H/2)/tan(fovy/2), fx = fy, cx = W/2, cy = H/2. ``cam_pos`` (3,)
    world. ``cam_mat`` (3, 3) world <- camera, columns = camera right, up,
    backward (OpenGL).
    """

    rgb: np.ndarray | None
    depth: np.ndarray | None
    seg: np.ndarray | None
    K: np.ndarray
    cam_pos: np.ndarray
    cam_mat: np.ndarray
    sim_time: float
    camera: str


@dataclass(frozen=True)
class GraspSpec:
    """One gripper in canonical ids: the input of the world-side grasp decision
    (``SimRuntime.grasp``), built by the gripper driver from its manifest block."""

    weld_body: int                    # the body a gripped object is attached to
    finger_bodies: tuple[int, ...]    # every one must touch the object
    jaw_joint: int                    # its q minus ``closed`` is the jaw opening
    closed: float
    min_opening: float                # held at least this far open by the object
    assist: bool = True              # False: contact evidence only; never attach


@dataclass(frozen=True)
class CompiledScene:
    """The canonical model of one simulation: what every backend loads.

    ``model`` is the compiled ``mujoco.MjModel``; this class only indexes its
    arrays, so it imports no simulator. Built by ``sim/scene`` (the one place that
    compiles), which also fills the name tables.
    """

    xml_path: Path
    model: Any
    timestep: float
    body_ids: Mapping[str, int]
    joint_ids: Mapping[str, int]
    cameras: Mapping[str, int]
    robot_root: int = -1                               # body id of the robot root; -1 = no robot
    grasp_pairs: tuple[tuple[int, int], ...] = ()      # (parent body, child body) that may be attached

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    def body(self, name: str) -> int:
        try:
            return self.body_ids[name]
        except KeyError:
            raise KeyError(f"no body {name!r}; bodies: {sorted(self.body_ids)}") from None

    def joint(self, name: str) -> int:
        try:
            return self.joint_ids[name]
        except KeyError:
            raise KeyError(f"no joint {name!r}; joints: {sorted(self.joint_ids)}") from None

    def body_name(self, body: int) -> str:
        for name, bid in self.body_ids.items():
            if bid == body:
                return name
        raise KeyError(f"no body with id {body}")

    def subtree(self, root: int) -> frozenset[int]:
        """*root* and every body below it in the kinematic tree."""
        parent = self.model.body_parentid
        out = set()
        for b in range(int(self.model.nbody)):
            a = b
            while a > 0 and a != root:
                a = int(parent[a])
            if a == root:
                out.add(b)
        return frozenset(out)

    def free_bodies(self) -> dict[str, int]:
        """Bodies whose joint is a free joint (movable objects and floating bases)."""
        m = self.model
        return {
            name: b for name, b in self.body_ids.items()
            if int(m.body_jntadr[b]) >= 0 and int(m.jnt_type[int(m.body_jntadr[b])]) == _JNT_FREE
        }

    def free_q(self, body: int) -> slice:
        """Slice of ``q`` holding a free body's pose: x y z, qw qx qy qz."""
        a = int(self.model.jnt_qposadr[self._free_joint(body)])
        return slice(a, a + 7)

    def free_qd(self, body: int) -> slice:
        """Slice of ``qd`` for a free body: linear WORLD (3), angular BODY (3)."""
        a = int(self.model.jnt_dofadr[self._free_joint(body)])
        return slice(a, a + 6)

    def joint_q(self, joint: int) -> int:
        """Index of a hinge / slide joint's position in ``q``."""
        return int(self.model.jnt_qposadr[joint])

    def joint_qd(self, joint: int) -> int:
        """Index of a hinge / slide joint's velocity in ``qd``."""
        return int(self.model.jnt_dofadr[joint])

    def _free_joint(self, body: int) -> int:
        j = int(self.model.body_jntadr[body])
        if j < 0 or int(self.model.jnt_type[j]) != _JNT_FREE:
            raise KeyError(f"body {body} has no free joint; free bodies: {sorted(self.free_bodies())}")
        return j


@runtime_checkable
class PhysicsWorld(Protocol):
    """One loaded scene inside one simulator. Not thread-safe: the SimRuntime that
    owns it is the only caller of every method except :meth:`render`."""

    name: str                       # "mujoco" | "pybullet"
    scene: CompiledScene
    caller_thread_only: bool        # True: no stepper thread may drive it (a main-thread viewer)

    def close(self) -> None:
        """Release the engine and any viewer. Idempotent."""

    def time(self) -> float:
        """Sim seconds since load (``reset`` does not change it)."""

    def step(self, n: int = 1) -> None:
        """Advance *n* physics steps of ``scene.timestep``. Commands are held for all
        *n* (zero-order hold): an engine that clears a command after one step
        re-issues it before each internal step."""

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """``(q, qd)`` copies of one instant, in the canonical layout (module docstring)."""

    def reset(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        """Write the whole state (``qd`` zero when omitted). Clears every torque, keeps
        the position targets and the attachments, refreshes kinematics and collision
        detection."""

    def set_targets(self, joints: Sequence[int], q: Sequence[float]) -> None:
        """Set-points, in joint coordinates, of the model's OWN position actuators for
        *joints*. Gains, control range and force limit are the model's and are
        evaluated at every physics step. A joint without a position actuator is a
        ``KeyError`` naming the joints that have one."""

    def targets(self, joints: Sequence[int]) -> np.ndarray:
        """The position set-points of *joints* as written (the control range clips
        what the servo tracks, not what is read back)."""

    def set_torque(self, joints: Sequence[int], tau: Sequence[float]) -> None:
        """Joint torques through the joints' motor actuators (``ctrl = tau / (gear *
        gain)``), clipped to the actuator's control range (then its force range) and
        held until replaced. A joint without a motor actuator is a ``KeyError``."""

    def contacts(self, body: int) -> frozenset[int]:
        """Bodies with at least one ENGINE contact point with *body* at signed distance
        <= :data:`CONTACT_DISTANCE_M`, as of the last :meth:`step` or :meth:`reset`.
        A penetration is always a contact, a gap up to the threshold is one, and a gap
        over it never is, whatever the geoms' own margins."""

    def attach(self, parent: int, child: int) -> None:
        """Hold *child* rigidly at its pose relative to *parent* as of the last
        :meth:`step` or :meth:`reset` (MuJoCo: the kinematics the last step computed,
        one physics step behind ``q``; current after a reset). Only pairs in
        ``scene.grasp_pairs``; any other pair is a ``KeyError`` listing them; a child
        attached to another parent is a ``ValueError``. World side only
        (``sim/grasp.py``): no driver, skill or perception module calls it."""

    def detach(self, child: int) -> None:
        """Release *child*; a no-op when it is not attached."""

    def attached(self) -> dict[int, int]:
        """``{child: parent}`` of every attachment that is active now."""

    def raycast(
        self, origins: np.ndarray, directions: np.ndarray, exclude_body: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast N rays (origins (N, 3), unit directions (N, 3)). Returns the distance
        to the first hit (-1 = miss) and the canonical id of the hit body (-1 = miss);
        *exclude_body*'s own geometry is transparent. The geometry tested is the
        engine's own (MuJoCo: every geom with alpha > 0, any group, as ``mj_ray``;
        PyBullet: collision shapes, and a hit beyond 100 m is a miss), so a lidar
        built on this is backend-specific and is never evidence."""

    def render(
        self, camera: str, width: int, height: int, kinds: tuple[str, ...] = ("rgb", "depth"),
    ) -> RGBDFrame:
        """Render ``scene.cameras[camera]`` at exactly (*width*, *height*); *kinds* is a
        subset of ``rgb``, ``depth``, ``seg`` (the others are None; anything else is a
        ``ValueError``, an unknown camera a ``KeyError``). Visible geoms are the ones a
        default MuJoCo render draws (group 0-2, alpha > 0); a pixel with nothing behind
        it has depth 0 and seg -1; pixel (row, col) samples its centre, so a point
        projects through ``K`` to (col + 0.5, row + 0.5). Observes one sim instant and
        may be called from any thread: a backend that cannot render while stepping
        serializes the two itself."""


@runtime_checkable
class SimBackend(Protocol):
    """A simulator plug-in: a row in ``vector_os_nano.sim.backends.BACKENDS``."""

    name: str

    def supports(self, robot: EmbodimentConfig, world: WorldManifest) -> str | None:
        """None when this backend can run *robot* (an EmbodimentConfig) in *world* (a
        WorldManifest), else the human-readable reason. The body x backend matrix the
        user sees is derived from this, never hand-written."""

    def load(self, scene: CompiledScene, *, gui: bool) -> PhysicsWorld:
        """Build a world from the canonical scene, with or without a viewer."""
