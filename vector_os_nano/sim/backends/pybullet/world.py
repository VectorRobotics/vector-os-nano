# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PyBulletWorld: ``interfaces.PhysicsWorld`` on Bullet, built from the compiled model.

Built from ``sim/scene/reader`` records, never by stepping MuJoCo:

* every world-child body tree that has a joint becomes one multibody (fixed base
  for a tree welded to the world, floating base for a free root); every static
  geom becomes its own mass-0 body. Bullet ids stay private: every id at the
  interface is canonical.
* inertia is forced to the model's (both Bullet load paths silently replace it);
  hinge armature is emulated by adding it to the link inertia about the joint
  axis (re-diagonalised); slide armature is not emulated.
* the model's actuators (motor and position, joint transmission) are evaluated
  in Python before EVERY physics step and re-issued as TORQUE_CONTROL (Bullet
  clears torques after each step); gravity compensation is an external force at
  each gravcomp body's COM, also every step; joint damping is Bullet's viscous
  joint damping; frictionloss is a zero-velocity motor with that force limit.
* mimic couplings (``q1 = c1 * q2``) are gear constraints; contact exclusions
  are pair filters over an all-pairs self-collision multibody.
* friction is per link (the max over its colliding geoms; condim 1 = 0) and
  Bullet multiplies the two sides; MuJoCo takes the max or the higher priority.
  ponytail: matches MuJoCo when the static side is 1.0 (the default), approximate otherwise.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client

from vector_os_nano.interfaces.physics import CONTACT_DISTANCE_M, CompiledScene, RGBDFrame
from vector_os_nano.sim.scene import canonical, reader
from vector_os_nano.sim.scene.reader import (
    EQ_JOINT, GEOM_BOX, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_MESH, GEOM_PLANE, GEOM_SPHERE,
    JNT_FREE, JNT_HINGE, JNT_SLIDE,
)

_KINDS = ("rgb", "depth", "seg")
RAY_RANGE_M = 100.0          # a ray longer than this is a miss
_RAY_PASSES = 16             # how many excluded surfaces one ray may pass through
_PLANE_VISUAL_HALF = 50.0    # half-size of an unbounded plane's visual


# -- quaternions (w x y z at the interface, x y z w inside Bullet) ----------------

def _xyzw(q: np.ndarray) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _wxyz(q: Sequence[float]) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _matq(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion w x y z (Shepperd)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def unsupported(m: Any, pickables: Iterable[int] = ()) -> str | None:
    """Why this backend cannot load the compiled model *m* (whose grasp pairs hold the
    bodies *pickables*), or None."""
    pickables = frozenset(pickables)
    why: list[str] = []
    joints = reader.joints(m)
    for b in reader.bodies(m)[1:]:
        if int(m.body_jntnum[b.id]) > 1:
            why.append(f"body {b.name!r} has {int(m.body_jntnum[b.id])} joints")
        if m.body_mocapid[b.id] >= 0:
            why.append(f"mocap body {b.name!r}")
    for j in joints:
        if j.type not in (JNT_FREE, JNT_HINGE, JNT_SLIDE):
            why.append(f"ball joint {j.name!r}")
        elif np.any(j.pos != 0):
            why.append(f"joint {j.name!r} is anchored off its body origin")
        if j.type != JNT_FREE and int(m.body_parentid[j.body]) == 0:  # (MuJoCo itself keeps free joints top level)
            why.append(f"joint {j.name!r}: a world child moves only by a free joint")
    for g in reader.geoms(m):
        if (g.collide or g.visible) and g.type not in (GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_BOX, GEOM_MESH):
            why.append(f"geom type {g.type} ({g.name or g.id})")
        if g.collide and (int(m.geom_contype[g.id]), int(m.geom_conaffinity[g.id])) != (1, 1):
            why.append(f"geom {g.name or g.id}: contype/conaffinity other than 0/0 or 1/1")
        if g.collide and g.type == GEOM_MESH and g.body in pickables:  # S2cV-01
            why.append(f"pickable geom {g.name or g.id} collides as a mesh: Bullet reads its hull as "
                       "touching ~3 mm away, and a grasp is decided on contact (use primitives)")
    for a in reader.actuators(m):
        if a.kind == "other":
            why.append(f"actuator {a.name!r} is neither a joint motor nor a joint position servo")
    for i in range(m.neq):
        joint_eq = int(m.eq_type[i]) == EQ_JOINT
        if joint_eq and m.eq_active0[i] and (m.eq_data[i, 0] != 0 or np.any(m.eq_data[i, 2:5] != 0) or m.eq_data[i, 1] == 0):
            why.append(f"joint equality {i} is not linear (q1 = c1 * q2)")
        elif not joint_eq and m.eq_active0[i]:
            why.append(f"active equality {i} of type {int(m.eq_type[i])}")
    if m.ntendon or m.npair or m.nhfield:
        why.append("tendons, explicit contact pairs and height fields are not supported")
    for c in reader.cameras(m):
        if not c.fixed:
            why.append(f"camera {c.name!r} is not fixed to its body")
    return "; ".join(dict.fromkeys(why)) or None


class PyBulletWorld:
    name = "pybullet"
    caller_thread_only = False

    def __init__(self, scene: CompiledScene, *, gui: bool = False) -> None:
        m = scene.model
        why = unsupported(m, (c for _p, c in scene.grasp_pairs))
        if why:
            raise ValueError(f"PyBullet cannot load {scene.xml_path}: {why}")
        self.scene = scene
        self._lock = threading.RLock()
        self._p = p = bullet_client.BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        self._tmp = Path(tempfile.mkdtemp(prefix="vos_pybullet_"))
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        p.setGravity(*[float(g) for g in m.opt.gravity])
        p.setTimeStep(scene.timestep)
        self._dt, self._steps = scene.timestep, 0
        self._near = float(m.vis.map.znear * m.stat.extent)
        self._far = float(m.vis.map.zfar * m.stat.extent)

        self._bodies = reader.bodies(m)
        self._joints = reader.joints(m)
        self._geoms = reader.geoms(m)
        self._cams = {c.name: c for c in reader.cameras(m)}
        self._mesh_files: dict[str, str] = {}
        self._link: dict[int, tuple[int, int]] = {}          # canonical body -> (uid, link)
        self._body_at: dict[tuple[int, int], int] = {}       # (uid, link) -> canonical body
        self._iframe: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # base body -> local inertial (pos, quat)
        pos0, quat0 = canonical.body_poses(scene, m.qpos0)
        self._static_pose = {b.id: (pos0[b.id], quat0[b.id]) for b in self._bodies}
        self._invisible = p.createVisualShape(pb.GEOM_SPHERE, radius=1e-4, rgbaColor=[0, 0, 0, 0])

        children: dict[int, list[int]] = defaultdict(list)
        for b in self._bodies[1:]:
            children[b.parent].append(b.id)
        static = [0]
        for root in children[0]:
            sub, stack = [], [root]
            while stack:
                b = stack.pop()
                sub.append(b)
                stack.extend(reversed(children[b]))
            sub.sort()  # model order: a parent before its children
            if any(self._bodies[b].joint >= 0 for b in sub):
                self._build_tree(sub)
            else:
                static.extend(sub)
        for b in static:
            self._build_static(b)
        self._build_couplings()
        self._build_actuation()
        self._build_collision_filter()
        self._constraints: dict[int, tuple[int, int]] = {}   # child -> (parent, constraint id)
        self._qref: dict[int, np.ndarray] = {}  # free joint -> last quaternion published or written
        self.reset(m.qpos0)

    # -- build -----------------------------------------------------------------

    def _obj(self, key: str, verts: np.ndarray, faces: np.ndarray) -> str:
        path = self._mesh_files.get(key)
        if path is None:
            normals = np.zeros_like(verts)
            fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
            for k in range(3):
                np.add.at(normals, faces[:, k], fn)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts]
            lines += [f"vn {x:.6g} {y:.6g} {z:.6g}" for x, y, z in normals]
            lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in faces + 1]
            path = str(self._tmp / f"{key}.obj")
            Path(path).write_text("\n".join(lines) + "\n")
            self._mesh_files[key] = path
        return path

    def _mesh_file(self, mesh: int) -> str:
        return self._obj(f"mesh{mesh}", *reader.mesh(self.scene.model, mesh))

    def _merged_mesh(self, gs: list[reader.Geom]) -> reader.Geom:
        """Mesh geoms of one body and colour as ONE mesh in the body frame (Bullet keeps
        at most 16 shapes per compound and drops the rest)."""
        verts, faces, n = [], [], 0
        for g in gs:
            v, f = reader.mesh(self.scene.model, g.mesh)
            verts.append(v @ _qmat(g.quat).T + g.pos)
            faces.append(f + n)
            n += len(v)
        path = self._obj("merged" + "_".join(str(g.id) for g in gs), np.concatenate(verts), np.concatenate(faces))
        return gs[0]._replace(pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]), mesh=path)

    def _shape_args(self, g: reader.Geom, visual: bool) -> dict[str, Any]:
        s = g.size
        if g.type == GEOM_PLANE:
            if visual:
                half = [float(s[0]) or _PLANE_VISUAL_HALF, float(s[1]) or _PLANE_VISUAL_HALF, 1e-4]
                return dict(shapeType=pb.GEOM_BOX, halfExtents=half, frame=(0.0, 0.0, -1e-4))
            return dict(shapeType=pb.GEOM_PLANE, planeNormal=[0, 0, 1])
        if g.type == GEOM_SPHERE:
            return dict(shapeType=pb.GEOM_SPHERE, radius=float(s[0]))
        if g.type in (GEOM_CAPSULE, GEOM_CYLINDER):
            kind = pb.GEOM_CAPSULE if g.type == GEOM_CAPSULE else pb.GEOM_CYLINDER
            return dict(shapeType=kind, radius=float(s[0]), length=2.0 * float(s[1]))
        if g.type == GEOM_BOX:
            return dict(shapeType=pb.GEOM_BOX, halfExtents=[float(v) for v in s[:3]])
        path = g.mesh if isinstance(g.mesh, str) else self._mesh_file(g.mesh)
        return dict(shapeType=pb.GEOM_MESH, fileName=path, meshScale=[1, 1, 1])

    def _shape_array(self, gs: list[reader.Geom], visual: bool) -> int:
        """One compound shape of *gs*, each placed by its pose in the body frame."""
        if not gs:
            return self._invisible if visual else -1
        if visual:  # one merged mesh per colour
            by_colour: dict[tuple, list[reader.Geom]] = defaultdict(list)
            for g in gs:
                if g.type == GEOM_MESH:
                    by_colour[tuple(np.round(g.rgba, 6))].append(g)
            gs = [g for g in gs if g.type != GEOM_MESH] + [self._merged_mesh(v) for v in by_colour.values()]
        if len(gs) > 16:
            raise ValueError(f"body {self._bodies[gs[0].body].name!r}: more than 16 "
                             f"{'visual' if visual else 'collision'} shapes (Bullet's compound limit)")
        cols: dict[str, list] = defaultdict(list)
        for g in gs:
            a = self._shape_args(g, visual)
            off = np.array(a.pop("frame", (0.0, 0.0, 0.0)))
            cols["shapeTypes"].append(a["shapeType"])
            cols["radii"].append(a.get("radius", 0.0))
            cols["halfExtents"].append(a.get("halfExtents", [0.0, 0.0, 0.0]))
            cols["lengths"].append(a.get("length", 0.0))
            cols["fileNames"].append(a.get("fileName", ""))
            cols["meshScales"].append(a.get("meshScale", [1, 1, 1]))
            cols["planeNormals"].append(a.get("planeNormal", [0, 0, 1]))
            pos = (g.pos + _qmat(g.quat) @ off).tolist()
            if visual:
                cols["rgbaColors"].append([float(c) for c in g.rgba])
                cols["visualFramePositions"].append(pos)
                cols["visualFrameOrientations"].append(_xyzw(g.quat))
            else:
                cols["collisionFramePositions"].append(pos)
                cols["collisionFrameOrientations"].append(_xyzw(g.quat))
        create = self._p.createVisualShapeArray if visual else self._p.createCollisionShapeArray
        return create(**cols)

    def _body_shapes(self, b: int) -> tuple[int, int]:
        gs = [g for g in self._geoms if g.body == b]
        return self._shape_array([g for g in gs if g.collide], False), self._shape_array([g for g in gs if g.visible], True)

    def _inertial(self, b: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """(mass, COM, principal axes quat, principal moments) with hinge armature emulated."""
        body = self._bodies[b]
        R = _qmat(body.iquat)
        tensor = R @ np.diag(body.inertia) @ R.T
        if body.joint >= 0 and self._joints[body.joint].type == JNT_HINGE:
            n = self._joints[body.joint].axis / np.linalg.norm(self._joints[body.joint].axis)
            tensor = tensor + self._joints[body.joint].armature * np.outer(n, n)
        w, V = np.linalg.eigh(tensor)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        return body.mass, body.ipos.copy(), _matq(V), w

    def _build_tree(self, sub: list[int]) -> None:
        p = self._p
        root, links = sub[0], sub[1:]
        floating = self._bodies[root].joint >= 0
        index = {root: 0, **{b: i + 1 for i, b in enumerate(links)}}
        shapes = {b: self._body_shapes(b) for b in sub}
        inert = {b: self._inertial(b) for b in sub}
        jtype = {JNT_HINGE: pb.JOINT_REVOLUTE, JNT_SLIDE: pb.JOINT_PRISMATIC}
        kw: dict[str, list] = defaultdict(list)
        for b in links:
            body = self._bodies[b]
            j = self._joints[body.joint] if body.joint >= 0 else None
            kw["linkMasses"].append(inert[b][0])
            kw["linkCollisionShapeIndices"].append(shapes[b][0])
            kw["linkVisualShapeIndices"].append(shapes[b][1])
            kw["linkPositions"].append(body.pos.tolist())
            kw["linkOrientations"].append(_xyzw(body.quat))
            kw["linkInertialFramePositions"].append(inert[b][1].tolist())
            kw["linkInertialFrameOrientations"].append(_xyzw(inert[b][2]))
            kw["linkParentIndices"].append(index[body.parent])
            kw["linkJointTypes"].append(pb.JOINT_FIXED if j is None else jtype[j.type])
            kw["linkJointAxis"].append([0.0, 0.0, 1.0] if j is None else j.axis.tolist())
        mass, ipos, iquat, _ = inert[root]
        uid = p.createMultiBody(
            baseMass=mass if floating else 0.0, baseCollisionShapeIndex=shapes[root][0],
            baseVisualShapeIndex=shapes[root][1], basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
            baseInertialFramePosition=ipos.tolist(), baseInertialFrameOrientation=_xyzw(iquat),
            flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_USE_SELF_COLLISION_INCLUDE_PARENT, **kw)
        self._iframe[root] = (ipos, iquat)
        # Body damping is per multibody, and passing it with a link index sends that call's
        # inertia to the BASE instead (measured, PyBullet 3.2.7): set it once, on its own.
        p.changeDynamics(uid, -1, linearDamping=0.0, angularDamping=0.0)
        for b in sub:
            link = index[b] - 1
            self._link[b] = (uid, link)
            self._body_at[(uid, link)] = b
            gs = [g for g in self._geoms if g.body == b and g.collide]
            dyn: dict[str, Any] = self._friction(gs)
            if link >= 0 or floating:
                dyn["localInertiaDiagonal"] = inert[b][3].tolist()  # Bullet replaced it on load
            body = self._bodies[b]
            if link >= 0 and body.joint >= 0:
                j = self._joints[body.joint]
                dyn["jointDamping"] = j.damping
                if j.range is not None:
                    dyn["jointLowerLimit"] = j.range[0] - j.qpos0
                    dyn["jointUpperLimit"] = j.range[1] - j.qpos0
            p.changeDynamics(uid, link, **dyn)
        movable = [index[b] - 1 for b in links if self._bodies[b].joint >= 0]
        if movable:
            loss = [self._joints[self._bodies[self._body_at[(uid, k)]].joint].frictionloss for k in movable]
            p.setJointMotorControlArray(uid, movable, pb.VELOCITY_CONTROL,
                                        targetVelocities=[0.0] * len(movable), forces=loss)

    def _friction(self, gs: list[reader.Geom]) -> dict[str, float]:
        if not gs:
            return {}
        return dict(
            lateralFriction=max(float(g.friction[0]) if g.condim > 1 else 0.0 for g in gs),
            spinningFriction=max(float(g.friction[1]) if g.condim > 3 else 0.0 for g in gs),
            rollingFriction=max(float(g.friction[2]) if g.condim > 4 else 0.0 for g in gs),
        )

    def _build_static(self, b: int) -> None:
        p = self._p
        bpos, bquat = self._static_pose[b]
        R = _qmat(bquat)
        for g in (g for g in self._geoms if g.body == b and (g.collide or g.visible)):
            pos, quat = bpos + R @ g.pos, _qmul(bquat, g.quat)
            col = vis = -1
            if g.collide:
                a = self._shape_args(g, False)
                if "length" in a:  # the single-shape call names it height
                    a["height"] = a.pop("length")
                col = p.createCollisionShape(**a)
            if g.visible:
                a = self._shape_args(g, True)
                off = np.array(a.pop("frame", (0.0, 0.0, 0.0)))
                vis = p.createVisualShape(rgbaColor=[float(c) for c in g.rgba], visualFramePosition=off.tolist(), **a)
            else:
                vis = self._invisible
            uid = p.createMultiBody(0.0, col, vis, pos.tolist(), _xyzw(quat))
            self._body_at[(uid, -1)] = b
            if g.collide:
                p.changeDynamics(uid, -1, **self._friction([g]))

    def _build_couplings(self) -> None:
        for c in reader.couplings(self.scene.model):  # q1 = c1 * q2; Bullet gear: q_A + r * q_B = 0
            (ua, la), (ub, lb) = self._link[self._joints[c.joint2].body], self._link[self._joints[c.joint1].body]
            cid = self._p.createConstraint(ua, la, ub, lb, pb.JOINT_GEAR, [0, 0, 1], [0, 0, 0], [0, 0, 0])
            self._p.changeConstraint(cid, gearRatio=-1.0 / float(c.polycoef[1]), maxForce=1e4)

    def _build_actuation(self) -> None:
        acts = reader.actuators(self.scene.model)
        self._ctrl = np.zeros(len(acts))
        self._position = {a.joint: a for a in acts if a.kind == "position"}
        self._motor = {a.joint: a for a in acts if a.kind == "motor"}
        # per multibody: the actuated links and, per actuator, its row in that list
        by_uid: dict[int, list[reader.Actuator]] = defaultdict(list)
        for a in acts:
            by_uid[self._link[self._joints[a.joint].body][0]].append(a)
        self._drive = []
        for uid, alist in by_uid.items():
            jnts = sorted({a.joint for a in alist})
            row = {j: i for i, j in enumerate(jnts)}
            lo = np.array([self._joints[j].actfrcrange[0] if self._joints[j].actfrcrange else -np.inf for j in jnts])
            hi = np.array([self._joints[j].actfrcrange[1] if self._joints[j].actfrcrange else np.inf for j in jnts])
            self._drive.append(dict(
                uid=uid, links=[self._link[self._joints[j].body][1] for j in jnts],
                q0=np.array([self._joints[j].qpos0 for j in jnts]),
                act=np.array([a.id for a in alist]), row=np.array([row[a.joint] for a in alist]),
                gear=np.array([a.gear for a in alist]), gain=np.array([a.gain for a in alist]),
                bias=np.array([a.bias for a in alist]),
                clo=np.array([a.ctrlrange[0] if a.ctrlrange else -np.inf for a in alist]),
                chi=np.array([a.ctrlrange[1] if a.ctrlrange else np.inf for a in alist]),
                flo=np.array([a.forcerange[0] if a.forcerange else -np.inf for a in alist]),
                fhi=np.array([a.forcerange[1] if a.forcerange else np.inf for a in alist]),
                tlo=lo, thi=hi, n=len(jnts),
            ))
        g = -np.asarray(self.scene.model.opt.gravity, dtype=np.float64)
        weld = self.scene.model.body_weldid
        self._gravcomp = [  # bodies welded to the world never move: no force needed
            (self._link[b.id], (b.gravcomp * b.mass * g).tolist())
            for b in self._bodies if b.gravcomp and b.mass > 0 and weld[b.id] != 0
        ]

    def _build_collision_filter(self) -> None:
        """MuJoCo's never-collide pairs (same weld, weld parent, <exclude>) as pair filters
        for stepping, and kept as ``_apart`` for contacts()."""
        self._places: dict[int, list[tuple[int, int]]] = defaultdict(list)  # canonical body -> its Bullet shapes
        for (uid, link), b in self._body_at.items():
            self._places[b].append((uid, link))
        self._apart = reader.no_collide_pairs(self.scene.model)
        for a, b in self._apart:
            for ua, la in self._places[a]:
                for ub, lb in self._places[b]:
                    if a in self._link or b in self._link:  # two static bodies never collide anyway
                        self._p.setCollisionFilterPair(ua, ub, la, lb, 0)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        p, self._p = self._p, None
        if p is not None:
            p.disconnect()
            shutil.rmtree(self._tmp, ignore_errors=True)

    def time(self) -> float:
        return self._steps * self._dt

    def step(self, n: int = 1) -> None:
        p = self._p
        for _ in range(int(n)):
            with self._lock:
                self._apply_forces()
                p.stepSimulation()
                self._steps += 1

    def _apply_forces(self) -> None:
        """Re-issued before every physics step: Bullet clears torques and forces after one."""
        p = self._p
        for dv in self._drive:
            st = p.getJointStates(dv["uid"], dv["links"])
            q = np.array([s[0] for s in st]) + dv["q0"]
            qd = np.array([s[1] for s in st])
            r = dv["row"]
            ctrl = np.clip(self._ctrl[dv["act"]], dv["clo"], dv["chi"])
            force = dv["gain"] * ctrl + dv["bias"][:, 0] + dv["bias"][:, 1] * dv["gear"] * q[r] + dv["bias"][:, 2] * dv["gear"] * qd[r]
            force = np.clip(force, dv["flo"], dv["fhi"])
            tau = np.zeros(dv["n"])
            np.add.at(tau, r, dv["gear"] * force)
            p.setJointMotorControlArray(dv["uid"], dv["links"], pb.TORQUE_CONTROL,
                                        forces=np.clip(tau, dv["tlo"], dv["thi"]).tolist())
        for (uid, link), f in self._gravcomp:
            com = p.getBasePositionAndOrientation(uid)[0] if link < 0 else p.getLinkState(uid, link)[0]
            p.applyExternalForce(uid, link, f, com, pb.WORLD_FRAME)

    # -- state -----------------------------------------------------------------

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.scene.model
        q, qd = np.zeros(m.nq), np.zeros(m.nv)
        p = self._p
        with self._lock:
            for j in self._joints:
                uid, link = self._link[j.body]
                if j.type == JNT_FREE:
                    ip, iq = self._iframe[j.body]
                    cp, cq = p.getBasePositionAndOrientation(uid)
                    v, w = p.getBaseVelocity(uid)
                    fq = _qmul(_wxyz(cq), _qconj(iq))
                    if np.dot(fq, self._qref[j.id]) < 0:  # Bullet may hand back -q: keep the caller's sign
                        fq = -fq
                    self._qref[j.id] = fq
                    R = _qmat(fq)
                    fp = np.array(cp) - R @ ip
                    q[j.qadr:j.qadr + 3], q[j.qadr + 3:j.qadr + 7] = fp, fq
                    qd[j.dadr:j.dadr + 3] = np.array(v) + np.cross(w, fp - np.array(cp))
                    qd[j.dadr + 3:j.dadr + 6] = R.T @ np.array(w)
                else:
                    pos, vel = p.getJointState(uid, link)[:2]
                    q[j.qadr], qd[j.dadr] = pos + j.qpos0, vel
        return q, qd

    def reset(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        m = self.scene.model
        q = np.asarray(q, dtype=np.float64).reshape(m.nq)
        qd = np.zeros(m.nv) if qd is None else np.asarray(qd, dtype=np.float64).reshape(m.nv)
        p = self._p
        with self._lock:
            for root, (ip, iq) in self._iframe.items():
                uid = self._link[root][0]
                body = self._bodies[root]
                if body.joint < 0:  # a tree welded to the world sits at its model pose
                    fp, fq = self._static_pose[root]
                    p.resetBasePositionAndOrientation(uid, (fp + _qmat(fq) @ ip).tolist(), _xyzw(_qmul(fq, iq)))
            for j in self._joints:
                uid, link = self._link[j.body]
                if j.type == JNT_FREE:
                    ip, iq = self._iframe[j.body]
                    fp, fq = q[j.qadr:j.qadr + 3], q[j.qadr + 3:j.qadr + 7]
                    self._qref[j.id] = fq.copy()
                    R = _qmat(fq)
                    cp = fp + R @ ip
                    w = R @ qd[j.dadr + 3:j.dadr + 6]
                    p.resetBasePositionAndOrientation(uid, cp.tolist(), _xyzw(_qmul(fq, iq)))
                    p.resetBaseVelocity(uid, (qd[j.dadr:j.dadr + 3] + np.cross(w, cp - fp)).tolist(), w.tolist())
                else:
                    p.resetJointState(uid, link, float(q[j.qadr] - j.qpos0), float(qd[j.dadr]))
            for a in self._motor.values():
                self._ctrl[a.id] = 0.0
            p.performCollisionDetection()

    # -- actuation -----------------------------------------------------------

    def _actuators(self, table: dict, joints: Sequence[int], what: str) -> list:
        try:
            return [table[int(j)] for j in joints]
        except KeyError as exc:
            have = sorted(self._joints[j].name for j in table)
            raise KeyError(f"joint {exc.args[0]} has no {what} actuator; joints with one: {have}") from None

    def set_targets(self, joints: Sequence[int], q: Sequence[float]) -> None:
        acts = self._actuators(self._position, joints, "position")
        q = np.asarray(q, dtype=np.float64).reshape(len(acts))
        for a, v in zip(acts, q):
            self._ctrl[a.id] = a.gear * v

    def targets(self, joints: Sequence[int]) -> np.ndarray:
        acts = self._actuators(self._position, joints, "position")
        return np.array([self._ctrl[a.id] / a.gear for a in acts])

    def set_torque(self, joints: Sequence[int], tau: Sequence[float]) -> None:
        acts = self._actuators(self._motor, joints, "motor")
        tau = np.asarray(tau, dtype=np.float64).reshape(len(acts))
        for a, t in zip(acts, tau):
            c = t / (a.gear * a.gain)
            self._ctrl[a.id] = c if a.ctrlrange is None else min(max(c, a.ctrlrange[0]), a.ctrlrange[1])

    # -- contact and attachment ----------------------------------------------

    def contacts(self, body: int) -> frozenset[int]:
        # A distance query over the shapes whose broadphase boxes meet this body's box, not
        # getContactPoints: Bullet keeps a separated point only inside its per-shape breaking
        # threshold, under 1 mm for small shapes (a 2 cm cylinder reads nothing at 0.2 mm).
        # The broadphase boxes carry Bullet's 2 cm padding (measured), so every shape within
        # the threshold is a candidate. getClosestPoints ignores the collision filter, so
        # MuJoCo's never-collide pairs are dropped here. Stepping is untouched.
        body, p, out = int(body), self._p, set()
        with self._lock:
            for ua, la in self._places.get(body, ()):
                for ub, lb in p.getOverlappingObjects(*p.getAABB(ua, la)) or ():
                    b = self._body_at[(ub, lb)]
                    if (b != body and b not in out and (min(b, body), max(b, body)) not in self._apart
                            and p.getClosestPoints(ua, ub, CONTACT_DISTANCE_M, la, lb)):
                        out.add(b)
        return frozenset(out)

    def _com(self, body: int) -> tuple[list[float], list[float]]:
        uid, link = self._link[body]
        if link < 0:
            return self._p.getBasePositionAndOrientation(uid)
        return self._p.getLinkState(uid, link, computeForwardKinematics=1)[:2]

    def attach(self, parent: int, child: int) -> None:
        pair = (int(parent), int(child))
        if pair not in self.scene.grasp_pairs:
            names = [(self.scene.body_name(a), self.scene.body_name(b)) for a, b in self.scene.grasp_pairs]
            raise KeyError(f"({parent}, {child}) is not a grasp pair; grasp pairs: {names}")
        held = self._constraints.get(pair[1])
        if held is not None:
            if held[0] != pair[0]:
                raise ValueError(f"body {child} is already attached to body {held[0]}; detach it first")
            self.detach(pair[1])
        p = self._p
        with self._lock:
            pp, pq = self._com(pair[0])
            cp, cq = self._com(pair[1])
            ip, iq = p.invertTransform(pp, pq)
            rel_p, rel_q = p.multiplyTransforms(ip, iq, cp, cq)  # frames are COM-relative in Bullet
            (pu, pl), (cu, cl) = self._link[pair[0]], self._link[pair[1]]
            cid = p.createConstraint(pu, pl, cu, cl, pb.JOINT_FIXED, [0, 0, 0], rel_p, [0, 0, 0], rel_q, [0, 0, 0, 1])
        self._constraints[pair[1]] = (pair[0], cid)

    def detach(self, child: int) -> None:
        held = self._constraints.pop(int(child), None)
        if held is not None:
            with self._lock:
                self._p.removeConstraint(held[1])

    def attached(self) -> dict[int, int]:
        return {c: p for c, (p, _) in self._constraints.items()}

    # -- sensing ---------------------------------------------------------------

    def raycast(
        self, origins: np.ndarray, directions: np.ndarray, exclude_body: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        v = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        dist, body = np.full(len(o), np.inf), np.full(len(o), -1, dtype=np.int64)

        def cast(idx: np.ndarray, **kw: int) -> list:
            out = []
            batch = int(pb.MAX_RAY_INTERSECTION_BATCH_SIZE)
            for k in range(0, len(idx), batch):
                part = idx[k:k + batch]
                with self._lock:
                    out += self._p.rayTestBatch(o[part].tolist(), (o[part] + RAY_RANGE_M * v[part]).tolist(), **kw)
            return out

        def keep(i: int, h: tuple) -> bool:
            """Record hit *h* of ray *i*; False when it is on the excluded body."""
            if h[0] < 0:
                return True
            b = self._body_at[(h[0], h[1])]
            if b == exclude_body:
                return False
            d = float(np.dot(np.array(h[3]) - o[i], v[i]))
            if d < dist[i]:
                dist[i], body[i] = d, b
            return True

        every = np.arange(len(o))
        hidden = [i for i, h in zip(every, cast(every)) if not keep(i, h)]
        # The excluded body's own geometry is transparent: walk every hit of those rays
        # (Bullet hands them back unordered) and keep the nearest other one.
        for k in range(_RAY_PASSES if hidden else 0):
            hits = cast(np.array(hidden), reportHitNumber=k)
            if all(h[0] < 0 for h in hits):
                break
            for i, h in zip(hidden, hits):
                keep(i, h)
        miss = ~np.isfinite(dist)
        dist[miss], body[miss] = -1.0, -1
        return dist, body

    def _frame(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        """The engine's body-frame pose (static bodies sit at their model pose)."""
        if body not in self._link:
            return self._static_pose[body]
        uid, link = self._link[body]
        if link >= 0:
            s = self._p.getLinkState(uid, link, computeForwardKinematics=1)
            return np.array(s[4]), _wxyz(s[5])
        ip, iq = self._iframe[body]
        cp, cq = self._p.getBasePositionAndOrientation(uid)
        fq = _qmul(_wxyz(cq), _qconj(iq))
        return np.array(cp) - _qmat(fq) @ ip, fq

    def render(
        self, camera: str, width: int, height: int, kinds: tuple[str, ...] = ("rgb", "depth"),
    ) -> RGBDFrame:
        kinds = tuple(kinds)
        bad = [k for k in kinds if k not in _KINDS]
        if bad:
            raise ValueError(f"unknown render kinds {bad}; kinds: {list(_KINDS)}")
        if camera not in self._cams:
            raise KeyError(f"no camera {camera!r}; cameras: {sorted(self._cams)}")
        cam, w, h = self._cams[camera], int(width), int(height)
        p = self._p
        with self._lock:
            bp, bq = self._frame(cam.body)
            R = _qmat(bq) @ _qmat(cam.quat)
            eye = bp + _qmat(bq) @ cam.pos
            view = p.computeViewMatrix(eye.tolist(), (eye - R[:, 2]).tolist(), R[:, 1].tolist())
            proj = list(p.computeProjectionMatrixFOV(cam.fovy, w / h, self._near, self._far))
            # TinyRenderer samples pixel corners: move the image half a pixel so pixel
            # (row, col) samples its centre and K's cx = W/2, cy = H/2 hold (measured)
            proj[8] += 1.0 / w
            proj[9] += 1.0 / h
            flags = pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if "seg" in kinds else pb.ER_NO_SEGMENTATION_MASK
            _, _, rgba, zbuf, seg = p.getCameraImage(w, h, view, proj, renderer=pb.ER_TINY_RENDERER, flags=flags, shadow=0)
            t = self.time()
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy() if "rgb" in kinds else None
        depth = None
        if "depth" in kinds:
            zb = np.asarray(zbuf, dtype=np.float64).reshape(h, w)
            z = self._far * self._near / (self._far - (self._far - self._near) * zb)
            depth = np.where(z >= 0.999 * self._far, 0.0, z).astype(np.float32)
        out_seg = None
        if "seg" in kinds:
            s = np.asarray(seg, dtype=np.int64).reshape(h, w)
            vals, inv = np.unique(s, return_inverse=True)
            lut = np.array([-1 if v < 0 else self._body_at.get((int(v) & 0xFFFFFF, (int(v) >> 24) - 1), -1) for v in vals])
            out_seg = lut[inv.reshape(h, w)].astype(np.int32)
        fy = (h / 2.0) / np.tan(np.radians(cam.fovy) / 2.0)
        return RGBDFrame(
            rgb=rgb, depth=depth, seg=out_seg,
            K=np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]]),
            cam_pos=eye, cam_mat=R, sim_time=t, camera=camera,
        )

    # -- test seams (conformance only) ----------------------------------------

    def _link_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """The engine's own body-frame poses; NaN for static bodies (no Bullet link)."""
        n = len(self._bodies)
        pos, quat = np.full((n, 3), np.nan), np.full((n, 4), np.nan)
        with self._lock:
            for b in self._link:
                pos[b], quat[b] = self._frame(b)
        return pos, quat

    def _inertials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The engine's own ``(mass, COM, inertia tensor)`` in the body frame, NaN where
        the body does not move (no Bullet link, or a fixed base)."""
        n = len(self._bodies)
        mass, com, tensor = np.full(n, np.nan), np.full((n, 3), np.nan), np.full((n, 3, 3), np.nan)
        for b, (uid, link) in self._link.items():
            if link < 0 and self._bodies[b].joint < 0:
                continue
            info = self._p.getDynamicsInfo(uid, link)
            R = _qmat(_wxyz(info[4]))
            mass[b], com[b], tensor[b] = info[0], info[3], R @ np.diag(info[2]) @ R.T
        return mass, com, tensor
