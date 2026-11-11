# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Read-only views of a compiled ``MjModel`` for backend authors.

A backend other than MuJoCo builds its world from these records instead of
parsing MJCF: every value is the COMPILED one (defaults applied, meshes fitted,
inertia in the principal frame). Ids are canonical (indices in the model).
Quaternions are ``w x y z``; poses are relative to the parent body frame.
"""
from __future__ import annotations

from typing import NamedTuple

import mujoco as mj
import numpy as np

_J, _G = mj.mjtJoint, mj.mjtGeom
JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE = (int(t) for t in (_J.mjJNT_FREE, _J.mjJNT_BALL, _J.mjJNT_SLIDE, _J.mjJNT_HINGE))
GEOM_PLANE, GEOM_HFIELD, GEOM_SPHERE, GEOM_CAPSULE, GEOM_ELLIPSOID, GEOM_CYLINDER, GEOM_BOX, GEOM_MESH = (
    int(t) for t in (_G.mjGEOM_PLANE, _G.mjGEOM_HFIELD, _G.mjGEOM_SPHERE, _G.mjGEOM_CAPSULE,
                     _G.mjGEOM_ELLIPSOID, _G.mjGEOM_CYLINDER, _G.mjGEOM_BOX, _G.mjGEOM_MESH))
EQ_WELD, EQ_JOINT = int(mj.mjtEq.mjEQ_WELD), int(mj.mjtEq.mjEQ_JOINT)
# Geom groups a default MuJoCo render shows (mjvOption.geomgroup).
VISIBLE_GROUPS = (0, 1, 2)
_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


class Body(NamedTuple):
    id: int
    name: str
    parent: int
    pos: np.ndarray          # frame in the parent frame
    quat: np.ndarray
    joint: int               # its joint id, -1 = welded to the parent
    mass: float
    ipos: np.ndarray         # centre of mass in the body frame
    iquat: np.ndarray        # principal axes in the body frame
    inertia: np.ndarray      # principal moments (3,)
    gravcomp: float          # fraction of gravity cancelled by an external force at the COM


class Joint(NamedTuple):
    id: int
    name: str
    type: int                # JNT_*
    body: int
    qadr: int
    dadr: int
    qpos0: float             # hinge / slide: the position at which the body sits at its model pose
    axis: np.ndarray         # in the body frame
    pos: np.ndarray          # anchor in the body frame
    range: tuple[float, float] | None
    damping: float
    armature: float
    frictionloss: float
    actfrcrange: tuple[float, float] | None   # clamp on the summed actuator force


class Geom(NamedTuple):
    id: int
    name: str
    body: int
    type: int                # GEOM_*
    size: np.ndarray
    pos: np.ndarray          # in the body frame
    quat: np.ndarray
    rgba: np.ndarray         # resolved: the geom's own colour, else its material's
    collide: bool            # contype or conaffinity non-zero
    visible: bool            # drawn by a default render: group 0-2 and alpha > 0
    friction: np.ndarray     # slide, spin, roll
    condim: int
    priority: int
    mesh: int                # mesh id, -1 = none


class Actuator(NamedTuple):
    id: int
    name: str
    joint: int
    kind: str                # "position" | "motor" | "other"
    gear: float
    gain: float              # force = gain * ctrl + bias[0] + bias[1] * length + bias[2] * velocity
    bias: np.ndarray         # (3,); length = gear * q, joint torque = gear * force
    ctrlrange: tuple[float, float] | None
    forcerange: tuple[float, float] | None


class Camera(NamedTuple):
    id: int
    name: str
    body: int
    pos: np.ndarray          # in the body frame; looks along -z, y up (OpenGL)
    quat: np.ndarray
    fovy: float              # degrees
    fixed: bool              # mode "fixed" (rigid on its body)


class Site(NamedTuple):
    id: int
    name: str
    body: int
    pos: np.ndarray
    quat: np.ndarray


class Coupling(NamedTuple):
    joint1: int              # q1 = sum(polycoef[k] * q2 ** k)
    joint2: int
    polycoef: np.ndarray     # (5,)


def _name(m: mj.MjModel, obj: mj.mjtObj, i: int) -> str:
    return mj.mj_id2name(m, obj, i) or ""


def _range(limited: bool, r: np.ndarray) -> tuple[float, float] | None:
    return (float(r[0]), float(r[1])) if limited else None


def bodies(m: mj.MjModel) -> list[Body]:
    return [
        Body(b, _name(m, mj.mjtObj.mjOBJ_BODY, b), int(m.body_parentid[b]), m.body_pos[b].copy(),
             m.body_quat[b].copy(), int(m.body_jntadr[b]) if m.body_jntnum[b] else -1,
             float(m.body_mass[b]), m.body_ipos[b].copy(), m.body_iquat[b].copy(),
             m.body_inertia[b].copy(), float(m.body_gravcomp[b]))
        for b in range(m.nbody)
    ]


def joints(m: mj.MjModel) -> list[Joint]:
    out = []
    for j in range(m.njnt):
        d, q = int(m.jnt_dofadr[j]), int(m.jnt_qposadr[j])
        out.append(Joint(
            j, _name(m, mj.mjtObj.mjOBJ_JOINT, j), int(m.jnt_type[j]), int(m.jnt_bodyid[j]), q, d,
            float(m.qpos0[q]), m.jnt_axis[j].copy(), m.jnt_pos[j].copy(),
            _range(bool(m.jnt_limited[j]), m.jnt_range[j]), float(m.dof_damping[d]),
            float(m.dof_armature[d]), float(m.dof_frictionloss[d]),
            _range(bool(m.jnt_actfrclimited[j]), m.jnt_actfrcrange[j]),
        ))
    return out


def geom_rgba(m: mj.MjModel, g: int) -> np.ndarray:
    """MuJoCo's colour rule: an explicit geom rgba wins, else the material's."""
    rgba = m.geom_rgba[g]
    if m.geom_matid[g] >= 0 and tuple(np.round(rgba, 6)) == _DEFAULT_RGBA:
        return m.mat_rgba[m.geom_matid[g]].copy()
    return rgba.copy()


def geoms(m: mj.MjModel) -> list[Geom]:
    out = []
    for g in range(m.ngeom):
        rgba = geom_rgba(m, g)
        out.append(Geom(
            g, _name(m, mj.mjtObj.mjOBJ_GEOM, g), int(m.geom_bodyid[g]), int(m.geom_type[g]),
            m.geom_size[g].copy(), m.geom_pos[g].copy(), m.geom_quat[g].copy(), rgba,
            bool(m.geom_contype[g] or m.geom_conaffinity[g]),
            int(m.geom_group[g]) in VISIBLE_GROUPS and rgba[3] > 0,
            m.geom_friction[g].copy(), int(m.geom_condim[g]), int(m.geom_priority[g]),
            int(m.geom_dataid[g]) if m.geom_type[g] == GEOM_MESH else -1,
        ))
    return out


def mesh(m: mj.MjModel, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V, 3) in the mesh frame (the geom pose places it) and faces (F, 3)."""
    va, vn = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
    fa, fn = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
    return m.mesh_vert[va:va + vn].astype(np.float64), m.mesh_face[fa:fa + fn].astype(np.int64)


def actuators(m: mj.MjModel) -> list[Actuator]:
    out = []
    for a in range(m.nu):
        gain, bias = float(m.actuator_gainprm[a, 0]), m.actuator_biasprm[a, :3].copy()
        simple = (m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT
                  and m.actuator_dyntype[a] == mj.mjtDyn.mjDYN_NONE
                  and m.actuator_gaintype[a] == mj.mjtGain.mjGAIN_FIXED)
        if simple and m.actuator_biastype[a] == mj.mjtBias.mjBIAS_NONE:
            kind = "motor"
        elif (simple and m.actuator_biastype[a] == mj.mjtBias.mjBIAS_AFFINE
              and gain > 0 and bias[0] == 0 and bias[1] == -gain):
            kind = "position"
        else:
            kind = "other"
        out.append(Actuator(
            a, _name(m, mj.mjtObj.mjOBJ_ACTUATOR, a),
            int(m.actuator_trnid[a, 0]) if m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT else -1,
            kind, float(m.actuator_gear[a, 0]), gain, bias,
            _range(bool(m.actuator_ctrllimited[a]), m.actuator_ctrlrange[a]),
            _range(bool(m.actuator_forcelimited[a]), m.actuator_forcerange[a]),
        ))
    return out


def cameras(m: mj.MjModel) -> list[Camera]:
    return [
        Camera(c, _name(m, mj.mjtObj.mjOBJ_CAMERA, c), int(m.cam_bodyid[c]), m.cam_pos[c].copy(),
               m.cam_quat[c].copy(), float(m.cam_fovy[c]), int(m.cam_mode[c]) == mj.mjtCamLight.mjCAMLIGHT_FIXED)
        for c in range(m.ncam)
    ]


def sites(m: mj.MjModel) -> list[Site]:
    return [
        Site(s, _name(m, mj.mjtObj.mjOBJ_SITE, s), int(m.site_bodyid[s]), m.site_pos[s].copy(), m.site_quat[s].copy())
        for s in range(m.nsite)
    ]


def couplings(m: mj.MjModel) -> list[Coupling]:
    """Active joint equalities (mimic joints): ``q1 = poly(q2)``."""
    return [
        Coupling(int(m.eq_obj1id[i]), int(m.eq_obj2id[i]), m.eq_data[i, :5].copy())
        for i in range(m.neq)
        if m.eq_type[i] == mj.mjtEq.mjEQ_JOINT and m.eq_active0[i]
    ]


def no_collide_pairs(m: mj.MjModel) -> frozenset[tuple[int, int]]:
    """Body pairs ``(a < b)``, both with colliding geoms, that MuJoCo never collides:
    the same weld group, a weld group and its parent's (filterparent), and every
    ``<exclude>``. Geom bitmasks are not folded in (see :func:`geoms`.collide)."""
    weld, parent = m.body_weldid, m.body_parentid
    filterparent = not (m.opt.disableflags & mj.mjtDisableBit.mjDSBL_FILTERPARENT)
    excluded = {(int(s >> 16), int(s & 0xFFFF)) for s in m.exclude_signature}
    colliding = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom)
                        if m.geom_contype[g] or m.geom_conaffinity[g]})
    out = set()
    for i, a in enumerate(colliding):
        wa, pa = int(weld[a]), int(weld[parent[weld[a]]])
        for b in colliding[i + 1:]:
            wb, pb = int(weld[b]), int(weld[parent[weld[b]]])
            if (wa == wb or (filterparent and wa and wb and (wa == pb or wb == pa))
                    or (a, b) in excluded or (b, a) in excluded):
                out.add((a, b))
    return frozenset(out)
