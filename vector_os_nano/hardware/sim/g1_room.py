# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R3 — a semantically structured G1 room scene with REAL collision.

Built programmatically with ``mujoco.MjSpec`` from the flat g1 gait scene
(keeps the downloaded asset paths + the policy's MJCF intact — the same trick
the R1 PROBE spike proved), then ADDS:
  - 4 perimeter walls (a closed ~7x6 m room) — static collision boxes;
  - obstacle boxes the gait must physically route AROUND (no pass-through);
  - 3 labeled target objects (named ``target_<label>``) the nav loop drives to.

Naming convention is the planner's contract (see g1_vgraph.obstacles_from_model):
walls ``wall_*`` and obstacles ``obstacle_*`` are routing geometry; ``target_*``
are goals (NOT obstacles). Everything is a STATIC body (no joint → welded to the
world), so the G1 collides with it but it never falls over. This module holds
NO control logic — it only assembles geometry; G1MuJoCoBase drives it.

Substrate-agnostic (campaign #8 DQ-10): MuJoCo physics + collision are reused
unchanged whether the owner picks substrate A (MuJoCo-as-world) or D (co-sim),
so building this does not pre-commit the gated photoreal decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Room layout (x forward, y left, metres). The G1 spawns at the origin facing
# +x; target_red sits straight ahead BEHIND the centre obstacle, so reaching it
# REQUIRES routing around real collision geometry (the honest avoidance test).
_ROOM_HALF_X = 3.6        # interior spans x in [-3.0, 4.2]-ish after wall inset
_ROOM_HALF_Y = 3.0
_WALL_T = 0.10            # wall half-thickness
_WALL_H = 0.6            # wall half-height
_WALL_CX = 1.0            # room centred a little ahead of the spawn
# Walls/obstacles/targets are tagged with this MuJoCo geom group so the lidar
# can ray-cast against ONLY the environment (mask this group) and never the
# G1's own body. Group is render/ray metadata — collision (contype) is
# unaffected, so physics/blocking still work.
ENV_GEOM_GROUP = 3
# Name of the first-person forward camera mounted on the pelvis (R9 vision).
HEAD_CAM = "g1_head_cam"


@dataclass(frozen=True)
class RoomObject:
    """A named static box in the room (obstacle or target)."""
    name: str
    cx: float
    cy: float
    hx: float
    hy: float
    hz: float
    rgba: tuple


# Obstacle boxes — the gait must physically avoid these (collision, no 穿模).
OBSTACLES: tuple = (
    # Grey/neutral so vision recognition (red/blue/green) is unambiguous — the
    # obstacles are COLLISION geometry, not colour targets. Collision is
    # unaffected by colour. The centre obstacle is offset off the x-axis so a
    # target straight ahead has clear line-of-sight for visual servoing.
    RoomObject("obstacle_center", 1.7, 0.9, 0.25, 0.5, 0.4, (0.55, 0.55, 0.58, 1)),
    RoomObject("obstacle_left", 2.9, 1.8, 0.3, 0.3, 0.4, (0.55, 0.55, 0.58, 1)),
    RoomObject("obstacle_right", 2.4, -1.3, 0.3, 0.3, 0.4, (0.55, 0.55, 0.58, 1)),
)

# Labeled target objects — nav goals (NOT obstacles). target_red is behind the
# centre obstacle so the planner must route around it.
TARGETS: tuple = (
    RoomObject("target_red", 3.7, 0.0, 0.12, 0.12, 0.25, (0.9, 0.1, 0.1, 1)),
    RoomObject("target_blue", 3.7, 2.0, 0.12, 0.12, 0.25, (0.1, 0.2, 0.9, 1)),
    RoomObject("target_green", 3.7, -2.0, 0.12, 0.12, 0.25, (0.1, 0.8, 0.2, 1)),
)


@dataclass(frozen=True)
class FurnitureTarget:
    """A named real-mesh furniture target (campaign #9 R1, track A).

    Unlike the colour boxes, these are recognisable household objects (Kenney
    CC0 meshes) so a REAL VLM (Qwen-VL) can ground a SEMANTIC class — 'chair',
    'sofa', 'potted plant' — not just a colour. ``label`` is the natural class
    name the VLM/seek query uses; ``mesh_file`` is relative to the furniture
    asset dir; ``cx, cy`` is the desired VISIBLE centre (the body is shifted so
    the mesh sits centred there and on the floor — meshes have off-origin pivots).
    """
    name: str          # MuJoCo body name (target_<label>)
    label: str         # semantic class for VLM query / world model
    mesh_file: str     # OBJ under the furniture asset dir
    scale: float
    cx: float
    cy: float


# Furniture asset dir (Kenney CC0 OBJ meshes), shared with the go2 room scene.
_FURNITURE_DIR = (Path(__file__).parent / "mjcf" / "go2" / "assets"
                  / "furniture")
# The Y-up OBJ → Z-up MuJoCo correction: a +90° rotation about world X. As a
# quat (w, x, y, z). (the go2 scene applies the same via euler="1.5708 0 0".)
_YUP_TO_ZUP_QUAT = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)

# 3 semantic furniture targets, laid out like the colour room (back of the room,
# spread in y) so the same explore→recognise→go loop applies. Distinct, easily
# named classes maximise a real VLM's recognition odds on MuJoCo's render.
FURNITURE: tuple = (
    FurnitureTarget("target_chair", "chair", "chair.obj", 2.1, 3.6, 0.0),
    FurnitureTarget("target_sofa", "sofa", "loungeSofa_carpet.obj", 2.2, 3.6, 2.0),
    FurnitureTarget("target_plant", "potted plant",
                    "pottedPlant_plant.obj", 1.8, 3.6, -2.0),
)


def furnished_targets() -> "dict[str, tuple[float, float]]":
    """{body_name: (cx, cy)} for the furnished room's semantic targets — the
    visible-centre coordinates the nav loop drives to (world-model GT)."""
    return {f.name: (f.cx, f.cy) for f in FURNITURE}


def room_bounds() -> "tuple[float, float, float, float]":
    """Interior (x_min, y_min, x_max, y_max) of the room — the occupancy
    grid's extent for exploration coverage."""
    cx, hx, hy = _WALL_CX, _ROOM_HALF_X, _ROOM_HALF_Y
    return (cx - hx, -hy, cx + hx, hy)


def target_position(label: str) -> "tuple[float, float] | None":
    """World (x, y) of a labeled target (colour box OR furniture), or None."""
    for t in TARGETS:
        if t.name == label or t.name == f"target_{label}":
            return (t.cx, t.cy)
    key = (label or "").strip().lower()
    for f in FURNITURE:
        if key in (f.name, f"target_{key}", f.label):
            return (f.cx, f.cy)
    return None


def _wall_specs() -> "list[RoomObject]":
    """4 perimeter walls as static collision boxes (closed room)."""
    cx, hx, hy = _WALL_CX, _ROOM_HALF_X, _ROOM_HALF_Y
    grey = (0.5, 0.5, 0.55, 1)
    return [
        RoomObject("wall_back", cx - hx, 0.0, _WALL_T, hy, _WALL_H, grey),
        RoomObject("wall_front", cx + hx, 0.0, _WALL_T, hy, _WALL_H, grey),
        RoomObject("wall_left", cx, hy, hx, _WALL_T, _WALL_H, grey),
        RoomObject("wall_right", cx, -hy, hx, _WALL_T, _WALL_H, grey),
    ]


def obstacles_from_model(
    model: Any,
    data: Any,
    prefixes: "tuple[str, ...]" = ("obstacle", "wall"),
    names_only: "set[str] | None" = None,
) -> "list[list[tuple[float, float]]]":
    """Enumerate routing polygons from a COMPILED MuJoCo model.

    Reads every BOX / CYLINDER geom whose owning body name starts with one of
    ``prefixes`` (default obstacle_*/wall_*) and returns its top-down polygon in
    world frame from the forwarded ``data`` (call mj_forward first). Targets
    (``target_*``) are deliberately excluded — they are nav GOALS, not
    obstacles. ``names_only`` restricts to an exact body-name set (tests).

    Lives HERE (the model-bridge module), not in g1_vgraph, so the planner
    stays pure-geometry / mujoco-free; it consumes box_polygon/cylinder_polygon
    from the pure planner.
    """
    import mujoco

    from vector_os_nano.hardware.sim import g1_vgraph as vg

    polys: list = []
    for gid in range(int(model.ngeom)):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if names_only is not None:
            if bname not in names_only:
                continue
        elif not any(bname.startswith(p) for p in prefixes):
            continue
        gtype = int(model.geom_type[gid])
        cx, cy = float(data.geom_xpos[gid][0]), float(data.geom_xpos[gid][1])
        size = model.geom_size[gid]
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            xm = data.geom_xmat[gid]
            import math
            yaw = math.atan2(float(xm[3]), float(xm[0]))   # R[1,0], R[0,0]
            polys.append(
                vg.box_polygon(cx, cy, float(size[0]), float(size[1]), yaw))
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            polys.append(vg.cylinder_polygon(cx, cy, float(size[0])))
    return polys


def _furniture_placement(mujoco: Any, asset_dir: Path
                         ) -> "dict[str, tuple[float, float, float]]":
    """Measure each furniture mesh's body-frame footprint (a throwaway compile)
    so the caller can centre it on (cx, cy) and rest it on the floor.

    Meshes have off-origin pivots: at body pos (0,0,0) the rotated mesh's
    bounding-box centre is some (ox, oy) and its lowest vertex is at oz. Returns
    {name: (ox, oy, oz)}; the real body pos is then (cx-ox, cy-oy, -oz)."""
    import numpy as np

    spec = mujoco.MjSpec.from_file(str(asset_dir / "scene.xml"))
    for f in FURNITURE:
        mesh = spec.add_mesh()
        mesh.name = f"m_{f.name}"
        mesh.file = str(_FURNITURE_DIR / f.mesh_file)
        mesh.scale = [f.scale, f.scale, f.scale]
        body = spec.worldbody.add_body(name=f.name, pos=[0.0, 0.0, 0.0])
        g = body.add_geom()
        g.name = f"{f.name}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = f"m_{f.name}"
        g.quat = list(_YUP_TO_ZUP_QUAT)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)     # populate geom_xpos/geom_xmat (incl. the
                                       # mesh-recentre geom_pos compensation)
    out: dict = {}
    for f in FURNITURE:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f.name}_geom")
        mid = int(model.geom_dataid[gid])
        adr = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        verts = np.array(model.mesh_vert[adr:adr + n]).reshape(-1, 3)
        rmat = np.array(data.geom_xmat[gid]).reshape(3, 3)
        world = verts @ rmat.T + np.array(data.geom_xpos[gid])  # body at origin
        ox = float((world[:, 0].min() + world[:, 0].max()) / 2.0)
        oy = float((world[:, 1].min() + world[:, 1].max()) / 2.0)
        oz = float(world[:, 2].min())
        out[f.name] = (ox, oy, oz)
    return out


def build_room_model(asset_dir: "Path | str", furnished: bool = False) -> Any:
    """Compile a G1 room MjModel from the flat gait scene + walls/obstacles/
    targets. Returns a ``mujoco.MjModel`` (asset paths stay intact).

    ``furnished`` (campaign #9 R1, track A): swap the 3 saturated colour boxes
    for 3 real Kenney furniture meshes (chair / sofa / potted plant) so a REAL
    VLM can ground a semantic class instead of a colour. Walls + the 3 grey
    collision obstacles are unchanged either way; the colour-box room (default)
    is byte-for-byte the campaign #8 scene."""
    import mujoco
    import numpy as np

    asset_dir = Path(asset_dir)
    spec = mujoco.MjSpec.from_file(str(asset_dir / "scene.xml"))
    statics = (*_wall_specs(), *OBSTACLES) if furnished else (
        *_wall_specs(), *OBSTACLES, *TARGETS)
    for obj in statics:
        body = spec.worldbody.add_body(name=obj.name, pos=[obj.cx, obj.cy, obj.hz])
        body.add_geom(
            name=f"{obj.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[obj.hx, obj.hy, obj.hz],
            rgba=list(obj.rgba),
            group=ENV_GEOM_GROUP,   # lidar masks to this group → ignores robot
        )
    if furnished:
        place = _furniture_placement(mujoco, asset_dir)
        for f in FURNITURE:
            ox, oy, oz = place[f.name]
            body = spec.worldbody.add_body(
                name=f.name, pos=[f.cx - ox, f.cy - oy, -oz])
            mesh = spec.add_mesh()
            mesh.name = f"m_{f.name}"
            mesh.file = str(_FURNITURE_DIR / f.mesh_file)
            mesh.scale = [f.scale, f.scale, f.scale]
            g = body.add_geom()
            g.name = f"{f.name}_geom"
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = f"m_{f.name}"
            g.quat = list(_YUP_TO_ZUP_QUAT)
            g.group = ENV_GEOM_GROUP   # lidar/render env group (same as targets)
    # First-person forward camera mounted on the pelvis (campaign #8 R9 — visual
    # recognition). mode=fixed so it rotates with the robot; xyaxes orient it to
    # look along the body +x (forward): cam right = body -y, up = body +z, so
    # cam -z (view dir) = +x. Mounted (not free-cam) avoids orbit-azimuth guesswork.
    import numpy as np
    pelvis = spec.body("pelvis")
    cam = pelvis.add_camera()
    cam.name = HEAD_CAM
    cam.pos = [0.18, 0.0, 0.45]      # in front of + above the pelvis
    cam.fovy = 70.0                  # widened (was 60) for more vertical reach
    # Orient to look along body +x (forward), up = body +z. The camera frame
    # axes in body coords: x_c=-y, y_c=+z, z_c=-x (so view dir -z_c = +x).
    # MjSpec takes a quat (xyaxes silently no-ops), so convert that matrix.
    # Pitched DOWN ~6° (was ~12°): a FAR target at range sits near frame centre
    # (not bottom-clipped) so its pixel area clears the detector threshold at
    # spawn — fixes Case 15 acquisition. The wider fovy + this shallower pitch
    # still keep a low box in frame during the approach, and arrival uses the
    # progress-stall signal (not close-up detection), so losing the box at the
    # very end is harmless.
    R = np.array([[0.0, 0.1045, -0.9945],
                  [-1.0, 0.0, 0.0],
                  [0.0, 0.9945, 0.1045]], dtype=np.float64)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    cam.quat = q.tolist()
    return spec.compile()
