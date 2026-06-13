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
    RoomObject("obstacle_center", 1.6, 0.0, 0.25, 0.7, 0.4, (0.8, 0.3, 0.2, 1)),
    RoomObject("obstacle_left", 2.9, 1.1, 0.3, 0.3, 0.4, (0.8, 0.4, 0.2, 1)),
    RoomObject("obstacle_right", 2.9, -1.1, 0.3, 0.3, 0.4, (0.8, 0.4, 0.2, 1)),
)

# Labeled target objects — nav goals (NOT obstacles). target_red is behind the
# centre obstacle so the planner must route around it.
TARGETS: tuple = (
    RoomObject("target_red", 3.7, 0.0, 0.12, 0.12, 0.25, (0.9, 0.1, 0.1, 1)),
    RoomObject("target_blue", 3.7, 2.0, 0.12, 0.12, 0.25, (0.1, 0.2, 0.9, 1)),
    RoomObject("target_green", 3.7, -2.0, 0.12, 0.12, 0.25, (0.1, 0.8, 0.2, 1)),
)


def target_position(label: str) -> "tuple[float, float] | None":
    """World (x, y) of a labeled target, or None if unknown."""
    for t in TARGETS:
        if t.name == label or t.name == f"target_{label}":
            return (t.cx, t.cy)
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


def build_room_model(asset_dir: "Path | str") -> Any:
    """Compile a G1 room MjModel from the flat gait scene + walls/obstacles/
    targets. Returns a ``mujoco.MjModel`` (asset paths stay intact)."""
    import mujoco

    asset_dir = Path(asset_dir)
    spec = mujoco.MjSpec.from_file(str(asset_dir / "scene.xml"))
    for obj in (*_wall_specs(), *OBSTACLES, *TARGETS):
        body = spec.worldbody.add_body(name=obj.name, pos=[obj.cx, obj.cy, obj.hz])
        body.add_geom(
            name=f"{obj.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[obj.hx, obj.hy, obj.hz],
            rgba=list(obj.rgba),
        )
    return spec.compile()
