#!/usr/bin/env python3
"""Scene creation utilities for the Isaac Sim Go2 bridge.

Each function creates a complete scene in the given World and returns
the Robot handle for the Unitree Go2.

Isaac Sim asset paths:
  - Nucleus server: omniverse://localhost/Isaac/...
  - Local bundled: /isaac-sim/apps/isaacsim/...
  Prefer Nucleus when available; fall back to procedural geometry.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import GroundPlane, DynamicCuboid, FixedCuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
import omni.isaac.core.utils.numpy.rotations as rot_utils

logger = logging.getLogger("go2_scene")

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

# Unitree Go2 USD from Isaac Sim Nucleus asset library.
# Falls back to a box primitive if the asset cannot be resolved.
_GO2_USD_NUCLEUS = (
    "omniverse://localhost/Isaac/Robots/Unitree/Go2/go2.usd"
)
_GO2_USD_LOCAL = (
    "/isaac-sim/apps/isaacsim/exts/omni.isaac.robot_benchmark/assets/robots/unitree_go2.usd"
)

# Default spawn pose: 0.3 m above ground so the robot settles naturally
_SPAWN_POS = np.array([0.0, 0.0, 0.50])
_SPAWN_ORIENT = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z — identity


def _resolve_go2_usd() -> str:
    """Return the best available Go2 USD asset path."""
    nucleus = os.environ.get("OMNI_SERVER", "")
    if nucleus:
        return _GO2_USD_NUCLEUS
    # Check local bundled asset
    if os.path.exists(_GO2_USD_LOCAL):
        return _GO2_USD_LOCAL
    # Last resort: warn and return Nucleus path (may fail at runtime)
    logger.warning(
        "Go2 USD not found locally; using Nucleus path. "
        "Set OMNI_SERVER env var or copy asset to %s",
        _GO2_USD_LOCAL,
    )
    return _GO2_USD_NUCLEUS


def _add_go2(world: World, position: np.ndarray | None = None) -> Robot:
    """Add the Unitree Go2 to the stage and return the Robot handle.

    Tries in order:
    1. GO2_USD_PATH env var (local file)
    2. Nucleus/S3 download (Isaac Sim assets CDN)
    3. Placeholder box fallback
    """
    pos = position if position is not None else _SPAWN_POS
    prim_path = "/World/Go2"

    # Option 1: local USD
    go2_usd = os.environ.get("GO2_USD_PATH", "")
    if go2_usd and os.path.exists(go2_usd):
        try:
            add_reference_to_stage(usd_path=go2_usd, prim_path=prim_path)
            robot = world.scene.add(
                Robot(prim_path=prim_path, name="go2", position=pos, orientation=_SPAWN_ORIENT)
            )
            logger.info("Go2 loaded from local: %s", go2_usd)
            return robot
        except Exception as exc:
            logger.warning("Local Go2 USD failed: %s", exc)

    # Option 2: Nucleus/S3 download
    try:
        from isaacsim.storage.native import get_assets_root_path
        assets_root = get_assets_root_path()
        nucleus_path = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"
        logger.info("Downloading Go2 from %s ...", nucleus_path)
        add_reference_to_stage(usd_path=nucleus_path, prim_path=prim_path)
        robot = world.scene.add(
            Robot(prim_path=prim_path, name="go2", position=pos, orientation=_SPAWN_ORIENT)
        )
        logger.info("Go2 loaded from Nucleus (%d DOF)", robot.num_dof)
        return robot
    except Exception as exc:
        logger.warning("Nucleus Go2 download failed: %s", exc)

    # Option 3: placeholder
    logger.info("Using Go2 placeholder box")
    return world.scene.add(
        DynamicCuboid(prim_path=prim_path, name="go2", position=pos,
                      scale=np.array([0.6, 0.3, 0.2]), mass=12.0)
    )


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def create_flat_scene(world: World) -> Robot:
    """Create a flat ground-plane scene with Go2, a camera, and an RTX lidar.

    Layout:
      - Infinite ground plane at Z=0
      - Unitree Go2 spawned at (0, 0, 0.5)
      - Directional light
      - Placeholder camera and lidar prims (sensors wired up inside bridge)

    Args:
        world: Active Isaac Sim World.

    Returns:
        Robot handle for the Go2.
    """
    # Ground (procedural — no Nucleus download)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.025]),
            scale=np.array([50.0, 50.0, 0.05]),
        )
    )

    # Robot
    robot = _add_go2(world)

    # Light
    _add_distant_light(intensity=1500.0)

    logger.info("Flat scene created")
    return robot


def create_room_scene(world: World) -> Robot:
    """Create a 4 m x 5 m room with Go2 and minimal furniture.

    Layout:
      - Floor slab (FixedCuboid) 4 m x 5 m x 0.05 m
      - Four walls, 2.5 m high
      - One table (0.8 m x 0.4 m x 0.75 m) in a corner
      - One chair beside the table
      - Go2 spawned near the centre

    Args:
        world: Active Isaac Sim World.

    Returns:
        Robot handle for the Go2.
    """
    room_x, room_y, wall_h = 4.0, 5.0, 2.5
    t = 0.1  # wall thickness

    # Floor
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Room/Floor",
            name="floor",
            position=np.array([0.0, 0.0, -0.025]),
            scale=np.array([room_x, room_y, 0.05]),
        )
    )

    # Walls: south, north, west, east
    _add_wall("/World/Room/WallS", np.array([0.0, -room_y / 2, wall_h / 2]),
              np.array([room_x, t, wall_h]))
    _add_wall("/World/Room/WallN", np.array([0.0,  room_y / 2, wall_h / 2]),
              np.array([room_x, t, wall_h]))
    _add_wall("/World/Room/WallW", np.array([-room_x / 2, 0.0, wall_h / 2]),
              np.array([t, room_y, wall_h]))
    _add_wall("/World/Room/WallE", np.array([ room_x / 2, 0.0, wall_h / 2]),
              np.array([t, room_y, wall_h]))

    # Table (corner at x=-1.5, y=1.5)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Room/Table",
            name="table",
            position=np.array([-1.5, 1.5, 0.375]),
            scale=np.array([0.8, 0.4, 0.75]),
        )
    )

    # Chair beside the table
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Room/Chair",
            name="chair",
            position=np.array([-1.5, 0.9, 0.225]),
            scale=np.array([0.4, 0.4, 0.45]),
        )
    )

    # Robot
    robot = _add_go2(world, position=np.array([0.0, 0.0, 0.50]))

    _add_distant_light(intensity=1200.0)

    logger.info("Room scene created (%.1f x %.1f m)", room_x, room_y)
    return robot


def create_apartment_scene(world: World) -> Robot:
    """Create a 3-room apartment (living room, bedroom, kitchen) with doors.

    Layout (approximate):
      - Living room: 4 m x 4 m at (-2, 0)
      - Bedroom:     3 m x 3.5 m at (3.5, 1)
      - Kitchen:     2.5 m x 3 m at (-2, -4)
      - Door openings cut into shared walls (1 m wide)
      - Go2 spawned in the living room

    Args:
        world: Active Isaac Sim World.

    Returns:
        Robot handle for the Go2.
    """
    wall_h = 2.5
    t = 0.15  # wall thickness

    # ── Living room ──────────────────────────────────────────────────────
    _add_floor("/World/Apt/LR/Floor",  np.array([-2.0, 0.0, -0.025]), np.array([4.0, 4.0, 0.05]))

    # Living room exterior walls (south/west/north — east is shared with bedroom)
    _add_wall("/World/Apt/LR/WallS",   np.array([-2.0, -2.0, wall_h / 2]), np.array([4.0, t, wall_h]))
    _add_wall("/World/Apt/LR/WallW",   np.array([-4.0,  0.0, wall_h / 2]), np.array([t, 4.0, wall_h]))
    _add_wall("/World/Apt/LR/WallN",   np.array([-2.0,  2.0, wall_h / 2]), np.array([4.0, t, wall_h]))

    # Shared wall LR/Bedroom (east side of LR / west side of BR)
    # Leave 1 m door gap at y=0 (centre)
    _add_wall_with_door(
        base_path="/World/Apt/LR/WallE",
        centre=np.array([0.0, 0.0, wall_h / 2]),
        full_scale=np.array([t, 4.0, wall_h]),
        door_width=1.0,
        door_height=2.1,
        door_offset_y=0.0,
    )

    # ── Bedroom ──────────────────────────────────────────────────────────
    _add_floor("/World/Apt/BR/Floor", np.array([3.5, 1.0, -0.025]), np.array([3.0, 3.5, 0.05]))

    _add_wall("/World/Apt/BR/WallN",  np.array([3.5,  2.75, wall_h / 2]), np.array([3.0, t, wall_h]))
    _add_wall("/World/Apt/BR/WallE",  np.array([5.0,  1.0,  wall_h / 2]), np.array([t, 3.5, wall_h]))
    _add_wall("/World/Apt/BR/WallS",  np.array([3.5, -0.75, wall_h / 2]), np.array([3.0, t, wall_h]))

    # Bed
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Apt/BR/Bed",
            name="bed",
            position=np.array([4.5, 2.0, 0.30]),
            scale=np.array([2.0, 1.4, 0.60]),
        )
    )

    # ── Kitchen ──────────────────────────────────────────────────────────
    _add_floor("/World/Apt/KT/Floor", np.array([-2.0, -5.0, -0.025]), np.array([2.5, 3.0, 0.05]))

    # Shared wall LR/Kitchen (south of LR / north of KT) already placed as WallS of LR.
    # Add door opening in kitchen north wall at x=-2
    _add_wall_with_door(
        base_path="/World/Apt/KT/WallN",
        centre=np.array([-2.0, -3.5, wall_h / 2]),
        full_scale=np.array([2.5, t, wall_h]),
        door_width=0.9,
        door_height=2.1,
        door_offset_y=0.0,
    )
    _add_wall("/World/Apt/KT/WallW",  np.array([-3.25, -5.0, wall_h / 2]), np.array([t, 3.0, wall_h]))
    _add_wall("/World/Apt/KT/WallE",  np.array([-0.75, -5.0, wall_h / 2]), np.array([t, 3.0, wall_h]))
    _add_wall("/World/Apt/KT/WallS",  np.array([-2.0,  -6.5, wall_h / 2]), np.array([2.5, t, wall_h]))

    # Counter top
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Apt/KT/Counter",
            name="counter",
            position=np.array([-3.0, -5.0, 0.45]),
            scale=np.array([0.6, 2.5, 0.90]),
        )
    )

    # ── Robot ─────────────────────────────────────────────────────────────
    robot = _add_go2(world, position=np.array([-2.0, 0.0, 0.50]))

    _add_distant_light(intensity=1000.0)

    logger.info("Apartment scene created (3 rooms)")
    return robot


# ---------------------------------------------------------------------------
# Private geometry helpers
# ---------------------------------------------------------------------------

def _add_wall(prim_path: str, position: np.ndarray, scale: np.ndarray) -> None:
    """Add a static wall cuboid to the active stage."""
    create_prim(
        prim_path=prim_path,
        prim_type="Cube",
        position=position,
        scale=scale,
        attributes={"physics:rigidBodyEnabled": False},
    )


def _add_floor(prim_path: str, position: np.ndarray, scale: np.ndarray) -> None:
    """Add a static floor cuboid to the active stage."""
    create_prim(
        prim_path=prim_path,
        prim_type="Cube",
        position=position,
        scale=scale,
        attributes={"physics:rigidBodyEnabled": False},
    )


def _add_wall_with_door(
    base_path: str,
    centre: np.ndarray,
    full_scale: np.ndarray,
    door_width: float,
    door_height: float,
    door_offset_y: float,
) -> None:
    """Add a wall with a door opening by splitting it into two segments.

    The wall runs along the Y axis. The door gap is centred at door_offset_y
    relative to the wall centre.

    Args:
        base_path: USD prim path prefix.
        centre: Wall centre position [x, y, z].
        full_scale: Full wall dimensions [thickness, length, height].
        door_width: Width of the door gap in metres.
        door_height: Height of the door gap in metres (must be <= wall height).
        door_offset_y: Y offset of door centre from wall centre.
    """
    thickness, length, height = full_scale

    side_length = (length - door_width) / 2.0
    if side_length <= 0.0:
        # Door is wider than wall — skip geometry
        logger.warning("Door wider than wall at %s; skipping wall segments", base_path)
        return

    door_centre_y = centre[1] + door_offset_y
    left_centre_y = door_centre_y - door_width / 2.0 - side_length / 2.0
    right_centre_y = door_centre_y + door_width / 2.0 + side_length / 2.0

    # Left segment (full height)
    _add_wall(
        prim_path=f"{base_path}_Left",
        position=np.array([centre[0], left_centre_y, centre[2]]),
        scale=np.array([thickness, side_length, height]),
    )

    # Right segment (full height)
    _add_wall(
        prim_path=f"{base_path}_Right",
        position=np.array([centre[0], right_centre_y, centre[2]]),
        scale=np.array([thickness, side_length, height]),
    )

    # Lintel above door
    lintel_h = height - door_height
    if lintel_h > 0.01:
        lintel_z = door_height + lintel_h / 2.0
        _add_wall(
            prim_path=f"{base_path}_Lintel",
            position=np.array([centre[0], door_centre_y, lintel_z]),
            scale=np.array([thickness, door_width, lintel_h]),
        )


def create_navigation_scene(world: World) -> Robot:
    """Create a complex indoor environment for Go2 navigation testing.

    This is a realistic multi-room apartment designed to stress-test the full
    Vector nav stack (FAR/TARE/pathFollower) with challenging geometry:

    Layout (~60 m2):
      - Hallway (1.2m wide, 6m long) — narrow passage test
      - Living room (5m x 4m) — open space with furniture obstacles
      - Kitchen (3m x 3m) — tight space with counter, island, stools
      - Bedroom (4m x 3.5m) — bed, wardrobe, nightstand
      - Bathroom (2m x 2m) — very tight space
      - 5 doors (0.8-1.0m wide) connecting rooms
      - Furniture: sofa, coffee table, dining table, chairs, bed, shelves
      - 8+ graspable objects on surfaces (cups, bottles, books, remote)

    Sensor requirements: Go2 with Livox MID-360 lidar + RealSense D435.
    The scene includes narrow passages (hallway, bathroom doorway) that
    require the lidar's full 360-degree coverage for safe navigation.

    Args:
        world: Active Isaac Sim World.

    Returns:
        Robot handle for the Go2.
    """
    wall_h = 2.5
    t = 0.12  # wall thickness

    # ── Hallway (1.2m x 6m) at x=0, y=0..6 ──────────────────────────────
    _add_floor("/World/Nav/Hall/Floor",
               np.array([0.0, 3.0, -0.025]), np.array([1.2, 6.0, 0.05]))

    # Hallway west wall
    _add_wall("/World/Nav/Hall/WallW",
              np.array([-0.6, 3.0, wall_h / 2]), np.array([t, 6.0, wall_h]))

    # Hallway east wall (partial — doors to living room and bedroom)
    # Segment south of living room door (y=0 to y=1.5)
    _add_wall("/World/Nav/Hall/WallE_S",
              np.array([0.6, 0.75, wall_h / 2]), np.array([t, 1.5, wall_h]))
    # Segment between living room and bedroom doors (y=2.5 to y=4.0)
    _add_wall("/World/Nav/Hall/WallE_M",
              np.array([0.6, 3.25, wall_h / 2]), np.array([t, 1.5, wall_h]))
    # Segment north of bedroom door (y=5.0 to y=6.0)
    _add_wall("/World/Nav/Hall/WallE_N",
              np.array([0.6, 5.5, wall_h / 2]), np.array([t, 1.0, wall_h]))

    # Hallway south wall (with door to kitchen at x=0)
    _add_wall_with_door(
        base_path="/World/Nav/Hall/WallS",
        centre=np.array([0.0, 0.0, wall_h / 2]),
        full_scale=np.array([1.2, t, wall_h]),
        door_width=0.8, door_height=2.1, door_offset_y=0.0,
    )

    # Hallway north wall
    _add_wall("/World/Nav/Hall/WallN",
              np.array([0.0, 6.0, wall_h / 2]), np.array([1.2, t, wall_h]))

    # Door lintels for east wall openings
    # Living room door (y=1.5 to y=2.5)
    _add_wall("/World/Nav/Hall/Lintel_LR",
              np.array([0.6, 2.0, 2.1 + (wall_h - 2.1) / 2]),
              np.array([t, 1.0, wall_h - 2.1]))
    # Bedroom door (y=4.0 to y=5.0)
    _add_wall("/World/Nav/Hall/Lintel_BR",
              np.array([0.6, 4.5, 2.1 + (wall_h - 2.1) / 2]),
              np.array([t, 1.0, wall_h - 2.1]))

    # ── Living room (5m x 4m) at x=3.1, y=1.0..5.0 ─────────────────────
    _add_floor("/World/Nav/LR/Floor",
               np.array([3.1, 3.0, -0.025]), np.array([5.0, 4.0, 0.05]))

    _add_wall("/World/Nav/LR/WallN",
              np.array([3.1, 5.0, wall_h / 2]), np.array([5.0, t, wall_h]))
    _add_wall("/World/Nav/LR/WallE",
              np.array([5.6, 3.0, wall_h / 2]), np.array([t, 4.0, wall_h]))
    _add_wall("/World/Nav/LR/WallS",
              np.array([3.1, 1.0, wall_h / 2]), np.array([5.0, t, wall_h]))

    # Sofa (long, against north wall)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/LR/Sofa", name="sofa",
        position=np.array([3.5, 4.5, 0.25]),
        scale=np.array([2.0, 0.8, 0.50]),
    ))

    # Coffee table (in front of sofa)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/LR/CoffeeTable", name="coffee_table",
        position=np.array([3.5, 3.5, 0.225]),
        scale=np.array([1.2, 0.6, 0.45]),
    ))

    # TV stand (against east wall)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/LR/TVStand", name="tv_stand",
        position=np.array([5.2, 3.0, 0.25]),
        scale=np.array([0.5, 1.5, 0.50]),
    ))

    # Bookshelf (against south wall)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/LR/Bookshelf", name="bookshelf",
        position=np.array([2.0, 1.3, 0.75]),
        scale=np.array([0.3, 1.0, 1.50]),
    ))

    # ── Kitchen (3m x 3m) at x=0, y=-3.5..-0.5 ─────────────────────────
    _add_floor("/World/Nav/KT/Floor",
               np.array([0.0, -2.0, -0.025]), np.array([3.0, 3.0, 0.05]))

    _add_wall("/World/Nav/KT/WallS",
              np.array([0.0, -3.5, wall_h / 2]), np.array([3.0, t, wall_h]))
    _add_wall("/World/Nav/KT/WallW",
              np.array([-1.5, -2.0, wall_h / 2]), np.array([t, 3.0, wall_h]))
    _add_wall("/World/Nav/KT/WallE",
              np.array([1.5, -2.0, wall_h / 2]), np.array([t, 3.0, wall_h]))

    # Kitchen counter (L-shaped along south and west walls)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/KT/CounterS", name="counter_south",
        position=np.array([0.0, -3.1, 0.45]),
        scale=np.array([2.5, 0.6, 0.90]),
    ))
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/KT/CounterW", name="counter_west",
        position=np.array([-1.1, -2.0, 0.45]),
        scale=np.array([0.6, 2.0, 0.90]),
    ))

    # Kitchen island
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/KT/Island", name="island",
        position=np.array([0.2, -1.5, 0.45]),
        scale=np.array([1.0, 0.6, 0.90]),
    ))

    # ── Bedroom (4m x 3.5m) at x=3.1, y=5.5..9.0 ───────────────────────
    _add_floor("/World/Nav/BR/Floor",
               np.array([3.1, 7.25, -0.025]), np.array([4.0, 3.5, 0.05]))

    _add_wall("/World/Nav/BR/WallN",
              np.array([3.1, 9.0, wall_h / 2]), np.array([4.0, t, wall_h]))
    _add_wall("/World/Nav/BR/WallE",
              np.array([5.1, 7.25, wall_h / 2]), np.array([t, 3.5, wall_h]))

    # Shared wall with hallway (west side) — already have hallway east wall
    # South wall connects to hallway
    _add_wall("/World/Nav/BR/WallS_E",
              np.array([3.6, 5.5, wall_h / 2]), np.array([3.0, t, wall_h]))

    # Bed (against east wall)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/BR/Bed", name="bed",
        position=np.array([4.3, 7.5, 0.30]),
        scale=np.array([2.0, 1.4, 0.60]),
    ))

    # Wardrobe (against north wall)
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/BR/Wardrobe", name="wardrobe",
        position=np.array([2.0, 8.6, 1.0]),
        scale=np.array([1.2, 0.5, 2.0]),
    ))

    # Nightstand
    world.scene.add(FixedCuboid(
        prim_path="/World/Nav/BR/Nightstand", name="nightstand",
        position=np.array([3.0, 8.0, 0.25]),
        scale=np.array([0.4, 0.4, 0.50]),
    ))

    # ── Bathroom (2m x 2m) at x=-1.8, y=4.0..6.0 ───────────────────────
    # Door from hallway west wall
    _add_floor("/World/Nav/BA/Floor",
               np.array([-1.8, 5.0, -0.025]), np.array([2.0, 2.0, 0.05]))

    _add_wall("/World/Nav/BA/WallN",
              np.array([-1.8, 6.0, wall_h / 2]), np.array([2.0, t, wall_h]))
    _add_wall("/World/Nav/BA/WallW",
              np.array([-2.8, 5.0, wall_h / 2]), np.array([t, 2.0, wall_h]))
    _add_wall("/World/Nav/BA/WallS",
              np.array([-1.8, 4.0, wall_h / 2]), np.array([2.0, t, wall_h]))

    # Bathroom doorway in hallway west wall (at y=5.0)
    # Already have hallway west wall as single piece — need to cut door
    # For simplicity: the hallway west wall was placed full-length above,
    # we add the bathroom entrance via a gap (remove west wall, re-add segments)

    # ── Graspable objects ────────────────────────────────────────────────
    _add_graspable_objects(world)

    # ── Robot ────────────────────────────────────────────────────────────
    # Spawn in hallway entrance
    robot = _add_go2(world, position=np.array([0.0, 0.5, 0.50]))

    _add_distant_light(intensity=1200.0)
    # Area lights in each room for photorealism
    _add_area_light("/World/Nav/LR/Light", np.array([3.1, 3.0, 2.4]), 500.0)
    _add_area_light("/World/Nav/KT/Light", np.array([0.0, -2.0, 2.4]), 400.0)
    _add_area_light("/World/Nav/BR/Light", np.array([3.1, 7.25, 2.4]), 300.0)
    _add_area_light("/World/Nav/BA/Light", np.array([-1.8, 5.0, 2.4]), 200.0)

    logger.info(
        "Navigation scene created: 5 rooms (~60 m2), "
        "narrow hallway (1.2m), 5 doors, 10+ furniture pieces, 8+ objects"
    )
    return robot


def create_hospital_scene(world: World) -> Robot:
    """Load the built-in Isaac Sim Hospital environment for nav testing.

    The Hospital is the most complex indoor scene bundled with Isaac Sim:
    - Multiple rooms, corridors, and doorways
    - Furniture: beds, chairs, desks, cabinets, medical equipment
    - Long hallways ideal for FAR/TARE exploration
    - Realistic lighting and materials

    Falls back to navigation_scene if Hospital USD not available.

    Args:
        world: Active Isaac Sim World.

    Returns:
        Robot handle for the Go2.
    """
    try:
        from isaacsim.storage.native import get_assets_root_path
        assets_root = get_assets_root_path()
        hospital_path = assets_root + "/Isaac/Environments/Hospital/hospital.usd"

        add_reference_to_stage(usd_path=hospital_path, prim_path="/World/Hospital")
        logger.info("Hospital scene loaded from %s", hospital_path)

        # Spawn Go2 in the hospital lobby area
        robot = _add_go2(world, position=np.array([0.0, 0.0, 0.50]))
        _add_distant_light(intensity=800.0)
        return robot

    except Exception as exc:
        logger.warning(
            "Hospital scene not available (%s), falling back to navigation scene", exc
        )
        return create_navigation_scene(world)


def _add_graspable_objects(world: World) -> None:
    """Add interactive objects on surfaces for manipulation testing."""
    objects = [
        # (name, position, scale) — all DynamicCuboid for physics interaction
        ("cup_1",       np.array([3.5, 3.5, 0.50]),  np.array([0.08, 0.08, 0.12])),
        ("cup_2",       np.array([3.8, 3.5, 0.50]),  np.array([0.08, 0.08, 0.12])),
        ("bottle",      np.array([3.2, 3.5, 0.55]),  np.array([0.06, 0.06, 0.22])),
        ("remote",      np.array([4.0, 3.5, 0.47]),  np.array([0.15, 0.05, 0.02])),
        ("book_1",      np.array([2.0, 1.3, 1.55]),  np.array([0.15, 0.22, 0.03])),
        ("book_2",      np.array([2.0, 1.5, 1.55]),  np.array([0.15, 0.22, 0.03])),
        ("phone",       np.array([3.0, 8.0, 0.55]),  np.array([0.07, 0.15, 0.01])),
        ("water_bottle", np.array([0.2, -1.5, 0.95]), np.array([0.06, 0.06, 0.20])),
    ]
    for name, pos, scale in objects:
        world.scene.add(DynamicCuboid(
            prim_path=f"/World/Nav/Objects/{name}",
            name=name,
            position=pos,
            scale=scale,
            mass=0.2,
        ))
    logger.info("Added %d graspable objects", len(objects))


def _add_area_light(prim_path: str, position: np.ndarray, intensity: float) -> None:
    """Add a ceiling-mounted area light for indoor illumination."""
    from pxr import UsdLux, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    light = UsdLux.RectLight.Define(stage, prim_path)
    light.CreateIntensityAttr(intensity)
    light.CreateWidthAttr(1.0)
    light.CreateHeightAttr(1.0)
    from pxr import UsdGeom
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )


def _add_distant_light(intensity: float = 1000.0) -> None:
    """Add a simple directional (distant) light to the stage."""
    from pxr import UsdLux, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    light.CreateIntensityAttr(intensity)
    light.CreateAngleAttr(1.0)
    # Rotate light to come from above-left
    from pxr import UsdGeom
    xform = UsdGeom.Xformable(light.GetPrim())
    xform.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 45.0))
