# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SCENARIOS — the playground's preset-scene catalog.

A small id -> Scenario registry. The kernel never imports this; the playground
world resolves a scenario by id and grounds its verify predicates against the
scenario's known ``object_names``. The scene XML path is derived from the
installed ``hardware/sim`` package (never a machine-specific absolute path).
"""

from __future__ import annotations

from pathlib import Path

from vector_os_nano.playground.scenario import Scenario

# Resolve bundled scene XMLs from the installed package, not a hardcoded path.
_SIM_DIR = Path(__file__).resolve().parent.parent / "hardware" / "sim"

# The bundled tabletop scene ships a table + six graspable free bodies. These
# names MUST match the MJCF body names in so101_mujoco.xml — they are the
# contract the sim-oracle predicates read ground truth against.
_TABLETOP_OBJECTS: tuple[str, ...] = (
    "banana",
    "mug",
    "bottle",
    "screwdriver",
    "duck",
    "lego",
)

TABLETOP = Scenario(
    id="tabletop",
    embodiment="arm",
    scene_xml=str(_SIM_DIR / "so101_mujoco.xml"),
    task_hint="Pick and place objects on a tabletop with the SO-101 arm.",
    object_names=_TABLETOP_OBJECTS,
)

# A named drop-zone over the same so101 scene: the tabletop spans x in
# [-0.10, 0.50], y in [-0.25, 0.25] (table centred at (0.20, 0) with half-sizes
# (0.30, 0.25)). The tray is the right-front quadrant, so placed_count() with no
# argument verifies "object resting INSIDE the tray" — a scene-defined region the
# sub-goal references by scenario instead of hand-passing raw coordinates.
_TRAY_REGION: tuple[float, float, float, float] = (0.20, 0.0, 0.50, 0.25)

TABLETOP_TRAY = Scenario(
    id="tabletop_tray",
    embodiment="arm",
    scene_xml=str(_SIM_DIR / "so101_mujoco.xml"),
    task_hint="Sort objects into the tray (right-front drop-zone) with the SO-101 arm.",
    object_names=_TABLETOP_OBJECTS,
    place_region=_TRAY_REGION,
)

# ---------------------------------------------------------------------------
# Go2 quadruped — a second embodiment (E-1). Proves the playground/seam
# generalizes beyond the arm: a mobile-base scene whose verify predicates read
# base ground truth (position / heading) instead of arm joints.
# ---------------------------------------------------------------------------

# The go2 room scene from the installed hardware package (never a machine path).
# QUIRK: loading the go2 sim rewrites scene_room_piper.xml's absolute asset paths;
# this scenario references scene_room.xml (the non-piper variant) but tests must
# still "git checkout" scene_room_piper.xml if any go2 load touches it.
_GO2_SIM_DIR = Path(__file__).resolve().parent.parent / "hardware" / "sim" / "mjcf" / "go2"

# Named rooms as axis-aligned (x_min, y_min, x_max, y_max) boxes in the world
# frame. These are the scenario-owned contract the base ``visited`` predicate
# reads against — a navigation sub-goal verifies "reached <room>" by name, and
# the GO2 rooms producer (PlaygroundWorld.ROOMS_STRATEGY) emits exactly this set
# as the list a "visit each room one by one" foreach iterates.
#
# REAL GEOMETRY — reconciled with the bundled scene_room.xml (the 20m x 14m house;
# see its top-down ASCII layout + ``f_*`` floor geoms). Each box is the room's
# floor extent in the world frame, derived directly from the matching ``f_<room>``
# plane geom ``pos +/- size`` (e.g. ``f_kitchen`` pos="17 2.5" size="3 2.5" =>
# (14, 0, 20, 5)). The Go2 spawns at (10, 3, 0.35) (MuJoCoGo2._reset sets
# data.qpos[0:3] = [10, 3, 0.35]) — inside the central ``hallway`` box. The
# hallway is the open central span (x=6..14, y=0..10) the perimeter rooms open
# onto; the bathroom box spans x=7..12 because the laundry was merged into it
# (see the XML's "LAUNDRY (merged into bathroom)" section). Room centres here
# match the canonical navigation room database in
# tests/unit/test_navigate_skill_nav2.py (kitchen=(17, 2.5), hallway=(10, 5),
# master_bedroom=(3.5, 12), ...). The mechanism (producer -> foreach -> visited)
# is geometry-agnostic, so this is a pure data swap.
_GO2_ROOMS: dict[str, tuple[float, float, float, float]] = {
    "living_room": (0.0, 0.0, 6.0, 5.0),
    "dining_room": (0.0, 5.0, 6.0, 10.0),
    "kitchen": (14.0, 0.0, 20.0, 5.0),
    "study": (14.0, 5.0, 20.0, 10.0),
    "master_bedroom": (0.0, 10.0, 7.0, 14.0),
    "guest_bedroom": (12.0, 10.0, 20.0, 14.0),
    "bathroom": (7.0, 10.0, 12.0, 14.0),
    "hallway": (6.0, 0.0, 14.0, 10.0),
}

GO2_ROOM = Scenario(
    id="go2_room",
    embodiment="go2",
    scene_xml=str(_GO2_SIM_DIR / "scene_room.xml"),
    task_hint="Navigate the room with the Go2 quadruped (walk, turn, explore).",
    rooms=_GO2_ROOMS,
)

# M2 (ADR-009): the first photoreal third-world preset — a license-free
# habitat test scene behind the conda-subprocess backend. embodiment="mobile"
# selects the base predicate family (a kinematic navmesh base, not the go2
# gait sim). rooms is honestly EMPTY: the test scene carries no semantic
# annotations; real named rooms arrive with HM3D-Semantics (DQ-1) in M3 —
# at_position/facing/geodesic_dist carry the verify story until then.
APARTMENT = Scenario(
    id="apartment",
    embodiment="mobile",
    scene_xml="",
    task_hint="Navigate a photoreal scanned apartment (habitat backend).",
    sim_backend="habitat",
    scene_ref="habitat-test-scenes/apartment_1.glb",
)

# N0 (campaign #2): the first LARGE composed scene — ReplicaCAD apt_1
# (CC-BY-4.0; ~13m x 7m navigable, 120 placed objects, authored navmesh).
# Loaded through the dataset config (stage + furniture + lighting), so
# scene_ref is the scene-instance NAME, not a file. Rooms are hand-labeled
# world-frame boxes derived from the scene instance's furniture anchors
# (tv/tvstand at x~-7.7, dining table_01 at (-5.2,-2.5), sofa strip at
# y~-4, fridge/kitchen_counter at y~2, door2/shoe rack at x~2) and
# live-verified against the authored navmesh (snap + geodesic harness).
_HOUSE_ROOMS: dict[str, tuple[float, float, float, float]] = {
    "tv_corner": (-8.2, -3.7, -6.6, -1.0),
    "dining": (-6.6, -3.6, -4.1, -1.2),
    "living_room": (-4.1, -4.7, 0.9, -2.4),
    "kitchen": (-3.6, 1.0, 0.2, 2.8),
    "entryway": (1.2, 1.2, 3.2, 2.8),
}

HOUSE = Scenario(
    id="house",
    embodiment="mobile",
    scene_xml="",
    task_hint="Navigate a large multi-room ReplicaCAD apartment (habitat backend).",
    rooms=_HOUSE_ROOMS,
    sim_backend="habitat",
    scene_ref="apt_1",
    scene_dataset_config="replica_cad/replicaCAD.scene_dataset_config.json",
)

# Campaign #5: the G1 humanoid REAL-gait scene. embodiment="g1" selects the
# base predicate family (a base, no arm); sim_backend stays the default
# "mujoco" — booted IN-PROCESS by g1_runtime.boot_g1_agent on the pretrained
# unitree_rl_gym policy. scene_xml points at the downloaded asset
# (scripts/setup_g1_gait.sh fetches it into assets/g1_gait/, gitignored);
# rooms is empty (flat scene) — at_position/facing carry the verify story,
# geodesic_dist fails safe to inf (no navmesh), same as the MuJoCo go2.
_G1_SCENE_XML = str(
    Path(__file__).resolve().parent.parent.parent
    / "assets" / "g1_gait" / "scene.xml"
)

G1_FLAT = Scenario(
    id="g1_flat",
    embodiment="g1",
    scene_xml=_G1_SCENE_XML,
    task_hint="Walk and turn with the G1 humanoid (real policy-driven gait).",
)

# id -> Scenario. Additive: new preset scenes append here.
SCENARIOS: dict[str, Scenario] = {
    TABLETOP.id: TABLETOP,
    TABLETOP_TRAY.id: TABLETOP_TRAY,
    GO2_ROOM.id: GO2_ROOM,
    APARTMENT.id: APARTMENT,
    HOUSE.id: HOUSE,
    G1_FLAT.id: G1_FLAT,
}


def get_scenario(scenario_id: str) -> Scenario:
    """Return the scenario registered under *scenario_id*.

    Fails loud with the valid set when unknown — never a silent fallback.
    """
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        valid = ", ".join(sorted(SCENARIOS)) or "<none>"
        raise KeyError(
            f"unknown playground scenario {scenario_id!r}; valid: {valid}"
        ) from None
