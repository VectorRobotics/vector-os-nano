# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Co-sim wiring shared by every embodiment (rule 7 — world-agnostic mechanism).

Both the G1 and the Go2/Piper bases render the SAME furnished room (the g1_room
``furnished_targets`` layout) with the SAME photoreal CC0 asset library, only
differing in which head camera they read. This factory builds the renderer +
spawns the Blender bridge once, so neither base duplicates the asset-mapping or
bridge-lifecycle logic. Heavy CC0 assets live under ``VECTOR_PHOTOREAL_ASSETS``
(never vendored to git); an unmapped target is simply skipped.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vector_os_nano.playground.photoreal.renderer import PhotorealRenderer
from vector_os_nano.playground.photoreal.scene import build_room_scene_spec


def photoreal_asset_dir() -> Path:
    """Local CC0 asset library (env-overridable; defaults to the R2/R3 sandbox)."""
    return Path(os.environ.get(
        "VECTOR_PHOTOREAL_ASSETS",
        str(Path.home() / "sandbox" / "c10-substrate-spike")))


def furnished_room_asset_map(asset_dir: "Path | None" = None) -> dict:
    """Map furnished-room target body names -> photoreal CC0 assets present on disk.

    Currently the PolyHaven armchair stands in for the ``target_chair`` (R3 proved
    it grounds 0.95). Missing assets are omitted — the scene builder skips them.
    """
    asset_dir = asset_dir or photoreal_asset_dir()
    asset_map: dict = {}
    chair = asset_dir / "armchair" / "ArmChair_01_4k.gltf"
    if chair.exists():
        asset_map["target_chair"] = {"path": str(chair), "scale": 1.0}
    return asset_map


def build_pick_scene_spec(objects: list, *, table: "dict | None" = None) -> dict:
    """Scene spec for the manipulation scene: a table + colored graspable
    primitives (cylinders/boxes) at their LIVE world poses. ``objects`` is a list
    of ``{type, pos, size, color}`` dicts (read from the sim by the base). The
    object positions render the scene AS IT IS — the VLM still has to find the
    object in the image; the grasp target is perception-derived, not GT (rule 5).
    """
    floor = {"size": 30, "color": [0.72, 0.70, 0.66]}
    walls = []
    if table is not None:
        walls = [{"pos": table.get("pos", [0, 0, 0.1]),
                  "scale": table.get("scale", [0.20, 0.25, 0.10]),
                  "color": table.get("color", [0.55, 0.40, 0.25])}]
    return build_room_scene_spec({}, {}, floor=floor, walls=walls, objects=objects)


def scene_renderer(
    scene_spec: dict,
    *,
    cam_name: str,
    bridge: "Any | None" = None,
    width: int = 640,
    height: int = 480,
    samples: int = 48,
):
    """Build a ``PhotorealRenderer`` for an explicit scene spec + spawn/reuse the
    bridge. Returns ``(renderer, bridge)``."""
    from vector_os_nano.playground.photoreal.bridge import (
        BlenderBridge, blender_available)

    if bridge is None:
        if not blender_available():
            raise RuntimeError(
                "photoreal requested but no Blender — set VECTOR_BLENDER "
                "(co-sim render server, campaign #10)")
        bridge = BlenderBridge()
        bridge.start()
    renderer = PhotorealRenderer(
        bridge, scene_spec, cam_name=cam_name,
        width=width, height=height, samples=samples)
    return renderer, bridge


def furnished_room_renderer(
    *,
    cam_name: str,
    bridge: "Any | None" = None,
    width: int = 640,
    height: int = 480,
    samples: int = 48,
):
    """Build a ``PhotorealRenderer`` for the furnished room and the spawned bridge.

    ``bridge`` may be injected (tests); otherwise a real ``BlenderBridge`` is
    spawned (fails loud if no Blender). Returns ``(renderer, bridge)`` so the
    caller owns teardown.
    """
    from vector_os_nano.hardware.sim.g1_room import furnished_targets
    from vector_os_nano.playground.photoreal.bridge import (
        BlenderBridge, blender_available)

    scene_spec = build_room_scene_spec(furnished_targets(), furnished_room_asset_map())
    if bridge is None:
        if not blender_available():
            raise RuntimeError(
                "photoreal requested but no Blender — set VECTOR_BLENDER "
                "(co-sim render server, campaign #10)")
        bridge = BlenderBridge()
        bridge.start()
    renderer = PhotorealRenderer(
        bridge, scene_spec, cam_name=cam_name,
        width=width, height=height, samples=samples)
    return renderer, bridge
