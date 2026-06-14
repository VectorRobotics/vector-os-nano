# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Build a Blender scene spec (the JSON the render server consumes) from a room's
furniture placements + a photoreal-asset map.

R2/R3 finding: *asset fidelity is the lever* — so this SUBSTITUTES photoreal CC0
PBR assets for the toy MuJoCo meshes, placed at the same visible centres. A label
with no mapped asset is skipped (partial asset libraries are valid — the seam
works with one asset or ten). The camera pose is NOT set here; it comes live from
MuJoCo per frame (see ``renderer.PhotorealRenderer``).
"""
from __future__ import annotations

_DEFAULT_FLOOR = {"size": 12, "color": [0.78, 0.74, 0.68]}
_DEFAULT_SUN = {"pos": [2.0, 2.0, 4.0], "energy": 3.0}
_DEFAULT_SKY = {"color": [0.62, 0.70, 0.82], "strength": 1.3}


def build_room_scene_spec(
    placements: dict,
    asset_map: dict,
    *,
    floor: dict | None = None,
    walls: list | None = None,
    sun: dict | None = None,
    sky: dict | None = None,
    floor_z: float = 0.0,
    objects: list | None = None,
    extra_assets: list | None = None,
) -> dict:
    """Map ``{label: (cx, cy)}`` + ``{label: {path, scale, ...}}`` to a scene spec.

    Each mapped asset is placed at ``(cx, cy, floor_z)`` and grounded (its lowest
    point rests on the floor). Unmapped labels are skipped.
    """
    assets = []
    for label, (cx, cy) in placements.items():
        amap = asset_map.get(label)
        if amap is None:
            continue
        assets.append({
            "path": amap["path"],
            "scale": amap.get("scale", 1.0),
            "pos": [float(cx), float(cy), float(floor_z)],
            "ground": amap.get("ground", True),
        })
    # Raw asset dicts with explicit 3D pos (the pick scene's graspable mesh sits
    # on the TABLE, not the room floor — placements/floor_z can't express that).
    for a in (extra_assets or []):
        assets.append(dict(a))
    return {
        "floor": floor if floor is not None else dict(_DEFAULT_FLOOR),
        "walls": walls if walls is not None else [],
        "assets": assets,
        # Primitive graspable objects (the pick scene's colored cylinders/boxes),
        # rendered photoreal by the Blender server — no mesh asset needed.
        "objects": list(objects) if objects else [],
        "sun": sun if sun is not None else dict(_DEFAULT_SUN),
        "sky": sky if sky is not None else dict(_DEFAULT_SKY),
    }
