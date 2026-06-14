#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Compose the Menagerie Unitree G1 into ONE rigid GLB for habitat (N3).

Offline, one-time asset build — run it from an isolated venv that has
``mujoco`` + ``trimesh`` (e.g. ``~/sandbox/g1_glb_build/.venv``); the repo
venv and the habitat conda env stay untouched. The OUTPUT lands under the
habitat data root (external runtime data — never vendored into git), next
to a minimal ``.object_config.json`` habitat's template manager loads.

Pose: the MJCF "home" keyframe (natural stance), visual geoms only
(group 2). Frames: MuJoCo is z-up, +x forward; glTF is y-up — vertices are
remapped ``(x, y, z) -> (x, z, -y)`` so the robot stands on y=0 facing +x.
The habitat server adds the agent-heading yaw (plus the +x -> -z forward
offset) at placement time.

Usage:
    python scripts/build_g1_glb.py \
        [--mjcf ~/Desktop/mujoco_menagerie/unitree_g1/g1.xml] \
        [--out  <VECTOR_HABITAT_DATA>/robots/unitree_g1/g1_rigid.glb]

License: Menagerie unitree_g1 is BSD-3 (Unitree Robotics) — redistribution
of the composed asset keeps that license; a LICENSE copy is written beside
the output.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np
import trimesh

_DEFAULT_MJCF = Path.home() / "Desktop/mujoco_menagerie/unitree_g1/g1.xml"
_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
_VISUAL_GROUP = 2  # menagerie convention: visual geoms group 2, collision 3


def compose_visual_mesh(mjcf_path: str) -> trimesh.Trimesh:
    """All visual mesh geoms at the 'home' keyframe, in one y-up mesh."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    parts: "list[trimesh.Trimesh]" = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[g] != _VISUAL_GROUP:
            continue
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va:va + vn].astype(np.float64)
        faces = model.mesh_face[fa:fa + fn].astype(np.int64)
        xmat = data.geom_xmat[g].reshape(3, 3)
        world = verts @ xmat.T + data.geom_xpos[g]
        parts.append(trimesh.Trimesh(world @ _ZUP_TO_YUP.T, faces, process=False))
    if not parts:
        raise RuntimeError(f"no visual mesh geoms (group {_VISUAL_GROUP}) in {mjcf_path}")
    merged = trimesh.util.concatenate(parts)
    # Feet exactly on the ground plane: the home keyframe leaves a few mm.
    merged.apply_translation([0.0, -float(merged.bounds[0][1]), 0.0])
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    data_root = os.environ.get(
        "VECTOR_HABITAT_DATA",
        str(Path.home() / "sandbox/habitat-spike/data/scene_datasets"),
    )
    ap.add_argument("--mjcf", default=str(_DEFAULT_MJCF))
    ap.add_argument(
        "--out", default=str(Path(data_root) / "robots/unitree_g1/g1_rigid.glb")
    )
    args = ap.parse_args()

    mesh = compose_visual_mesh(args.mjcf)
    lo, hi = mesh.bounds
    size = hi - lo
    print(f"composed: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"size x={size[0]:.2f} y(height)={size[1]:.2f} z={size[2]:.2f}")
    if not (0.8 < size[1] < 2.0):
        raise RuntimeError(f"implausible robot height {size[1]:.2f} — frame bug?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    cfg = out.with_name(out.stem + ".object_config.json")
    cfg.write_text(json.dumps({"render_asset": out.name, "is_collidable": False}))
    lic = Path(args.mjcf).parent / "LICENSE"
    if lic.exists():
        shutil.copy(lic, out.parent / "LICENSE")
    print(f"wrote {out}\n      {cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
