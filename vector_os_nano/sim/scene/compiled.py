# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The one compile step: an MJCF file -> :class:`CompiledScene` (the canonical model).

Scene COMPOSITION (world + robot + arm, generated grasp welds) happens before this,
into one MJCF file; this compiles that file with MuJoCo and indexes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco as mj

from vector_os_nano.interfaces.physics import CompiledScene


def _names(model: mj.MjModel, obj: mj.mjtObj, n: int) -> dict[str, int]:
    out = {}
    for i in range(n):
        name = mj.mj_id2name(model, obj, i)
        if name:
            out[name] = i
    return out


def compiled_scene(
    xml_path: str | Path,
    *,
    robot_root: str | None = None,
    grasp_pairs: Sequence[tuple[str, str]] = (),
    timestep: float | None = None,
    model: mj.MjModel | None = None,
) -> CompiledScene:
    """Compile *xml_path* and fill the name tables.

    ``grasp_pairs`` are ``(parent body, child body)`` names; each must be an INACTIVE
    weld equality ``body1=parent body2=child`` in the file (the MuJoCo backend
    activates it on attach) and the child must be a free body. ``timestep``
    overrides the file's (the robot policy's timestep wins over the world's).
    """
    xml_path = Path(xml_path)
    if model is None:
        model = mj.MjModel.from_xml_path(str(xml_path))
    if timestep is not None:
        model.opt.timestep = float(timestep)
    bodies = _names(model, mj.mjtObj.mjOBJ_BODY, model.nbody)
    scene = CompiledScene(
        xml_path=xml_path,
        model=model,
        timestep=float(model.opt.timestep),
        body_ids=bodies,
        joint_ids=_names(model, mj.mjtObj.mjOBJ_JOINT, model.njnt),
        cameras=_names(model, mj.mjtObj.mjOBJ_CAMERA, model.ncam),
    )
    root = scene.body(robot_root) if robot_root is not None else -1
    welds = {
        (int(model.eq_obj1id[i]), int(model.eq_obj2id[i])): bool(model.eq_active0[i])
        for i in range(model.neq)
        if model.eq_type[i] == mj.mjtEq.mjEQ_WELD and model.eq_objtype[i] == mj.mjtObj.mjOBJ_BODY
    }
    pairs = []
    for parent, child in grasp_pairs:
        pair = (scene.body(parent), scene.body(child))
        if pair not in welds:
            raise ValueError(
                f"{xml_path}: grasp pair ({parent}, {child}) has no weld equality "
                f"body1={parent!r} body2={child!r} in the scene"
            )
        if welds[pair]:
            raise ValueError(f"{xml_path}: the grasp weld ({parent}, {child}) must start inactive")
        if child not in scene.free_bodies():
            raise ValueError(f"{xml_path}: grasp child {child!r} is not a free body")
        pairs.append(pair)
    return CompiledScene(
        xml_path=scene.xml_path, model=model, timestep=scene.timestep,
        body_ids=scene.body_ids, joint_ids=scene.joint_ids, cameras=scene.cameras,
        robot_root=root, grasp_pairs=tuple(pairs),
    )
