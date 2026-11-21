# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scene composition (charter section 2): world scene ⊕ one robot at the world's spawn
⊕ the manifest's cameras ⊕ one inactive grasp pair per (gripper weld body × world
pickable), written to the scene cache and compiled once into a ``CompiledScene``.

The robot is attached with ``MjSpec.attach`` (no prefix: every manifest name is the
model file's own). Composing a base with an arm at a mount is not done here yet.
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import mujoco as mj

from vector_os_nano.embodiments.assets import model_root
from vector_os_nano.interfaces.physics import CompiledScene
from vector_os_nano.schemas import paths
from vector_os_nano.sim.scene.compiled import compiled_scene


def scene_cache_dir() -> Path:
    """Where generated scenes live: the state directory's cache, never the package tree."""
    return paths.scene_cache()


def write_scene(stem: str, text: str) -> Path:
    """Write *text* to ``<cache>/<stem>-<hash>.xml`` atomically and return the path.

    Content-addressed, so processes that build the same scene share one file and one
    that builds a different scene can never clobber it; ``mkstemp`` + ``os.replace``
    means a reader never sees a half-written file.
    """
    out = scene_cache_dir() / f"{stem}-{hashlib.sha1(text.encode()).hexdigest()[:12]}.xml"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{stem}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out


def normalize_attach_defaults(xml: str) -> str:
    """Hoist anonymous (class-less) nested ``<default>`` wrappers from attach.

    MuJoCo 3.9's ``MjSpec.to_xml`` serializes an attached robot whose defaults nest
    under a class as ``<default><default><default class=...>``; the middle class-less
    ``<default>`` breaks ``from_xml_path`` ("empty class name"). Splicing its children
    up one level keeps the default tree's semantics; a no-op without such a wrapper.
    """
    root = ET.fromstring(xml)
    for top in root.findall("default"):  # the single class-less root <default>
        changed = True
        while changed:
            changed = False
            for child in list(top):
                if child.tag == "default" and child.get("class") is None:
                    idx = list(top).index(child)
                    top.remove(child)
                    for offset, grandchild in enumerate(child):
                        top.insert(idx + offset, grandchild)
                    changed = True
                    break
    return ET.tostring(root, encoding="unicode")


def robot_spec(model_path: Path | str, meshes_dir: Path | str | None = None) -> Any:
    """Load a robot or world, resolving mesh and texture files before caching it."""
    spec = mj.MjSpec.from_file(str(model_path))
    directory = Path(model_path).resolve().parent
    base = Path(meshes_dir) if meshes_dir is not None else directory / (spec.meshdir or "")
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = str((base / mesh.file).resolve())
    spec.meshdir = ""
    textures = directory / (spec.texturedir or "")
    for texture in spec.textures:
        if texture.file and not os.path.isabs(texture.file):
            texture.file = str((textures / texture.file).resolve())
        texture.cubefiles = [str((textures / f).resolve()) if f and not os.path.isabs(f) else f
                             for f in texture.cubefiles]
    spec.texturedir = ""
    return spec


def model_files(robot: Any) -> tuple[Path, Path | None]:
    """``(model file, meshes folder or None)`` of *robot* (an EmbodimentConfig), resolved
    by ``embodiments.assets.model_root``."""
    root = model_root(robot)
    return root / robot.model.path, root / robot.model.meshes_dir if robot.model.meshes_dir else None


def place_cameras(world: Any, model: Any, robot: Any) -> None:
    """Add *robot*'s placed cameras (a manifest camera with ``pos``) to its *model* spec on
    their ``body``, or to the *world* spec when they name none."""
    for cam in robot.sensors:
        if getattr(cam, "pos", None) is not None and cam.role == "camera":
            spec = world if cam.body is None else model
            body = spec.worldbody if cam.body is None else spec.body(cam.body)
            if body is None:
                raise KeyError(f"{robot.id}: camera {cam.name!r} body {cam.body!r} is not in {robot.model.path}")
            body.add_camera(name=cam.name, pos=list(cam.pos), quat=list(cam.quat), fovy=cam.fovy)


def compose(world_scene: Path | str, robot: Any, *, spawn: Sequence[float],
            pickables: Sequence[str] = (), timestep: float | None = None) -> CompiledScene:
    """*robot* (an EmbodimentConfig) mounted in *world_scene* at *spawn* (x, y, z,
    heading), its placed cameras added, one inactive grasp weld per gripper weld body
    and pickable; compiled with its root body as the scene's robot root."""
    world = robot_spec(world_scene)
    model = robot_spec(*model_files(robot))
    place_cameras(world, model, robot)
    x, y, z, heading = (float(v) for v in spawn)
    frame = world.worldbody.add_frame(pos=[x, y, z], quat=[math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
    world.attach(model, prefix="", frame=frame)
    gripper = robot.arm.gripper if robot.arm is not None else None
    pairs = [(gripper.weld_body, p) for p in pickables] if gripper is not None else []
    for parent, child in pairs:
        eq = world.add_equality()
        eq.type, eq.objtype = mj.mjtEq.mjEQ_WELD, mj.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2, eq.active = parent, child, False
    world.compile()
    path = write_scene(f"scene_{robot.id}_{Path(world_scene).stem}", normalize_attach_defaults(world.to_xml()))
    return compiled_scene(path, robot_root=robot.model.root_body, grasp_pairs=pairs, timestep=timestep)


def check_fingers(scene: CompiledScene, fingers: Sequence[str]) -> None:
    """Every finger body has a colliding geom (the grasp is gated on their contacts);
    a ``ValueError`` naming the one that has none."""
    m = scene.model
    colliding = {int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]}
    for name in fingers:
        if scene.body(name) not in colliding:
            raise ValueError(f"{scene.xml_path}: finger body {name!r} has no colliding geom")
