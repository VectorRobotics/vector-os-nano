# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scene composition (charter section 2): world scene ⊕ one robot at the world's spawn
⊕ the manifest's cameras ⊕ one inactive grasp pair per (gripper weld body × world
pickable), written to the scene cache and compiled once into a ``CompiledScene``.

Robot and arm namespaces are applied once; all downstream drivers read compiled names.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import mujoco as mj

from vector_os_nano.embodiments.assets import model_root
from vector_os_nano.schemas.robot import EmbodimentConfig, EmbodimentConfigError
from vector_os_nano.schemas.world import WorldManifest
from vector_os_nano.interfaces.physics import CompiledScene
from vector_os_nano.sim.scene.compiled import compiled_scene
from vector_os_nano.sim.scene.cache import cache_path, atomic_write


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
    for asset in (*spec.hfields, *spec.skins):
        if asset.file and not os.path.isabs(asset.file):
            asset.file = str((directory / asset.file).resolve())
    return spec


def model_files(robot: Any) -> tuple[Path, Path | None]:
    """``(model file, meshes folder or None)`` of *robot* (an EmbodimentConfig), resolved
    by ``embodiments.assets.model_root``."""
    root = model_root(robot)
    return root / robot.model.file, None


def robot_model(robot: EmbodimentConfig) -> Any:
    """Build only the physical robot, retaining each composed child's asset origin."""
    c = robot.compose
    if c is None:
        model = robot_spec(*model_files(robot))
        if robot.model.prefix:
            parent = mj.MjSpec()
            parent.attach(model, prefix=robot.model.prefix, frame=parent.worldbody.add_frame())
            model = parent
        return model
    model = robot_model(c.base)
    body = model.body(c.base.model.prefix + c.body)
    if body is None:
        raise EmbodimentConfigError(f"{robot.path}.compose.mount.body: {c.body!r}; valid: {[b.name for b in model.bodies]}")
    frame = body.add_frame(pos=c.pos, quat=c.quat)
    model.attach(robot_model(c.arm), prefix=c.prefix, frame=frame)
    return model


def check_model_names(robot: EmbodimentConfig, model: Any) -> None:
    """Reject misspelled morphology names before any backend can start moving."""
    bodies, joints, sites = ({e.name for e in getattr(model, kind) if e.name} for kind in ('bodies', 'joints', 'sites'))
    checks = [('model.root_body', [robot.model.root_body], bodies)]
    checks.append(('initial_joints', robot.initial_joints, joints))
    checks.extend((f'joint_groups.{name}.joints', spec.joints, joints)
                  for name, spec in robot.joint_groups.items())
    if robot.base:
        checks.append(('base.joints', robot.base.joints, joints))
    if robot.arm:
        checks.extend([('arm.joints', robot.arm.joints, joints), ('arm.ee.site', [robot.arm.ee_site], sites)])
        g = robot.arm.gripper
        if g:
            checks.extend([('arm.gripper.joint', [g.joint], joints),
                           ('arm.gripper.finger_bodies', g.finger_bodies, bodies),
                           ('arm.gripper.weld_body', [g.weld_body], bodies)])
    for i, sensor in enumerate(robot.sensors):
        if sensor.body:
            checks.append((f'sensors[{i}].body', [sensor.body], bodies))
        if sensor.role == 'camera' and sensor.pos is None:
            checks.append((f'sensors[{i}].name', [sensor.name], {c.name for c in model.cameras}))
    for key, names, valid in checks:
        missing = set(names) - valid
        if missing:
            raise EmbodimentConfigError(f'{robot.path}.{key}: unknown {sorted(missing)}; valid: {sorted(valid)}')


def place_cameras(world: Any, model: Any, robot: Any) -> None:
    """Add *robot*'s placed cameras (a manifest camera with ``pos``) to its *model* spec on
    their ``body``, or to the *world* spec when they name none."""
    for cam in robot.sensors:
        if getattr(cam, "pos", None) is not None and cam.role == "camera":
            spec = world if cam.body is None else model
            body = spec.worldbody if cam.body is None else spec.body(cam.body)
            if body is None:
                raise KeyError(f"{robot.id}: camera {cam.name!r} body {cam.body!r} is not in {robot.model.file}")
            body.add_camera(name=cam.name, pos=list(cam.pos), quat=list(cam.quat), fovy=cam.fovy)


def compose(robot: EmbodimentConfig, manifest: WorldManifest) -> CompiledScene:
    """Compile one world and one robot using the world's spawn and the robot's sensors."""
    if manifest.scene is None or manifest.spawn is None:
        raise ValueError(f"{manifest.path}: scene and spawn are required")
    world = robot_spec(manifest.scene)
    model = robot_model(robot)
    check_model_names(robot, model)
    place_cameras(world, model, robot)
    x, y, z, heading = (float(v) for v in manifest.spawn)
    frame = world.worldbody.add_frame(pos=[x, y, z], quat=[math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
    world.attach(model, prefix="", frame=frame)
    gripper = robot.arm.gripper if robot.arm is not None else None
    pairs = [(gripper.weld_body, p) for p in manifest.pickables] if gripper is not None and gripper.assist else []
    for parent, child in pairs:
        eq = world.add_equality()
        eq.name = f"grasp_{child}"
        eq.type, eq.objtype = mj.mjtEq.mjEQ_WELD, mj.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2, eq.active = parent, child, False
    path = cache_path(robot, manifest, world)
    binary = path.with_suffix('.mjb')
    timestep = robot.base.policy.timestep if robot.base is not None else None
    args = dict(robot_root=robot.model.root_body, grasp_pairs=pairs, timestep=timestep)
    if path.is_file() and binary.is_file():
        try:
            model = mj.MjModel.from_binary_path(str(binary))
        except ValueError:  # an interrupted or corrupt cache is regenerable
            pass
        else:
            return compiled_scene(path, model=model, **args)
    world.compile()
    xml = normalize_attach_defaults(world.to_xml())
    atomic_write(path, lambda tmp: Path(tmp).write_text(xml))
    scene = compiled_scene(path, **args)
    atomic_write(binary, lambda tmp: mj.mj_saveModel(scene.model, tmp, None))
    return scene


def check_fingers(scene: CompiledScene, fingers: Sequence[str]) -> None:
    """Every finger body has a colliding geom (the grasp is gated on their contacts);
    a ``ValueError`` naming the one that has none."""
    m = scene.model
    colliding = {int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]}
    for name in fingers:
        if scene.body(name) not in colliding:
            raise ValueError(f"{scene.xml_path}: finger body {name!r} has no colliding geom")
