# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unified MjSpec.attach scene builder — ONE scene-build path for every body.

Rule 11 (embodiments are config, not code): a robot is stood up in the go2
house room by ATTACHING its MuJoCo model into a room-only base spec, never by a
per-robot scene builder. ``build_room_scene`` is world-agnostic: it takes the
robot model path + attach prefix + spawn xy + optional welds/camera (all sourced
from the embodiment config by the caller) and returns the compiled ``MjModel``
plus, optionally, a reloadable scene file on disk.

Why attach (not ``<include>``): the include path is go2-only and cannot prefix
or weld programmatically. ``MjSpec.attach`` + ``compile()`` is uniform across
quadruped / biped / arm and is proven byte-identical to the legacy go2 include
build (same nq/nv/nu/nbody/ngeom/ncam/neq, identical name-sets, identical
class-default-derived joint ranges / geom params).

Why ``normalize_attach_defaults``: MuJoCo 3.9's ``MjSpec.to_xml`` serializes an
attached robot whose defaults nest under a class (go2's ``<default class="go2">``
tree) by wrapping them in an ANONYMOUS (class-less) nested ``<default>``. That
text reloads with ``XML Error: empty class name``. The in-memory compiled model
is unaffected — only the serialized text is malformed — so we repair the text by
hoisting the anonymous wrapper's children up one level before writing. g1's flat
default tree has no such wrapper, so the repair is a harmless no-op for it.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from vector_os_nano.interfaces.physics import CompiledScene
from vector_os_nano.sim.scene.compiled import compiled_scene
from vector_os_nano.sim.scene.compose import normalize_attach_defaults, robot_spec, write_scene

# A weld is (equality_name, body1, body2); created inactive (gripper activates it).
Weld = tuple[str, str, str]


def _room_only_spec(
    mj: Any, room_template_path: Path, room_assets_dir: Path
) -> Any:
    """Resolve the go2_room template to a robot-free MjSpec (room geometry only).

    Strips the ``<include>`` robot (the robot is attached programmatically) and
    fills the asset-dir token. GRASP_WELDS is emptied — welds are added on the
    composite spec after attach so they reference the prefixed robot body.
    """
    xml = room_template_path.read_text()
    xml = xml.replace(
        '<include file="GO2_MODEL_PATH"/>',
        "<!-- robot attached programmatically (scene_builder) -->",
    )
    xml = xml.replace("GO2_ASSETS_DIR", str(room_assets_dir))
    xml = xml.replace("GRASP_WELDS", "")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, prefix="room_only_"
    ) as fh:
        fh.write(xml)
        room_tmp = fh.name
    try:
        return mj.MjSpec.from_file(room_tmp)
    finally:
        try:
            os.unlink(room_tmp)
        except OSError:
            pass


def build_room_scene(
    robot_model_path: Path | str,
    room_template_path: Path | str,
    room_assets_dir: Path | str,
    *,
    attach_prefix: str,
    spawn_xy: Sequence[float],
    welds: Sequence[Weld] = (),
    cache_stem: str | None = None,
    robot_meshes_dir: Path | str | None = None,
    camera: Mapping[str, Any] | None = None,
) -> tuple[Any, Path | None]:
    """Attach a robot model into the go2 house room; return ``(MjModel, scene_path)``.

    Args:
        robot_model_path: MJCF of the robot to stand up (go2.xml / go2_piper.xml /
            g1_12dof.xml).
        room_template_path: the go2_room.xml template (GO2_MODEL_PATH /
            GO2_ASSETS_DIR / GRASP_WELDS tokens).
        room_assets_dir: directory the room's furniture/texture assets resolve from.
        attach_prefix: name prefix for every attached robot element ("" for go2,
            "g1_" for g1). go2 keeps unprefixed names (base_link, FL_hip, ...).
        spawn_xy: (x, y) of the attach frame in the room (z=0; connect() sets the
            standing height via qpos).
        welds: ``(name, body1, body2)`` triples added as inactive WELD equalities
            after attach (grasp welds; bodies must exist post-attach).
        cache_stem: if given, the compiled scene is serialized (with the attach-
            default repair) for from_xml_path consumers, content-addressed under
            this file stem in the scene cache (``sim/scene/compose.write_scene``).
        robot_meshes_dir: base directory for relative robot ``mesh.file`` paths.
            Defaults to the model file's directory + its ``meshdir`` (MuJoCo's rule).
        camera: optional ``{mount_body, name, pos, xyaxes}`` head camera added to
            the named robot body before attach.

    Returns:
        ``(compiled MjModel, scene file or None)``.
    """
    import mujoco as mj  # noqa: PLC0415

    room_spec = _room_only_spec(mj, Path(room_template_path), Path(room_assets_dir))
    robot = robot_spec(robot_model_path, robot_meshes_dir)

    if camera is not None:
        mount = camera["mount_body"]
        body = next((b for b in robot.bodies if b.name == mount), None)
        if body is None:
            raise RuntimeError(
                f"scene_builder: camera mount_body {mount!r} not found in "
                f"{robot_model_path}"
            )
        body.add_camera(
            name=camera["name"], pos=list(camera["pos"]), xyaxes=list(camera["xyaxes"])
        )

    frame = room_spec.worldbody.add_frame(pos=[float(spawn_xy[0]), float(spawn_xy[1]), 0.0])
    room_spec.attach(robot, prefix=attach_prefix, frame=frame)

    for name, body1, body2 in welds:
        eq = room_spec.add_equality()
        eq.type = mj.mjtEq.mjEQ_WELD
        eq.name = name
        eq.objtype = mj.mjtObj.mjOBJ_BODY
        eq.name1 = body1
        eq.name2 = body2
        eq.active = False

    model = room_spec.compile()

    if cache_stem is None:
        return model, None
    return model, write_scene(cache_stem, normalize_attach_defaults(room_spec.to_xml()))


_ROOM_XML = Path(__file__).with_name("go2_room.xml")
_ROOM_ASSETS = Path(__file__).parent / "mjcf" / "go2" / "assets"
_REPO = Path(__file__).resolve().parents[3]


def room_scene(robot: Any, world: Any) -> CompiledScene:
    """The apartment room (go2_room.xml) with *robot* (an EmbodimentConfig with a base)
    attached at its spawn and one inactive grasp weld per (gripper weld body x pickable
    of *world*), compiled at the robot's policy timestep."""
    gripper = robot.arm.gripper if robot.arm is not None else None
    pairs = [(gripper.weld_body, p) for p in world.pickables] if gripper is not None else []
    meshes = robot.model.meshes_dir
    _model, path = build_room_scene(
        robot_model_path=_REPO / robot.model.path, room_template_path=_ROOM_XML,
        room_assets_dir=_ROOM_ASSETS, attach_prefix="", spawn_xy=robot.spawn.xy,
        welds=[(f"grasp_{child}", parent, child) for parent, child in pairs],
        cache_stem=f"scene_{robot.id}_room", robot_meshes_dir=_REPO / meshes if meshes else None,
    )
    assert path is not None
    return compiled_scene(path, robot_root=robot.model.root_body, grasp_pairs=pairs,
                          timestep=robot.base.policy.timestep)
