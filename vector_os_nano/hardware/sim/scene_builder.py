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

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

from vector_os_nano.schemas import paths

# A weld is (equality_name, body1, body2); created inactive (gripper activates it).
Weld = tuple[str, str, str]


def scene_cache_dir() -> Path:
    """Where generated scenes live: the state directory's cache, never the package tree."""
    return paths.scene_cache()


def write_scene(stem: str, text: str) -> Path:
    """Write *text* to ``<cache>/<stem>-<hash>.xml`` atomically and return the path.

    Content-addressed, so processes that build the same scene share one file and
    one that builds a different scene can never clobber it; ``mkstemp`` +
    ``os.replace`` means a reader never sees a half-written file.
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

    MuJoCo 3.9's ``MjSpec.attach`` emits ``<default><default><default class=...>``
    when serializing a robot whose defaults nest under a class. The middle
    class-less ``<default>`` breaks ``from_xml_path`` ("empty class name"). This
    repair splices any such wrapper's children up into its parent, preserving the
    default tree's semantics (proven byte-identical via in-memory compile). It is
    a no-op when there is no class-less nested ``<default>`` (g1's flat tree).
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
            this file stem in :func:`scene_cache_dir` (see :func:`write_scene`).
        robot_meshes_dir: base directory for relative robot ``mesh.file`` paths.
            Defaults to the model file's directory + its ``meshdir`` (MuJoCo's rule).
        camera: optional ``{mount_body, name, pos, xyaxes}`` head camera added to
            the named robot body before attach.

    Returns:
        ``(compiled MjModel, scene file or None)``.
    """
    import mujoco as mj  # noqa: PLC0415

    room_spec = _room_only_spec(mj, Path(room_template_path), Path(room_assets_dir))
    robot_spec = mj.MjSpec.from_file(str(robot_model_path))

    # Resolve every relative mesh path to an absolute one so the attached scene is
    # loadable from any directory. Default base = MuJoCo's own rule: the model
    # file's directory joined with its meshdir. Committed MJCF must stay relative
    # (absolute paths only ever work on the machine that generated them).
    meshes_dir = (
        Path(robot_meshes_dir)
        if robot_meshes_dir is not None
        else Path(robot_model_path).resolve().parent / (robot_spec.meshdir or "")
    )
    for mesh in robot_spec.meshes:
        if not os.path.isabs(mesh.file):
            mesh.file = str((meshes_dir / mesh.file).resolve())
    robot_spec.meshdir = ""

    if camera is not None:
        mount = camera["mount_body"]
        body = next((b for b in robot_spec.bodies if b.name == mount), None)
        if body is None:
            raise RuntimeError(
                f"scene_builder: camera mount_body {mount!r} not found in "
                f"{robot_model_path}"
            )
        body.add_camera(
            name=camera["name"], pos=list(camera["pos"]), xyaxes=list(camera["xyaxes"])
        )

    frame = room_spec.worldbody.add_frame(pos=[float(spawn_xy[0]), float(spawn_xy[1]), 0.0])
    room_spec.attach(robot_spec, prefix=attach_prefix, frame=frame)

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
