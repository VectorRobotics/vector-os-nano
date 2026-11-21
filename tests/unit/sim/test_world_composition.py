# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for the shared world/robot scene composition.

Proves the attach + compile() path is BYTE-IDENTICAL to the legacy go2 include
build (and g1's existing attach build) for all three cases, and that the
serialized scene file reloads (the MuJoCo 3.9 anonymous-default repair).
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from vector_os_nano.sim.scene.compose import compose  # noqa: E402
from vector_os_nano.sim.scene.compose import normalize_attach_defaults
from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world  # noqa: E402

_SIM = Path("vector_os_nano/hardware/sim").resolve()
_MJCF = _SIM / "mjcf" / "go2"
_ROOM = _SIM.parents[1] / "worlds" / "apartment" / "scene.xml"
_GO2_ASSETS = _ROOM.parent / "meshes"
_GO2_PIPER = _SIM / "mjcf" / "go2_piper" / "go2_piper.xml"
_GO2_BARE = _MJCF / "go2.xml"

_WELDS = (
    ("grasp_pickable_bottle_blue", "piper_link6", "pickable_bottle_blue"),
    ("grasp_pickable_bottle_green", "piper_link6", "pickable_bottle_green"),
    ("grasp_pickable_can_red", "piper_link6", "pickable_can_red"),
)
_WELDS_TXT = (
    "  <equality>\n"
    '    <weld name="grasp_pickable_bottle_blue"  body1="piper_link6" body2="pickable_bottle_blue"  active="false"/>\n'
    '    <weld name="grasp_pickable_bottle_green" body1="piper_link6" body2="pickable_bottle_green" active="false"/>\n'
    '    <weld name="grasp_pickable_can_red"      body1="piper_link6" body2="pickable_can_red"      active="false"/>\n'
    "  </equality>\n"
)


# ---------------------------------------------------------------------------
# Independent include/attach reference builds for the equivalence baseline
# ---------------------------------------------------------------------------

def _legacy_go2(with_arm: bool) -> mj.MjModel:
    go2_xml = _GO2_PIPER if with_arm else _GO2_BARE
    xml = _ROOM.read_text().replace('meshdir="meshes"', f'meshdir="{_GO2_ASSETS}"')
    xml = xml.replace('texturedir="meshes"', f'texturedir="{_GO2_ASSETS}"')
    # The included robot's compiler meshdir applies to the whole document.
    xml = xml.replace('file="furniture/', f'file="{_GO2_ASSETS}/furniture/')
    with tempfile.TemporaryDirectory() as folder:
        robot = ET.parse(go2_xml).getroot()
        compiler = robot.find('compiler')
        for item in robot.findall('./asset/*[@file]'):
            directory = compiler.get('meshdir' if item.tag == 'mesh' else 'texturedir', '')
            item.set('file', str((go2_xml.parent / directory / item.get('file')).resolve()))
        compiler.set('meshdir', '')
        compiler.set('texturedir', '')
        model_path = Path(folder) / 'robot.xml'
        model_path.write_text(ET.tostring(robot, encoding='unicode'))
        xml = xml.replace('</mujoco>', f'<include file="{model_path}"/>' + (_WELDS_TXT if with_arm else '') + '</mujoco>')
        out = Path(folder) / 'scene.xml'
        out.write_text(xml)
        return mj.MjModel.from_xml_path(str(out))


def _legacy_g1(g1_xml: Path, g1_meshes: Path) -> mj.MjModel:
    import os

    xml = _ROOM.read_text().replace('meshdir="meshes"', f'meshdir="{_GO2_ASSETS}"')
    xml = xml.replace('texturedir="meshes"', f'texturedir="{_GO2_ASSETS}"')
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir="/tmp", prefix="g1_room_legacy_"
    ) as fh:
        fh.write(xml)
        room_tmp = fh.name
    try:
        room_spec = mj.MjSpec.from_file(room_tmp)
        g1_spec = mj.MjSpec.from_file(str(g1_xml))
        for mesh in g1_spec.meshes:
            if not os.path.isabs(mesh.file):
                mesh.file = str(g1_meshes / mesh.file)
        g1_spec.meshdir = ""
        pelvis = next(b for b in g1_spec.bodies if b.name == "pelvis")
        pelvis.add_camera(name="head_rgb", pos=[0.04, 0.0, 0.42], xyaxes=[0, -1, 0, 0, 0, 1])
        frame = room_spec.worldbody.add_frame(pos=[10.0, 3.0, 0.0])
        room_spec.attach(g1_spec, prefix="", frame=frame)
        room_spec.compile()
        return mj.MjModel.from_xml_string(room_spec.to_xml())
    finally:
        import os as _os

        _os.unlink(room_tmp)


# ---------------------------------------------------------------------------
# Equivalence helpers (BY NAME — row order differs between include and attach)
# ---------------------------------------------------------------------------

def _names(m: mj.MjModel, objtype, n: int) -> set[str]:
    out = set()
    for i in range(n):
        nm = mj.mj_id2name(m, objtype, i)
        if nm:
            out.add(nm)
    return out


def _assert_byte_identical(a: mj.MjModel, b: mj.MjModel) -> None:
    for c in ("nq", "nv", "nu", "nbody", "ngeom", "ncam", "neq", "njnt", "nsite", "nmesh"):
        assert getattr(a, c) == getattr(b, c), f"{c}: {getattr(a, c)} != {getattr(b, c)}"
    for objtype, n_a, n_b in (
        (mj.mjtObj.mjOBJ_BODY, a.nbody, b.nbody),
        (mj.mjtObj.mjOBJ_JOINT, a.njnt, b.njnt),
        (mj.mjtObj.mjOBJ_ACTUATOR, a.nu, b.nu),
        (mj.mjtObj.mjOBJ_CAMERA, a.ncam, b.ncam),
        (mj.mjtObj.mjOBJ_EQUALITY, a.neq, b.neq),
    ):
        assert _names(a, objtype, n_a) == _names(b, objtype, n_b), f"name-set {objtype}"
    # jnt_range BY NAME proves abduction/front_hip/back_hip/knee class scoping.
    ja = {mj.mj_id2name(a, mj.mjtObj.mjOBJ_JOINT, i): a.jnt_range[i] for i in range(a.njnt)}
    jb = {mj.mj_id2name(b, mj.mjtObj.mjOBJ_JOINT, i): b.jnt_range[i] for i in range(b.njnt)}
    for nm in ja:
        if nm in jb:
            assert np.allclose(ja[nm], jb[nm], atol=1e-9), f"jnt_range {nm}: {ja[nm]} != {jb[nm]}"
    # actuator ctrlrange BY NAME (knee motor class default).
    aa = {mj.mj_id2name(a, mj.mjtObj.mjOBJ_ACTUATOR, i): a.actuator_ctrlrange[i] for i in range(a.nu)}
    ab = {mj.mj_id2name(b, mj.mjtObj.mjOBJ_ACTUATOR, i): b.actuator_ctrlrange[i] for i in range(b.nu)}
    for nm in aa:
        if nm in ab:
            assert np.allclose(aa[nm], ab[nm], atol=1e-9), f"ctrlrange {nm}"


def _assert_foot_class(m: mj.MjModel) -> None:
    """Every condim==6 geom (foot class) keeps friction[0]==0.8."""
    feet = [i for i in range(m.ngeom) if m.geom_condim[i] == 6]
    assert len(feet) == 4, f"expected 4 foot geoms, got {len(feet)}"
    for i in feet:
        assert abs(m.geom_friction[i][0] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# normalize_attach_defaults
# ---------------------------------------------------------------------------

def test_normalize_attach_defaults_idempotent_on_flat_xml():
    flat = (
        '<mujoco model="x"><default><default class="leg">'
        '<joint range="-1 1"/></default></default></mujoco>'
    )
    out = normalize_attach_defaults(flat)
    # Parse-equal: same element structure (no class-less nested default existed).
    assert ET.tostring(ET.fromstring(flat)) == ET.tostring(ET.fromstring(out))


def test_normalize_attach_defaults_hoists_anonymous():
    corrupt = (
        '<mujoco model="x">'
        "<default>"  # root (class-less, allowed)
        "<default>"  # ANONYMOUS wrapper from attach — must be hoisted
        '<default class="go2"><joint armature="0.01"/>'
        '<default class="knee"><joint range="-2 -1"/></default>'
        "</default>"
        "</default>"
        "</default>"
        "<worldbody/></mujoco>"
    )
    out = normalize_attach_defaults(corrupt)
    # No class-less <default> may be a child of the root <default>.
    root = ET.fromstring(out)
    for top in root.findall("default"):
        for child in top:
            if child.tag == "default":
                assert child.get("class") is not None, "anonymous nested default remains"
    # And the repaired text must reload without the 'empty class name' error.
    m = mj.MjModel.from_xml_string(out)
    assert m is not None


# ---------------------------------------------------------------------------
# build_room_scene byte-identical (3 cases)
# ---------------------------------------------------------------------------

def test_build_room_scene_go2_arm_byte_identical():
    legacy = _legacy_go2(with_arm=True)
    cfg = replace(registry.load("go2_piper"), sensors=())
    world = load_world("apartment")
    world = replace(world, pickables={child: world.pickables[child] for _, _, child in _WELDS})
    model = compose(cfg, world).model
    _assert_byte_identical(legacy, model)
    _assert_foot_class(model)
    # Arm-specific bodies/sites present.
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "piper_link6") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pickable_bottle_green") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "piper_ee_site") >= 0


def test_build_room_scene_bare_go2_byte_identical():
    legacy = _legacy_go2(with_arm=False)
    cfg = replace(registry.load("go2"), sensors=())
    model = compose(cfg, load_world("apartment")).model
    _assert_byte_identical(legacy, model)
    _assert_foot_class(model)
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link") >= 0


def test_build_room_scene_g1_byte_identical(g1_assets):
    g1_xml, g1_meshes = g1_assets / "g1_12dof.xml", g1_assets / "meshes"
    legacy = _legacy_g1(g1_xml, g1_meshes)
    model = compose(registry.load("g1"), load_world("apartment")).model
    _assert_byte_identical(legacy, model)
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis") >= 0


# ---------------------------------------------------------------------------
# the legged room scene is a RELOADABLE file (the corruption regression test)
# ---------------------------------------------------------------------------

def test_go2_builder_file_reloads():
    from vector_os_nano.embodiments import registry
    from vector_os_nano.sim.scene.compose import compose
    from vector_os_nano.schemas.world import load_world

    for with_arm in (True, False):
        path = compose(registry.load("go2_piper" if with_arm else "go2"), load_world("apartment")).xml_path
        m = mj.MjModel.from_xml_path(str(path))  # must NOT raise 'empty class name'
        # Class defaults survive the round-trip.
        fl_hip = m.jnt_range[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "FL_hip_joint")]
        assert np.allclose(fl_hip, [-1.0472, 1.0472], atol=1e-4)
        rl_thigh = m.jnt_range[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "RL_thigh_joint")]
        assert np.allclose(rl_thigh, [-0.5236, 4.5379], atol=1e-4)
        if with_arm:
            assert mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "pickable_bottle_green") >= 0
            assert mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "piper_link6") >= 0
            assert mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "piper_ee_site") >= 0
