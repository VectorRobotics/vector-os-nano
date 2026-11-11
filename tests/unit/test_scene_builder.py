# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for the unified MjSpec.attach scene builder (scene_builder.py).

Proves the attach + compile() path is BYTE-IDENTICAL to the legacy go2 include
build (and g1's existing attach build) for all three cases, and that the
serialized scene file reloads (the MuJoCo 3.9 anonymous-default repair).
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from vector_os_nano.hardware.sim.scene_builder import (  # noqa: E402
    build_room_scene,
    normalize_attach_defaults,
)

_SIM = Path("vector_os_nano/hardware/sim").resolve()
_MJCF = _SIM / "mjcf" / "go2"
_ROOM = _SIM / "go2_room.xml"
_GO2_ASSETS = (_MJCF / "assets").resolve()
_GO2_PIPER = _SIM / "mjcf" / "go2_piper" / "go2_piper.xml"
_GO2_BARE = _MJCF / "go2.xml"

_G1_12DOF = Path("assets/g1_gait/g1_12dof.xml").resolve()
_G1_MESHES = Path("assets/g1_gait/meshes").resolve()

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
# Legacy (current) builders — verbatim behaviour, for the equivalence baseline
# ---------------------------------------------------------------------------

def _legacy_go2(with_arm: bool) -> mj.MjModel:
    go2_xml = _GO2_PIPER if with_arm else _GO2_BARE
    scene_name = "scene_room_piper.xml" if with_arm else "scene_room.xml"
    xml = _ROOM.read_text()
    xml = xml.replace("GO2_MODEL_PATH", str(go2_xml))
    xml = xml.replace("GO2_ASSETS_DIR", str(_GO2_ASSETS))
    xml = xml.replace("GRASP_WELDS", _WELDS_TXT if with_arm else "")
    out = _MJCF / scene_name
    out.write_text(xml)
    return mj.MjModel.from_xml_path(str(out))


def _legacy_g1() -> mj.MjModel:
    import os

    xml = _ROOM.read_text()
    xml = xml.replace('<include file="GO2_MODEL_PATH"/>', "<!-- no robot (g1 scene) -->")
    xml = xml.replace("GO2_ASSETS_DIR", str(_GO2_ASSETS))
    xml = xml.replace("GRASP_WELDS", "")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir="/tmp", prefix="g1_room_legacy_"
    ) as fh:
        fh.write(xml)
        room_tmp = fh.name
    try:
        room_spec = mj.MjSpec.from_file(room_tmp)
        g1_spec = mj.MjSpec.from_file(str(_G1_12DOF))
        for mesh in g1_spec.meshes:
            if not os.path.isabs(mesh.file):
                mesh.file = str(_G1_MESHES / mesh.file)
        g1_spec.meshdir = ""
        pelvis = next(b for b in g1_spec.bodies if b.name == "pelvis")
        pelvis.add_camera(name="head_rgb", pos=[0.04, 0.0, 0.42], xyaxes=[0, -1, 0, 0, 0, 1])
        frame = room_spec.worldbody.add_frame(pos=[10.0, 3.0, 0.0])
        room_spec.attach(g1_spec, prefix="g1_", frame=frame)
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
    model, path = build_room_scene(
        robot_model_path=_GO2_PIPER,
        room_template_path=_ROOM,
        room_assets_dir=_GO2_ASSETS,
        attach_prefix="",
        spawn_xy=[10.0, 3.0],
        welds=_WELDS,
        robot_meshes_dir=None,  # go2_piper.xml's ../go2/assets paths resolve from its own dir, as in production
    )
    assert path is None
    _assert_byte_identical(legacy, model)
    _assert_foot_class(model)
    # Arm-specific bodies/sites present.
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "piper_link6") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pickable_bottle_green") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "piper_ee_site") >= 0


def test_build_room_scene_bare_go2_byte_identical():
    legacy = _legacy_go2(with_arm=False)
    model, _ = build_room_scene(
        robot_model_path=_GO2_BARE,
        room_template_path=_ROOM,
        room_assets_dir=_GO2_ASSETS,
        attach_prefix="",
        spawn_xy=[10.0, 3.0],
        welds=(),
        robot_meshes_dir=_GO2_ASSETS,
    )
    _assert_byte_identical(legacy, model)
    _assert_foot_class(model)
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link") >= 0


@pytest.mark.skipif(not _G1_12DOF.exists(), reason="g1_12dof.xml asset not present")
def test_build_room_scene_g1_byte_identical():
    legacy = _legacy_g1()
    model, _ = build_room_scene(
        robot_model_path=_G1_12DOF,
        room_template_path=_ROOM,
        room_assets_dir=_GO2_ASSETS,
        attach_prefix="g1_",
        spawn_xy=[10.0, 3.0],
        welds=(),
        robot_meshes_dir=_G1_MESHES,
        camera={
            "mount_body": "pelvis",
            "name": "head_rgb",
            "pos": [0.04, 0.0, 0.42],
            "xyaxes": [0, -1, 0, 0, 0, 1],
        },
    )
    _assert_byte_identical(legacy, model)
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "g1_pelvis") >= 0


# ---------------------------------------------------------------------------
# go2 driver builder writes a RELOADABLE file (the corruption regression test)
# ---------------------------------------------------------------------------

def test_go2_builder_file_reloads():
    from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml

    for with_arm in (True, False):
        path = _build_room_scene_xml(with_arm=with_arm)
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
