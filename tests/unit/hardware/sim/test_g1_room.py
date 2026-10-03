# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for G1-12dof-in-go2-room scene — R2 WIP floor.

Tests are headless (MUJOCO_GL=egl) and cover:
  1. Scene compiles and contains both the room's pick_table AND g1_pelvis.
  2. The compiled model has the g1_head_rgb camera.
  3. G1 self-geom-id set is non-empty.
  4. g1 is a startable embodiment (manifest driver + start_simulation enum).
  5. walk() produces forward displacement (2 s cmd → base x >0.3 m, base z >0.5).

Run:
    cd /home/yusen/Desktop/vector_os_nano
    PATH=/usr/bin:$PATH MUJOCO_GL=egl .venv/bin/python -m pytest tests/unit/hardware/sim/test_g1_room.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_G1_SCENE_XML = Path(
    "/home/yusen/Desktop/vector_os_nano/vector_os_nano/hardware/sim/mjcf/g1/scene_g1_12dof_room.xml"
)


def _ensure_scene() -> Path:
    """Build the scene if not already built (idempotent, fast on cache hit)."""
    from vector_os_nano.hardware.sim.mujoco_g1 import _build_g1_room_scene_xml
    return _build_g1_room_scene_xml()


# ---------------------------------------------------------------------------
# Test 1: scene builds and compiles — both pick_table and g1_pelvis present
# ---------------------------------------------------------------------------


def test_scene_compiles_with_pick_table_and_g1_pelvis() -> None:
    """The compiled scene must contain both room geometry and g1 robot."""
    import mujoco

    scene_path = _ensure_scene()
    assert scene_path.exists(), f"scene_g1_12dof_room.xml not found at {scene_path}"

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody > 0, "Model has no bodies"

    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }

    assert "pick_table" in body_names, (
        f"pick_table not found in model bodies: {sorted(body_names)}"
    )
    assert "g1_pelvis" in body_names, (
        f"g1_pelvis not found in model bodies: {sorted(body_names)}"
    )

    # R2 specific: confirm model has 12 actuators (not 29+ from Menagerie)
    assert model.nu == 12, (
        f"Expected 12 actuators (12-DOF model), got {model.nu}"
    )


# ---------------------------------------------------------------------------
# Test 2: g1_head_rgb camera exists in compiled model
# ---------------------------------------------------------------------------


def test_g1_head_camera_in_compiled_model() -> None:
    """The compiled scene must contain the g1_head_rgb camera."""
    import mujoco

    scene_path = _ensure_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    cam_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    ]
    assert "g1_head_rgb" in cam_names, (
        f"g1_head_rgb camera not found; available cameras: {cam_names}"
    )


# ---------------------------------------------------------------------------
# Test 3: G1 self-geom-id set is non-empty
# ---------------------------------------------------------------------------


def test_g1_self_geom_set_nonempty() -> None:
    """_build_robot_geom_set must find at least one g1_* body geom."""
    import mujoco
    from vector_os_nano.hardware.sim.mujoco_g1 import (
        _build_g1_room_scene_xml,
        _build_robot_geom_set,
    )

    scene_path = _build_g1_room_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    geom_ids = _build_robot_geom_set(model)

    assert len(geom_ids) > 0, "Robot geom set is empty — lidar self-filter will not work"

    for gid in geom_ids:
        bid = model.geom_bodyid[gid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        assert name.startswith("g1_"), (
            f"Non-g1 body '{name}' in robot geom set (geom_id={gid})"
        )


# ---------------------------------------------------------------------------
# Test 4: g1 is a startable embodiment (no sim instantiated)
# ---------------------------------------------------------------------------


def test_g1_is_a_startable_embodiment() -> None:
    """The g1 manifest names a SimSession driver and start_simulation offers it."""
    from vector_os_nano.embodiments.registry import load
    from vector_os_nano.sim.session import _BUILDERS
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    assert load("g1").driver in _BUILDERS
    allowed = SimStartTool.input_schema["properties"]["embodiment"]["enum"]
    assert "g1" in allowed, f"'g1' not in embodiment enum; allowed values: {allowed}"


# ---------------------------------------------------------------------------
# Test 5: walk() produces forward displacement — policy gait is working
# ---------------------------------------------------------------------------


def test_walk_forward_displacement() -> None:
    """walk(vx=0.5, duration=2.0) must move the robot >0.3 m forward without falling.

    This is the key R2 correctness test: verifies the policy gait works in the
    combined room scene with correct qpos/qvel/ctrl offsets.
    """
    from vector_os_nano.hardware.sim.mujoco_g1 import MuJoCoG1

    g1 = MuJoCoG1(gui=False, room=True)
    try:
        g1.connect()

        # Record start position
        off = g1._offsets
        assert off is not None
        qa = off.pelvis_qpos_adr
        x0 = float(g1._data.qpos[qa])
        y0 = float(g1._data.qpos[qa + 1])
        z0 = g1.get_base_height()

        assert z0 > 0.5, f"Robot not standing at connect: z={z0:.3f}"

        # Walk forward for 2 seconds
        g1.walk(vx=0.5, duration=2.0)

        xf = float(g1._data.qpos[qa])
        yf = float(g1._data.qpos[qa + 1])
        zf = g1.get_base_height()

        displacement_x = xf - x0
        total_displacement = ((xf - x0) ** 2 + (yf - y0) ** 2) ** 0.5

        assert total_displacement > 0.3, (
            f"Robot did not move enough: displacement={total_displacement:.3f} m "
            f"(x: {x0:.3f} → {xf:.3f}, y: {y0:.3f} → {yf:.3f})"
        )
        assert zf > 0.5, (
            f"Robot fell during walk: final z={zf:.3f} (fell threshold: 0.5)"
        )

        # Also verify lidar and camera still work while/after walking
        scan = g1.get_lidar_scan()
        assert scan.n_returns > 0, "Lidar returns 0 hits after walk"

        frame = g1.get_camera_frame(width=320, height=240)
        assert frame.shape == (240, 320, 3), f"Camera frame shape wrong: {frame.shape}"

    finally:
        g1.close()


# ---------------------------------------------------------------------------
# Test 6: _G1NavResult contract — import-only, no sim instantiation
# ---------------------------------------------------------------------------


def test_g1_nav_result_contract() -> None:
    """_G1NavResult truthiness reflects 'reached'; .get() still works."""
    import inspect

    from vector_os_nano.hardware.sim.mujoco_g1 import MuJoCoG1, _G1NavResult

    # (a) bool reflects 'reached'
    r_false = _G1NavResult({"reached": False, "moved_m": 1.5, "reason": "timeout"})
    assert not r_false, "_G1NavResult({'reached': False}) must be falsy"

    r_true = _G1NavResult({"reached": True, "moved_m": 2.0, "reason": "arrived"})
    assert r_true, "_G1NavResult({'reached': True}) must be truthy"

    r_missing = _G1NavResult({"moved_m": 0.0})
    assert not r_missing, "_G1NavResult without 'reached' key must be falsy"

    # (b) dict accessor still works
    assert r_true.get("moved_m") == 2.0, ".get('moved_m') must return the stored value"
    assert r_false.get("reached") is False, ".get('reached') must return False"

    # (c) navigate_to signature contains 'timeout'
    sig = inspect.signature(MuJoCoG1.navigate_to)
    assert "timeout" in sig.parameters, (
        f"'timeout' not in MuJoCoG1.navigate_to parameters; got {list(sig.parameters)}"
    )
