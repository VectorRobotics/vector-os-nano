# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""G1's pinned policy, room, camera and lidar on the shared driver and runtime."""
from __future__ import annotations

import math

import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.hardware.sim.scene_builder import room_scene
from vector_os_nano.schemas.world import load_world


def test_g1_scene_and_sensors_compile(g1_assets):
    cfg = registry.load("g1")
    scene = room_scene(cfg, load_world(cfg.world))
    assert scene.body("pick_table") != scene.robot_root == scene.body("pelvis")
    assert scene.model.nu == 12 and cfg.camera.name in scene.cameras
    assert scene.body(cfg.lidar.body) in scene.subtree(scene.robot_root)


def test_g1_is_a_startable_embodiment(g1_assets):
    from vector_os_nano.sim.session import SimSession, SimSpec
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    SimSession.validate(SimSpec("g1", gui=False))
    assert "g1" in SimStartTool.input_schema["properties"]["embodiment"]["enum"]


@pytest.mark.sim
def test_walk_forward_displacement_and_sensors(g1_assets, monkeypatch):
    from tests.hardware.sim.dog import stand_up
    from vector_os_nano.sim.drivers.camera import Camera

    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    robot = stand_up("g1")
    try:
        base = robot.base
        x0, y0, z0 = base.get_position()
        assert z0 > 0.5
        assert base.walk(vx=0.5, duration=2.0)
        x, y, z = base.get_position()
        assert math.hypot(x - x0, y - y0) > 0.3 and x - x0 > 0.3
        assert z > 0.5
        scan = base.get_lidar_scan()
        assert any(r < scan.range_max for r in scan.ranges)
        camera = Camera(robot.cfg.camera, robot.rt)
        frame = camera.capture()
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.depth.shape == (480, 640)
    finally:
        robot.close()
