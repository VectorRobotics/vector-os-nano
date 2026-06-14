# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The shared co-sim factory (campaign #10 R6) — embodiment-agnostic wiring used
by both the G1 and Go2/Piper bases. Tested with an injected fake bridge."""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.playground.photoreal import cosim


class _FakeBridge:
    def render(self, pose, scene, *, width=640, height=480, samples=32):
        return np.zeros((height, width, 3), np.uint8)


def test_asset_map_present_when_armchair_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_PHOTOREAL_ASSETS", str(tmp_path))
    (tmp_path / "armchair").mkdir()
    (tmp_path / "armchair" / "ArmChair_01_4k.gltf").write_text("x")
    amap = cosim.furnished_room_asset_map()
    assert "target_chair" in amap and amap["target_chair"]["path"].endswith(".gltf")


def test_asset_map_empty_when_assets_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_PHOTOREAL_ASSETS", str(tmp_path))
    assert cosim.furnished_room_asset_map() == {}


def test_factory_builds_renderer_with_injected_bridge_and_cam(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_PHOTOREAL_ASSETS", str(tmp_path))   # empty lib ok
    fake = _FakeBridge()
    renderer, bridge = cosim.furnished_room_renderer(
        cam_name="d435_rgb", bridge=fake, width=320, height=240, samples=8)
    assert bridge is fake
    assert renderer._cam_name == "d435_rgb"
    # the scene spec carries the furnished-room floor + (skipped) assets
    assert "floor" in renderer._scene and "assets" in renderer._scene
    rgb = renderer.render_from_pose([0, 0, 1.0], np.eye(3).reshape(-1), 55.0)
    assert rgb.shape == (240, 320, 3)


def test_factory_without_blender_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_PHOTOREAL_ASSETS", str(tmp_path))
    monkeypatch.setenv("VECTOR_BLENDER", "/nonexistent/blender")
    with pytest.raises(RuntimeError, match="no Blender"):
        cosim.furnished_room_renderer(cam_name="d435_rgb")
