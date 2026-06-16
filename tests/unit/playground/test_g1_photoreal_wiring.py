# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""G1 photoreal co-sim wiring (campaign #10 R3) — tested WITHOUT booting MuJoCo
or Blender. Constructing G1MuJoCoBase only checks the gait assets exist; the
``_photoreal_swap`` logic is exercised with an injected fake renderer.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.hardware.sim.mujoco_g1 import G1MuJoCoBase, g1_assets_ready

pytestmark = pytest.mark.skipif(
    not g1_assets_ready(), reason="g1 gait assets absent (setup_g1_gait.sh)")


class _FakeRenderer:
    def __init__(self):
        self.calls = []
        self._frame = np.full((4, 4, 3), 99, dtype=np.uint8)

    def render_from_pose(self, cam_pos, cam_mat, fovy):
        self.calls.append((np.asarray(cam_pos), np.asarray(cam_mat), float(fovy)))
        return self._frame


def _obs():
    return {"rgb": np.zeros((4, 4, 3), np.uint8),
            "depth": np.ones((4, 4), np.float32),
            "cam_pos": np.array([1.0, 2.0, 0.9]),
            "cam_mat": np.eye(3).reshape(-1),
            "fovy": 55.0}


def test_swap_is_noop_when_photoreal_off():
    base = G1MuJoCoBase(photoreal=False)
    obs = _obs()
    out = base._photoreal_swap(obs)
    assert out is obs                      # untouched — existing behaviour identical


def test_swap_replaces_rgb_from_same_pose_keeps_depth():
    base = G1MuJoCoBase(photoreal=True)
    base._photoreal_renderer = _FakeRenderer()   # inject; no Blender, no boot
    obs = _obs()
    out = base._photoreal_swap(obs)
    # rgb is now the photoreal frame; depth/pose are the MuJoCo originals (hybrid)
    assert np.array_equal(out["rgb"], np.full((4, 4, 3), 99, np.uint8))
    assert np.array_equal(out["rgb_mujoco"], obs["rgb"])
    assert out["depth"] is obs["depth"]
    np.testing.assert_array_equal(out["cam_pos"], obs["cam_pos"])
    # the photoreal RGB was rendered from EXACTLY the MuJoCo depth's pose
    cp, cm, fv = base._photoreal_renderer.calls[0]
    np.testing.assert_array_equal(cp, obs["cam_pos"])
    np.testing.assert_array_equal(cm, obs["cam_mat"])
    assert fv == obs["fovy"]


def test_photoreal_requested_without_blender_fails_loud(monkeypatch):
    base = G1MuJoCoBase(photoreal=True)     # no injected bridge
    monkeypatch.setenv("VECTOR_BLENDER", "/nonexistent/blender")
    monkeypatch.delenv("VECTOR_PHOTOREAL_ASSETS", raising=False)
    with pytest.raises(RuntimeError, match="no Blender"):
        base._photoreal_swap(_obs())
