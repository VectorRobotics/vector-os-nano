# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2/Piper photoreal hook (campaign #10 R6) — the second embodiment on the
shared co-sim factory. Tested with injected fakes (no MuJoCo boot, no Blender)."""
from __future__ import annotations

import numpy as np

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2


class _FakeRenderer:
    def __init__(self):
        self.calls = []
        self._frame = np.full((6, 6, 3), 42, dtype=np.uint8)

    def render_from_pose(self, cam_pos, cam_mat, fovy):
        self.calls.append((np.asarray(cam_pos), np.asarray(cam_mat), float(fovy)))
        return self._frame


class _FakeMj:
    """Minimal stand-in for the go2 model/data with one camera pose."""
    class _M:
        cam_fovy = np.array([58.0])
    class _D:
        cam_xpos = np.array([[0.3, 0.0, 0.45]])
        cam_xmat = np.eye(3).reshape(1, 9)
    model = _M()
    data = _D()


def test_photoreal_frame_renders_from_live_cam_pose():
    base = MuJoCoGo2(photoreal=True)
    base._photoreal_renderer = _FakeRenderer()   # inject; skip factory/Blender
    base._mj = _FakeMj()
    frame = base._photoreal_frame(cam_id=0, cam_name="d435_rgb")
    assert np.array_equal(frame, np.full((6, 6, 3), 42, np.uint8))
    cp, cm, fv = base._photoreal_renderer.calls[0]
    np.testing.assert_array_equal(cp, [0.3, 0.0, 0.45])     # the live cam pose
    np.testing.assert_array_equal(cm, np.eye(3).reshape(9))
    assert fv == 58.0


def test_photoreal_flag_defaults_off():
    assert MuJoCoGo2()._photoreal is False
    assert MuJoCoGo2(photoreal=True)._photoreal is True
