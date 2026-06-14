# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RED-first: the honest photoreal-perception-driven grasp (campaign #10 R10).

Both existing pick paths read GT object poses; this closes the loop honestly —
a VLM recognises the object on the PHOTOREAL frame, depth back-projects its bbox
to a 3D world point, and that PERCEPTION-derived point (never GT) is handed to
PickTopDownSkill as ``target_xyz``. Tested with fakes (no Blender/VLM/MuJoCo)."""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.perception.target_locate import locate_xyz_from_depth
from vector_os_nano.skills.recognize_pick import RecognizePickSkill


# -- locate_xyz_from_depth: same back-projection as locate_from_depth, keeps z --

def test_locate_xyz_returns_three_d_point_centre_pixel():
    # Camera at origin, identity pose (looks down world -Z, +Y up, +X right);
    # a flat depth of 2.0 m everywhere. Centre pixel (x_norm=y_norm=0) → straight
    # ahead along -Z → world (0, 0, -2).
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    xyz = locate_xyz_from_depth(0.0, 0.0, depth, [0.0, 0.0, 0.0],
                                np.eye(3).reshape(-1), 60.0)
    assert xyz is not None
    x, y, z = xyz
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    assert abs(z - (-2.0)) < 1e-6           # 2 m straight along -Z


def test_locate_xyz_none_on_empty_depth():
    depth = np.zeros((10, 10), dtype=np.float32)   # all invalid (<0.1)
    assert locate_xyz_from_depth(0.0, 0.0, depth, [0, 0, 0],
                                 np.eye(3).reshape(-1), 60.0) is None


# -- RecognizePickSkill: perception -> target_xyz -> pick --------------------

class _FakeDetector:
    def __init__(self, dets):
        self._dets = dets
    def detect_targets(self, rgb, query):
        return list(self._dets)


class _FakePick:
    def __init__(self):
        self.params = None
    def execute(self, params, context):
        self.params = params
        from vector_os_nano.core.types import SkillResult
        return SkillResult(success=True, result_data={"grasped": True})


class _FakeBase:
    """Exposes a g1-style get_camera_observation (rgb+depth+pose)."""
    def __init__(self):
        self._depth = np.full((48, 64), 1.5, dtype=np.float32)
    def get_camera_frame(self):
        return np.zeros((48, 64, 3), np.uint8)
    def get_camera_observation(self):
        return {"rgb": np.zeros((48, 64, 3), np.uint8), "depth": self._depth,
                "cam_pos": np.array([1.0, 2.0, 0.9]), "cam_mat": np.eye(3).reshape(-1),
                "fovy": 60.0}


class _Ctx:
    def __init__(self, base):
        self.base = base
        self.world_model = None
        self.perception = None
        self.calibration = None


def test_recognize_pick_locates_then_delegates_target_xyz():
    base = _FakeBase()
    det = _FakeDetector([{"label": "red can", "x_norm": 0.0, "y_norm": 0.0,
                          "area_frac": 0.05}])
    pick = _FakePick()
    skill = RecognizePickSkill(detector=det, pick=pick)
    res = skill.execute({"label": "red can"}, _Ctx(base))
    assert res.success
    # the pick was handed a PERCEPTION-derived target_xyz (not GT)
    assert pick.params is not None and "target_xyz" in pick.params
    x, y, z = pick.params["target_xyz"]
    # centre pixel, depth 1.5, cam at (1,2,0.9) identity → (1, 2, 0.9-1.5)
    assert abs(x - 1.0) < 1e-3 and abs(y - 2.0) < 1e-3 and abs(z - (-0.6)) < 1e-3


def test_recognize_pick_fails_when_not_recognised():
    skill = RecognizePickSkill(detector=_FakeDetector([]), pick=_FakePick())
    skill._DETECT_BACKOFF_S = 0.0       # don't sleep through the retries in tests
    res = skill.execute({"label": "red can"}, _Ctx(_FakeBase()))
    assert not res.success
    assert res.result_data.get("diagnosis") in {"not_found", "not_located"}


def test_recognize_pick_no_base():
    skill = RecognizePickSkill(detector=_FakeDetector([]), pick=_FakePick())
    res = skill.execute({"label": "x"}, _Ctx(None))
    assert not res.success and res.result_data.get("diagnosis") == "no_base"
