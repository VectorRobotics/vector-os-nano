# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R3 — honest target localisation from recognition bearing + lidar.

The recognize→navigate pivot (Case 21): the VLM gives a bearing (which direction
the object is), the LIDAR gives range (how far the nearest surface in that
direction is). Combining them yields a WORLD (x,y) for the recognised object —
sensor-derived, never ground truth. The reliable navigate_to then drives there.
Pure numpy; deterministic; no sim. Lidar points are world-frame (x,y,z,intensity)
per MuJoCoLivox360 (intensity>0 = a real hit; 0 = a free-ray miss to ignore).
"""
from __future__ import annotations

import math

import numpy as np

from vector_os_nano.perception.target_locate import (
    locate_from_bearing,
    locate_from_depth,
)


def _pts(rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


class TestLocate:
    def test_picks_hit_in_bearing_direction(self):
        # robot at origin facing +x; target dead ahead (x_norm 0) → world angle 0.
        # a hit at (2.0, 0.0) in that direction, plus a distractor hit to the side.
        pts = _pts([[2.0, 0.0, 0.3, 1.0],     # ahead — the target surface
                    [0.5, 2.0, 0.3, 1.0],     # off to the left — different bearing
                    [3.0, 0.0, 0.3, 0.0]])     # a MISS (intensity 0) — ignored
        xy = locate_from_bearing((0.0, 0.0), 0.0, 0.0, pts)
        assert xy is not None
        assert abs(xy[0] - 2.0) < 1e-3 and abs(xy[1] - 0.0) < 1e-3

    def test_bearing_to_the_right(self):
        # x_norm>0 = target on the right → world angle = heading - x_norm*half_fov < 0.
        # hit down-right at (1.5, -1.0) (~ -33°); x_norm chosen to point there.
        ang = math.atan2(-1.0, 1.5)
        x_norm = -ang / 0.80     # invert world_angle = heading - x_norm*half_fov
        pts = _pts([[1.5, -1.0, 0.3, 1.0], [1.5, 1.0, 0.3, 1.0]])
        xy = locate_from_bearing((0.0, 0.0), 0.0, x_norm, pts)
        assert xy is not None and xy[1] < 0          # picked the right-side hit

    def test_nearest_hit_when_several_in_window(self):
        # two hits roughly ahead — the NEAREST surface is the object front.
        pts = _pts([[3.0, 0.1, 0.3, 1.0], [1.8, -0.05, 0.3, 1.0]])
        xy = locate_from_bearing((0.0, 0.0), 0.0, 0.0, pts)
        assert abs(xy[0] - 1.8) < 1e-3

    def test_none_when_nothing_in_direction(self):
        pts = _pts([[0.0, 3.0, 0.3, 1.0]])           # only a side hit
        assert locate_from_bearing((0.0, 0.0), 0.0, 0.0, pts) is None

    def test_none_on_empty_or_all_miss(self):
        assert locate_from_bearing((0.0, 0.0), 0.0, 0.0, _pts([])) is None
        assert locate_from_bearing((0.0, 0.0), 0.0, 0.0,
                                   _pts([[2.0, 0.0, 0.3, 0.0]])) is None

    def test_respects_robot_pose(self):
        # robot at (1,1) facing +y (heading=pi/2); target ahead → hit at (1, 3).
        pts = _pts([[1.0, 3.0, 0.3, 1.0]])
        xy = locate_from_bearing((1.0, 1.0), math.pi / 2, 0.0, pts)
        assert xy is not None
        assert abs(xy[0] - 1.0) < 1e-3 and abs(xy[1] - 3.0) < 1e-3


# --- depth-at-bbox (R5): back-project the depth at the recognised pixel ---
# Camera looking along world +x, up = +z (cam -z = +x): cam_mat columns are the
# camera axes in world = x_right=(0,-1,0), y_up=(0,0,1), z=(-1,0,0).
_CAM_MAT_FWD_X = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)


def _depth(H=240, W=320, fill=40.0):
    return np.full((H, W), fill, dtype=np.float32)


class TestLocateFromDepth:
    def test_centre_pixel_projects_forward(self):
        # chair dead-ahead: bbox centre (x_norm=0,y_norm=0) at depth 3.0, cam at
        # (0,0,1) looking +x → world (3, 0, 1); navigate uses (x, y)=(3,0).
        d = _depth()
        d[110:130, 150:170] = 3.0       # a patch at the centre
        xy = locate_from_depth(0.0, 0.0, d, (0.0, 0.0, 1.0),
                               _CAM_MAT_FWD_X.flatten(), 70.0)
        assert xy is not None
        assert abs(xy[0] - 3.0) < 0.15 and abs(xy[1] - 0.0) < 0.15

    def test_right_of_centre_projects_to_negative_y(self):
        # a target on the image RIGHT (x_norm>0) is at world -y when facing +x.
        d = _depth()
        d[110:130, 230:250] = 3.0
        xy = locate_from_depth(0.6, 0.0, d, (0.0, 0.0, 1.0),
                               _CAM_MAT_FWD_X.flatten(), 70.0)
        assert xy is not None and xy[1] < 0

    def test_respects_cam_position(self):
        d = _depth()
        d[110:130, 150:170] = 2.0
        xy = locate_from_depth(0.0, 0.0, d, (1.0, 5.0, 1.0),
                               _CAM_MAT_FWD_X.flatten(), 70.0)
        assert abs(xy[0] - 3.0) < 0.15 and abs(xy[1] - 5.0) < 0.15

    def test_none_on_farclip_or_invalid_depth(self):
        d = _depth(fill=40.0)           # all far-clip → no valid surface
        assert locate_from_depth(0.0, 0.0, d, (0.0, 0.0, 1.0),
                                 _CAM_MAT_FWD_X.flatten(), 70.0,
                                 max_depth=20.0) is None
        z = np.zeros((240, 320), dtype=np.float32)   # all zero (no return)
        assert locate_from_depth(0.0, 0.0, z, (0.0, 0.0, 1.0),
                                 _CAM_MAT_FWD_X.flatten(), 70.0) is None
