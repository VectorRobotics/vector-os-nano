# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R3 — honest target localisation (recognition bearing + lidar range).

The recognize→navigate pivot (tricky Case 21): VLM visual-servoing the last metres
is flaky (slow + noisy perception × gait drift). Instead, the VLM RECOGNISES the
object and gives its BEARING (x_norm), and the LIDAR gives the RANGE to the nearest
surface in that direction — together a world (x, y) for the object. A reliable
``navigate_to`` (odometry + planner, collision-stop) then drives there.

The estimate is SENSOR-DERIVED — it NEVER consults ground-truth object coordinates
(rule 5). If nothing is sensed in the recognised direction it returns None and the
caller advances / fails honestly (the at_position GT check stays the judge).

Pure numpy; lidar points are world-frame ``(N, 4)`` ``[x, y, z, intensity]``
(intensity > 0 = a real hit; 0 = a free-ray miss, ignored).
"""
from __future__ import annotations

import math

import numpy as np

# Half the recognition camera's FOV (g1 HEAD_CAM / go2 RECOG_CAM are ~70–75°),
# mapping x_norm∈[-1, 1] to a bearing offset — shared with vision_seek._CAM_HALF_FOV.
_HALF_FOV = 0.80
_ANG_WINDOW = 0.30       # rad — accept lidar hits within this of the target bearing
_MIN_RANGE = 0.15        # ignore self/near specks below this range


def locate_from_bearing(
    robot_xy: "tuple[float, float]",
    robot_heading: float,
    x_norm: float,
    lidar_points: "np.ndarray | None",
    half_fov: float = _HALF_FOV,
    ang_window: float = _ANG_WINDOW,
) -> "tuple[float, float] | None":
    """World (x, y) of the recognised object, or None if not sensed in its
    direction.

    ``x_norm`` is the detection's horizontal centre in [-1, 1] (>0 = right). The
    target's world bearing is ``robot_heading - x_norm*half_fov`` (yaw+ = left,
    so a right-of-centre target is at a lower yaw). Among lidar HITS whose
    direction-from-robot falls within ``ang_window`` of that bearing, return the
    NEAREST one (the front surface of the object). None if there are no such hits
    (e.g. the object is beyond lidar range — the caller should advance first)."""
    if lidar_points is None:
        return None
    pts = np.asarray(lidar_points, dtype=np.float64).reshape(-1, 4)
    if pts.shape[0] == 0:
        return None
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    target_ang = robot_heading - float(x_norm) * half_fov

    best_xy = None
    best_rng = float("inf")
    for p in pts:
        if p[3] <= 0.0:                      # miss (free-ray endpoint) → ignore
            continue
        dx, dy = p[0] - rx, p[1] - ry
        rng = math.hypot(dx, dy)
        if rng < _MIN_RANGE:
            continue
        ang = math.atan2(dy, dx)
        d = math.atan2(math.sin(ang - target_ang), math.cos(ang - target_ang))
        if abs(d) <= ang_window and rng < best_rng:
            best_rng = rng
            best_xy = (float(p[0]), float(p[1]))
    return best_xy
