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


def locate_from_depth(
    x_norm: float,
    y_norm: float,
    depth: "np.ndarray",
    cam_pos,
    cam_mat,
    fovy_deg: float,
    win: int = 6,
    max_depth: float = 30.0,
) -> "tuple[float, float] | None":
    """World (x, y) of the RECOGNISED object by back-projecting the depth at its
    bbox centre — the semantically-correct estimate (depth at the object's OWN
    pixels, so an intervening obstacle is NOT mistaken for the target, unlike
    nearest-lidar; tricky Case 22). Depth is the MuJoCo renderer's per-pixel
    distance (metres); the recognition camera is OpenGL-convention (looks along
    its -z, +x right, +y up). ``cam_pos`` (3,) and ``cam_mat`` (9 row-major or
    3x3) are the camera world pose captured WITH the frame.

    ``x_norm, y_norm`` are the bbox centre in [-1, 1] (>0 = right / down). A
    small ``win``×``win`` window median rejects single bad pixels; far-clip /
    zero / >max_depth samples are dropped. None if no valid depth at the bbox."""
    if depth is None:
        return None
    dimg = np.asarray(depth, dtype=np.float64)
    if dimg.ndim != 2:
        return None
    H, W = dimg.shape
    px = int(round((float(x_norm) + 1.0) * 0.5 * (W - 1)))
    py = int(round((float(y_norm) + 1.0) * 0.5 * (H - 1)))
    px = max(0, min(W - 1, px))
    py = max(0, min(H - 1, py))
    x0, x1 = max(0, px - win), min(W, px + win + 1)
    y0, y1 = max(0, py - win), min(H, py + win + 1)
    patch = dimg[y0:y1, x0:x1].ravel()
    valid = patch[(patch > 0.1) & (patch < max_depth)]
    if valid.size == 0:
        return None
    # NEAREST surface in the bbox window, not the median: a thin object (chair
    # legs/back) lets the bbox see THROUGH to the wall behind, biasing a median
    # too far (and to an unreachable point past the target — R5). A low
    # percentile is the object's own FRONT FACE, robust to a few near specks.
    z = float(np.percentile(valid, 20))

    # OpenGL pinhole back-projection. fovy is the VERTICAL field of view.
    f = (H / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    xc = (px - W / 2.0) / f
    yc = -(py - H / 2.0) / f          # image y is down → camera y is up
    p_cam = np.array([xc * z, yc * z, -z], dtype=np.float64)   # looks along -z
    rot = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    p_world = np.asarray(cam_pos, dtype=np.float64) + rot @ p_cam
    return (float(p_world[0]), float(p_world[1]))


def locate_xyz_from_depth(
    x_norm: float,
    y_norm: float,
    depth: "np.ndarray",
    cam_pos,
    cam_mat,
    fovy_deg: float,
    win: int = 6,
    max_depth: float = 30.0,
) -> "tuple[float, float, float] | None":
    """Like ``locate_from_depth`` but returns the full world ``(x, y, z)`` — the
    3D grasp target for the recognised object (campaign #10 R10). Same OpenGL
    back-projection of the bbox-centre depth; z is the object's own surface
    height, never read from ground truth (rule 5)."""
    if depth is None:
        return None
    dimg = np.asarray(depth, dtype=np.float64)
    if dimg.ndim != 2:
        return None
    H, W = dimg.shape
    px = max(0, min(W - 1, int(round((float(x_norm) + 1.0) * 0.5 * (W - 1)))))
    py = max(0, min(H - 1, int(round((float(y_norm) + 1.0) * 0.5 * (H - 1)))))
    x0, x1 = max(0, px - win), min(W, px + win + 1)
    y0, y1 = max(0, py - win), min(H, py + win + 1)
    patch = dimg[y0:y1, x0:x1].ravel()
    valid = patch[(patch > 0.1) & (patch < max_depth)]
    if valid.size == 0:
        return None
    z = float(np.percentile(valid, 20))    # nearest surface (object front face)
    f = (H / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    xc = (px - W / 2.0) / f
    yc = -(py - H / 2.0) / f
    p_cam = np.array([xc * z, yc * z, -z], dtype=np.float64)
    rot = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    p_world = np.asarray(cam_pos, dtype=np.float64) + rot @ p_cam
    return (float(p_world[0]), float(p_world[1]), float(p_world[2]))
