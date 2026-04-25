# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Virtual Livox Mid-360 lidar against a MuJoCo scene.

``mj_ray`` is fired in a polar grid pattern that mimics the Mid-360
field of view (HFoV 360 degrees, VFoV roughly -7 degrees to +52 degrees);
hits are returned in world (== ``map``) frame because the simulator
already provides ground-truth pose. The bridge subprocess wraps the
emitted :class:`LidarSample` into a ROS2 ``sensor_msgs/PointCloud2`` —
this module deliberately avoids rclpy at module load time so unit tests
run without sourcing a ROS2 workspace.

Layout reference for the encoded PointCloud2 data bytes (field offsets
match what consumers like ``sensor_msgs_py.point_cloud2`` expect):

    offset 0  : float32 x
    offset 4  : float32 y
    offset 8  : float32 z
    offset 12 : float32 intensity
    point_step = 16
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


# PointField datatype constants — stable values in sensor_msgs/PointField
# (FLOAT32 = 7). We avoid importing the ROS2 message at module load so
# tests run without a sourced workspace.
_PCD_FLOAT32 = 7
_POINT_STEP = 16


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LidarSample:
    """Snapshot of a virtual lidar scan in world frame.

    ``points`` shape is ``(N, 4)`` float32: ``[x, y, z, intensity]``.
    The bridge subprocess converts this to ``sensor_msgs/PointCloud2``
    via :meth:`pointcloud2_data_bytes`.
    """

    stamp_seconds: float
    frame_id: str
    points: np.ndarray              # (N, 4) float32

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def point_step(self) -> int:
        return _POINT_STEP

    @property
    def is_dense(self) -> bool:
        return True

    @property
    def is_bigendian(self) -> bool:
        return False

    @property
    def field_layout(self) -> list[tuple[str, int, int]]:
        """List of (name, offset_bytes, datatype) for each PointField."""
        return [
            ("x", 0, _PCD_FLOAT32),
            ("y", 4, _PCD_FLOAT32),
            ("z", 8, _PCD_FLOAT32),
            ("intensity", 12, _PCD_FLOAT32),
        ]

    def pointcloud2_data_bytes(self) -> bytes:
        """Serialise the points as the ``data`` field of PointCloud2.

        ``points`` is enforced to float32 if not already, then flattened
        in C order (row-major).
        """
        arr = self.points
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr.tobytes()


# ---------------------------------------------------------------------------
# Lidar publisher
# ---------------------------------------------------------------------------


class MuJoCoLivox360:
    """Polar-grid ray cast against a MuJoCo ``MjModel``.

    Args:
        model: ``mujoco.MjModel`` instance.
        data: ``mujoco.MjData`` paired with *model*.
        body_name: name of the body the lidar is mounted on.
        offset: (x, y, z) offset in body frame from the body origin to
            the lidar centre. Default is +0.10 m up — typical Mid-360
            mount on Go2 trunk.
        h_resolution: number of azimuth samples per spin (default 360).
        v_layers: number of elevation layers (default 16).
        v_min_deg: lower elevation bound, degrees (default -7).
        v_max_deg: upper elevation bound, degrees (default 52).
        max_range: rays beyond this are discarded (default 30 m).
        rate_hz: maximum sample rate (default 10).
        frame_id: header frame for the emitted PointCloud2 (default ``map``).

    Raises:
        ValueError: ``body_name`` not found in *model*; resolution
            parameters non-positive; v_min_deg >= v_max_deg.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        body_name: str = "trunk",
        offset: tuple[float, float, float] = (0.0, 0.0, 0.10),
        h_resolution: int = 360,
        v_layers: int = 16,
        v_min_deg: float = -7.0,
        v_max_deg: float = 52.0,
        max_range: float = 30.0,
        rate_hz: float = 10.0,
        frame_id: str = "map",
    ) -> None:
        import mujoco

        if h_resolution <= 0 or v_layers <= 0:
            raise ValueError("h_resolution and v_layers must be positive")
        if v_min_deg > v_max_deg:
            raise ValueError("v_min_deg must be <= v_max_deg")
        if max_range <= 0:
            raise ValueError("max_range must be positive")

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"body {body_name!r} not found in model")

        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._body_id = int(body_id)
        self._offset = tuple(float(v) for v in offset)
        self._h_resolution = int(h_resolution)
        self._v_layers = int(v_layers)
        self._v_min = math.radians(v_min_deg)
        self._v_max = math.radians(v_max_deg)
        self._max_range = float(max_range)
        self._rate_hz = float(rate_hz)
        self._frame_id = str(frame_id)

        self._ray_dirs_body = _build_ray_dirs(
            self._h_resolution, self._v_layers, self._v_min, self._v_max,
        )
        self._last_step_t: float | None = None
        self._cached: LidarSample | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def num_rays(self) -> int:
        return int(self._ray_dirs_body.shape[0])

    @property
    def ray_dirs_body(self) -> np.ndarray:
        """Read-only copy of the body-frame ray directions."""
        return self._ray_dirs_body.copy()

    def due(self, now: float | None = None) -> bool:
        now = self._mono(now)
        if self._cached is None:
            return True
        return (now - self._last_step_t) >= (1.0 / self._rate_hz)

    def step(self, now: float | None = None) -> LidarSample:
        """Cast all rays once. Returns cached sample within rate-limit window."""
        now = self._mono(now)
        if self._cached is not None and not self.due(now):
            return self._cached

        # Body pose in world frame
        body_pos = np.array(self._data.xpos[self._body_id], dtype=np.float64)
        body_quat = np.array(self._data.xquat[self._body_id], dtype=np.float64)
        body_R = _quat_wxyz_to_rot(body_quat)

        origin = body_pos + body_R @ np.array(self._offset, dtype=np.float64)
        ray_dirs_world = (body_R @ self._ray_dirs_body.T).T

        hits = self._cast_rays(origin, ray_dirs_world)
        sample = LidarSample(
            stamp_seconds=now,
            frame_id=self._frame_id,
            points=hits,
        )
        self._cached = sample
        self._last_step_t = now
        return sample

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _cast_rays(
        self, origin: np.ndarray, dirs_world: np.ndarray,
    ) -> np.ndarray:
        """Vectorised mj_ray loop (Python — fast enough at 5760 rays/frame).

        Returns (N, 4) float32 ``[x, y, z, intensity]`` for hits inside
        ``max_range``. Misses are discarded.
        """
        n = dirs_world.shape[0]
        geomid_buf = np.zeros(1, dtype=np.int32)
        out = np.empty((n, 4), dtype=np.float32)
        write = 0
        origin_buf = origin.astype(np.float64)
        for i in range(n):
            direction = np.ascontiguousarray(dirs_world[i], dtype=np.float64)
            dist = self._mujoco.mj_ray(
                self._model, self._data, origin_buf, direction,
                None, 1, -1, geomid_buf,
            )
            if dist < 0.0 or dist > self._max_range:
                continue
            hit = origin_buf + direction * dist
            out[write, 0] = hit[0]
            out[write, 1] = hit[1]
            out[write, 2] = hit[2]
            out[write, 3] = 1.0       # intensity placeholder
            write += 1
        return out[:write]

    @staticmethod
    def _mono(now: float | None) -> float:
        return float(now) if now is not None else time.monotonic()


# ---------------------------------------------------------------------------
# Pure helpers (testable in isolation)
# ---------------------------------------------------------------------------


def _build_ray_dirs(
    h_resolution: int, v_layers: int, v_min_rad: float, v_max_rad: float,
) -> np.ndarray:
    """Build a ``(h_resolution * v_layers, 3)`` array of unit ray directions.

    Azimuth excludes the +π endpoint to avoid duplicating the -π column;
    elevation includes both endpoints.
    """
    azimuths = np.linspace(-math.pi, math.pi, h_resolution, endpoint=False)
    if v_layers == 1:
        elevations = np.array([0.5 * (v_min_rad + v_max_rad)])
    else:
        elevations = np.linspace(v_min_rad, v_max_rad, v_layers)
    az_grid, el_grid = np.meshgrid(azimuths, elevations)
    cos_el = np.cos(el_grid)
    cx = cos_el * np.cos(az_grid)
    cy = cos_el * np.sin(az_grid)
    cz = np.sin(el_grid)
    return np.stack([cx, cy, cz], axis=-1).reshape(-1, 3).astype(np.float64)


def _quat_wxyz_to_rot(wxyz: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
