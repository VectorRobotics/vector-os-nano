# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Ground-truth odometry from MuJoCo body pose.

Skips SLAM entirely: the simulator already knows where bodies are.
This is appropriate for sim-only deployments (v2.4 SysNav-on-MuJoCo).
The real-robot bringup must use ``arise_slam_mid360`` instead — same
output topic ``/state_estimation`` so downstream consumers (SysNav
``semantic_mapping_node``) cannot tell the difference.

Reads MuJoCo ``data.xpos[body_id]`` and ``data.xquat[body_id]`` each
``step()``; finite-differences position to estimate twist; converts
the (w, x, y, z) MuJoCo quaternion to (x, y, z, w) ROS convention.

No rclpy import at module load. Tests construct a tiny MJCF with a
single body — no Go2 model dependency.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OdomSample:
    """Plain-Python mirror of ``nav_msgs/Odometry`` payload.

    Field names match ROS but units are plain Python floats.
    """

    stamp_seconds: float
    frame_id: str
    child_frame_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]   # (qx, qy, qz, qw) — ROS order
    linear_twist: tuple[float, float, float]
    angular_twist: tuple[float, float, float]


class GroundTruthOdomPublisher:
    """Read MuJoCo body pose; emit Odometry samples at fixed rate.

    Args:
        model: ``mujoco.MjModel`` instance.
        data:  ``mujoco.MjData`` paired with *model*.
        body_name: name of the body whose pose is published.
        rate_hz: maximum sample rate. Calls within ``1 / rate_hz`` of
            the previous accepted sample return the cached result.
        frame_id: ROS parent frame for the pose.
        child_frame_id: ROS child frame for the pose.

    Raises:
        ValueError: if ``body_name`` is not in *model*.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        body_name: str = "trunk",
        rate_hz: float = 50.0,
        frame_id: str = "map",
        child_frame_id: str = "sensor",
    ) -> None:
        # Lazy mujoco import to keep module-load cheap
        import mujoco

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"body {body_name!r} not found in model")

        self._model = model
        self._data = data
        self._body_id = int(body_id)
        self._rate_hz = float(rate_hz)
        self._frame_id = str(frame_id)
        self._child_frame_id = str(child_frame_id)

        self._last_step_t: float | None = None
        self._last_xpos: tuple[float, float, float] | None = None
        self._last_xquat_wxyz: tuple[float, float, float, float] | None = None
        self._cached: OdomSample | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def body_id(self) -> int:
        """MuJoCo body id resolved at construction."""
        return self._body_id

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    def due(self, now: float | None = None) -> bool:
        """Whether ``step()`` should produce a fresh sample.

        Useful when the bridge subprocess wants to avoid the work of
        reading MuJoCo state when the rate-limit has not elapsed.
        """
        now = self._monotonic(now)
        if self._cached is None:
            return True
        return (now - self._last_step_t) >= (1.0 / self._rate_hz)

    def step(self, now: float | None = None) -> OdomSample:
        """Return the latest odom sample.

        First call always computes a sample. Subsequent calls within
        ``1 / rate_hz`` of the previous accepted sample return the
        cached result so callers can poll without computing twice.
        """
        now = self._monotonic(now)

        if self._cached is not None and not self.due(now):
            return self._cached

        xpos = (
            float(self._data.xpos[self._body_id, 0]),
            float(self._data.xpos[self._body_id, 1]),
            float(self._data.xpos[self._body_id, 2]),
        )
        # MuJoCo stores xquat as (w, x, y, z)
        wxyz = self._data.xquat[self._body_id]
        qw, qx, qy, qz = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm < 1e-9:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        # Twist via finite difference between this sample and the previous
        if (
            self._last_xpos is None
            or self._last_xquat_wxyz is None
            or self._last_step_t is None
        ):
            linear = (0.0, 0.0, 0.0)
            angular = (0.0, 0.0, 0.0)
        else:
            dt = max(now - self._last_step_t, 1e-3)
            linear = (
                (xpos[0] - self._last_xpos[0]) / dt,
                (xpos[1] - self._last_xpos[1]) / dt,
                (xpos[2] - self._last_xpos[2]) / dt,
            )
            angular = self._angular_twist(
                self._last_xquat_wxyz, (qw, qx, qy, qz), dt
            )

        sample = OdomSample(
            stamp_seconds=now,
            frame_id=self._frame_id,
            child_frame_id=self._child_frame_id,
            position=xpos,
            orientation=(qx, qy, qz, qw),    # ROS xyzw order
            linear_twist=linear,
            angular_twist=angular,
        )

        self._last_step_t = now
        self._last_xpos = xpos
        self._last_xquat_wxyz = (qw, qx, qy, qz)
        self._cached = sample
        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _monotonic(now: float | None) -> float:
        return float(now) if now is not None else time.monotonic()

    @staticmethod
    def _angular_twist(
        prev_wxyz: tuple[float, float, float, float],
        curr_wxyz: tuple[float, float, float, float],
        dt: float,
    ) -> tuple[float, float, float]:
        """Return small-angle approximation of angular velocity in body frame.

        Uses ``omega ≈ 2 * (curr * prev_conj).vec / dt``, valid for the
        small per-step rotations seen in 50 Hz odometry.
        """
        pw, px, py, pz = prev_wxyz
        cw, cx, cy, cz = curr_wxyz
        # prev conjugate
        cpw, cpx, cpy, cpz = pw, -px, -py, -pz
        # quaternion product curr * prev_conj
        rw = cw * cpw - cx * cpx - cy * cpy - cz * cpz
        rx = cw * cpx + cx * cpw + cy * cpz - cz * cpy
        ry = cw * cpy - cx * cpz + cy * cpw + cz * cpx
        rz = cw * cpz + cx * cpy - cy * cpx + cz * cpw
        # If rw < 0, flip — keeps the rotation in the [0, π] range
        if rw < 0.0:
            rx, ry, rz = -rx, -ry, -rz
        return (2.0 * rx / dt, 2.0 * ry / dt, 2.0 * rz / dt)
