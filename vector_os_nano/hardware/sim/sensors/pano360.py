# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Virtual 360-degree RGB camera against a MuJoCo scene.

Six 90-degree-FoV cube faces are rendered with ``mujoco.Renderer`` and
stitched into a 1920 × 640 equirectangular image (HFoV 360 deg, VFoV
120 deg cropped) — matching the output format SysNav's
``cloud_image_fusion.CAMERA_PARA`` expects (``hfov=360, vfov=120,
width=1920, height=640``).

Depth is intentionally omitted: SysNav projects lidar voxels onto the
RGB pano pixel-by-pixel via known camera intrinsics; it does not
consume a separate depth image. A future cycle can add metric depth
output if downstream consumers need it.

Module-load is rclpy-free; the bridge converts :class:`PanoSample` to
``sensor_msgs/Image`` at publish time.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


_FACE_FORWARD = 0
_FACE_RIGHT = 1
_FACE_BACK = 2
_FACE_LEFT = 3
_FACE_UP = 4
_FACE_DOWN = 5

# Spec §3.3 default crop: HFoV=360°, VFoV=120° (60° above, 60° below)
_DEFAULT_VFOV_RAD = math.radians(120.0)


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanoSample:
    """Snapshot of a virtual panoramic camera frame.

    ``image`` is ``(H, W, 3)`` uint8 RGB in equirectangular layout.
    """

    stamp_seconds: float
    frame_id: str
    image: np.ndarray              # (H, W, 3) uint8

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def encoding(self) -> str:
        return "rgb8"


# ---------------------------------------------------------------------------
# Pano camera publisher
# ---------------------------------------------------------------------------


class MuJoCoPano360:
    """Cube-faced equirectangular RGB camera mounted on a MuJoCo body.

    Args:
        model: ``mujoco.MjModel`` instance.
        data:  paired ``mujoco.MjData``.
        body_name: body the pano is mounted on.
        offset: (x, y, z) body-frame mount offset.
        out_w: equirectangular output width (default 1920 — SysNav spec).
        out_h: equirectangular output height (default 640 — SysNav spec).
        face_size: cube-face render size (default 480).
        vfov_deg: vertical field of view of the output (default 120).
        rate_hz: max sample rate.
        frame_id: ROS header frame.
    """

    _FACE_AZIMUTH_DEG: dict[int, float] = {
        _FACE_FORWARD: 0.0,
        _FACE_RIGHT: -90.0,
        _FACE_BACK: 180.0,
        _FACE_LEFT: 90.0,
        _FACE_UP: 0.0,
        _FACE_DOWN: 0.0,
    }
    _FACE_ELEVATION_DEG: dict[int, float] = {
        _FACE_FORWARD: 0.0,
        _FACE_RIGHT: 0.0,
        _FACE_BACK: 0.0,
        _FACE_LEFT: 0.0,
        _FACE_UP: 90.0,
        _FACE_DOWN: -90.0,
    }

    def __init__(
        self,
        model: Any,
        data: Any,
        body_name: str = "trunk",
        offset: tuple[float, float, float] = (0.0, 0.0, 0.185),
        out_w: int = 1920,
        out_h: int = 640,
        face_size: int = 480,
        vfov_deg: float = 120.0,
        rate_hz: float = 5.0,
        frame_id: str = "map",
    ) -> None:
        import mujoco

        if out_w <= 0 or out_h <= 0:
            raise ValueError("out_w and out_h must be positive")
        if face_size <= 0:
            raise ValueError("face_size must be positive")
        if not (0 < vfov_deg <= 180):
            raise ValueError("vfov_deg must be in (0, 180]")

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"body {body_name!r} not found in model")

        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._body_id = int(body_id)
        self._offset = tuple(float(v) for v in offset)
        self._out_w = int(out_w)
        self._out_h = int(out_h)
        self._face_size = int(face_size)
        self._vfov_rad = math.radians(float(vfov_deg))
        self._rate_hz = float(rate_hz)
        self._frame_id = str(frame_id)

        # Lazy renderer: defer GL context creation until step() is first
        # called so unit tests that only exercise LUT math do not allocate
        # a renderer.
        self._renderer: Any = None
        self._depth_renderer: Any = None    # lazy, M4 feed-source depth
        self._cube_cam: Any = None

        self._face_idx, self._fx, self._fy = _build_equirec_lut(
            self._out_w, self._out_h, self._face_size, self._vfov_rad,
        )

        self._last_step_t: float | None = None
        self._cached: PanoSample | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def out_w(self) -> int:
        return self._out_w

    @property
    def out_h(self) -> int:
        return self._out_h

    @property
    def face_size(self) -> int:
        return self._face_size

    @property
    def lut(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read-only access to (face_idx, fx, fy) LUT for testing."""
        return self._face_idx.copy(), self._fx.copy(), self._fy.copy()

    def due(self, now: float | None = None) -> bool:
        now = self._mono(now)
        if self._cached is None:
            return True
        return (now - self._last_step_t) >= (1.0 / self._rate_hz)

    def step(self, now: float | None = None) -> PanoSample:
        now = self._mono(now)
        if self._cached is not None and not self.due(now):
            return self._cached

        if self._renderer is None:
            self._init_renderer()

        cube_imgs = self._render_cube_faces()
        pano = _stitch_equirec(cube_imgs, self._face_idx, self._fx, self._fy)
        sample = PanoSample(
            stamp_seconds=now,
            frame_id=self._frame_id,
            image=pano,
        )
        self._cached = sample
        self._last_step_t = now
        return sample

    def render_rgbd(self) -> "tuple[np.ndarray, np.ndarray]":
        """Render an equirect (rgb, depth) pair NOW (campaign #11 M4 feed source).

        rgb is (out_h, out_w, 3) uint8 (same as step().image); depth is
        (out_h, out_w) float32 metres — the per-pixel cube-face depth stitched
        with the SAME LUT. Depth is the renderer's z-buffer distance per face
        (good enough for the SysNav feed's registered-scan; the bridge's
        unproject handles the equirect geometry). Used by base.get_pano()."""
        if self._renderer is None:
            self._init_renderer()
        if self._depth_renderer is None:
            self._depth_renderer = self._mujoco.Renderer(
                self._model, height=self._face_size, width=self._face_size)
            self._depth_renderer.enable_depth_rendering()
        body_pos = np.array(self._data.xpos[self._body_id], dtype=np.float64)
        body_quat = np.array(self._data.xquat[self._body_id], dtype=np.float64)
        eye = body_pos + _quat_wxyz_to_rot(body_quat) @ np.array(
            self._offset, dtype=np.float64)
        rgb_faces = np.zeros(
            (6, self._face_size, self._face_size, 3), dtype=np.uint8)
        depth_faces = np.zeros(
            (6, self._face_size, self._face_size), dtype=np.float32)
        for face in range(6):
            self._cube_cam.lookat[:] = eye
            self._cube_cam.azimuth = self._FACE_AZIMUTH_DEG[face]
            self._cube_cam.elevation = self._FACE_ELEVATION_DEG[face]
            self._renderer.update_scene(self._data, camera=self._cube_cam)
            rgb_faces[face] = self._renderer.render()
            self._depth_renderer.update_scene(self._data, camera=self._cube_cam)
            depth_faces[face] = self._depth_renderer.render()
        rgb = _stitch_equirec(rgb_faces, self._face_idx, self._fx, self._fy)
        # Depth stitch: same LUT but NEAREST-pixel gather (the LUT's fx/fy are
        # fractional for the rgb bilinear sample; depth must not interpolate
        # across depth discontinuities). Round + clip, then clamp far-plane.
        fidx = self._face_idx.astype(np.intp)
        fy_i = np.clip(np.rint(self._fy).astype(np.intp), 0, self._face_size - 1)
        fx_i = np.clip(np.rint(self._fx).astype(np.intp), 0, self._face_size - 1)
        depth = depth_faces[fidx, fy_i, fx_i].astype(np.float32)
        depth[(depth <= 0.0) | (depth > 30.0)] = 0.0
        return rgb, depth

    def close(self) -> None:
        """Release the GL renderer(s) (if allocated)."""
        for attr in ("_renderer", "_depth_renderer"):
            r = getattr(self, attr, None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._cube_cam = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_renderer(self) -> None:
        # Set the global free-camera FoV to 90 deg so every cube face
        # gets a clean square FoV. Saved/restored to avoid clobbering
        # other consumers (best-effort — sim is the only consumer of
        # vis.global_.fovy in practice).
        try:
            self._saved_fovy = float(self._model.vis.global_.fovy)
            self._model.vis.global_.fovy = 90.0
        except Exception:
            self._saved_fovy = None

        self._renderer = self._mujoco.Renderer(
            self._model, height=self._face_size, width=self._face_size,
        )
        self._cube_cam = self._mujoco.MjvCamera()
        self._cube_cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        self._cube_cam.distance = 0.0    # camera AT lookat

    def _render_cube_faces(self) -> np.ndarray:
        """Render 6 faces, return ``(6, face_size, face_size, 3)`` uint8."""
        body_pos = np.array(self._data.xpos[self._body_id], dtype=np.float64)
        body_quat = np.array(self._data.xquat[self._body_id], dtype=np.float64)
        body_R = _quat_wxyz_to_rot(body_quat)
        eye = body_pos + body_R @ np.array(self._offset, dtype=np.float64)

        out = np.zeros(
            (6, self._face_size, self._face_size, 3), dtype=np.uint8
        )
        for face in range(6):
            self._cube_cam.lookat[:] = eye
            self._cube_cam.azimuth = self._FACE_AZIMUTH_DEG[face]
            self._cube_cam.elevation = self._FACE_ELEVATION_DEG[face]
            self._renderer.update_scene(self._data, camera=self._cube_cam)
            out[face] = self._renderer.render()
        return out

    @staticmethod
    def _mono(now: float | None) -> float:
        return float(now) if now is not None else time.monotonic()


# ---------------------------------------------------------------------------
# Pure helpers — testable in isolation
# ---------------------------------------------------------------------------


def _build_equirec_lut(
    out_w: int, out_h: int, face_size: int, vfov_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (face_idx, fx, fy) per output pixel.

    For each output ``(v, u)`` pixel, compute the spherical direction
    on the camera's unit sphere, find which cube face it samples, and
    record the within-face pixel coordinates.

    Output arrays have shape ``(out_h, out_w)``.

    Coordinate convention (matches the per-face camera setup):
        * forward (face 0): +X
        * right   (face 1): -Y     (camera azimuth -90 deg)
        * back    (face 2): -X
        * left    (face 3): +Y     (camera azimuth +90 deg)
        * up      (face 4): +Z
        * down    (face 5): -Z
    """
    # u in [0, 1) → azimuth phi in [-pi, pi)
    u_norm = (np.arange(out_w) + 0.5) / out_w
    phi = (u_norm - 0.5) * 2.0 * math.pi

    # v in [0, 1) → elevation theta in [+vfov/2, -vfov/2]
    v_norm = (np.arange(out_h) + 0.5) / out_h
    theta = (0.5 - v_norm) * vfov_rad

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x = np.cos(theta_grid) * np.cos(phi_grid)
    y = np.cos(theta_grid) * np.sin(phi_grid)
    z = np.sin(theta_grid)

    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    face_idx = np.zeros((out_h, out_w), dtype=np.int8)
    fx = np.zeros((out_h, out_w), dtype=np.float32)
    fy = np.zeros((out_h, out_w), dtype=np.float32)

    half = (face_size - 1) / 2.0

    for face in range(6):
        if face == _FACE_FORWARD:
            mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
            sx = -y[mask] / x[mask]
            sy = -z[mask] / x[mask]
        elif face == _FACE_RIGHT:
            mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y < 0)
            sx = x[mask] / -y[mask]
            sy = -z[mask] / -y[mask]
        elif face == _FACE_BACK:
            mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
            sx = y[mask] / -x[mask]
            sy = -z[mask] / -x[mask]
        elif face == _FACE_LEFT:
            mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
            sx = -x[mask] / y[mask]
            sy = -z[mask] / y[mask]
        elif face == _FACE_UP:
            mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
            sx = y[mask] / z[mask]
            sy = x[mask] / z[mask]
        else:           # _FACE_DOWN
            mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)
            sx = y[mask] / -z[mask]
            sy = -x[mask] / -z[mask]

        # Map (sx, sy) ∈ [-1, 1] → face pixel ∈ [0, face_size - 1]
        face_idx[mask] = face
        fx[mask] = (sx + 1.0) * half
        fy[mask] = (sy + 1.0) * half

    fx = np.clip(fx, 0, face_size - 1)
    fy = np.clip(fy, 0, face_size - 1)
    return face_idx, fx, fy


def _stitch_equirec(
    cube_imgs: np.ndarray,
    face_idx: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
) -> np.ndarray:
    """Combine 6 cube faces into one equirectangular image via the LUT.

    ``cube_imgs`` shape is ``(6, face_size, face_size, 3)`` uint8.
    """
    fx_int = fx.astype(np.int32)
    fy_int = fy.astype(np.int32)
    out = np.zeros((face_idx.shape[0], face_idx.shape[1], 3), dtype=np.uint8)
    for face in range(6):
        mask = face_idx == face
        if not np.any(mask):
            continue
        face_img = cube_imgs[face]
        out[mask] = face_img[fy_int[mask], fx_int[mask]]
    return out


def _quat_wxyz_to_rot(wxyz: np.ndarray) -> np.ndarray:
    """Convert ``(w, x, y, z)`` quaternion → 3x3 rotation matrix."""
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
