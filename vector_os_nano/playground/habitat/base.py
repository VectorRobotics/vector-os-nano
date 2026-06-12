# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""HabitatBase — kinematic BaseProtocol over the habitat bridge (M2).

Satisfies the kernel's narrow provider specs (``BaseStateProvider`` /
``BaseMotionProvider``, W3.3) and the full ``hardware/base.py`` HAL surface,
so every existing base predicate / locomotion primitive / skill works
against the photoreal world unchanged. Locomotion fidelity deliberately
lives in MuJoCo; this base is navmesh-kinematic (the VLN convention):

- ``walk`` blocks while the SERVER integrates vx/vyaw with per-step
  ``pathfinder.try_step`` (slides along walls, never leaves the mesh).
- ``vy`` is unsupported (``supports_holonomic`` is False); a non-zero vy is
  ignored with a log line rather than faked.
- ``set_velocity`` is a NON-BLOCKING streaming command (N1): the server's
  50 Hz integration thread applies it until superseded or its 0.6 s
  deadman stops the robot. State reads and velocity writes go over the
  bridge's STREAM channel, so they never queue behind a pano render.
- Odometry is exact ground truth from the simulator state.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from vector_os_nano.core.types import Odometry
from vector_os_nano.playground.habitat.bridge import HabitatBridge

logger = logging.getLogger(__name__)


class HabitatBase:
    """Kinematic mobile base in a photoreal habitat scene."""

    def __init__(
        self,
        scene: str,
        gui: bool = False,
        dataset_config: str = "",
        navmesh: str = "",
        robot_glb: str = "",
        viewer_mode: str = "",
        viewer_size: int = 0,
    ) -> None:
        self._bridge = HabitatBridge(
            scene, gui=gui, dataset_config=dataset_config, navmesh=navmesh,
            robot_glb=robot_glb, viewer_mode=viewer_mode, viewer_size=viewer_size,
        )
        self._connected = False
        self._warned_vy = False

    @property
    def name(self) -> str:
        return "habitat_kinematic"

    # -- lifecycle -------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        self._bridge.start()
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._bridge.close()
        self._connected = False

    def stop(self) -> None:
        # Kinematic base: nothing is streaming; stop must never raise.
        try:
            if self._connected:
                self._bridge.request({"op": "stop"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("HabitatBase.stop: %s", exc)

    # -- motion -----------------------------------------------------------
    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        self._require_connected()
        if abs(vy) > 1e-6:
            logger.info("HabitatBase.walk: vy=%.3f unsupported (kinematic), ignored", vy)
        try:
            self._bridge.request(
                {"op": "walk", "vx": vx, "vyaw": vyaw, "duration": duration}
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("HabitatBase.walk failed: %s", exc)
            return False

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Non-blocking streaming command (N1): the server's integration
        thread applies it until superseded; its deadman stops a stale
        stream. Safe at nav-stack rates — never blocks on a render."""
        self._require_connected()
        if abs(vy) > 1e-6 and not self._warned_vy:
            logger.info(
                "HabitatBase.set_velocity: vy unsupported (kinematic), ignored"
            )
            self._warned_vy = True
        try:
            self._bridge.stream_request(
                {"op": "set_velocity", "vx": vx, "vyaw": vyaw}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HabitatBase.set_velocity failed: %s", exc)

    def set_markers(self, markers: "list[dict]") -> None:
        """Labelled world-frame markers overlaid on the viewer window
        (SysNav detections made visible). Replaces the whole set; rides the
        STREAM channel so a ROS callback thread never queues behind a paced
        walk or a pano render. Best-effort — display must never raise."""
        self._require_connected()
        try:
            self._bridge.stream_request({"op": "set_markers", "markers": markers})
        except Exception as exc:  # noqa: BLE001
            logger.warning("HabitatBase.set_markers failed: %s", exc)

    # -- state ------------------------------------------------------------
    def get_position(self) -> list[float]:
        st = self._state()
        return [float(v) for v in st["pos"]]

    def get_heading(self) -> float:
        return float(self._state()["heading"])

    def get_velocity(self) -> list[float]:
        vel = self._state().get("vel")
        return [float(v) for v in vel] if vel else [0.0, 0.0, 0.0]

    def get_state(self) -> dict:
        """One-roundtrip pose+velocity snapshot (stream channel): the 50 Hz
        odom publisher's read — pos / heading / vel in one request."""
        return self._state()

    def get_odometry(self) -> Odometry:
        import math

        st = self._state()
        x, y, z = (float(v) for v in st["pos"])
        yaw = float(st["heading"])
        return Odometry(
            timestamp=time.time(),
            x=x, y=y, z=z,
            qz=math.sin(yaw / 2.0), qw=math.cos(yaw / 2.0),
        )

    def get_lidar_scan(self) -> Any:
        return None  # no lidar in this world (SysNav inputs come from depth, M4)

    # -- capability flags ---------------------------------------------------
    @property
    def supports_holonomic(self) -> bool:
        return False

    @property
    def supports_lidar(self) -> bool:
        return False

    # -- navigation + perception (M3) ---------------------------------------
    def navigate_to(self, x: float, y: float, tol: float = 0.2) -> dict:
        """Shortest-path navigate to world (x, y). Blocking; returns the
        server's outcome dict: reached / remaining geodesic / final state.
        Honest failure: an unreachable or wall-stuck goal returns
        reached=False — the verify predicate is the judge, never this flag."""
        self._require_connected()
        return self._bridge.request(
            {"op": "navigate_to", "x": x, "y": y, "tol": tol}
        )

    def render_rgb_png(self) -> bytes:
        """Egocentric 256x256 RGB as PNG bytes (the VLM perception input)."""
        import base64

        self._require_connected()
        resp = self._bridge.request({"op": "render"})
        return base64.b64decode(resp["png_base64"])

    def viewer_frame_png(self) -> bytes:
        """Viewer-camera frame as PNG bytes (N3: the third-person view that
        shows the robot body; headless acceptance + owner artifacts)."""
        import base64

        self._require_connected()
        resp = self._bridge.request({"op": "viewer_frame"})
        if "png_base64" not in resp:
            raise RuntimeError(resp.get("error", "viewer frame unavailable"))
        return base64.b64decode(resp["png_base64"])

    def get_pano(self) -> dict:
        """Pose-synced equirect panorama (M4): rgb uint8 HxWx3, depth f32 HxW,
        plus the world pos/heading captured in the SAME frame."""
        import base64

        import numpy as np

        self._require_connected()
        r = self._bridge.request({"op": "pano"})
        h, w = int(r["height"]), int(r["width"])
        return {
            "rgb": np.frombuffer(
                base64.b64decode(r["rgb_b64"]), dtype=np.uint8
            ).reshape(h, w, 3),
            "depth": np.frombuffer(
                base64.b64decode(r["depth_b64"]), dtype=np.float32
            ).reshape(h, w),
            "pos": [float(v) for v in r["pos"]],
            "heading": float(r["heading"]),
        }

    # -- oracle passthroughs (verify predicates, M2 part 2) -----------------
    def geodesic_distance(self, a: list[float], b: list[float]) -> float:
        self._require_connected()
        return float(
            self._bridge.request({"op": "geodesic_distance", "a": a, "b": b})["distance"]
        )

    def snap_point(self, p: list[float]) -> list[float]:
        self._require_connected()
        return [float(v) for v in self._bridge.request({"op": "snap_point", "p": p})["point"]]

    def get_semantic_objects(self) -> list[dict]:
        self._require_connected()
        return list(self._bridge.request({"op": "objects"})["objects"])

    # -- internals ----------------------------------------------------------
    def _state(self) -> dict:
        self._require_connected()
        return self._bridge.stream_request({"op": "get_state"})

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("HabitatBase not connected (call connect())")
