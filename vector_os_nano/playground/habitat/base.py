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
- ``set_velocity`` applies ONE 0.1 s kinematic step per call (documented
  v1 limitation — there is no physics thread; streaming nav-stack control
  is out of M2 scope).
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

    def __init__(self, scene: str) -> None:
        self._bridge = HabitatBridge(scene)
        self._connected = False

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
        # v1: one instantaneous 0.1 s kinematic step per call (no physics loop).
        self._require_connected()
        try:
            self._bridge.request(
                {"op": "walk", "vx": vx, "vyaw": vyaw, "duration": 0.1}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HabitatBase.set_velocity failed: %s", exc)

    # -- state ------------------------------------------------------------
    def get_position(self) -> list[float]:
        st = self._state()
        return [float(v) for v in st["pos"]]

    def get_heading(self) -> float:
        return float(self._state()["heading"])

    def get_velocity(self) -> list[float]:
        return [0.0, 0.0, 0.0]  # kinematic: no persistent velocity state

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
        return self._bridge.request({"op": "get_state"})

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("HabitatBase not connected (call connect())")
