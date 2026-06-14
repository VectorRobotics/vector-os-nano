# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""BlenderBridge — repo-side client for the Blender Cycles/OptiX render server.

The repo venv (py3.12) NEVER imports ``bpy``: ``server.py`` runs under a
standalone Blender (GPL) as a subprocess — process isolation keeps Blender's
GPL out of the Apache repo, the same split #9 uses for habitat. This client
speaks one-JSON-object-per-line over a localhost socket; the server hands back
a base64 PNG that this side decodes to an ``(H,W,3)`` uint8 RGB array.

Resolve the Blender binary via ``VECTOR_BLENDER`` (falls back to the sandbox
spike build used in R2/R3). ``blender_available()`` lets tests skip cleanly on
a box with no Blender.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import socket
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vector_os_nano.playground.photoreal.camera import BlenderCameraPose

logger = logging.getLogger(__name__)

_DEFAULT_BLENDER = str(
    Path.home() / "sandbox" / "c10-substrate-spike"
    / "blender-4.5.10-linux-x64" / "blender"
)
_SERVER_PATH = Path(__file__).with_name("server.py")


def blender_bin() -> str:
    """Blender binary that runs the render server (env-overridable)."""
    return os.environ.get("VECTOR_BLENDER", _DEFAULT_BLENDER)


def blender_available() -> bool:
    """True when a Blender binary exists on this machine."""
    return os.path.exists(blender_bin())


class BlenderBridgeError(RuntimeError):
    """Subprocess spawn / handshake / protocol failure — always loud."""


# Render is the only slow op (OptiX warmup ~2.3 s first frame, then ~1 s); a
# generous per-op budget avoids a spurious timeout on the warmup frame.
_OP_TIMEOUT = {"render": 120.0}
_DEFAULT_OP_TIMEOUT = 30.0
_BOOT_TIMEOUT = 180.0   # Blender cold start + addon load can be slow


class BlenderBridge:
    """Spawn + talk to one Blender render server. One bridge per session."""

    def __init__(self, *, boot_timeout: float = _BOOT_TIMEOUT) -> None:
        self._boot_timeout = boot_timeout
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._wfile: Any = None
        self._port: int | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        binv = blender_bin()
        if not os.path.exists(binv):
            raise BlenderBridgeError(
                f"blender binary not found: {binv} — set VECTOR_BLENDER")
        if not _SERVER_PATH.exists():
            raise BlenderBridgeError(f"server script missing: {_SERVER_PATH}")
        self._proc = subprocess.Popen(
            [binv, "--background", "--python", str(_SERVER_PATH), "--", "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        port = self._read_port_handshake()
        self._connect(port)
        pong = self.request({"op": "ping"})
        if not pong.get("pong"):
            raise BlenderBridgeError(f"handshake ping failed: {pong}")

    def _connect(self, port: int) -> None:
        """Open the line-protocol socket to a server already listening on ``port``.

        Used by ``start()`` after the handshake, and directly by tests that run a
        fake server instead of a real Blender.
        """
        self._port = port
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=30.0)
        self._sock.settimeout(60.0)
        self._rfile = self._sock.makefile("r", encoding="utf-8")
        self._wfile = self._sock.makefile("w", encoding="utf-8")

    def _read_port_handshake(self) -> int:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.time() + self._boot_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise BlenderBridgeError(
                    f"blender server exited rc={self._proc.returncode} before handshake")
            line = self._proc.stdout.readline()
            if line.startswith("PORT "):
                return int(line.split()[1])
        raise BlenderBridgeError(
            f"blender server produced no PORT handshake within {self._boot_timeout}s")

    def close(self) -> None:
        """Graceful shutdown; idempotent; escalates to kill."""
        try:
            if self._wfile is not None:
                self.request({"op": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        for f in (self._rfile, self._wfile, self._sock):
            try:
                if f is not None:
                    f.close()
            except Exception:  # noqa: BLE001
                pass
        self._rfile = self._wfile = self._sock = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None

    def __enter__(self) -> "BlenderBridge":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- protocol ---------------------------------------------------------
    def request(self, payload: dict) -> dict:
        if self._wfile is None or self._sock is None:
            raise BlenderBridgeError("bridge not connected (call start())")
        timeout = _OP_TIMEOUT.get(payload.get("op", ""), _DEFAULT_OP_TIMEOUT)
        self._sock.settimeout(timeout)
        self._wfile.write(json.dumps(payload) + "\n")
        self._wfile.flush()
        line = self._rfile.readline()
        if not line:
            raise BlenderBridgeError("blender server closed the connection")
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BlenderBridgeError(f"torn protocol line: {exc}") from None
        if not resp.get("ok", False):
            raise BlenderBridgeError(
                f"server error for {payload.get('op')}: {resp.get('error', '<none>')}")
        return resp

    def ping(self) -> bool:
        return bool(self.request({"op": "ping"}).get("pong"))

    def render(
        self,
        pose: BlenderCameraPose,
        scene: dict,
        *,
        width: int = 640,
        height: int = 480,
        samples: int = 32,
    ) -> np.ndarray:
        """Render one frame from ``pose`` of ``scene`` and return ``(H,W,3)`` uint8.

        ``scene`` is a JSON-able spec (floor + assets + lighting) the server
        builds; the camera pose is exact so the frame is shot from MuJoCo's own
        camera (no drift — see ``camera.py``).
        """
        resp = self.request({
            "op": "render",
            "matrix_world": pose.flat16(),
            "fovy_rad": float(pose.fovy_rad),
            "width": int(width),
            "height": int(height),
            "samples": int(samples),
            "scene": scene,
        })
        png = base64.b64decode(resp["png_b64"])
        img = Image.open(BytesIO(png)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
