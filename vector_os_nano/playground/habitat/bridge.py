# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""HabitatBridge — repo-side client for the habitat conda subprocess (ADR-009).

The repo venv (py3.12) never imports habitat: ``server.py`` runs under the
pinned conda interpreter (``VECTOR_HABITAT_PYTHON``, default the
``habitat-spike`` env) as a subprocess — the proven go2-sim process split —
and this client speaks one-JSON-object-per-line over a localhost socket.
The subprocess inherits ``VECTOR_RUN_ID``, so the W2.2 watchdog can sweep it
after an abnormal exit.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_HABITAT_PYTHON = str(
    Path.home() / "miniconda3" / "envs" / "habitat-spike" / "bin" / "python"
)
_SERVER_PATH = Path(__file__).with_name("server.py")


def habitat_python() -> str:
    """Interpreter that runs the habitat server (env-overridable)."""
    return os.environ.get("VECTOR_HABITAT_PYTHON", _DEFAULT_HABITAT_PYTHON)


def habitat_available() -> bool:
    """True when the pinned habitat interpreter exists on this machine."""
    return os.path.exists(habitat_python())


class HabitatBridgeError(RuntimeError):
    """Subprocess spawn/handshake/protocol failure — always loud."""


class HabitatBridge:
    """Spawn + talk to one habitat server. One bridge per scene/run."""

    def __init__(self, scene: str, *, boot_timeout: float = 120.0) -> None:
        self._scene = scene
        self._boot_timeout = boot_timeout
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._wfile: Any = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        py = habitat_python()
        if not os.path.exists(py):
            raise HabitatBridgeError(
                f"habitat interpreter not found: {py} — set VECTOR_HABITAT_PYTHON "
                f"or create the pinned conda env (ADR-009)"
            )
        if not _SERVER_PATH.exists():
            raise HabitatBridgeError(f"server script missing: {_SERVER_PATH}")
        self._proc = subprocess.Popen(
            [py, "-u", str(_SERVER_PATH), "--scene", self._scene],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        port = self._read_port_handshake()
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=30.0)
        self._sock.settimeout(60.0)
        self._rfile = self._sock.makefile("r", encoding="utf-8")
        self._wfile = self._sock.makefile("w", encoding="utf-8")
        pong = self.request({"op": "ping"})
        if not pong.get("pong"):
            raise HabitatBridgeError(f"handshake ping failed: {pong}")

    def _read_port_handshake(self) -> int:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.time() + self._boot_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise HabitatBridgeError(
                    f"habitat server exited rc={self._proc.returncode} before handshake"
                )
            line = self._proc.stdout.readline()
            if line.startswith("PORT "):
                return int(line.split()[1])
        raise HabitatBridgeError(
            f"habitat server produced no PORT handshake within {self._boot_timeout}s"
        )

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

    # -- protocol ---------------------------------------------------------
    def request(self, payload: dict) -> dict:
        if self._wfile is None or self._rfile is None:
            raise HabitatBridgeError("bridge not connected (call start())")
        self._wfile.write(json.dumps(payload) + "\n")
        self._wfile.flush()
        line = self._rfile.readline()
        if not line:
            raise HabitatBridgeError("habitat server closed the connection")
        resp = json.loads(line)
        if not resp.get("ok", False):
            raise HabitatBridgeError(f"server error for {payload.get('op')}: "
                                     f"{resp.get('error', '<none>')}")
        return resp
