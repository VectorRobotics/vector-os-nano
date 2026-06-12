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
import threading
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


# #15 — per-op read-timeout tiers (seconds). The old flat 60 s socket timeout
# vs unbounded paced motion ("walk 20 m" = 67 s) raised a BARE socket.timeout
# and permanently offset the line protocol by one response.
_OP_TIMEOUT_TIERS: "dict[str, float]" = {
    "navigate_to": 300.0,     # paced cross-apartment path, generous bound
    "pano": 120.0,            # equirect pair render
    "render": 120.0,
    "viewer_frame": 120.0,
}
_DEFAULT_OP_TIMEOUT = 60.0
_WALK_TIMEOUT_MARGIN = 30.0


class _BufferedLineReader:
    """Line reader over a raw socket that SURVIVES read timeouts.

    A ``makefile()`` reader refuses any further read after one timeout
    ("cannot read from timed out object") and may tear a half-received line.
    Here partial data stays buffered across calls, so the late response a
    timeout abandoned is read INTACT by the next request and discarded by
    rid (#15 resync path).
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def readline(self, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                line = self._buf[: i + 1].decode("utf-8", "replace")
                del self._buf[: i + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout(f"no full line within {timeout:.1f}s")
            self._sock.settimeout(remaining)
            chunk = self._sock.recv(65536)
            if not chunk:
                return ""  # EOF
            self._buf.extend(chunk)


class _StreamChannel:
    """A second connection to the server, served by a dedicated server-side
    thread with a restricted op set ({ping, get_state, set_velocity}) that
    never touches the simulator — 50 Hz state polls and cmd_vel writes never
    queue behind a multi-hundred-ms pano render on the main channel (N1)."""

    def __init__(self, port: int) -> None:
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
        self._sock.settimeout(10.0)
        self._rfile = self._sock.makefile("r", encoding="utf-8")
        self._wfile = self._sock.makefile("w", encoding="utf-8")
        self._lock = threading.Lock()

    def request(self, payload: dict) -> dict:
        with self._lock:
            self._wfile.write(json.dumps(payload) + "\n")
            self._wfile.flush()
            line = self._rfile.readline()
        if not line:
            raise HabitatBridgeError("habitat server closed the stream channel")
        resp = json.loads(line)
        if not resp.get("ok", False):
            raise HabitatBridgeError(
                f"server error for {payload.get('op')}: {resp.get('error', '<none>')}"
            )
        return resp

    def close(self) -> None:
        for f in (self._rfile, self._wfile, self._sock):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass


class HabitatBridge:
    """Spawn + talk to one habitat server. One bridge per scene/run."""

    def __init__(
        self,
        scene: str,
        *,
        boot_timeout: float = 120.0,
        gui: bool = False,
        dataset_config: str = "",
        navmesh: str = "",
        robot_glb: str = "",
        viewer_mode: str = "",
        viewer_size: int = 0,
    ) -> None:
        self._scene = scene
        self._boot_timeout = boot_timeout
        self._gui = gui
        self._dataset_config = dataset_config
        self._navmesh = navmesh
        self._robot_glb = robot_glb
        self._viewer_mode = viewer_mode
        self._viewer_size = int(viewer_size)
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._wfile: Any = None
        self._port: int | None = None
        self._stream: _StreamChannel | None = None
        # One in-flight request at a time: the M5 demo drives navigation and
        # the pano timer from different threads over this single socket —
        # interleaved writes would corrupt the line protocol.
        self._lock = threading.Lock()
        # #15 — request/response pairing. Monotonic rid injected per request
        # and echoed by the server; stale lines (a late answer to a request
        # that timed out) are discarded by rid. ``_dead`` flips on protocol
        # corruption (torn JSON / rid from the future): a main-channel
        # reconnect is impossible by design (closing it shuts the server
        # down), so corruption is terminal and LOUD — restart the world.
        self._rid = 0
        self._dead = False
        self._dead_reason = ""   # cause shown by the dead gate (R5: EOF ≠ desync)
        self._reader: _BufferedLineReader | None = None  # lazy, over _sock

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
            self._server_argv(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        port = self._read_port_handshake()
        self._port = port
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=30.0)
        self._sock.settimeout(60.0)
        self._rfile = self._sock.makefile("r", encoding="utf-8")
        self._wfile = self._sock.makefile("w", encoding="utf-8")
        pong = self.request({"op": "ping"})
        if not pong.get("pong"):
            raise HabitatBridgeError(f"handshake ping failed: {pong}")

    def _server_argv(self) -> "list[str]":
        argv = [habitat_python(), "-u", str(_SERVER_PATH), "--scene", self._scene]
        if self._gui:
            argv.append("--gui")  # live first-person viewer (server-side window)
        if self._dataset_config:
            argv += ["--dataset-config", self._dataset_config]
        if self._navmesh:
            argv += ["--navmesh", self._navmesh]
        if self._robot_glb:
            argv += ["--robot-glb", self._robot_glb]
        if self._viewer_mode:
            argv += ["--viewer-mode", self._viewer_mode]
        if self._viewer_size > 0:
            argv += ["--viewer-size", str(self._viewer_size)]
        return argv

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
        if self._stream is not None:
            self._stream.close()
            self._stream = None
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
        self._reader = None
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
    def _op_timeout(self, payload: dict) -> float:
        """Estimated read timeout for this op (#15) — never a flat 60 s."""
        op = payload.get("op", "")
        if op == "walk":
            return float(payload.get("duration", 1.0)) + _WALK_TIMEOUT_MARGIN
        return _OP_TIMEOUT_TIERS.get(op, _DEFAULT_OP_TIMEOUT)

    def request(self, payload: dict) -> dict:
        with self._lock:
            if self._dead:
                raise HabitatBridgeError(
                    f"bridge dead ({self._dead_reason or 'protocol failure'}) "
                    "— restart the habitat world (stop_simulation / "
                    "start_simulation)")
            if self._wfile is None or self._sock is None:
                raise HabitatBridgeError("bridge not connected (call start())")
            if self._reader is None:
                self._reader = _BufferedLineReader(self._sock)
            self._rid += 1
            rid = self._rid
            timeout = self._op_timeout(payload)
            self._wfile.write(json.dumps(dict(payload, rid=rid)) + "\n")
            self._wfile.flush()
            resp = self._read_matching(rid, timeout, payload.get("op", ""))
        if not resp.get("ok", False):
            raise HabitatBridgeError(f"server error for {payload.get('op')}: "
                                     f"{resp.get('error', '<none>')}")
        return resp

    def _read_matching(self, rid: int, timeout: float, op: str) -> dict:
        """Read lines until the response paired to ``rid`` arrives.

        - stale rid (< ours): a late answer to a timed-out request — discard,
          keep reading (the resync path; the protocol is line-aligned again).
        - no rid in the response: legacy/fake server — accept as before.
        - rid from the future / torn JSON: protocol corruption → bridge dead.
        - read timeout: HabitatBridgeError naming the op and budget; the
          socket stays open — the late line is discarded by rid next call.
        Caller holds ``self._lock``.
        """
        while True:
            try:
                line = self._reader.readline(timeout)
            except socket.timeout:
                raise HabitatBridgeError(
                    f"read timeout ({timeout:.0f}s) for op {op!r} (rid {rid}) "
                    f"— a late response will be discarded by rid") from None
            if not line:
                self._dead = True
                self._dead_reason = "server closed the connection"
                raise HabitatBridgeError("habitat server closed the connection")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                self._dead = True
                self._dead_reason = "torn protocol line"
                raise HabitatBridgeError(
                    f"protocol corruption (torn line) for op {op!r}: {exc} "
                    f"— bridge dead, restart the world") from None
            got = resp.get("rid")
            if got is None or got == rid:
                return resp
            if isinstance(got, int) and got < rid:
                logger.warning(
                    "habitat bridge: discarding stale response rid=%s "
                    "(expecting %s)", got, rid)
                continue
            self._dead = True
            self._dead_reason = "rid desync"
            raise HabitatBridgeError(
                f"protocol desync for op {op!r}: got rid {got!r}, expected "
                f"{rid} — bridge dead, restart the world")

    def stream_request(self, payload: dict) -> dict:
        """Request on the STREAM channel (lazy-opened second connection).

        Falls back to the main channel when no port is known — fake-server
        tests bypass ``start()`` and old single-accept fakes stay valid.
        """
        if self._stream is None:
            if self._port is None:
                return self.request(payload)
            self._stream = _StreamChannel(self._port)
        return self._stream.request(payload)
