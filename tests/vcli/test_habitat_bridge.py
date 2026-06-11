# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M2 — habitat bridge/base protocol tests (NO habitat dependency).

A pure-python FakeHabitatServer speaks the same one-JSON-per-line protocol
on a flat infinite floor (``try_step`` = identity), so the bridge framing,
the kinematic walk semantics, and HabitatBase's provider conformance are
all pinned in the py3.12 venv without the conda env. The real-sim e2e lives
in test_habitat_e2e.py (skipped where the pinned interpreter is absent).
"""
from __future__ import annotations

import json
import math
import socket
import threading

import pytest

from vector_os_nano.playground.habitat.base import HabitatBase
from vector_os_nano.playground.habitat.bridge import HabitatBridge, HabitatBridgeError
from vector_os_nano.vcli.providers import BaseMotionProvider, BaseStateProvider


class FakeHabitatServer(threading.Thread):
    """Flat-floor kinematic re-implementation of server.py's contract."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self.pos = [0.0, 0.0, 0.0]
        self.heading = 0.0

    def run(self) -> None:  # pragma: no cover - thread body
        conn, _ = self._sock.accept()
        rf, wf = conn.makefile("r"), conn.makefile("w")
        for line in rf:
            req = json.loads(line)
            resp = self.handle(req)
            wf.write(json.dumps(resp) + "\n")
            wf.flush()
            if req.get("op") == "shutdown":
                break
        conn.close()
        self._sock.close()

    def _state(self) -> dict:
        return {"ok": True, "pos": list(self.pos), "heading": self.heading}

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "pong": True}
        if op in ("get_state", "stop"):
            return self._state()
        if op == "walk":
            vx, vyaw, dur = req["vx"], req["vyaw"], req["duration"]
            steps = max(1, int(round(dur / 0.1)))
            dt = dur / steps
            for _ in range(steps):
                self.heading += vyaw * dt
                self.pos[0] += vx * dt * math.cos(self.heading)
                self.pos[1] += vx * dt * math.sin(self.heading)
            return self._state()
        if op == "geodesic_distance":
            a, b = req["a"], req["b"]
            return {"ok": True, "distance": math.dist(a[:2], b[:2])}
        if op == "snap_point":
            return {"ok": True, "point": list(req["p"])}
        if op == "objects":
            return {"ok": True, "objects": [{"name": "sofa_1", "category": "sofa",
                                             "pos": [1.0, 2.0, 0.4]}]}
        if op == "navigate_to":
            # flat floor: teleport-walk straight to the goal, always reached
            self.pos[0], self.pos[1] = req["x"], req["y"]
            return {"reached": True, "remaining": 0.0, **self._state()}
        if op == "render":
            import base64, io, struct, zlib
            # minimal valid 1x1 PNG (no PIL dep in the py3.12 venv)
            sig = b"\x89PNG\r\n\x1a\n"
            def chunk(t, d):
                return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
            ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
            png = sig + ihdr + idat + chunk(b"IEND", b"")
            return {"ok": True, "png_base64": base64.b64encode(png).decode(),
                    "width": 1, "height": 1}
        if op == "shutdown":
            return {"ok": True, "bye": True}
        return {"ok": False, "error": f"unknown op {op!r}"}


@pytest.fixture()
def fake_base(monkeypatch):
    server = FakeHabitatServer()
    server.start()

    def _fake_start(bridge_self: HabitatBridge) -> None:
        bridge_self._sock = socket.create_connection(("127.0.0.1", server.port))
        bridge_self._rfile = bridge_self._sock.makefile("r", encoding="utf-8")
        bridge_self._wfile = bridge_self._sock.makefile("w", encoding="utf-8")
        bridge_self.request({"op": "ping"})

    monkeypatch.setattr(HabitatBridge, "start", _fake_start)
    base = HabitatBase(scene="fake.glb")
    base.connect()
    yield base, server
    base.disconnect()


class TestProviderConformance:
    def test_base_satisfies_narrow_specs(self, fake_base) -> None:
        base, _ = fake_base
        assert isinstance(base, BaseStateProvider)
        assert isinstance(base, BaseMotionProvider)

    def test_capability_flags(self, fake_base) -> None:
        base, _ = fake_base
        assert base.supports_holonomic is False
        assert base.supports_lidar is False


class TestKinematics:
    def test_walk_forward_moves_along_heading(self, fake_base) -> None:
        base, server = fake_base
        assert base.walk(vx=0.5, duration=2.0) is True
        pos = base.get_position()
        assert pos[0] == pytest.approx(1.0, abs=1e-6)
        assert pos[1] == pytest.approx(0.0, abs=1e-6)

    def test_turn_then_walk(self, fake_base) -> None:
        base, _ = fake_base
        base.walk(vyaw=math.pi / 2, duration=1.0)  # face +y
        assert base.get_heading() == pytest.approx(math.pi / 2, abs=1e-6)
        base.walk(vx=1.0, duration=1.0)
        pos = base.get_position()
        assert pos[1] == pytest.approx(1.0, abs=1e-3)

    def test_vy_is_ignored_not_faked(self, fake_base) -> None:
        base, _ = fake_base
        base.walk(vy=1.0, duration=1.0)
        assert base.get_position()[1] == pytest.approx(0.0, abs=1e-6)

    def test_odometry_quaternion_matches_heading(self, fake_base) -> None:
        base, _ = fake_base
        base.walk(vyaw=math.pi / 2, duration=1.0)
        odo = base.get_odometry()
        yaw = 2.0 * math.atan2(odo.qz, odo.qw)
        assert yaw == pytest.approx(math.pi / 2, abs=1e-6)


class TestOraclePassthroughs:
    def test_geodesic_and_snap_and_objects(self, fake_base) -> None:
        base, _ = fake_base
        assert base.geodesic_distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)
        assert base.snap_point([1, 2, 3]) == [1.0, 2.0, 3.0]
        objs = base.get_semantic_objects()
        assert objs and objs[0]["category"] == "sofa"


class TestFailureModes:
    def test_unknown_op_is_loud(self, fake_base) -> None:
        base, _ = fake_base
        with pytest.raises(HabitatBridgeError, match="unknown op"):
            base._bridge.request({"op": "fly"})

    def test_unconnected_base_is_loud(self) -> None:
        base = HabitatBase(scene="fake.glb")
        with pytest.raises(RuntimeError, match="not connected"):
            base.get_position()

    def test_spawn_failure_names_the_interpreter(self, monkeypatch) -> None:
        monkeypatch.setenv("VECTOR_HABITAT_PYTHON", "/nonexistent/python")
        bridge = HabitatBridge(scene="x.glb")
        with pytest.raises(HabitatBridgeError, match="VECTOR_HABITAT_PYTHON"):
            bridge.start()


class TestM3Ops:
    def test_navigate_to_reaches_goal(self, fake_base) -> None:
        base, _ = fake_base
        out = base.navigate_to(3.0, 4.0)
        assert out["reached"] is True
        assert base.get_position()[:2] == [3.0, 4.0]

    def test_render_returns_png_bytes(self, fake_base) -> None:
        base, _ = fake_base
        png = base.render_rgb_png()
        assert png.startswith(b"\x89PNG")
