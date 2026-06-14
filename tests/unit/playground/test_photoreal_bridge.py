# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""BlenderBridge client protocol — tested WITHOUT Blender via a fake socket
server, and (when a real Blender is present) one true end-to-end frame.

The fake server speaks the same one-JSON-per-line wire protocol the real
``server.py`` does, so the client's request/response + base64-PNG decode path is
exercised deterministically on any box (CI has no Blender). The Blender-gated
test proves the spawn/handshake/render plumbing against a real OptiX render and
skips cleanly where Blender is absent.
"""
from __future__ import annotations

import base64
import json
import socket
import threading
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vector_os_nano.playground.photoreal.bridge import (
    BlenderBridge,
    BlenderBridgeError,
    blender_available,
)
from vector_os_nano.playground.photoreal.camera import mujoco_camera_to_blender


def _solid_png_b64(w: int, h: int, rgb) -> str:
    img = Image.new("RGB", (w, h), tuple(rgb))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeServer:
    """A localhost socket server echoing the photoreal wire protocol."""

    def __init__(self, render_color=(10, 200, 30), render_size=(64, 48)) -> None:
        self._color = render_color
        self._w, self._h = render_size
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self.seen: list[dict] = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        conn, _ = self._srv.accept()
        rfile = conn.makefile("r", encoding="utf-8")
        wfile = conn.makefile("w", encoding="utf-8")
        for line in rfile:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            self.seen.append(req)
            op = req.get("op")
            if op == "ping":
                resp = {"ok": True, "pong": True}
            elif op == "render":
                resp = {"ok": True,
                        "png_b64": _solid_png_b64(self._w, self._h, self._color),
                        "width": self._w, "height": self._h}
            elif op == "shutdown":
                wfile.write(json.dumps({"ok": True}) + "\n")
                wfile.flush()
                break
            elif op == "boom":
                resp = {"ok": False, "error": "kaboom"}
            else:
                resp = {"ok": False, "error": f"unknown op {op}"}
            wfile.write(json.dumps(resp) + "\n")
            wfile.flush()
        conn.close()
        self._srv.close()


@pytest.fixture
def fake_bridge():
    srv = _FakeServer()
    bridge = BlenderBridge()
    bridge._connect(srv.port)
    try:
        yield bridge, srv
    finally:
        try:
            bridge.close()
        except Exception:  # noqa: BLE001
            pass


def test_ping_round_trips(fake_bridge):
    bridge, _ = fake_bridge
    assert bridge.ping() is True


def test_render_decodes_png_to_rgb_array(fake_bridge):
    bridge, srv = fake_bridge
    pose = mujoco_camera_to_blender([0, 0, 1.0], np.eye(3).reshape(-1), 60.0)
    frame = bridge.render(pose, {"floor": {}}, width=64, height=48, samples=8)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    # the fake server's solid colour survives the encode->b64->decode round-trip
    np.testing.assert_array_equal(frame[0, 0], [10, 200, 30])


def test_render_forwards_exact_pose_and_params(fake_bridge):
    bridge, srv = fake_bridge
    pose = mujoco_camera_to_blender([1.5, -0.5, 0.85],
                                    np.array([0, -1, 0, 1, 0, 0, 0, 0, 1.0]), 45.0)
    bridge.render(pose, {"assets": []}, width=320, height=240, samples=16)
    req = next(r for r in srv.seen if r["op"] == "render")
    assert req["matrix_world"] == pose.flat16()      # no drift over the wire
    assert req["width"] == 320 and req["height"] == 240 and req["samples"] == 16
    assert abs(req["fovy_rad"] - pose.fovy_rad) < 1e-12


def test_server_error_is_loud(fake_bridge):
    bridge, _ = fake_bridge
    with pytest.raises(BlenderBridgeError, match="kaboom"):
        bridge.request({"op": "boom"})


def test_request_before_connect_raises():
    bridge = BlenderBridge()
    with pytest.raises(BlenderBridgeError, match="not connected"):
        bridge.request({"op": "ping"})


@pytest.mark.skipif(not blender_available(),
                    reason="no Blender binary (set VECTOR_BLENDER)")
def test_end_to_end_real_blender_renders_a_frame():
    """Spawn a real Blender, render one OptiX frame of a floor — non-empty RGB."""
    bridge = BlenderBridge()
    bridge.start()
    try:
        assert bridge.ping() is True
        pose = mujoco_camera_to_blender(
            [0.0, -3.0, 1.2], np.array([1, 0, 0, 0, 0, 1, 0, -1, 0.0]), 50.0)
        scene = {"floor": {"size": 12, "color": [0.8, 0.5, 0.3]},
                 "sun": {"pos": [2, 2, 4], "energy": 3.0}}
        frame = bridge.render(pose, scene, width=160, height=120, samples=16)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8
        assert float(frame.mean()) > 1.0     # not an all-black frame
    finally:
        bridge.close()
