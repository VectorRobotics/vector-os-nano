"""Tests for web dashboard."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from fastapi.testclient import TestClient

from vector_os_nano.core.agent import Agent
from vector_os_nano.core.config import load_config
from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
from vector_os_nano.web.app import create_app
from vector_os_nano.web.chat import ChatManager


@pytest.fixture
def app():
    """Create test app with sim arm."""
    arm = MuJoCoArm(gui=False)
    arm.connect()
    gripper = MuJoCoGripper(arm)
    cfg = load_config(None)
    agent = Agent(arm=arm, gripper=gripper, config=cfg)
    app = create_app(agent, cfg)
    yield app
    arm.disconnect()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHTTPEndpoints:
    def test_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "VECTOR OS NANO" in resp.text

    def test_api_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "mode" in data
        assert "arm" in data
        assert "objects" in data
        assert data["mode"] == "sim"

    def test_api_status_has_objects(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        assert len(data["objects"]) >= 6

    def test_api_status_has_joints(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        assert len(data["arm"]["joints"]) == 5

    def test_api_history_empty(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        assert resp.json() == []


class TestWebSocket:
    def test_chat_connect(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            # Should receive welcome message
            data = ws.receive_json()
            assert data["type"] == "response"
            assert "Welcome" in data["content"]

    def test_status_connect(self, client):
        with client.websocket_connect("/ws/status") as ws:
            # Status WS accepts connection
            pass  # connection successful = pass


class TestChatManager:
    def test_is_command_pick(self):
        cm = ChatManager.__new__(ChatManager)
        cm._history = []
        assert cm.is_command("抓杯子") is True
        assert cm.is_command("pick the mug") is True
        assert cm.is_command("home") is True
        assert cm.is_command("open") is True

    def test_is_command_chat(self):
        cm = ChatManager.__new__(ChatManager)
        cm._history = []
        assert cm.is_command("hello") is False
        assert cm.is_command("what time is it") is False
        assert cm.is_command("你好") is False

    def test_history_trim(self):
        cm = ChatManager.__new__(ChatManager)
        cm._max_history = 5
        cm._history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        cm._trim_history()
        assert len(cm._history) == 5


class TestFrontend:
    def test_html_has_chat_panel(self, client):
        resp = client.get("/")
        assert "chat-messages" in resp.text
        assert "chat-input" in resp.text

    def test_html_has_status_panel(self, client):
        resp = client.get("/")
        assert "joint-list" in resp.text
        assert "object-list" in resp.text

    def test_html_has_websocket_js(self, client):
        resp = client.get("/")
        assert "ws/chat" in resp.text
        assert "ws/status" in resp.text

    def test_html_has_dark_theme(self, client):
        resp = client.get("/")
        assert "#08080d" in resp.text or "#0a0a0f" in resp.text
        assert "#00b4b4" in resp.text
