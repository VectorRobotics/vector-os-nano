"""Tests for GazeboGo2Proxy — Go2 control via Gazebo Harmonic + ROS2.

Level: Gazebo-L1
All tests mock ROS2 and subprocess — no external dependencies required.
"""
from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proxy() -> Any:
    """Create GazeboGo2Proxy with all external dependencies mocked."""
    from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy
    return GazeboGo2Proxy()


# ---------------------------------------------------------------------------
# 1. Inheritance and identity
# ---------------------------------------------------------------------------


class TestGazeboGo2ProxyIdentity:
    """GazeboGo2Proxy must inherit from Go2ROS2Proxy and expose correct identity."""

    def test_inherits_go2ros2proxy(self) -> None:
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy
        assert issubclass(GazeboGo2Proxy, Go2ROS2Proxy)

    def test_name_returns_gazebo_go2(self) -> None:
        proxy = _make_proxy()
        assert proxy.name == "gazebo_go2"

    def test_supports_lidar_true(self) -> None:
        proxy = _make_proxy()
        assert proxy.supports_lidar is True

    def test_node_name_is_gazebo_go2_proxy(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy
        assert GazeboGo2Proxy._NODE_NAME == "gazebo_go2_proxy"

    def test_name_overrides_parent_name(self) -> None:
        proxy = _make_proxy()
        assert proxy.name != "go2_ros2_proxy"
        assert proxy.name == "gazebo_go2"

    def test_supports_lidar_overrides_parent_false(self) -> None:
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        parent = Go2ROS2Proxy()
        proxy = _make_proxy()
        assert parent.supports_lidar is False
        assert proxy.supports_lidar is True


# ---------------------------------------------------------------------------
# 2. is_gazebo_running
# ---------------------------------------------------------------------------


class TestIsGazeboRunning:
    """is_gazebo_running() detects /clock topic without external deps."""

    def test_is_gazebo_running_false_no_process(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            assert GazeboGo2Proxy.is_gazebo_running() is False

    def test_is_gazebo_running_true_clock_topic(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        mock_result = MagicMock()
        mock_result.stdout = "/clock\n/cmd_vel\n"

        with patch("subprocess.run", return_value=mock_result):
            assert GazeboGo2Proxy.is_gazebo_running() is True

    def test_is_gazebo_running_handles_timeout(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ros2", 5)):
            assert GazeboGo2Proxy.is_gazebo_running() is False

    def test_is_gazebo_running_false_when_clock_absent(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        mock_result = MagicMock()
        mock_result.stdout = "/parameter_events\n/rosout\n"

        with patch("subprocess.run", return_value=mock_result):
            assert GazeboGo2Proxy.is_gazebo_running() is False

    def test_is_gazebo_running_handles_file_not_found(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        with patch("subprocess.run", side_effect=FileNotFoundError("ros2 not found")):
            assert GazeboGo2Proxy.is_gazebo_running() is False

    def test_is_gazebo_running_uses_topic_list_command(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            GazeboGo2Proxy.is_gazebo_running()
            call_args = mock_run.call_args[0][0]
            assert "ros2" in call_args
            assert "topic" in call_args
            assert "list" in call_args

    def test_is_gazebo_running_uses_timeout(self) -> None:
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            GazeboGo2Proxy.is_gazebo_running()
            call_kwargs = mock_run.call_args[1]
            assert "timeout" in call_kwargs
            assert call_kwargs["timeout"] > 0


# ---------------------------------------------------------------------------
# 3. Lifecycle — connect / disconnect
# ---------------------------------------------------------------------------


class TestGazeboGo2ProxyLifecycle:
    """connect() checks Gazebo first; disconnect() delegates to parent."""

    def test_connect_raises_when_not_running(self) -> None:
        proxy = _make_proxy()
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ConnectionError) as exc_info:
                proxy.connect()
            assert "gazebo" in str(exc_info.value).lower()

    def test_connect_calls_super_when_running(self) -> None:
        proxy = _make_proxy()
        mock_result = MagicMock()
        mock_result.stdout = "/clock\n"

        with patch("subprocess.run", return_value=mock_result), \
             patch.object(
                 proxy.__class__.__bases__[0], "connect"
             ) as mock_parent_connect:
            proxy.connect()
            mock_parent_connect.assert_called_once()

    def test_connect_error_mentions_gazebo(self) -> None:
        proxy = _make_proxy()
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ConnectionError) as exc_info:
                proxy.connect()
            msg = str(exc_info.value).lower()
            assert "gazebo" in msg or "running" in msg

    def test_disconnect_delegates_to_parent(self) -> None:
        proxy = _make_proxy()
        mock_node = MagicMock()
        proxy._node = mock_node
        proxy._connected = True

        proxy.disconnect()

        mock_node.destroy_node.assert_called_once()
        assert proxy._node is None
        assert proxy._connected is False

    def test_disconnect_safe_when_not_connected(self) -> None:
        proxy = _make_proxy()
        # Never connected — must not raise
        proxy.disconnect()
        assert proxy._connected is False

    def test_initial_state_not_connected(self) -> None:
        proxy = _make_proxy()
        assert proxy._connected is False
        assert proxy._node is None


# ---------------------------------------------------------------------------
# 4. __init__.py export
# ---------------------------------------------------------------------------


class TestInitPyExportsGazeboProxy:
    """GazeboGo2Proxy is lazily importable from vector_os_nano.hardware.sim."""

    def test_init_py_exports_gazebo_proxy(self) -> None:
        # The import should succeed (module exists on path)
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy
        assert GazeboGo2Proxy is not None

    def test_gazebo_proxy_accessible_via_sim_package(self) -> None:
        # After the lazy import block in __init__.py is added, GazeboGo2Proxy
        # should appear in the sim package namespace.
        import vector_os_nano.hardware.sim as sim_pkg
        assert hasattr(sim_pkg, "GazeboGo2Proxy"), (
            "GazeboGo2Proxy not exported from vector_os_nano.hardware.sim — "
            "add lazy import to __init__.py"
        )
