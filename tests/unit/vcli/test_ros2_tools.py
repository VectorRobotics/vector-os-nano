"""Tests for ROS2 diagnostic tools.

TDD RED phase — tests written before implementation exists.
All subprocess calls are mocked — no real ros2 CLI is invoked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.vcli.tools.base import ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> MagicMock:
    ctx = MagicMock(spec=ToolContext)
    ctx.app_state = {}
    return ctx


def _make_completed(stdout: str, returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def topics_tool():
    from vector_os_nano.vcli.tools.ros2_tools import Ros2TopicsTool
    return Ros2TopicsTool()


@pytest.fixture()
def nodes_tool():
    from vector_os_nano.vcli.tools.ros2_tools import Ros2NodesTool
    return Ros2NodesTool()


@pytest.fixture()
def log_tool():
    from vector_os_nano.vcli.tools.ros2_tools import Ros2LogTool
    return Ros2LogTool()


# ---------------------------------------------------------------------------
# Test: ros2_topics list
# ---------------------------------------------------------------------------


class TestRos2TopicsList:
    def test_topics_list(self, topics_tool):
        """list action returns parsed topic list from subprocess stdout."""
        mock_output = "/cmd_vel [geometry_msgs/msg/Twist]\n/scan [sensor_msgs/msg/LaserScan]"
        with patch("subprocess.run", return_value=_make_completed(mock_output)):
            result = topics_tool.execute({"action": "list"}, _make_context())

        assert not result.is_error
        assert "/cmd_vel" in result.content
        assert "/scan" in result.content

    def test_topics_list_calls_correct_args(self, topics_tool):
        """list action calls ros2 topic list -t."""
        mock_output = "/chatter [std_msgs/msg/String]"
        with patch("subprocess.run", return_value=_make_completed(mock_output)) as mock_run:
            topics_tool.execute({"action": "list"}, _make_context())

        called_args = mock_run.call_args[0][0]
        assert called_args == ["ros2", "topic", "list", "-t"]


# ---------------------------------------------------------------------------
# Test: ros2_topics hz
# ---------------------------------------------------------------------------


class TestRos2TopicsHz:
    def test_topics_hz(self, topics_tool):
        """hz action parses rate from subprocess stdout."""
        mock_output = "average rate: 10.003\n\tmin: 0.099s max: 0.101s std dev: 0.00001s window: 5"
        with patch("subprocess.run", return_value=_make_completed(mock_output)):
            result = topics_tool.execute(
                {"action": "hz", "topic": "/cmd_vel"}, _make_context()
            )

        assert not result.is_error
        assert "average rate: 10.003" in result.content

    def test_topics_hz_missing_topic(self, topics_tool):
        """hz without topic param returns error."""
        result = topics_tool.execute({"action": "hz"}, _make_context())
        assert result.is_error
        assert "topic" in result.content.lower()


# ---------------------------------------------------------------------------
# Test: ros2_topics echo
# ---------------------------------------------------------------------------


class TestRos2TopicsEcho:
    def test_topics_echo(self, topics_tool):
        """echo action returns message content."""
        mock_output = "linear:\n  x: 0.5\n  y: 0.0\n  z: 0.0"
        with patch("subprocess.run", return_value=_make_completed(mock_output)):
            result = topics_tool.execute(
                {"action": "echo", "topic": "/cmd_vel"}, _make_context()
            )

        assert not result.is_error
        assert "linear" in result.content

    def test_topics_echo_truncation(self, topics_tool):
        """echo output longer than 2000 chars is truncated."""
        long_output = "x: 0.1\n" * 400  # ~2800 chars
        with patch("subprocess.run", return_value=_make_completed(long_output)):
            result = topics_tool.execute(
                {"action": "echo", "topic": "/scan"}, _make_context()
            )

        assert not result.is_error
        assert len(result.content) <= 2020  # 2000 + "... (truncated)" overhead
        assert "truncated" in result.content

    def test_topics_echo_missing_topic(self, topics_tool):
        """echo without topic param returns error."""
        result = topics_tool.execute({"action": "echo"}, _make_context())
        assert result.is_error


# ---------------------------------------------------------------------------
# Test: ros2_nodes list
# ---------------------------------------------------------------------------


class TestRos2NodesList:
    def test_nodes_list(self, nodes_tool):
        """list action returns node names from subprocess stdout."""
        mock_output = "/vector_nav\n/go2_driver\n/rosm_bridge"
        with patch("subprocess.run", return_value=_make_completed(mock_output)):
            result = nodes_tool.execute({"action": "list"}, _make_context())

        assert not result.is_error
        assert "/vector_nav" in result.content
        assert "/go2_driver" in result.content

    def test_nodes_list_calls_correct_args(self, nodes_tool):
        """list action calls ros2 node list."""
        mock_output = "/some_node"
        with patch("subprocess.run", return_value=_make_completed(mock_output)) as mock_run:
            nodes_tool.execute({"action": "list"}, _make_context())

        called_args = mock_run.call_args[0][0]
        assert called_args == ["ros2", "node", "list"]


# ---------------------------------------------------------------------------
# Test: ros2_nodes info
# ---------------------------------------------------------------------------


class TestRos2NodesInfo:
    def test_nodes_info(self, nodes_tool):
        """info action returns node details from subprocess stdout."""
        mock_output = (
            "/vector_nav\n"
            "  Subscribers:\n    /odom: nav_msgs/msg/Odometry\n"
            "  Publishers:\n    /cmd_vel: geometry_msgs/msg/Twist"
        )
        with patch("subprocess.run", return_value=_make_completed(mock_output)):
            result = nodes_tool.execute(
                {"action": "info", "node": "/vector_nav"}, _make_context()
            )

        assert not result.is_error
        assert "Subscribers" in result.content
        assert "Publishers" in result.content

    def test_nodes_info_missing_node(self, nodes_tool):
        """info without node param returns error."""
        result = nodes_tool.execute({"action": "info"}, _make_context())
        assert result.is_error
        assert "node" in result.content.lower()


# ---------------------------------------------------------------------------
# Test: ros2_log read
# ---------------------------------------------------------------------------


class TestRos2LogRead:
    def test_log_read(self, log_tool, tmp_path):
        """Reading an existing log file returns last N lines."""
        log_file = tmp_path / "vector_vnav_bridge.log"
        lines = [f"line {i}" for i in range(100)]
        log_file.write_text("\n".join(lines))

        # Patch the _LOG_MAP to point at tmp file
        original_map = log_tool._LOG_MAP.copy()
        log_tool._LOG_MAP = {"bridge": str(log_file)}
        try:
            result = log_tool.execute({"log_name": "bridge", "lines": 10}, _make_context())
        finally:
            log_tool._LOG_MAP = original_map

        assert not result.is_error
        returned_lines = result.content.splitlines()
        assert len(returned_lines) == 10
        assert "line 99" in result.content
        assert "line 90" in result.content

    def test_log_default_lines(self, log_tool, tmp_path):
        """Default line count is 50 when not specified."""
        log_file = tmp_path / "vector_tare.log"
        lines = [f"entry {i}" for i in range(200)]
        log_file.write_text("\n".join(lines))

        original_map = log_tool._LOG_MAP.copy()
        log_tool._LOG_MAP = {"tare": str(log_file)}
        try:
            result = log_tool.execute({"log_name": "tare"}, _make_context())
        finally:
            log_tool._LOG_MAP = original_map

        assert not result.is_error
        assert len(result.content.splitlines()) == 50

    def test_log_missing_file(self, log_tool):
        """Nonexistent log file returns error with path info."""
        original_map = log_tool._LOG_MAP.copy()
        log_tool._LOG_MAP = {"bridge": "/tmp/nonexistent_vector_abc123.log"}
        try:
            result = log_tool.execute({"log_name": "bridge"}, _make_context())
        finally:
            log_tool._LOG_MAP = original_map

        assert result.is_error
        assert "not found" in result.content.lower()

    def test_log_unknown_name(self, log_tool):
        """Unknown log_name returns error listing available logs."""
        result = log_tool.execute({"log_name": "nonexistent_log"}, _make_context())
        assert result.is_error
        assert "Unknown log" in result.content


# ---------------------------------------------------------------------------
# Test: error handling — ros2 not available
# ---------------------------------------------------------------------------


class TestRos2NotAvailable:
    def test_ros2_not_available_topics(self, topics_tool):
        """FileNotFoundError from subprocess returns graceful error."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = topics_tool.execute({"action": "list"}, _make_context())

        assert result.is_error
        assert "ros2" in result.content.lower()
        assert "not available" in result.content.lower()

    def test_ros2_not_available_nodes(self, nodes_tool):
        """FileNotFoundError from subprocess returns graceful error."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = nodes_tool.execute({"action": "list"}, _make_context())

        assert result.is_error
        assert "ros2" in result.content.lower()


# ---------------------------------------------------------------------------
# Test: subprocess timeout
# ---------------------------------------------------------------------------


class TestSubprocessTimeout:
    def test_subprocess_timeout_topics(self, topics_tool):
        """TimeoutExpired from subprocess returns graceful error."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ros2"], timeout=10),
        ):
            result = topics_tool.execute({"action": "list"}, _make_context())

        assert result.is_error
        assert "timed out" in result.content.lower()

    def test_subprocess_timeout_nodes(self, nodes_tool):
        """TimeoutExpired from subprocess returns graceful error."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ros2"], timeout=10),
        ):
            result = nodes_tool.execute({"action": "list"}, _make_context())

        assert result.is_error
        assert "timed out" in result.content.lower()
