"""Tests for FoxgloveTool — Foxglove Bridge management.

All subprocess calls are mocked — no real foxglove_bridge is invoked.
"""
from __future__ import annotations

import subprocess
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


def _ss_output_with_8765() -> MagicMock:
    proc = MagicMock()
    proc.stdout = 'LISTEN 0 0 0.0.0.0:8765 0.0.0.0:* users:(("foxglove",pid=123,fd=5))'
    proc.returncode = 0
    return proc


def _ss_output_empty() -> MagicMock:
    proc = MagicMock()
    proc.stdout = ""
    proc.returncode = 0
    return proc


# ---------------------------------------------------------------------------
# Tests: status
# ---------------------------------------------------------------------------


class TestFoxgloveStatus:
    def test_status_not_running(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        tool = FoxgloveTool()
        ctx = _make_context()

        with patch("subprocess.run", return_value=_ss_output_empty()):
            result = tool.execute({"action": "status"}, ctx)
            assert "not running" in result.content

    def test_status_running(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        tool = FoxgloveTool()
        ctx = _make_context()

        with patch("subprocess.run", return_value=_ss_output_with_8765()):
            result = tool.execute({"action": "status"}, ctx)
            assert "running" in result.content
            assert "8765" in result.content


# ---------------------------------------------------------------------------
# Tests: start
# ---------------------------------------------------------------------------


class TestFoxgloveStart:
    def test_start_already_running(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        tool = FoxgloveTool()
        ctx = _make_context()

        with patch(
            "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
            return_value=True,
        ):
            result = tool.execute({"action": "start"}, ctx)
            assert "already running" in result.content
            assert not result.is_error

    def test_start_no_ros2(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        tool = FoxgloveTool()
        ctx = _make_context()

        with (
            patch(
                "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
                return_value=False,
            ),
            patch("shutil.which", return_value=None),
        ):
            result = tool.execute({"action": "start"}, ctx)
            assert result.is_error
            assert "ros2 not found" in result.content

    def test_start_success(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool, _foxglove_proc
        import vector_os_nano.vcli.tools.viz_tool as viz_mod

        tool = FoxgloveTool()
        ctx = _make_context()

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ros2", timeout=2)

        with (
            patch(
                "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
                return_value=False,
            ),
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            result = tool.execute({"action": "start"}, ctx)
            assert not result.is_error
            assert "ws://localhost:8765" in result.content
            assert "app.foxglove.dev" in result.content
            mock_popen.assert_called_once()

        # Cleanup module state
        viz_mod._foxglove_proc = None

    def test_start_exits_immediately(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool
        import vector_os_nano.vcli.tools.viz_tool as viz_mod

        tool = FoxgloveTool()
        ctx = _make_context()

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1  # exits immediately

        with (
            patch(
                "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
                return_value=False,
            ),
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            result = tool.execute({"action": "start"}, ctx)
            assert result.is_error
            assert "exited immediately" in result.content

        viz_mod._foxglove_proc = None


# ---------------------------------------------------------------------------
# Tests: stop
# ---------------------------------------------------------------------------


class TestFoxgloveStop:
    def test_stop_no_process(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool
        import vector_os_nano.vcli.tools.viz_tool as viz_mod

        viz_mod._foxglove_proc = None
        tool = FoxgloveTool()
        ctx = _make_context()

        with patch(
            "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
            return_value=False,
        ):
            result = tool.execute({"action": "stop"}, ctx)
            assert "not running" in result.content

    def test_stop_with_process(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool
        import vector_os_nano.vcli.tools.viz_tool as viz_mod

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        viz_mod._foxglove_proc = mock_proc

        tool = FoxgloveTool()
        ctx = _make_context()

        result = tool.execute({"action": "stop"}, ctx)
        assert "stopped" in result.content.lower()
        mock_proc.terminate.assert_called_once()
        assert viz_mod._foxglove_proc is None

    def test_stop_port_kill_fallback(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool
        import vector_os_nano.vcli.tools.viz_tool as viz_mod

        viz_mod._foxglove_proc = None
        tool = FoxgloveTool()
        ctx = _make_context()

        with (
            patch(
                "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
                return_value=True,
            ),
            patch("subprocess.run") as mock_run,
        ):
            result = tool.execute({"action": "stop"}, ctx)
            assert "stopped" in result.content.lower()
            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: default action
# ---------------------------------------------------------------------------


class TestFoxgloveDefaults:
    def test_default_action_is_start(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        tool = FoxgloveTool()
        ctx = _make_context()

        with patch(
            "vector_os_nano.vcli.tools.viz_tool._is_bridge_running",
            return_value=True,
        ):
            result = tool.execute({}, ctx)
            assert "already running" in result.content

    def test_tool_metadata(self):
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        tool = FoxgloveTool()
        assert tool.name == "open_foxglove"
        assert "Foxglove" in tool.description
