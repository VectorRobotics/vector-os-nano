# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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

    def test_stop_kills_its_own_launch_tree(self):
        """stop reaches the whole process group it started (ros2 launch spawns
        children), not just the leader."""
        import os
        import time

        import vector_os_nano.vcli.tools.viz_tool as viz_mod
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        proc = subprocess.Popen(
            ["bash", "-c", "sleep 30 & echo $!; wait"],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        child = int(proc.stdout.readline())
        viz_mod._foxglove_proc = proc
        try:
            result = FoxgloveTool().execute({"action": "stop"}, _make_context())
            assert "stopped" in result.content.lower()
            assert viz_mod._foxglove_proc is None
            assert proc.poll() is not None
            for _ in range(50):
                try:
                    with open(f"/proc/{child}/stat") as fh:
                        if fh.read().split()[2] == "Z":
                            break
                except FileNotFoundError:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("the bridge's child survived stop")
        finally:
            for pid in (proc.pid, child):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
            proc.wait()
            proc.stdout.close()

    def test_stop_never_kills_a_bridge_it_did_not_start(self):
        """Another stack's bridge on :8765 is reported, never killed by port or name."""
        import vector_os_nano.vcli.tools.viz_tool as viz_mod
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        viz_mod._foxglove_proc = None
        with (
            patch("vector_os_nano.vcli.tools.viz_tool._is_bridge_running", return_value=True),
            patch("subprocess.run") as mock_run,
            patch("os.kill") as mock_kill,
            patch("os.killpg") as mock_killpg,
        ):
            result = FoxgloveTool().execute({"action": "stop"}, _make_context())
        assert "did not start" in result.content
        mock_run.assert_not_called()
        mock_kill.assert_not_called()
        mock_killpg.assert_not_called()

    def test_start_runs_the_bridge_in_its_own_session(self):
        import vector_os_nano.vcli.tools.viz_tool as viz_mod
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ros2", timeout=2)
        with (
            patch("vector_os_nano.vcli.tools.viz_tool._is_bridge_running", return_value=False),
            patch("shutil.which", return_value="/usr/bin/ros2"),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            FoxgloveTool().execute({"action": "start"}, _make_context())
        assert mock_popen.call_args.kwargs.get("start_new_session") is True
        viz_mod._foxglove_proc = None

    def test_open_foxglove_asks(self):
        from vector_os_nano.vcli.permissions import PermissionContext
        from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

        assert PermissionContext().check(FoxgloveTool(), {"action": "start"}).behavior == "ask"


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
