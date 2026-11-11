# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for NavStateTool and TerrainStatusTool.

TDD RED phase — all tests must fail before implementation is written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vector_os_nano.vcli.tools.base import ToolContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_context() -> MagicMock:
    ctx = MagicMock(spec=ToolContext)
    ctx.app_state = {}
    return ctx


@pytest.fixture()
def nav_tool():
    from vector_os_nano.vcli.tools.nav_tools import NavStateTool
    return NavStateTool()


@pytest.fixture()
def terrain_tool():
    from vector_os_nano.vcli.tools.nav_tools import TerrainStatusTool
    return TerrainStatusTool()


# ---------------------------------------------------------------------------
# NavStateTool tests
# ---------------------------------------------------------------------------


class TestNavStateTool:

    @patch("vector_os_nano.vcli.tools.nav_tools._is_exploring", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._is_nav_stack_running", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._get_explored_rooms", return_value=["kitchen", "hallway"])
    def test_nav_state_all_fields(self, mock_rooms, mock_nav, mock_explore, nav_tool):
        """Output must include all required diagnostic fields."""
        ctx = _make_context()
        result = nav_tool.execute({}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert "exploring" in data
        assert "nav_stack_running" in data
        assert "nav_flag_active" in data
        assert "explored_rooms" in data
        assert "tare_running" in data

    @patch("vector_os_nano.vcli.tools.nav_tools._is_exploring", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._is_nav_stack_running", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._get_explored_rooms", return_value=["kitchen", "hallway"])
    def test_nav_state_values(self, mock_rooms, mock_nav, mock_explore, nav_tool):
        """Mocked values must flow through to the output JSON."""
        ctx = _make_context()
        result = nav_tool.execute({}, ctx)
        data = json.loads(result.content)
        assert data["exploring"] is True
        assert data["nav_stack_running"] is True
        assert data["explored_rooms"] == ["kitchen", "hallway"]

    def test_nav_state_no_explore_module(self, nav_tool):
        """Tool must succeed gracefully even when explore module is unavailable.

        When _is_exploring / _is_nav_stack_running / _get_explored_rooms raise
        ImportError internally they return False / []. The tool itself must never
        raise and must return a valid JSON result.
        """
        ctx = _make_context()
        # Patch helpers to simulate ImportError behaviour (returns defaults)
        with patch("vector_os_nano.vcli.tools.nav_tools._is_exploring", return_value=False), \
             patch("vector_os_nano.vcli.tools.nav_tools._is_nav_stack_running", return_value=False), \
             patch("vector_os_nano.vcli.tools.nav_tools._get_explored_rooms", return_value=[]):
            result = nav_tool.execute({}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert data["exploring"] is False
        assert data["explored_rooms"] == []

    def test_nav_tool_is_read_only(self, nav_tool):
        assert nav_tool.is_read_only({}) is True

    def test_nav_tool_is_concurrency_safe(self, nav_tool):
        assert nav_tool.is_concurrency_safe({}) is True


# ---------------------------------------------------------------------------
# TerrainStatusTool tests
# ---------------------------------------------------------------------------


class TestTerrainStatusTool:

    def test_terrain_status_file_exists(self, terrain_tool, tmp_path):
        """When terrain file exists, file_exists=True and size is reported."""
        npz_file = tmp_path / "terrain_map.npz"
        ix = np.array([1, 2, 3])
        np.savez(str(npz_file), ix=ix)

        ctx = _make_context()
        with patch("vector_os_nano.vcli.tools.nav_tools._TERRAIN_PATH", str(npz_file)):
            result = terrain_tool.execute({}, ctx)

        assert not result.is_error
        data = json.loads(result.content)
        assert data["file_exists"] is True
        assert data["file_size_kb"] > 0
        assert data["voxel_count"] == 3

    def test_terrain_status_file_missing(self, terrain_tool, tmp_path):
        """When terrain file is absent, file_exists=False with graceful output."""
        missing = tmp_path / "no_terrain_here.npz"
        ctx = _make_context()
        with patch("vector_os_nano.vcli.tools.nav_tools._TERRAIN_PATH", str(missing)):
            result = terrain_tool.execute({}, ctx)

        assert not result.is_error
        data = json.loads(result.content)
        assert data["file_exists"] is False
        assert data["file_size_kb"] == 0
        assert data["voxel_count"] == 0

    def test_terrain_status_fields(self, terrain_tool, tmp_path):
        """All expected fields must be present in the output."""
        npz_file = tmp_path / "terrain_map.npz"
        np.savez(str(npz_file), ix=np.array([10, 20]))

        ctx = _make_context()
        with patch("vector_os_nano.vcli.tools.nav_tools._TERRAIN_PATH", str(npz_file)):
            result = terrain_tool.execute({}, ctx)

        data = json.loads(result.content)
        for key in ("file_exists", "file_path", "file_size_kb", "replay_triggered", "voxel_count"):
            assert key in data, f"Missing field: {key}"

    def test_terrain_tool_is_read_only(self, terrain_tool):
        assert terrain_tool.is_read_only({}) is True

    def test_terrain_tool_is_concurrency_safe(self, terrain_tool):
        assert terrain_tool.is_concurrency_safe({}) is True


class TestGoToRoom:
    """go_to_room: a room name in any language -> the map's free point -> navigate."""

    @staticmethod
    def _agent(skills):
        from types import SimpleNamespace

        from vector_os_nano.core.skill import SkillRegistry

        reg = SkillRegistry()
        for s in skills:
            reg.register(s)
        return SimpleNamespace(
            _skill_registry=reg, _spatial_memory=None, _base=None, _config={"world": "apartment"},
            _build_context=lambda: SimpleNamespace(), _sync_robot_state=lambda: None,
        )

    @staticmethod
    def _ctx(agent):
        import threading

        return ToolContext(agent=None, cwd=Path("/tmp"), session=None, permissions=None,
                           abort=threading.Event(), app_state={"agent": agent})

    def test_room_alias_drives_navigate_to_the_map_point(self):
        from types import SimpleNamespace

        from vector_os_nano.core.types import SkillResult
        from vector_os_nano.vcli.tools.nav_tools import GoToRoomTool

        calls = []
        nav = SimpleNamespace(name="navigate", parameters={},
                              execute=lambda p, c: calls.append(p) or SkillResult(success=True))
        res = GoToRoomTool().execute({"room": "厨房"}, self._ctx(self._agent([nav])))
        assert not res.is_error, res.content
        assert calls == [{"x": 17.0, "y": 1.7}]  # config/worlds/apartment.yaml kitchen seed
        assert "verify(in_room('kitchen'))" in res.content

    def test_unknown_room_fails_loud_with_the_valid_set(self):
        from vector_os_nano.vcli.tools.nav_tools import GoToRoomTool

        res = GoToRoomTool().execute({"room": "garage"}, self._ctx(self._agent([])))
        assert res.is_error and "kitchen" in res.content and "garage" in res.content

    def test_no_robot(self):
        from vector_os_nano.vcli.tools.nav_tools import GoToRoomTool

        res = GoToRoomTool().execute({"room": "kitchen"}, self._ctx(None))
        assert res.is_error and "start_simulation" in res.content
