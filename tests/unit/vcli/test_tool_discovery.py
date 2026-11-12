# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for vcli.tools discovery functions — Task 9 (Tool Discovery Update).

TDD RED phase: these tests are written before the implementation is updated.
"""
from __future__ import annotations

import pytest


class TestDiscoverAllTools:
    """Tests for the backward-compatible discover_all_tools() flat list."""

    def test_discover_all_tools_count(self) -> None:
        """discover_all_tools() must return at least 10 tools (existing + new)."""
        from vector_os_nano.vcli.tools import discover_all_tools

        tools = discover_all_tools()
        assert len(tools) >= 10

    def test_discover_all_tools_includes_new(self) -> None:
        """New Wave 1-2 tools must appear in the flat list by name."""
        from vector_os_nano.vcli.tools import discover_all_tools

        tools = discover_all_tools()
        names = {t.name for t in tools}
        expected_new = {
            "scene_graph_query",
            "nav_state",
            "terrain_status",
            "skill_reload",
        }
        missing = expected_new - names
        assert missing == set(), f"Missing tools: {missing}"

    def test_backward_compat_returns_flat_list(self) -> None:
        """discover_all_tools() must still return a plain list (backward compat)."""
        from vector_os_nano.vcli.tools import discover_all_tools

        result = discover_all_tools()
        assert isinstance(result, list)


class TestToolCategories:
    """Each tool declares its own category; the registry files it there (no hand map)."""

    def test_categories_code(self) -> None:
        """'code' category contains file_read, file_write, file_edit, bash, glob, grep."""
        from vector_os_nano.vcli.runtime import build_registry

        categories = build_registry().list_categories()
        assert "code" in categories
        code_tools = set(categories["code"])
        expected = {"file_read", "file_write", "file_edit", "bash", "glob", "grep"}
        missing = expected - code_tools
        assert missing == set(), f"'code' category missing: {missing}"

    def test_categories_diag(self) -> None:
        """'diag' is exactly nav_state and terrain_status (the ROS diagnostic tools are deleted)."""
        from vector_os_nano.vcli.runtime import build_registry

        categories = build_registry().list_categories()
        assert "diag" in categories
        diag_tools = set(categories["diag"])
        assert diag_tools == {"nav_state", "terrain_status"}

    def test_categories_system(self) -> None:
        """'system' holds robot infra; sim lifecycle lives in its own 'sim' category."""
        from vector_os_nano.vcli.runtime import build_registry

        categories = build_registry().list_categories()
        assert "system" in categories
        system_tools = set(categories["system"])
        expected = {"robot_status", "skill_reload", "open_foxglove"}
        missing = expected - system_tools
        assert missing == set(), f"'system' category missing: {missing}"
        # start/stop_simulation moved to the always-enabled 'sim' category so the
        # user can spin up a sim from the bare dev world.
        assert {"start_simulation", "stop_simulation"} <= set(categories.get("sim", []))
        assert "start_simulation" not in system_tools

    def test_categories_general(self) -> None:
        """'general' is the domain-general bucket and holds web_fetch (moved out of 'system')."""
        from vector_os_nano.vcli.runtime import build_registry

        categories = build_registry().list_categories()
        assert "general" in categories
        assert "web_fetch" in set(categories["general"])
        # web_fetch must no longer be advertised as a robot/system tool
        assert "web_fetch" not in set(categories.get("system", []))

    def test_categories_robot(self) -> None:
        """'robot' category contains scene_graph_query, the one object/scene query tool."""
        from vector_os_nano.vcli.runtime import build_registry

        categories = build_registry().list_categories()
        assert "robot" in categories
        robot_tools = set(categories["robot"])
        assert "scene_graph_query" in robot_tools, (
            f"'robot' category missing 'scene_graph_query'; has: {robot_tools}"
        )

    def test_every_built_in_tool_declares_a_category(self) -> None:
        """producers:tool-category-map-handauthored: a new tool cannot land in
        'default' silently; its decorator names its category."""
        from vector_os_nano.vcli.tools import discover_all_tools

        assert [t.name for t in discover_all_tools() if t.category == "default"] == []
