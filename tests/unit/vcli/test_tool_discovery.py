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
            "ros2_topics",
            "ros2_nodes",
            "ros2_log",
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


class TestDiscoverCategorizedTools:
    """Tests for the new discover_categorized_tools() function."""

    def test_discover_categorized_tools_signature(self) -> None:
        """discover_categorized_tools() must return a 2-tuple."""
        from vector_os_nano.vcli.tools import discover_categorized_tools

        result = discover_categorized_tools()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_discover_categorized_tools_structure(self) -> None:
        """First element is list of tools, second is dict of categories."""
        from vector_os_nano.vcli.tools import discover_categorized_tools

        tools_list, categories_dict = discover_categorized_tools()
        assert isinstance(tools_list, list)
        assert isinstance(categories_dict, dict)

    def test_categories_code(self) -> None:
        """'code' category contains file_read, file_write, file_edit, bash, glob, grep."""
        from vector_os_nano.vcli.tools import discover_categorized_tools

        _, categories = discover_categorized_tools()
        assert "code" in categories
        code_tools = set(categories["code"])
        expected = {"file_read", "file_write", "file_edit", "bash", "glob", "grep"}
        missing = expected - code_tools
        assert missing == set(), f"'code' category missing: {missing}"

    def test_categories_diag(self) -> None:
        """'diag' category contains ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status."""
        from vector_os_nano.vcli.tools import discover_categorized_tools

        _, categories = discover_categorized_tools()
        assert "diag" in categories
        diag_tools = set(categories["diag"])
        expected = {"ros2_topics", "ros2_nodes", "ros2_log", "nav_state", "terrain_status"}
        missing = expected - diag_tools
        assert missing == set(), f"'diag' category missing: {missing}"

    def test_categories_system(self) -> None:
        """'system' category contains robot_status, start_simulation, web_fetch, skill_reload."""
        from vector_os_nano.vcli.tools import discover_categorized_tools

        _, categories = discover_categorized_tools()
        assert "system" in categories
        system_tools = set(categories["system"])
        expected = {"robot_status", "start_simulation", "web_fetch", "skill_reload"}
        missing = expected - system_tools
        assert missing == set(), f"'system' category missing: {missing}"

    def test_categories_robot(self) -> None:
        """'robot' category contains scene_graph_query (world_query may also be here)."""
        from vector_os_nano.vcli.tools import discover_categorized_tools

        _, categories = discover_categorized_tools()
        assert "robot" in categories
        robot_tools = set(categories["robot"])
        assert "scene_graph_query" in robot_tools, (
            f"'robot' category missing 'scene_graph_query'; has: {robot_tools}"
        )
