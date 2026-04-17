"""Unit tests for vcli robot_context.py — TDD RED phase.

Covers:
- test_context_with_base_includes_position
- test_context_with_base_includes_heading
- test_context_with_base_and_sg_includes_room
- test_context_with_sg_includes_stats
- test_context_without_base_graceful
- test_context_block_format
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockBase:
    def get_position(self) -> tuple[float, float, float]:
        return (10.2, 5.3, 0.28)

    def get_heading(self) -> float:
        return 0.4  # radians


class MockSceneGraph:
    def nearest_room(self, x: float, y: float) -> str:
        return "kitchen"

    def stats(self) -> dict:
        return {"rooms": 8, "visited_rooms": 6, "viewpoints": 3, "objects": 12}

    def get_all_doors(self) -> dict:
        return {("kitchen", "hallway"): (13.5, 3.0)}

    def get_room_summary(self) -> str:
        return "kitchen (10 visits), hallway (5 visits), bedroom (3 visits)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(base=None, sg=None):
    from vector_os_nano.vcli.robot_context import RobotContextProvider

    return RobotContextProvider(base=base, scene_graph=sg)


# ---------------------------------------------------------------------------
# Block format
# ---------------------------------------------------------------------------


class TestBlockFormat:
    def test_context_block_format(self) -> None:
        """get_context_block() returns a dict with 'type' and 'text' keys."""
        provider = _make_provider()
        block = provider.get_context_block()
        assert isinstance(block, dict)
        assert block.get("type") == "text"
        assert isinstance(block.get("text"), str)
        assert len(block["text"]) > 0


# ---------------------------------------------------------------------------
# Base-only
# ---------------------------------------------------------------------------


class TestBaseOnly:
    def test_context_with_base_includes_position(self) -> None:
        """When base provides get_position(), coordinates appear in output."""
        provider = _make_provider(base=MockBase())
        block = provider.get_context_block()
        text = block["text"]
        assert "10.2" in text
        assert "5.3" in text

    def test_context_with_base_includes_heading(self) -> None:
        """When base provides get_heading(), heading in degrees appears in output."""
        provider = _make_provider(base=MockBase())
        block = provider.get_context_block()
        text = block["text"]
        # 0.4 rad = ~22.9 degrees
        expected_deg = round(math.degrees(0.4))
        assert str(expected_deg) in text

    def test_context_without_base_graceful(self) -> None:
        """No base connected -> 'No hardware connected' message in output."""
        provider = _make_provider(base=None, sg=None)
        block = provider.get_context_block()
        text = block["text"]
        assert "No hardware connected" in text


# ---------------------------------------------------------------------------
# Base + SceneGraph
# ---------------------------------------------------------------------------


class TestBaseAndSceneGraph:
    def test_context_with_base_and_sg_includes_room(self) -> None:
        """When base + SceneGraph present, current room name appears in output."""
        provider = _make_provider(base=MockBase(), sg=MockSceneGraph())
        block = provider.get_context_block()
        text = block["text"]
        assert "kitchen" in text

    def test_context_with_sg_includes_stats(self) -> None:
        """When SceneGraph present, room/object counts appear in output."""
        provider = _make_provider(base=MockBase(), sg=MockSceneGraph())
        block = provider.get_context_block()
        text = block["text"]
        # rooms=8, visited_rooms=6, objects=12
        assert "8" in text
        assert "6" in text
        assert "12" in text

    def test_context_with_sg_includes_room_summary(self) -> None:
        """When SceneGraph provides room summary, it appears in output."""
        provider = _make_provider(base=MockBase(), sg=MockSceneGraph())
        block = provider.get_context_block()
        text = block["text"]
        assert "hallway" in text

    def test_context_sg_only_no_base(self) -> None:
        """SceneGraph without base: stats still appear, position absent."""
        provider = _make_provider(base=None, sg=MockSceneGraph())
        block = provider.get_context_block()
        text = block["text"]
        # No position since no base
        assert "Position" not in text
        # Stats still there
        assert "8" in text

    def test_context_block_header(self) -> None:
        """Output starts with '[Robot State]' header."""
        provider = _make_provider(base=MockBase())
        block = provider.get_context_block()
        assert block["text"].startswith("[Robot State]")
