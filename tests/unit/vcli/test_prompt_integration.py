"""Unit tests for robot_context integration in prompt.py — TDD RED phase.

Covers:
- test_build_prompt_with_robot_context: RobotContextProvider block appears in output
- test_build_prompt_without_robot_context: robot_context=None produces no robot state block
- test_robot_context_block_position: robot state block appears after world model, before VECTOR.md
- test_backward_compat: existing calls without robot_context kwarg still work

Uses mock base/sg matching the RobotContextProvider protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vector_os_nano.vcli.robot_context import RobotContextProvider


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockBase:
    def get_position(self) -> tuple[float, float, float]:
        return (10.0, 5.0, 0.28)

    def get_heading(self) -> float:
        return 0.5


class MockSG:
    def nearest_room(self, x: float, y: float) -> str:
        return "hallway"

    def stats(self) -> dict[str, Any]:
        return {
            "rooms": 8,
            "visited_rooms": 6,
            "viewpoints": 3,
            "objects": 12,
        }

    def get_all_doors(self) -> dict[tuple[str, str], tuple[float, float]]:
        return {("a", "b"): (1.0, 2.0)}

    def get_room_summary(self) -> str:
        return "kitchen, hallway, bedroom"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(**kwargs: Any) -> list[dict]:
    from vector_os_nano.vcli.prompt import build_system_prompt

    return build_system_prompt(**kwargs)


def _combined(blocks: list[dict]) -> str:
    return " ".join(b["text"] for b in blocks)


def _has_robot_state_block(blocks: list[dict]) -> bool:
    """Check if there is a dedicated [Robot State] block (not just a mention in instructions)."""
    for b in blocks:
        text = b.get("text", "")
        # A dedicated robot state block STARTS with [Robot State]
        if text.strip().startswith("[Robot State]"):
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPromptWithRobotContext:
    """build_system_prompt(robot_context=provider) includes the robot state block."""

    def test_robot_state_block_appears_in_output(self) -> None:
        """When robot_context is provided, [Robot State] header appears in output."""
        provider = RobotContextProvider(base=MockBase(), scene_graph=MockSG())
        result = _build(robot_context=provider)
        combined = _combined(result)
        assert "[Robot State]" in combined

    def test_robot_state_contains_position(self) -> None:
        """Robot state block includes position from MockBase."""
        provider = RobotContextProvider(base=MockBase(), scene_graph=MockSG())
        result = _build(robot_context=provider)
        combined = _combined(result)
        # MockBase.get_position() returns (10.0, 5.0, 0.28)
        assert "10.0" in combined
        assert "5.0" in combined

    def test_robot_state_contains_room(self) -> None:
        """Robot state block includes current room from MockSG."""
        provider = RobotContextProvider(base=MockBase(), scene_graph=MockSG())
        result = _build(robot_context=provider)
        combined = _combined(result)
        assert "hallway" in combined

    def test_robot_state_contains_scene_graph_stats(self) -> None:
        """Robot state block includes SceneGraph room/object counts."""
        provider = RobotContextProvider(base=MockBase(), scene_graph=MockSG())
        result = _build(robot_context=provider)
        combined = _combined(result)
        # MockSG.stats() → 8 rooms, 6 visited, 12 objects
        assert "8" in combined
        assert "12" in combined


class TestBuildPromptWithoutRobotContext:
    """build_system_prompt(robot_context=None) does not inject a robot state block."""

    def test_no_robot_state_block_when_none(self) -> None:
        """robot_context=None → no dedicated [Robot State] block in output."""
        result = _build(robot_context=None)
        assert not _has_robot_state_block(result)

    def test_default_is_none(self) -> None:
        """Calling build_system_prompt() without robot_context still works, no robot state block."""
        result = _build()
        assert not _has_robot_state_block(result)


class TestRobotContextBlockPosition:
    """Robot state block must appear after world model and before VECTOR.md."""

    def test_robot_state_block_after_world_model(self, tmp_path: Path) -> None:
        """Robot state block index > world model block index."""
        class _MockObjectState:
            label = "chair"
            x = 1.0
            y = 2.0
            z = 0.0

        class _MockWorldModel:
            def get_objects(self):
                return [_MockObjectState()]

        class _MockAgent:
            _arm = None
            _gripper = None
            _base = None
            _perception = None
            _world_model = _MockWorldModel()

            class _MockRegistry:
                def list_skills(self):
                    return []

            _skill_registry = _MockRegistry()

        provider = RobotContextProvider(base=MockBase(), scene_graph=MockSG())
        result = _build(agent=_MockAgent(), robot_context=provider)

        world_idx = next(
            (i for i, b in enumerate(result) if "World Model" in b["text"]), None
        )
        robot_idx = next(
            (i for i, b in enumerate(result) if b["text"].strip().startswith("[Robot State]")), None
        )

        assert world_idx is not None, "World Model block not found"
        assert robot_idx is not None, "[Robot State] block not found"
        assert robot_idx > world_idx, (
            f"Expected robot state (idx={robot_idx}) after world model (idx={world_idx})"
        )

    def test_robot_state_block_before_vector_md(self, tmp_path: Path) -> None:
        """Robot state block index < VECTOR.md block index."""
        vector_md = tmp_path / "VECTOR.md"
        vector_md.write_text("# Project Info\nSome project content.")

        provider = RobotContextProvider(base=MockBase(), scene_graph=MockSG())
        result = _build(cwd=tmp_path, robot_context=provider)

        robot_idx = next(
            (i for i, b in enumerate(result) if "[Robot State]" in b["text"]), None
        )
        vector_idx = next(
            (i for i, b in enumerate(result) if "Project Context" in b["text"]), None
        )

        assert robot_idx is not None, "[Robot State] block not found"
        assert vector_idx is not None, "Project Context (VECTOR.md) block not found"
        assert robot_idx < vector_idx, (
            f"Expected robot state (idx={robot_idx}) before VECTOR.md (idx={vector_idx})"
        )


class TestBackwardCompat:
    """Existing call signatures without robot_context must continue to work."""

    def test_no_args_still_works(self) -> None:
        """build_system_prompt() with no args returns valid list[dict]."""
        result = _build()
        assert isinstance(result, list)
        assert len(result) >= 2
        for block in result:
            assert "type" in block
            assert "text" in block

    def test_agent_only_still_works(self) -> None:
        """build_system_prompt(agent=...) without robot_context works unchanged."""
        class _MinimalAgent:
            _arm = None
            _gripper = None
            _base = None
            _perception = None
            _world_model = type("WM", (), {"get_objects": lambda self: []})()
            _skill_registry = type("SR", (), {"list_skills": lambda self: []})()

        result = _build(agent=_MinimalAgent())
        assert isinstance(result, list)
        assert not _has_robot_state_block(result)

    def test_cwd_only_still_works(self, tmp_path: Path) -> None:
        """build_system_prompt(cwd=...) without robot_context works unchanged."""
        vector_md = tmp_path / "VECTOR.md"
        vector_md.write_text("# Project\nSome info.")
        result = _build(cwd=tmp_path)
        assert isinstance(result, list)
        combined = _combined(result)
        assert "Project Context" in combined
        assert not _has_robot_state_block(result)

    def test_robot_context_kwarg_is_optional(self) -> None:
        """robot_context is a keyword-only optional param — signature is backward-compat."""
        from vector_os_nano.vcli.prompt import build_system_prompt
        import inspect

        sig = inspect.signature(build_system_prompt)
        params = sig.parameters
        assert "robot_context" in params, "robot_context parameter missing from build_system_prompt"
        assert params["robot_context"].default is None, "robot_context default must be None"


class TestRobotContextErrorHandling:
    """Errors in robot_context.get_context_block() must not crash prompt building."""

    def test_exception_in_get_context_block_is_swallowed(self) -> None:
        """If get_context_block() raises, build_system_prompt() still returns valid output."""
        class _FailingProvider:
            def get_context_block(self) -> dict:
                raise RuntimeError("hardware disconnected")

        result = _build(robot_context=_FailingProvider())
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_empty_block_from_provider_is_excluded(self) -> None:
        """If get_context_block() returns empty dict/falsy, no dedicated robot state block added."""
        class _EmptyProvider:
            def get_context_block(self) -> dict:
                return {}

        result = _build(robot_context=_EmptyProvider())
        assert not _has_robot_state_block(result)
