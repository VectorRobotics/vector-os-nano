"""Unit tests for vcli prompt.py — TDD RED phase.

Covers:
- test_returns_list_of_dicts: build_system_prompt() returns list[dict] with "type" and "text" keys
- test_includes_role_description: first block contains "Vector" and "robotics"
- test_includes_tool_instructions: mentions tools and permissions
- test_static_sections_have_cache_control: blocks with cache_control key exist
- test_includes_hardware_state: when agent has arm, output includes arm info
- test_includes_skills_list: when agent has skills, lists skill names + descriptions
- test_includes_world_model: when world model has objects, they appear
- test_loads_vector_md: when VECTOR.md exists in cwd, its content is included
- test_handles_no_agent: agent=None -> still returns valid prompt (no hardware section)
- test_handles_no_vector_md: no VECTOR.md -> no error, section omitted

Uses only mock objects — no real hardware imported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockSkill:
    """Minimal skill mock."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


class MockSkillRegistry:
    """Minimal skill registry mock."""

    def __init__(self, skills: list[MockSkill] | None = None) -> None:
        self._skills: dict[str, MockSkill] = {s.name: s for s in (skills or [])}

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def get(self, name: str) -> MockSkill | None:
        return self._skills.get(name)


class MockObjectState:
    """Detected object with position."""

    def __init__(self, label: str, x: float = 0.5, y: float = 0.3, z: float = 0.1) -> None:
        self.label = label
        self.x = x
        self.y = y
        self.z = z


class MockWorldModel:
    """Minimal world model with objects."""

    def __init__(self, objects: list[MockObjectState] | None = None) -> None:
        self._objects = objects or []

    def get_objects(self) -> list[MockObjectState]:
        return list(self._objects)


class MockArm:
    """Minimal arm mock."""

    name: str = "SO101Arm"
    dof: int = 5

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 5


class MockBase:
    """Minimal mobile base mock."""

    name: str = "MuJoCoGo2"


class MockAgent:
    """Minimal agent mock used by prompt builder."""

    def __init__(
        self,
        arm: Any | None = None,
        gripper: Any | None = None,
        base: Any | None = None,
        perception: Any | None = None,
        skills: list[MockSkill] | None = None,
        objects: list[MockObjectState] | None = None,
    ) -> None:
        self._arm = arm
        self._gripper = gripper
        self._base = base
        self._perception = perception
        self._skill_registry = MockSkillRegistry(skills)
        self._world_model = MockWorldModel(objects)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(**kwargs: Any) -> list[dict]:
    from vector_os_nano.vcli.prompt import build_system_prompt

    return build_system_prompt(**kwargs)


# ---------------------------------------------------------------------------
# Core structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_returns_list_of_dicts(self) -> None:
        """build_system_prompt() returns a non-empty list of dicts."""
        result = _build()
        assert isinstance(result, list)
        assert len(result) > 0
        for block in result:
            assert isinstance(block, dict)

    def test_every_block_has_type_and_text(self) -> None:
        """Every block has 'type' and 'text' keys."""
        result = _build()
        for block in result:
            assert "type" in block, f"Block missing 'type': {block}"
            assert "text" in block, f"Block missing 'text': {block}"

    def test_all_type_values_are_text(self) -> None:
        """All blocks have type=='text' (Anthropic text block format)."""
        result = _build()
        for block in result:
            assert block["type"] == "text"

    def test_all_text_values_are_strings(self) -> None:
        """All text values are non-empty strings."""
        result = _build()
        for block in result:
            assert isinstance(block["text"], str)
            assert len(block["text"]) > 0


# ---------------------------------------------------------------------------
# Content checks
# ---------------------------------------------------------------------------


class TestContentPresence:
    def test_includes_role_description(self) -> None:
        """First block contains 'Vector' and 'robotics' (role description)."""
        result = _build()
        first_text = result[0]["text"]
        assert "Vector" in first_text
        assert "robotics" in first_text.lower()

    def test_includes_tool_instructions(self) -> None:
        """Some block mentions 'tool' and 'permission'."""
        result = _build()
        combined = " ".join(b["text"] for b in result).lower()
        assert "tool" in combined
        assert "permission" in combined


# ---------------------------------------------------------------------------
# Cache control
# ---------------------------------------------------------------------------


class TestCacheControl:
    def test_static_sections_have_cache_control(self) -> None:
        """At least one block has a 'cache_control' key (static cacheable section)."""
        result = _build()
        cached = [b for b in result if "cache_control" in b]
        assert len(cached) >= 1

    def test_cache_control_has_ephemeral_type(self) -> None:
        """All cache_control blocks use type='ephemeral'."""
        result = _build()
        for block in result:
            if "cache_control" in block:
                assert block["cache_control"].get("type") == "ephemeral"

    def test_at_least_two_cached_blocks(self) -> None:
        """Both ROLE_PROMPT and TOOL_INSTRUCTIONS are cached (>=2 cached blocks)."""
        result = _build()
        cached = [b for b in result if "cache_control" in b]
        assert len(cached) >= 2


# ---------------------------------------------------------------------------
# Hardware state
# ---------------------------------------------------------------------------


class TestHardwareState:
    def test_includes_hardware_state_when_arm_present(self) -> None:
        """When agent has arm, hardware section appears and mentions arm info."""
        arm = MockArm()
        agent = MockAgent(arm=arm)
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        # Should mention arm or SO101
        assert "arm" in combined.lower() or "SO101" in combined

    def test_hardware_section_absent_when_no_hardware(self) -> None:
        """Agent with no hardware components: no 'Current Hardware' section."""
        agent = MockAgent()  # No arm, gripper, base, perception
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "Current Hardware" not in combined

    def test_hardware_includes_base_when_present(self) -> None:
        """When agent has base, hardware section mentions the base."""
        base = MockBase()
        agent = MockAgent(base=base)
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "MuJoCoGo2" in combined or "base" in combined.lower()


# ---------------------------------------------------------------------------
# Skills list
# ---------------------------------------------------------------------------


class TestSkillsList:
    def test_includes_skills_list_when_skills_present(self) -> None:
        """When agent has skills, skill names appear in the prompt."""
        skills = [
            MockSkill("pick", "Pick up an object"),
            MockSkill("place", "Place an object"),
        ]
        agent = MockAgent(skills=skills)
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "pick" in combined
        assert "place" in combined

    def test_includes_skill_descriptions(self) -> None:
        """Skill descriptions appear alongside skill names."""
        skills = [MockSkill("scan", "Scan the environment for objects")]
        agent = MockAgent(skills=skills)
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "Scan the environment" in combined

    def test_no_skills_section_when_empty_registry(self) -> None:
        """When skill registry is empty, no 'Available Skills' section appears."""
        agent = MockAgent(skills=[])
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "Available Skills" not in combined


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


class TestWorldModel:
    def test_includes_world_model_when_objects_present(self) -> None:
        """When world model has objects, they appear in the prompt."""
        objects = [
            MockObjectState("red_cube", x=0.3, y=0.1, z=0.05),
            MockObjectState("blue_bottle", x=-0.2, y=0.4, z=0.12),
        ]
        agent = MockAgent(objects=objects)
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "red_cube" in combined
        assert "blue_bottle" in combined

    def test_no_world_section_when_no_objects(self) -> None:
        """Empty world model: no 'World Model' section in prompt."""
        agent = MockAgent(objects=[])
        result = _build(agent=agent)
        combined = " ".join(b["text"] for b in result)
        assert "World Model" not in combined


# ---------------------------------------------------------------------------
# VECTOR.md loading
# ---------------------------------------------------------------------------


class TestVectorMd:
    def test_loads_vector_md_from_cwd(self, tmp_path: Path) -> None:
        """When VECTOR.md exists in cwd, its content is included in the prompt."""
        vector_md = tmp_path / "VECTOR.md"
        vector_md.write_text("# My Project\nThis is a robotics project.")

        result = _build(cwd=tmp_path)
        combined = " ".join(b["text"] for b in result)
        assert "My Project" in combined
        assert "robotics project" in combined

    def test_handles_no_vector_md_gracefully(self, tmp_path: Path) -> None:
        """When no VECTOR.md exists, no error and section is omitted."""
        # tmp_path is empty — no VECTOR.md
        result = _build(cwd=tmp_path)
        assert isinstance(result, list)
        combined = " ".join(b["text"] for b in result)
        assert "Project Context" not in combined

    def test_vector_md_section_label(self, tmp_path: Path) -> None:
        """When VECTOR.md exists, section is labelled 'Project Context'."""
        vector_md = tmp_path / "VECTOR.md"
        vector_md.write_text("# Robot info")
        result = _build(cwd=tmp_path)
        combined = " ".join(b["text"] for b in result)
        assert "Project Context" in combined


# ---------------------------------------------------------------------------
# No-agent handling
# ---------------------------------------------------------------------------


class TestNoAgent:
    def test_handles_no_agent(self) -> None:
        """agent=None -> returns valid prompt with no hardware/skills/world sections."""
        result = _build(agent=None)
        assert isinstance(result, list)
        assert len(result) >= 2
        combined = " ".join(b["text"] for b in result)
        # Static sections still present
        assert "Vector" in combined

    def test_no_agent_no_hardware_section(self) -> None:
        """agent=None -> 'Current Hardware' section does not appear."""
        result = _build(agent=None)
        combined = " ".join(b["text"] for b in result)
        assert "Current Hardware" not in combined

    def test_no_agent_no_skills_section(self) -> None:
        """agent=None -> 'Available Skills' section does not appear."""
        result = _build(agent=None)
        combined = " ".join(b["text"] for b in result)
        assert "Available Skills" not in combined
