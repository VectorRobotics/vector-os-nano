"""Unit tests for dynamic constraint injection in build_planning_prompt().

TDD — written before implementation. Tests verify that:
- Enum constraints from skill schemas appear in the prompt
- Available object labels from world state appear in the prompt
- Gripper held-object state is reflected in the prompt
- Empty world state produces appropriate constraint text
- Empty gripper shows must-pick message
"""
from __future__ import annotations

from vector_os_nano.llm.prompts import build_planning_prompt


def test_planning_prompt_includes_enum_constraints():
    """Prompt contains enum values from skill schemas."""
    schemas = [
        {"name": "pick", "parameters": {"mode": {"type": "string", "enum": ["drop", "hold"]}}},
        {"name": "place", "parameters": {"location": {"type": "string", "enum": ["front", "left", "right"]}}},
    ]
    world = {"objects": [], "robot": {}}
    prompt = build_planning_prompt(schemas, world)
    assert "VALID VALUES for pick.mode: drop, hold" in prompt
    assert "VALID VALUES for place.location: front, left, right" in prompt


def test_planning_prompt_includes_available_objects():
    """Prompt contains object labels from world state."""
    schemas = [{"name": "pick", "parameters": {}}]
    world = {"objects": [{"label": "banana"}, {"label": "mug"}], "robot": {}}
    prompt = build_planning_prompt(schemas, world)
    assert "AVAILABLE OBJECTS: banana, mug" in prompt
    assert "MUST be one of these exact names" in prompt


def test_planning_prompt_includes_gripper_state():
    """When robot holds an object, prompt reflects it."""
    schemas = [{"name": "pick", "parameters": {}}]
    world = {"objects": [{"label": "mug"}], "robot": {"held_object": "banana_0", "gripper_state": "holding"}}
    prompt = build_planning_prompt(schemas, world)
    assert "holding banana_0" in prompt
    assert "Can place directly" in prompt


def test_planning_prompt_empty_world():
    """Empty world state produces appropriate constraint text."""
    schemas = [{"name": "pick", "parameters": {}}]
    world = {"objects": [], "robot": {"gripper_state": "open"}}
    prompt = build_planning_prompt(schemas, world)
    assert "AVAILABLE OBJECTS: none detected" in prompt
    assert "MUST start with scan + detect" in prompt


def test_planning_prompt_gripper_empty():
    """Empty gripper shows must-pick message."""
    schemas = []
    world = {"objects": [{"label": "cup"}], "robot": {"gripper_state": "open"}}
    prompt = build_planning_prompt(schemas, world)
    assert "not holding anything" in prompt
    assert "Must pick before place" in prompt
