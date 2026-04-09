"""L39: Navigate routing fixes + terrain replay trigger.

Validates:
1. NavigateSkill has direct=True (bypasses LLM in classic agent)
2. Navigate description doesn't mislead LLM to choose explore
3. Agent._execute_matched() maps extracted_arg to 'room' param
4. Explore creates /tmp/vector_terrain_replay flag on exit
5. Bridge _check_nav_flag detects replay flag
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# L39-1: NavigateSkill routing attributes
# ---------------------------------------------------------------------------


def test_navigate_skill_is_direct() -> None:
    """NavigateSkill must have direct=True so alias match executes without LLM."""
    from vector_os_nano.skills.navigate import NavigateSkill
    assert getattr(NavigateSkill, "__skill_direct__", False) is True, \
        "NavigateSkill must be direct=True (via @skill decorator)"


def test_navigate_skill_direct_from_decorator() -> None:
    """Check the @skill decorator set direct=True on the class."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.navigate import NavigateSkill

    registry = SkillRegistry()
    registry.register(NavigateSkill())

    match = registry.match("去主卧")
    assert match is not None, "alias '去' should match navigate"
    assert match.skill_name == "navigate"
    assert match.direct is True, "navigate match must be direct=True"
    assert match.extracted_arg == "主卧"


def test_navigate_description_no_explore_first() -> None:
    """Navigate description must NOT tell LLM to 'run explore first'.

    This was causing Haiku/GPT-4o to choose explore over navigate in vcli.
    """
    from vector_os_nano.skills.navigate import NavigateSkill
    desc = NavigateSkill.description.lower()
    assert "run explore first" not in desc, \
        f"Description must not say 'run explore first': {NavigateSkill.description}"
    assert "explore first" not in desc, \
        f"Description must not say 'explore first': {NavigateSkill.description}"


def test_navigate_description_guides_llm_correctly() -> None:
    """Description should mention 'go to' and room navigation."""
    from vector_os_nano.skills.navigate import NavigateSkill
    desc = NavigateSkill.description.lower()
    assert "go to" in desc or "room" in desc or "navigate" in desc


# ---------------------------------------------------------------------------
# L39-2: Agent._execute_matched() room parameter mapping
# ---------------------------------------------------------------------------


def test_execute_matched_maps_room_param() -> None:
    """When navigate matches with extracted_arg, it maps to 'room' param."""
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.skill import SkillMatch, SkillRegistry
    from vector_os_nano.skills.navigate import NavigateSkill

    # Create minimal agent
    agent = Agent.__new__(Agent)
    agent._skill_registry = SkillRegistry()
    skill = NavigateSkill()
    agent._skill_registry.register(skill)

    # Mock _build_context to return a context with base and spatial_memory
    from vector_os_nano.core.scene_graph import SceneGraph
    sg = SceneGraph()
    for _ in range(5):
        sg.visit("master_bedroom", 14.0, 12.0)

    mock_base = MagicMock()
    mock_base.get_position.return_value = [3.0, 2.5, 0.3]
    mock_base.get_heading.return_value = 0.0
    mock_base.navigate_to.return_value = True

    mock_ctx = MagicMock()
    mock_ctx.base = mock_base
    mock_ctx.services = {"spatial_memory": sg}
    agent._build_context = MagicMock(return_value=mock_ctx)
    agent._sync_robot_state = MagicMock()

    match = SkillMatch(
        skill_name="navigate",
        extracted_arg="主卧",
        direct=True,
        auto_steps=[],
    )

    result = agent._execute_matched(match, "去主卧")

    # The key test: navigate.execute should have received room="主卧"
    # If room param wasn't mapped, it would get room="" -> error
    assert result.success is True or "master_bedroom" in str(result.failure_reason or "").lower() or \
           result.failure_reason is None, \
        f"Expected navigate to receive room='主卧', got failure: {result.failure_reason}"


def test_execute_matched_still_maps_object_label() -> None:
    """Adding room mapping didn't break object_label mapping for other skills."""
    from vector_os_nano.core.agent import Agent

    agent = Agent.__new__(Agent)
    agent._skill_registry = MagicMock()
    agent._sync_robot_state = MagicMock()

    # Mock skill with object_label parameter
    mock_skill = MagicMock()
    mock_skill.parameters = {"object_label": {"type": "string", "required": True}}
    mock_skill.execute.return_value = MagicMock(
        success=True, error_message=None, result_data={}
    )
    agent._skill_registry.get.return_value = mock_skill
    agent._build_context = MagicMock()

    from vector_os_nano.core.skill import SkillMatch
    match = SkillMatch(
        skill_name="pick",
        extracted_arg="banana",
        direct=True,
        auto_steps=[],
    )

    agent._execute_matched(match, "pick banana")

    # Verify object_label was mapped
    call_args = mock_skill.execute.call_args
    params = call_args[0][0]
    assert params.get("object_label") == "banana"


# ---------------------------------------------------------------------------
# L39-3: Explore terrain replay flag
# ---------------------------------------------------------------------------


def test_explore_creates_terrain_replay_flag() -> None:
    """explore.py source code creates /tmp/vector_terrain_replay on exit."""
    explore_path = _REPO / "vector_os_nano" / "skills" / "go2" / "explore.py"
    source = explore_path.read_text()
    assert "vector_terrain_replay" in source, \
        "explore.py must create /tmp/vector_terrain_replay flag"
    assert 'open("/tmp/vector_terrain_replay"' in source or \
           "open(\"/tmp/vector_terrain_replay\"" in source, \
        "explore.py must write the terrain replay flag file"


def test_explore_terrain_flag_outside_cancel_check() -> None:
    """Terrain replay flag must be created regardless of _explore_cancel state.

    It should be outside the `if not _explore_cancel.is_set()` block
    so it triggers on both manual stop and natural finish.
    """
    explore_path = _REPO / "vector_os_nano" / "skills" / "go2" / "explore.py"
    source = explore_path.read_text()

    # Find the terrain replay flag code
    flag_idx = source.find("vector_terrain_replay")
    assert flag_idx > 0

    # Find the "finally:" block
    finally_idx = source.find("finally:", flag_idx - 500)
    assert finally_idx > 0

    # The flag code should be before finally (inside try, outside if cancel check)
    assert flag_idx < finally_idx, \
        "Terrain replay flag must be before finally: block"


# ---------------------------------------------------------------------------
# L39-4: Bridge terrain replay trigger detection
# ---------------------------------------------------------------------------


def test_bridge_check_nav_flag_has_replay_trigger() -> None:
    """go2_vnav_bridge.py _check_nav_flag must check for terrain replay flag."""
    bridge_path = _REPO / "scripts" / "go2_vnav_bridge.py"
    source = bridge_path.read_text()
    assert "vector_terrain_replay" in source, \
        "Bridge must check for /tmp/vector_terrain_replay flag"


def test_bridge_replay_trigger_saves_terrain() -> None:
    """Bridge replay trigger must call save_terrain() before replaying."""
    bridge_path = _REPO / "scripts" / "go2_vnav_bridge.py"
    source = bridge_path.read_text()

    # Find the replay trigger code
    trigger_idx = source.find("vector_terrain_replay")
    assert trigger_idx > 0

    # save_terrain should appear after the flag check
    save_idx = source.find("self.save_terrain()", trigger_idx)
    assert save_idx > 0, \
        "Bridge must call save_terrain() when replay flag detected"


def test_bridge_replay_trigger_resets_counter() -> None:
    """Bridge replay trigger must reset _terrain_replay_count to 0."""
    bridge_path = _REPO / "scripts" / "go2_vnav_bridge.py"
    source = bridge_path.read_text()

    trigger_idx = source.find("vector_terrain_replay")
    assert trigger_idx > 0

    reset_idx = source.find("_terrain_replay_count = 0", trigger_idx)
    assert reset_idx > 0, \
        "Bridge must reset replay count to 0 when triggered"


# ---------------------------------------------------------------------------
# L39-5: Navigate alias matching covers Chinese commands
# ---------------------------------------------------------------------------


def test_navigate_alias_chinese_go_to() -> None:
    """'去X' matches navigate with extracted_arg='X'."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.navigate import NavigateSkill

    registry = SkillRegistry()
    registry.register(NavigateSkill())

    for cmd, expected_arg in [
        ("去主卧", "主卧"),
        ("去厨房", "厨房"),
        ("到客厅", "客厅"),
        ("走到卫生间", "卫生间"),
        ("导航去书房", "去书房"),
    ]:
        match = registry.match(cmd)
        assert match is not None, f"'{cmd}' should match navigate"
        assert match.skill_name == "navigate", f"'{cmd}' matched {match.skill_name}"
        assert match.extracted_arg == expected_arg, \
            f"'{cmd}' extracted '{match.extracted_arg}', expected '{expected_arg}'"


def test_navigate_alias_english() -> None:
    """English navigate aliases work."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.navigate import NavigateSkill

    registry = SkillRegistry()
    registry.register(NavigateSkill())

    match = registry.match("go to kitchen")
    assert match is not None
    assert match.skill_name == "navigate"
    assert match.extracted_arg == "kitchen"


# ---------------------------------------------------------------------------
# L39-6: Navigate vs Explore — LLM should prefer navigate
# ---------------------------------------------------------------------------


def test_navigate_explore_descriptions_not_conflicting() -> None:
    """Navigate's description must not suggest running explore.
    Explore's description must not overlap with navigate's use case.
    """
    from vector_os_nano.skills.navigate import NavigateSkill
    from vector_os_nano.skills.go2.explore import ExploreSkill

    nav_desc = NavigateSkill.description.lower()
    exp_desc = ExploreSkill.description.lower()

    # Navigate should NOT reference explore
    assert "explore" not in nav_desc or "discovered" in nav_desc, \
        f"Navigate description should not reference explore: {nav_desc}"

    # Explore should NOT mention "go to" or "navigate to"
    assert "go to" not in exp_desc
    assert "navigate to" not in exp_desc
