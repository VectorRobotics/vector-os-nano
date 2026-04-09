"""Level-49 harness tests: ExperienceCompiler + TemplateLibrary (AC-16 to AC-23).

TDD — tests written first, implementation follows.
"""
from __future__ import annotations

import tempfile
import os
import pytest

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.cognitive.experience_compiler import (
    ExperienceCompiler,
    GoalTemplate,
    SubGoalTemplate,
)
from vector_os_nano.vcli.cognitive.template_library import TemplateLibrary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(room: str, obj: str, success: bool = True) -> ExecutionTrace:
    """Create a standard find-object-in-room trace."""
    tree = GoalTree(
        goal=f"find {obj} in {room}",
        sub_goals=(
            SubGoal(
                name=f"reach_{room}",
                description=f"go to {room}",
                verify=f"nearest_room() == '{room}'",
                strategy="navigate_skill",
            ),
            SubGoal(
                name=f"observe_{room}",
                description=f"look around {room}",
                verify="'table' in describe_scene()",
                depends_on=(f"reach_{room}",),
                strategy="look_skill",
            ),
            SubGoal(
                name=f"detect_{obj}",
                description=f"find {obj}",
                verify=f"len(detect_objects('{obj}')) > 0",
                depends_on=(f"observe_{room}",),
            ),
        ),
    )
    steps = tuple(
        StepRecord(
            sub_goal_name=sg.name,
            strategy="navigate_skill",
            success=success,
            verify_result=success,
            duration_sec=10.0,
        )
        for sg in tree.sub_goals
    )
    return ExecutionTrace(
        goal_tree=tree,
        steps=steps,
        success=success,
        total_duration_sec=30.0,
    )


def _make_patrol_trace() -> ExecutionTrace:
    """Create a structurally different trace (4 sub-goals)."""
    tree = GoalTree(
        goal="patrol",
        sub_goals=(
            SubGoal(name="stand_up", description="Stand up", verify="is_standing()"),
            SubGoal(name="walk_forward", description="Walk forward", verify="moved() > 0"),
            SubGoal(name="turn_around", description="Turn 180", verify="heading_changed()"),
            SubGoal(name="sit_down", description="Sit down", verify="is_sitting()"),
        ),
    )
    steps = tuple(
        StepRecord(
            sub_goal_name=sg.name,
            strategy="locomotion",
            success=True,
            verify_result=True,
            duration_sec=5.0,
        )
        for sg in tree.sub_goals
    )
    return ExecutionTrace(
        goal_tree=tree,
        steps=steps,
        success=True,
        total_duration_sec=20.0,
    )


# ---------------------------------------------------------------------------
# GoalTemplate.success_rate property
# ---------------------------------------------------------------------------

def test_success_rate_with_counts() -> None:
    """GoalTemplate.success_rate returns correct fraction."""
    t = GoalTemplate(
        name="test_template",
        description="desc",
        parameters=("room",),
        sub_goal_templates=(),
        success_count=3,
        fail_count=1,
    )
    assert t.success_rate == pytest.approx(0.75)


def test_success_rate_zero_total() -> None:
    """GoalTemplate.success_rate is 0.0 when no executions recorded."""
    t = GoalTemplate(
        name="test_template",
        description="desc",
        parameters=(),
        sub_goal_templates=(),
    )
    assert t.success_rate == 0.0


# ---------------------------------------------------------------------------
# SubGoalTemplate
# ---------------------------------------------------------------------------

def test_sub_goal_template_frozen() -> None:
    """SubGoalTemplate is immutable."""
    sgt = SubGoalTemplate(
        name_pattern="reach_${room}",
        description_pattern="go to ${room}",
        verify_pattern="nearest_room() == '${room}'",
        strategy="navigate_skill",
        timeout_sec=30.0,
        depends_on=(),
        fail_action="",
    )
    with pytest.raises((AttributeError, TypeError)):
        sgt.strategy = "other"  # type: ignore[misc]


def test_sub_goal_template_depends_on_preserved() -> None:
    """SubGoalTemplate preserves depends_on tuple."""
    sgt = SubGoalTemplate(
        name_pattern="observe_${room}",
        description_pattern="look in ${room}",
        verify_pattern="'table' in describe_scene()",
        depends_on=("reach_${room}",),
    )
    assert sgt.depends_on == ("reach_${room}",)


# ---------------------------------------------------------------------------
# AC-18: Failed traces filtered out
# ---------------------------------------------------------------------------

def test_ac18_failed_traces_filtered() -> None:
    """ExperienceCompiler ignores failed traces."""
    traces = [
        _make_trace("kitchen", "cup", success=True),
        _make_trace("bedroom", "book", success=False),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    # Only 1 success trace remains — produces 1 concrete template (no params)
    assert len(templates) == 1


def test_empty_traces_returns_empty() -> None:
    """ExperienceCompiler with empty input returns empty list."""
    compiler = ExperienceCompiler()
    templates = compiler.compile([])
    assert templates == []


def test_all_failed_traces_returns_empty() -> None:
    """ExperienceCompiler with all-failed traces returns empty list."""
    traces = [
        _make_trace("kitchen", "cup", success=False),
        _make_trace("bedroom", "book", success=False),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    assert templates == []


# ---------------------------------------------------------------------------
# AC-16: 3 similar traces → 1 parameterized template
# ---------------------------------------------------------------------------

def test_ac16_three_similar_traces_one_template() -> None:
    """3 traces with same structure produce at least 1 template with sub-goals."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_trace("bedroom", "book"),
        _make_trace("hallway", "key"),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    assert len(templates) >= 1
    # The parameterized template has sub_goal_templates matching original count
    parameterized = [t for t in templates if len(t.parameters) > 0]
    assert len(parameterized) >= 1
    t = parameterized[0]
    assert len(t.sub_goal_templates) == 3  # reach, observe, detect


# ---------------------------------------------------------------------------
# AC-17: Template has parameters (room and/or object detected)
# ---------------------------------------------------------------------------

def test_ac17_template_has_parameters() -> None:
    """Compiled template from similar traces has at least one parameter."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_trace("bedroom", "book"),
        _make_trace("hallway", "key"),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    parameterized = [t for t in templates if len(t.parameters) > 0]
    assert len(parameterized) >= 1
    t = parameterized[0]
    # Should detect room and/or object as parameters
    assert len(t.parameters) >= 1


# ---------------------------------------------------------------------------
# AC-19: Structurally different traces not merged
# ---------------------------------------------------------------------------

def test_ac19_different_structures_not_merged() -> None:
    """Traces with different sub-goal counts produce separate templates."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_patrol_trace(),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    # Each structurally distinct trace group produces its own template
    assert len(templates) == 2


# ---------------------------------------------------------------------------
# Single trace → concrete template (0 parameters)
# ---------------------------------------------------------------------------

def test_single_trace_concrete_template() -> None:
    """A single successful trace produces a concrete template with no parameters."""
    traces = [_make_trace("kitchen", "cup")]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    assert len(templates) == 1
    t = templates[0]
    assert len(t.parameters) == 0
    assert len(t.sub_goal_templates) == 3


def test_concrete_template_name_patterns_match_original() -> None:
    """Concrete template preserves original sub-goal names as patterns."""
    traces = [_make_trace("kitchen", "cup")]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    t = templates[0]
    names = [sgt.name_pattern for sgt in t.sub_goal_templates]
    assert "reach_kitchen" in names
    assert "detect_cup" in names


# ---------------------------------------------------------------------------
# AC-20: TemplateLibrary add + match
# ---------------------------------------------------------------------------

def test_ac20_library_add_and_match() -> None:
    """TemplateLibrary.match() finds a template matching the task description."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_trace("bedroom", "book"),
        _make_trace("hallway", "key"),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    parameterized = [t for t in templates if len(t.parameters) > 0]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        for t in parameterized:
            lib.add(t)

        # Task mentions known room and object — should match
        result = lib.match("find cup in kitchen")
        assert result is not None
        tmpl, params = result
        assert isinstance(tmpl, GoalTemplate)
        assert isinstance(params, dict)
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# AC-21: instantiate produces correct GoalTree
# ---------------------------------------------------------------------------

def test_ac21_instantiate_correct_goal_tree() -> None:
    """instantiate() fills in parameters to produce concrete GoalTree."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_trace("bedroom", "book"),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)
    parameterized = [t for t in templates if len(t.parameters) > 0]
    assert parameterized, "Expected at least one parameterized template"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        lib.add(parameterized[0])

        tree = lib.instantiate(parameterized[0], {"room": "kitchen", "object": "cup"})
        assert tree is not None
        names = [sg.name for sg in tree.sub_goals]
        # After substitution, "kitchen" should appear in some sub-goal
        assert any("kitchen" in name for name in names)
        # Verify expression should also be instantiated
        assert any("kitchen" in sg.verify for sg in tree.sub_goals)
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# AC-22: save + load roundtrip
# ---------------------------------------------------------------------------

def test_ac22_save_load_roundtrip() -> None:
    """TemplateLibrary persists templates and reloads them correctly."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_trace("bedroom", "book"),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        for t in templates:
            lib.add(t)
        lib.save()

        lib2 = TemplateLibrary(persist_path=tmpfile)
        assert len(lib2._templates) == len(lib._templates)
        # Names should match
        original_names = {t.name for t in lib._templates}
        loaded_names = {t.name for t in lib2._templates}
        assert original_names == loaded_names
    finally:
        os.unlink(tmpfile)


def test_save_load_preserves_success_counts() -> None:
    """Roundtrip preserves success_count and fail_count."""
    t = GoalTemplate(
        name="nav_template",
        description="navigate",
        parameters=("room",),
        sub_goal_templates=(
            SubGoalTemplate(
                name_pattern="reach_${room}",
                description_pattern="go to ${room}",
                verify_pattern="nearest_room() == '${room}'",
            ),
        ),
        success_count=5,
        fail_count=2,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        lib.add(t)
        lib.save()

        lib2 = TemplateLibrary(persist_path=tmpfile)
        loaded = lib2._templates[0]
        assert loaded.success_count == 5
        assert loaded.fail_count == 2
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# AC-23: No match → None
# ---------------------------------------------------------------------------

def test_ac23_no_match_returns_none() -> None:
    """TemplateLibrary.match() returns None for unrecognized tasks."""
    traces = [
        _make_trace("kitchen", "cup"),
        _make_trace("bedroom", "book"),
    ]
    compiler = ExperienceCompiler()
    templates = compiler.compile(traces)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        for t in templates:
            lib.add(t)

        result = lib.match("stand up")
        assert result is None
    finally:
        os.unlink(tmpfile)


def test_match_empty_library_returns_none() -> None:
    """match() on an empty library returns None."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        assert lib.match("find cup in kitchen") is None
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# TemplateLibrary add — replace existing by name
# ---------------------------------------------------------------------------

def test_add_replaces_existing_template() -> None:
    """Adding a template with the same name replaces the old one."""
    t1 = GoalTemplate(
        name="my_template",
        description="v1",
        parameters=(),
        sub_goal_templates=(),
        success_count=1,
    )
    t2 = GoalTemplate(
        name="my_template",
        description="v2",
        parameters=(),
        sub_goal_templates=(),
        success_count=99,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        lib.add(t1)
        lib.add(t2)
        assert len(lib._templates) == 1
        assert lib._templates[0].description == "v2"
        assert lib._templates[0].success_count == 99
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# instantiate — concrete template (no parameters)
# ---------------------------------------------------------------------------

def test_instantiate_concrete_template() -> None:
    """instantiate() with empty params on a concrete template preserves original names."""
    sgt = SubGoalTemplate(
        name_pattern="reach_kitchen",
        description_pattern="go to kitchen",
        verify_pattern="nearest_room() == 'kitchen'",
    )
    t = GoalTemplate(
        name="kitchen_template",
        description="Go to kitchen",
        parameters=(),
        sub_goal_templates=(sgt,),
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmpfile = f.name
    try:
        lib = TemplateLibrary(persist_path=tmpfile)
        lib.add(t)
        tree = lib.instantiate(t, {})
        assert tree.sub_goals[0].name == "reach_kitchen"
        assert "kitchen" in tree.sub_goals[0].verify
    finally:
        os.unlink(tmpfile)
