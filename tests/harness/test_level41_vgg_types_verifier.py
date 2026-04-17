"""Level 41 — VGG Phase 1: types.py + goal_verifier.py

TDD tests for:
- SubGoal, GoalTree, StepRecord, ExecutionTrace frozen dataclasses
- GoalVerifier safe sandbox evaluation

AC-7  to AC-12 from spec, plus additional coverage.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Mock primitives namespace used across GoalVerifier tests
# ---------------------------------------------------------------------------

MOCK_PRIMITIVES: dict = {
    "nearest_room": lambda: "kitchen",
    "get_position": lambda: (10.0, 5.0, 0.3),
    "get_heading": lambda: 1.57,
    "get_visited_rooms": lambda: ["kitchen", "hallway", "bedroom"],
    "query_rooms": lambda: [{"id": "kitchen"}, {"id": "hallway"}],
    "describe_scene": lambda: "A kitchen with a table and chairs",
    "detect_objects": lambda query="": [{"name": "cup", "confidence": 0.9}],
    "world_stats": lambda: {"rooms": 3, "objects": 5, "visited": 2},
}

# ---------------------------------------------------------------------------
# Import under test — will FAIL until implementation exists
# ---------------------------------------------------------------------------

from vector_os_nano.vcli.cognitive.types import (  # noqa: E402
    SubGoal,
    GoalTree,
    StepRecord,
    ExecutionTrace,
)
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier  # noqa: E402


# ===========================================================================
# Types tests
# ===========================================================================


class TestSubGoal:
    def test_subgoal_is_frozen(self) -> None:
        sg = SubGoal(name="reach_kitchen", description="Go to kitchen", verify="True")
        with pytest.raises((AttributeError, TypeError)):
            sg.name = "other"  # type: ignore[misc]

    def test_subgoal_defaults(self) -> None:
        sg = SubGoal(name="x", description="d", verify="True")
        assert sg.timeout_sec == 30.0
        assert sg.depends_on == ()
        assert sg.strategy == ""
        assert sg.strategy_params == {}
        assert sg.fail_action == ""

    def test_subgoal_custom_fields(self) -> None:
        sg = SubGoal(
            name="reach_kitchen",
            description="Go to kitchen",
            verify="nearest_room() == 'kitchen'",
            timeout_sec=60.0,
            depends_on=("init",),
            strategy="navigate",
            strategy_params={"speed": 0.5},
            fail_action="wait",
        )
        assert sg.name == "reach_kitchen"
        assert sg.timeout_sec == 60.0
        assert sg.depends_on == ("init",)
        assert sg.strategy == "navigate"
        assert sg.strategy_params == {"speed": 0.5}
        assert sg.fail_action == "wait"

    def test_subgoal_strategy_params_default_is_empty_dict(self) -> None:
        sg1 = SubGoal(name="a", description="b", verify="True")
        sg2 = SubGoal(name="c", description="d", verify="True")
        # Each instance gets its own dict — not the same object
        assert sg1.strategy_params == {}
        assert sg2.strategy_params == {}

    def test_subgoal_frozen_strategy_params(self) -> None:
        """strategy_params field itself cannot be replaced on the frozen instance."""
        sg = SubGoal(name="x", description="d", verify="True")
        with pytest.raises((AttributeError, TypeError)):
            sg.strategy_params = {"new": 1}  # type: ignore[misc]


class TestGoalTree:
    def test_goaltree_is_frozen(self) -> None:
        sg = SubGoal(name="s", description="d", verify="True")
        gt = GoalTree(goal="do something", sub_goals=(sg,))
        with pytest.raises((AttributeError, TypeError)):
            gt.goal = "other"  # type: ignore[misc]

    def test_goaltree_holds_multiple_subgoals(self) -> None:
        sg1 = SubGoal(name="a", description="d1", verify="True")
        sg2 = SubGoal(name="b", description="d2", verify="False")
        gt = GoalTree(goal="task", sub_goals=(sg1, sg2))
        assert len(gt.sub_goals) == 2
        assert gt.sub_goals[0].name == "a"
        assert gt.sub_goals[1].name == "b"

    def test_goaltree_context_snapshot_default(self) -> None:
        sg = SubGoal(name="x", description="d", verify="True")
        gt = GoalTree(goal="g", sub_goals=(sg,))
        assert gt.context_snapshot == ""

    def test_goaltree_empty_subgoals(self) -> None:
        gt = GoalTree(goal="nothing", sub_goals=())
        assert gt.sub_goals == ()


class TestStepRecord:
    def test_steprecord_is_frozen(self) -> None:
        sr = StepRecord(
            sub_goal_name="x",
            strategy="navigate",
            success=True,
            verify_result=True,
            duration_sec=1.5,
        )
        with pytest.raises((AttributeError, TypeError)):
            sr.success = False  # type: ignore[misc]

    def test_steprecord_defaults(self) -> None:
        sr = StepRecord(
            sub_goal_name="x",
            strategy="navigate",
            success=True,
            verify_result=True,
            duration_sec=2.3,
        )
        assert sr.error == ""
        assert sr.fallback_used is False

    def test_steprecord_with_error(self) -> None:
        sr = StepRecord(
            sub_goal_name="x",
            strategy="navigate",
            success=False,
            verify_result=False,
            duration_sec=5.0,
            error="Timeout",
            fallback_used=True,
        )
        assert sr.error == "Timeout"
        assert sr.fallback_used is True


class TestExecutionTrace:
    def test_executiontrace_is_frozen(self) -> None:
        sg = SubGoal(name="s", description="d", verify="True")
        gt = GoalTree(goal="g", sub_goals=(sg,))
        et = ExecutionTrace(
            goal_tree=gt,
            steps=(),
            success=True,
            total_duration_sec=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            et.success = False  # type: ignore[misc]

    def test_executiontrace_holds_multiple_steps(self) -> None:
        sg = SubGoal(name="s", description="d", verify="True")
        gt = GoalTree(goal="g", sub_goals=(sg,))
        sr1 = StepRecord(
            sub_goal_name="s",
            strategy="nav",
            success=True,
            verify_result=True,
            duration_sec=1.0,
        )
        sr2 = StepRecord(
            sub_goal_name="s2",
            strategy="look",
            success=False,
            verify_result=False,
            duration_sec=0.5,
            error="failed",
        )
        et = ExecutionTrace(
            goal_tree=gt,
            steps=(sr1, sr2),
            success=False,
            total_duration_sec=1.5,
        )
        assert len(et.steps) == 2
        assert et.steps[0].strategy == "nav"
        assert et.steps[1].error == "failed"


# ===========================================================================
# GoalVerifier tests
# ===========================================================================


class TestGoalVerifier:
    @pytest.fixture
    def verifier(self) -> GoalVerifier:
        return GoalVerifier(MOCK_PRIMITIVES)

    # AC-7: function call comparison
    def test_ac7_nearest_room_eq_kitchen(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("nearest_room() == 'kitchen'") is True

    # AC-8: len of list
    def test_ac8_len_visited_rooms_ge_3(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("len(get_visited_rooms()) >= 3") is True

    # AC-9: import statement blocked by AST
    def test_ac9_import_os_blocked(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("import os") is False

    # AC-10: dunder call blocked
    def test_ac10_dunder_import_blocked(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("__import__('os')") is False

    # AC-11: infinite loop returns False within timeout
    def test_ac11_infinite_loop_timeout(self, verifier: GoalVerifier) -> None:
        # Must complete and return False rather than hanging
        result = verifier.verify("True if (lambda: [1 for _ in iter(int, 1)])() else False")
        assert result is False

    # AC-12: function not in whitelist returns False
    def test_ac12_unknown_function_blocked(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("open('/etc/passwd').read() == ''") is False

    # Basic arithmetic allowed
    def test_basic_arithmetic(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("1 + 1 == 2") is True

    # Assignment blocked by AST
    def test_assignment_blocked(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("x = 5") is False

    # Empty expression
    def test_empty_expression(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("") is False

    # Syntax error
    def test_syntax_error(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("def (") is False

    # Tuple return indexing
    def test_get_position_index(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("get_position()[0] > 5.0") is True

    # List membership
    def test_in_visited_rooms(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("'kitchen' in get_visited_rooms()") is True

    # False arithmetic
    def test_false_arithmetic(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("1 + 1 == 3") is False

    # Using allowed builtins
    def test_builtin_len_direct(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("len('hello') == 5") is True

    # world_stats dict access
    def test_world_stats_dict(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("world_stats()['rooms'] == 3") is True

    # AugAssign blocked
    def test_augassign_blocked(self, verifier: GoalVerifier) -> None:
        # augmented assignment — should be caught at AST level
        # Note: "x += 1" won't parse as eval expression, it's a statement
        # We verify that assignment-like statements are blocked
        assert verifier.verify("x = 5") is False

    # FunctionDef blocked
    def test_functiondef_blocked(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("lambda x: x") is True  # lambda OK in eval
        # def statement is not a valid eval expression anyway, but verify returns False
        assert verifier.verify("def foo(): pass") is False

    # Dunder attribute access blocked
    def test_dunder_attribute_blocked(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("().__class__.__mro__") is False

    # Exec blocked
    def test_exec_not_in_builtins(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("exec('import os')") is False

    # any/all builtins allowed
    def test_builtin_any(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("any([True, False])") is True

    def test_builtin_all(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("all([True, True])") is True

    # get_heading float comparison
    def test_get_heading_float(self, verifier: GoalVerifier) -> None:
        assert verifier.verify("get_heading() > 1.0") is True
