"""Level 43 — VGG Phase 1: GoalDecomposer

TDD tests for:
- GoalDecomposer.decompose() — LLM-backed task decomposition
- Prompt construction (system prompt includes strategies + verify fns)
- JSON extraction from markdown fences
- Validation: verify expressions, strategies, depends_on, count limits
- Graceful fallback on JSON parse failure

AC-1 to AC-6 from spec, plus additional coverage.
"""
from __future__ import annotations

import json
import pytest

# ---------------------------------------------------------------------------
# Imports under test — will FAIL until implementation exists
# ---------------------------------------------------------------------------

from vector_os_nano.vcli.cognitive.types import SubGoal, GoalTree  # noqa: E402
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer  # noqa: E402


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------

class MockBackend:
    """Mock LLMBackend that returns a fixed response string."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict] = []
        self.last_system: list[dict] = []

    def call(
        self,
        messages,
        tools,
        system,
        max_tokens,
        on_text=None,
    ):
        """Record call args and return fixed response via a mock LLMResponse."""
        self.last_messages = messages
        self.last_system = system

        class _FakeResponse:
            text = self._response

        return _FakeResponse()

    # Allow attribute access on self
    @property
    def response(self):
        return self._response


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_valid_json(goal: str, sub_goals: list[dict]) -> str:
    """Build a valid GoalTree JSON string."""
    return json.dumps({"goal": goal, "sub_goals": sub_goals})


KITCHEN_CUP_SUBGOALS = [
    {
        "name": "reach_kitchen",
        "description": "导航到厨房",
        "verify": "nearest_room() == 'kitchen'",
        "strategy": "navigate_skill",
        "timeout_sec": 60,
    },
    {
        "name": "observe_table",
        "description": "观察厨房",
        "verify": "'table' in describe_scene()",
        "depends_on": ["reach_kitchen"],
        "strategy": "look_skill",
        "timeout_sec": 15,
    },
    {
        "name": "detect_cup",
        "description": "检测杯子",
        "verify": "len(detect_objects('cup')) > 0",
        "depends_on": ["observe_table"],
        "timeout_sec": 10,
    },
]

KITCHEN_CUP_JSON = _make_valid_json("去厨房看看有没有杯子", KITCHEN_CUP_SUBGOALS)


# ===========================================================================
# AC-1: Valid GoalTree JSON for 去厨房看看有没有杯子
# ===========================================================================

class TestAC1ValidDecomposition:
    def test_returns_goal_tree(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "world_context: empty")
        assert isinstance(result, GoalTree)

    def test_goal_field_matches(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "world_context: empty")
        assert result.goal == "去厨房看看有没有杯子"

    def test_has_three_sub_goals(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "world_context: empty")
        assert len(result.sub_goals) == 3

    def test_sub_goal_names_present(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "world_context: empty")
        names = {sg.name for sg in result.sub_goals}
        assert "reach_kitchen" in names
        assert "observe_table" in names
        assert "detect_cup" in names

    def test_sub_goals_are_frozen(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "world_context: empty")
        with pytest.raises((AttributeError, TypeError)):
            result.sub_goals[0].name = "mutated"  # type: ignore[misc]


# ===========================================================================
# AC-2: Single sub_goal for simple task "站起来"
# ===========================================================================

class TestAC2SingleSubGoal:
    STAND_JSON = json.dumps({
        "goal": "站起来",
        "sub_goals": [
            {
                "name": "stand_up",
                "description": "让机器人站立",
                "verify": "get_position()[2] > 0.3",
                "strategy": "stand_skill",
                "timeout_sec": 10,
            }
        ],
    })

    def test_simple_task_has_one_sub_goal(self):
        backend = MockBackend(self.STAND_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("站起来", "world_context: standing")
        assert len(result.sub_goals) == 1

    def test_single_sub_goal_correct_strategy(self):
        backend = MockBackend(self.STAND_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("站起来", "world_context: standing")
        assert result.sub_goals[0].strategy == "stand_skill"


# ===========================================================================
# AC-3: Valid Python verify expressions (ast.parse succeeds)
# ===========================================================================

class TestAC3ValidVerifyExpressions:
    def test_all_verify_are_valid_python(self):
        import ast
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "world_context: empty")
        for sg in result.sub_goals:
            # Should not raise
            ast.parse(sg.verify, mode="eval")

    def test_invalid_verify_expression_removed(self):
        """Sub-goal with unparseable verify should be dropped or have verify cleared."""
        bad_json = json.dumps({
            "goal": "test",
            "sub_goals": [
                {
                    "name": "good_step",
                    "description": "desc",
                    "verify": "nearest_room() == 'kitchen'",
                    "timeout_sec": 10,
                },
                {
                    "name": "bad_step",
                    "description": "bad verify",
                    "verify": "def bad_syntax(",
                    "timeout_sec": 10,
                },
            ],
        })
        backend = MockBackend(bad_json)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("test", "")
        # bad_step must be removed or have verify cleared
        for sg in result.sub_goals:
            if sg.name == "bad_step":
                # If kept, verify must be parseable
                import ast
                ast.parse(sg.verify, mode="eval")


# ===========================================================================
# AC-4: Strategy fields from KNOWN_STRATEGIES or empty
# ===========================================================================

class TestAC4KnownStrategies:
    def test_known_strategies_kept(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "")
        for sg in result.sub_goals:
            if sg.strategy:
                assert sg.strategy in GoalDecomposer.KNOWN_STRATEGIES, (
                    f"Unknown strategy: {sg.strategy!r}"
                )

    def test_unknown_strategy_cleared(self):
        """LLM returns unknown strategy 'fly_to_moon' → cleared to ''."""
        bad_strategy_json = json.dumps({
            "goal": "test",
            "sub_goals": [
                {
                    "name": "step1",
                    "description": "desc",
                    "verify": "nearest_room() == 'kitchen'",
                    "strategy": "fly_to_moon",
                    "timeout_sec": 10,
                }
            ],
        })
        backend = MockBackend(bad_strategy_json)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("test", "")
        assert result.sub_goals[0].strategy == ""

    def test_empty_strategy_kept(self):
        """Empty strategy string is always valid."""
        no_strategy_json = json.dumps({
            "goal": "test",
            "sub_goals": [
                {
                    "name": "step1",
                    "description": "desc",
                    "verify": "nearest_room() == 'kitchen'",
                    "strategy": "",
                    "timeout_sec": 10,
                }
            ],
        })
        backend = MockBackend(no_strategy_json)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("test", "")
        assert result.sub_goals[0].strategy == ""


# ===========================================================================
# AC-5: Max sub_goals limit = 8
# ===========================================================================

class TestAC5MaxSubGoals:
    def _make_ten_subgoals_json(self) -> str:
        sub_goals = [
            {
                "name": f"step_{i}",
                "description": f"step {i}",
                "verify": "nearest_room() == 'kitchen'",
                "timeout_sec": 10,
            }
            for i in range(10)
        ]
        return json.dumps({"goal": "ten steps", "sub_goals": sub_goals})

    def test_ten_sub_goals_truncated_to_eight(self):
        backend = MockBackend(self._make_ten_subgoals_json())
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("ten steps", "")
        assert len(result.sub_goals) <= GoalDecomposer.MAX_SUB_GOALS

    def test_max_sub_goals_constant_is_eight(self):
        assert GoalDecomposer.MAX_SUB_GOALS == 8


# ===========================================================================
# AC-6: Graceful fallback on garbage JSON
# ===========================================================================

class TestAC6GarbageFallback:
    def test_garbage_response_returns_goal_tree(self):
        backend = MockBackend("This is not JSON at all!!! @#$%")
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("some task", "context")
        assert isinstance(result, GoalTree)

    def test_garbage_response_has_one_sub_goal(self):
        backend = MockBackend("totally invalid")
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("some task", "context")
        assert len(result.sub_goals) == 1

    def test_garbage_fallback_sub_goal_contains_task(self):
        """Fallback sub_goal description should reference the original task."""
        backend = MockBackend("{bad json")
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("go to kitchen", "context")
        # Either goal or description mentions the original task
        task_referenced = (
            "go to kitchen" in result.goal
            or "go to kitchen" in result.sub_goals[0].description
            or "go to kitchen" in result.sub_goals[0].name
        )
        assert task_referenced

    def test_fallback_sub_goal_has_valid_verify(self):
        """Fallback sub_goal must have a parseable verify expression."""
        import ast
        backend = MockBackend("{ not json")
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("sit down", "ctx")
        ast.parse(result.sub_goals[0].verify, mode="eval")


# ===========================================================================
# Additional tests: JSON in markdown fences
# ===========================================================================

class TestMarkdownFenceExtraction:
    def test_json_in_backtick_fence(self):
        wrapped = f"```json\n{KITCHEN_CUP_JSON}\n```"
        backend = MockBackend(wrapped)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "")
        assert isinstance(result, GoalTree)
        assert len(result.sub_goals) == 3

    def test_json_in_plain_backtick_fence(self):
        wrapped = f"```\n{KITCHEN_CUP_JSON}\n```"
        backend = MockBackend(wrapped)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "")
        assert isinstance(result, GoalTree)

    def test_json_with_leading_text(self):
        wrapped = f"Here is the decomposition:\n\n{KITCHEN_CUP_JSON}"
        backend = MockBackend(wrapped)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "")
        assert isinstance(result, GoalTree)


# ===========================================================================
# Additional tests: verify function whitelist
# ===========================================================================

class TestVerifyFunctionWhitelist:
    def test_import_statement_in_verify_rejected(self):
        """verify with 'import os' is invalid — sub_goal should be dropped."""
        import_json = json.dumps({
            "goal": "test",
            "sub_goals": [
                {
                    "name": "safe_step",
                    "description": "desc",
                    "verify": "nearest_room() == 'kitchen'",
                    "timeout_sec": 10,
                },
                {
                    "name": "dangerous_step",
                    "description": "desc",
                    "verify": "__import__('os').system('rm -rf /')",
                    "timeout_sec": 10,
                },
            ],
        })
        backend = MockBackend(import_json)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("test", "")
        # dangerous_step must not appear with the malicious verify
        for sg in result.sub_goals:
            if sg.name == "dangerous_step":
                # verify must be cleared/emptied or sub_goal removed
                assert "__import__" not in sg.verify and "os.system" not in sg.verify

    def test_unknown_function_call_rejected(self):
        """verify calling a function not in VERIFY_FUNCTIONS is invalid."""
        unknown_fn_json = json.dumps({
            "goal": "test",
            "sub_goals": [
                {
                    "name": "bad_fn",
                    "description": "desc",
                    "verify": "unknown_function() == True",
                    "timeout_sec": 10,
                }
            ],
        })
        backend = MockBackend(unknown_fn_json)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("test", "")
        # Sub_goal with unknown function should be dropped or verify cleared
        for sg in result.sub_goals:
            if sg.name == "bad_fn":
                # If kept, verify must be empty or cleared
                assert sg.verify == "" or sg.verify is None or "unknown_function" not in sg.verify


# ===========================================================================
# Additional tests: depends_on validation
# ===========================================================================

class TestDependsOnValidation:
    def test_valid_depends_on_preserved(self):
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("去厨房看看有没有杯子", "")
        observe = next(sg for sg in result.sub_goals if sg.name == "observe_table")
        assert "reach_kitchen" in observe.depends_on

    def test_invalid_depends_on_reference_removed(self):
        """depends_on references a name not in the tree → reference removed."""
        invalid_dep_json = json.dumps({
            "goal": "test",
            "sub_goals": [
                {
                    "name": "step1",
                    "description": "desc",
                    "verify": "nearest_room() == 'kitchen'",
                    "depends_on": ["nonexistent_step"],
                    "timeout_sec": 10,
                }
            ],
        })
        backend = MockBackend(invalid_dep_json)
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("test", "")
        step1 = next(sg for sg in result.sub_goals if sg.name == "step1")
        assert "nonexistent_step" not in step1.depends_on


# ===========================================================================
# Additional tests: empty task string
# ===========================================================================

class TestEdgeCases:
    def test_empty_task_returns_valid_goal_tree(self):
        backend = MockBackend(json.dumps({
            "goal": "",
            "sub_goals": [
                {
                    "name": "noop",
                    "description": "no-op",
                    "verify": "world_stats() is not None",
                    "timeout_sec": 5,
                }
            ],
        }))
        decomposer = GoalDecomposer(backend)
        result = decomposer.decompose("", "")
        assert isinstance(result, GoalTree)
        assert len(result.sub_goals) >= 1


# ===========================================================================
# Additional tests: prompt construction
# ===========================================================================

class TestPromptConstruction:
    def test_system_prompt_includes_strategies(self):
        """System prompt should list all KNOWN_STRATEGIES."""
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        decomposer.decompose("去厨房看看有没有杯子", "some world context")

        # Reconstruct system as string from last call
        system_text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in (backend.last_system or [])
        )
        for strategy in GoalDecomposer.KNOWN_STRATEGIES:
            assert strategy in system_text, (
                f"Strategy {strategy!r} missing from system prompt"
            )

    def test_system_prompt_includes_verify_functions(self):
        """System prompt should list all VERIFY_FUNCTIONS."""
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        decomposer.decompose("去厨房看看有没有杯子", "some world context")

        system_text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in (backend.last_system or [])
        )
        for fn in GoalDecomposer.VERIFY_FUNCTIONS:
            assert fn in system_text, (
                f"Verify function {fn!r} missing from system prompt"
            )

    def test_user_message_includes_task_and_context(self):
        """User message must include both task and world_context."""
        backend = MockBackend(KITCHEN_CUP_JSON)
        decomposer = GoalDecomposer(backend)
        decomposer.decompose("go to kitchen", "map: room A connected to room B")

        user_messages = [
            m for m in backend.last_messages
            if m.get("role") == "user"
        ]
        assert len(user_messages) >= 1
        user_text = " ".join(
            str(m.get("content", "")) for m in user_messages
        )
        assert "go to kitchen" in user_text
        assert "map: room A connected to room B" in user_text
