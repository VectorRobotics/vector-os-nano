# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 75 — Phase D: hallucinated strategy is rejected LOUDLY on the replan path.

Bug C (found live on deepseek-v4-flash): a replan emitted a non-existent skill
("scan_to_look"). The decomposer's validator already CLEARS an unknown strategy to
"" and records a fail-loud ``GoalTree.validation_note`` (and does so on EVERY
decompose, including replan, because the harness re-decomposes through the same
validator). But the *cleared* step then SILENTLY reached the executor:

  - on a baseless arm world the empty strategy fell through to the opaque
    ``fallback`` -> ``unmatched`` error (which never named the hallucination), and
  - on a base/registry-less go2 world the empty strategy was RE-ROUTED by keyword
    matching to a phantom skill (e.g. ``scan_to_look`` -> ``look``).

Either way the hallucination reached execution as a phantom rather than being
rejected loudly (rule 8). The fix stamps the offending name on
``SubGoal.cleared_strategy`` when clearing; the selector resolves any such step to
the LOUD ``invalid`` route (clear, named error + valid set) instead of keyword /
registry / fallback routing — on the first decompose AND every replan / retry,
without weakening the existing clear-to-"" + validation_note contract.

Hermetic: no real LLM, no mujoco. Real GoalDecomposer + StrategySelector +
GoalExecutor + GoalVerifier + VGGHarness over a stub arm skill registry.
"""
from __future__ import annotations

import json
from typing import Any

from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness
from vector_os_nano.vcli.cognitive.vocab_from_registry import build_decompose_vocab


# ---------------------------------------------------------------------------
# Arm-like skill registry (no mobile base) + mock backends
# ---------------------------------------------------------------------------

_ARM_SCHEMAS: list[dict[str, Any]] = [
    {"name": "pick", "description": "Pick up an object with the gripper.", "parameters": {}},
    {"name": "place", "description": "Place the held object.", "parameters": {}},
    {"name": "scan", "description": "Sweep the camera across the workspace.", "parameters": {}},
    {"name": "detect", "description": "Detect objects matching a query.", "parameters": {}},
    {"name": "home", "description": "Return the arm to its home pose.", "parameters": {}},
]
_ARM_VERIFY_SIGS = {"gripper_holding": "gripper_holding() -> bool  # gripper holds an object"}
_ARM_SKILL_NAMES = frozenset(s["name"] for s in _ARM_SCHEMAS)


class _SkillResult:
    def __init__(self, success: bool, result_data: dict, error: str = "") -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = error


class _StubSkill:
    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        return _SkillResult(success=True, result_data={"ran": self.name})


class _ArmRegistry:
    def __init__(self) -> None:
        self._skills = {n: _StubSkill(n) for n in _ARM_SKILL_NAMES}

    def list_skills(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> _StubSkill | None:
        return self._skills.get(name)

    def match(self, _description: str) -> None:
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in _ARM_SCHEMAS]


class _FixedBackend:
    """Returns one fixed JSON response on every call."""

    def __init__(self, response: str) -> None:
        self._response = response

    def call(self, messages, tools, system, max_tokens, on_text=None):
        class _R:
            text = self._response

        return _R()


class _ScriptedBackend:
    """Returns a different JSON response per call (first decompose, then replan)."""

    def __init__(self, responses: list[str]) -> None:
        self._it = iter(responses)

    def call(self, messages, tools, system, max_tokens, on_text=None):
        text = next(self._it)

        class _R:
            pass

        r = _R()
        r.text = text
        return r


def _arm_vocab_kwargs() -> dict[str, Any]:
    return build_decompose_vocab(_ARM_SCHEMAS, _ARM_VERIFY_SIGS, has_base=False).as_kwargs()


def _arm_decomposer(backend: Any) -> GoalDecomposer:
    return GoalDecomposer(
        backend, skill_registry=_ArmRegistry(), has_base=False, **_arm_vocab_kwargs()
    )


def _arm_executor() -> GoalExecutor:
    registry = _ArmRegistry()
    return GoalExecutor(
        strategy_selector=StrategySelector(skill_registry=registry, has_base=False),
        verifier=GoalVerifier({}),
        skill_registry=registry,
    )


# A plan a replan might emit: the single step names a non-existent skill.
_HALLUCINATED_PLAN = {
    "goal": "scan the bench",
    "sub_goals": [
        {
            "name": "look_around",
            "description": "scan to look at the bench",
            "verify": "True",
            "strategy": "scan_to_look",  # HALLUCINATION — not an arm skill
            "depends_on": [],
            "strategy_params": {},
        }
    ],
}


# ---------------------------------------------------------------------------
# 1. A (re-)decompose with an unknown strategy clears it + records a note AND
#    stamps the offending name so it can fail loud — same on every decompose.
# ---------------------------------------------------------------------------


def test_hallucinated_strategy_cleared_and_marked() -> None:
    tree = _arm_decomposer(_FixedBackend(json.dumps(_HALLUCINATED_PLAN))).decompose(
        "scan the bench", "world"
    )
    sg = tree.sub_goals[0]
    # Existing contract preserved: strategy cleared to "" + fail-loud note.
    assert sg.strategy == ""
    assert any("scan_to_look" in n and "not valid" in n for n in tree.validation_notes)
    # NEW: the offending name is stamped so the step can fail loud at execution
    # instead of silently re-routing the cleared (empty) strategy.
    assert sg.cleared_strategy == "scan_to_look"


# ---------------------------------------------------------------------------
# 2. The cleared hallucination does NOT reach execution as a phantom — it fails
#    LOUD naming the original strategy + the valid set (baseless arm world).
# ---------------------------------------------------------------------------


def test_cleared_hallucination_fails_loud_at_execution() -> None:
    tree = _arm_decomposer(_FixedBackend(json.dumps(_HALLUCINATED_PLAN))).decompose(
        "scan the bench", "world"
    )
    trace = _arm_executor().execute(tree)

    assert trace.success is False
    step = trace.steps[0]
    assert step.success is False
    # The step surfaces the ORIGINAL hallucinated name, not the opaque 'unmatched'.
    assert step.strategy == "scan_to_look"
    assert "scan_to_look" in step.error
    assert "is not a skill in this world" in step.error
    assert "unmatched" not in step.error
    # Names a real arm skill in the valid set.
    assert "pick" in step.error


# ---------------------------------------------------------------------------
# 3. Selector resolves a cleared+marked step to 'invalid', not keyword/fallback.
#    Covers BOTH a baseless arm world AND a registry-less base (go2) world — the
#    latter previously keyword-re-routed 'scan_to_look' -> phantom 'look' skill.
# ---------------------------------------------------------------------------


def test_selector_routes_cleared_marker_to_invalid_arm() -> None:
    sel = StrategySelector(skill_registry=_ArmRegistry(), has_base=False)
    sg = SubGoal(
        name="look_around",
        description="scan to look",
        verify="True",
        strategy="",  # cleared
        cleared_strategy="scan_to_look",
    )
    result = sel.select(sg)
    assert result.executor_type == "invalid"
    assert result.name == "scan_to_look"
    assert "pick" in result.params["valid_strategies"]


def test_selector_routes_cleared_marker_to_invalid_base_world() -> None:
    # Registry-less base world (default go2 selector). The bug: a cleared empty
    # strategy with name/desc containing 'scan'/'look' keyword-re-routed to a
    # phantom 'look' skill. With the marker it must resolve to 'invalid' instead.
    sel = StrategySelector(has_base=True)
    sg = SubGoal(
        name="scan_to_look",
        description="scan around to look",
        verify="True",
        strategy="",
        cleared_strategy="scan_to_look",
    )
    result = sel.select(sg)
    assert result.executor_type == "invalid"
    assert result.name == "scan_to_look"
    # Never re-routed to the phantom 'look' skill.
    assert result.name != "look"


# ---------------------------------------------------------------------------
# 4. End-to-end through the harness REPLAN: a first plan fails, the replan
#    hallucinates an unknown strategy -> it is cleared + noted AND fails loud at
#    execution (never silently reaches the executor as an 'unmatched' phantom).
# ---------------------------------------------------------------------------


def test_replan_hallucination_fails_loud_end_to_end() -> None:
    # First plan: a valid arm skill but verify=False -> the step FAILS, forcing a
    # pipeline retry (replan). Replan: hallucinates 'scan_to_look'.
    first_plan = {
        "goal": "g",
        "sub_goals": [
            {
                "name": "s0",
                "description": "detect the mug",
                "verify": "False",
                "strategy": "detect_skill",
                "depends_on": [],
                "strategy_params": {},
            }
        ],
    }
    decomposer = _arm_decomposer(
        _ScriptedBackend([json.dumps(first_plan), json.dumps(_HALLUCINATED_PLAN)])
    )
    registry = _ArmRegistry()
    selector = StrategySelector(skill_registry=registry, has_base=False)
    executor = GoalExecutor(
        strategy_selector=selector, verifier=GoalVerifier({}), skill_registry=registry
    )
    harness = VGGHarness(
        decomposer=decomposer,
        executor=executor,
        selector=selector,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=1),
    )
    trace = harness.run("g", "world")

    assert trace.success is False
    # The replan's hallucinated step fails LOUD (named + valid set), never as the
    # opaque 'unmatched' fallback. The replan tree is the one that contains it.
    last = trace.steps[-1]
    assert "unmatched" not in last.error

    # Confirm the replan path applies the SAME validation as the first decompose:
    # a fresh decompose of the hallucinated plan clears + notes + marks it.
    replan_tree = _arm_decomposer(
        _FixedBackend(json.dumps(_HALLUCINATED_PLAN))
    ).decompose("g", "world")
    replan_sg = replan_tree.sub_goals[0]
    assert replan_sg.strategy == ""
    assert replan_sg.cleared_strategy == "scan_to_look"
    assert any(
        "scan_to_look" in n and "not valid" in n for n in replan_tree.validation_notes
    )


# ---------------------------------------------------------------------------
# 5. Retry preserves the loud-invalid marker — a hallucination stays loudly
#    invalid across step retries, never re-routing to a phantom on retry.
# ---------------------------------------------------------------------------


def test_retry_preserves_loud_invalid_marker() -> None:
    registry = _ArmRegistry()
    selector = StrategySelector(skill_registry=registry, has_base=False)
    executor = GoalExecutor(
        strategy_selector=selector, verifier=GoalVerifier({}), skill_registry=registry
    )
    # Pre-built tree carrying a cleared hallucination (as the decomposer would
    # produce). max_step_retries=2 so the step is retried twice.
    tree = GoalTree(
        goal="g",
        sub_goals=(
            SubGoal(
                name="look_around",
                description="scan to look",
                verify="True",
                strategy="",
                cleared_strategy="scan_to_look",
            ),
        ),
    )
    harness = VGGHarness(
        decomposer=object(),
        executor=executor,
        selector=selector,
        config=HarnessConfig(max_step_retries=2, max_pipeline_retries=0),
    )
    trace = harness._execute_with_retry(tree, [])
    step = trace.steps[0]
    assert step.success is False
    # Every attempt stayed loudly invalid — never demoted to 'unmatched'/phantom.
    assert step.strategy == "scan_to_look"
    assert "is not a skill in this world" in step.error
    assert "unmatched" not in step.error


# ---------------------------------------------------------------------------
# 6. Regression: a legitimately strategy-less step (pure check / foreach owner)
#    is NOT marked invalid — the path is byte-identical for non-hallucinations.
# ---------------------------------------------------------------------------


def test_pure_check_step_not_marked_invalid() -> None:
    # A step with NO strategy at all (a pure verify check) — cleared_strategy stays
    # empty and the selector keeps its existing keyword/registry/fallback routing.
    plan = {
        "goal": "g",
        "sub_goals": [
            {
                "name": "check",
                "description": "confirm the gripper holds something",
                "verify": "gripper_holding()",
                "strategy": "",  # intentionally empty — a pure check
                "depends_on": [],
                "strategy_params": {},
            }
        ],
    }
    tree = _arm_decomposer(_FixedBackend(json.dumps(plan))).decompose("g", "world")
    sg = tree.sub_goals[0]
    assert sg.strategy == ""
    assert sg.cleared_strategy == ""  # no hallucination -> not marked
    # The selector does NOT resolve it to 'invalid' (no marker).
    sel = StrategySelector(skill_registry=_ArmRegistry(), has_base=False)
    result = sel.select(sg)
    assert result.executor_type != "invalid"


def test_foreach_owner_not_marked_invalid_even_if_strategy_hallucinated() -> None:
    # A foreach LOOP OWNER legitimately carries an empty strategy. Even if the LLM
    # additionally named a hallucinated strategy on the owner, the owner must NOT be
    # marked invalid (the body does the work + is validated independently).
    plan = {
        "goal": "g",
        "sub_goals": [
            {
                "name": "detect_items",
                "description": "detect every object",
                "verify": "True",
                "strategy": "detect_skill",
                "depends_on": [],
                "strategy_params": {},
            },
            {
                "name": "act_each",
                "description": "act on each item",
                "verify": "True",
                "strategy": "bogus_loop_skill",  # hallucinated on a loop owner
                "depends_on": ["detect_items"],
                "strategy_params": {},
                "foreach": {
                    "source_step": "detect_items",
                    "source_path": "objects",
                    "var": "item",
                    "body": [
                        {
                            "name": "act_item",
                            "description": "pick the item",
                            "verify": "True",
                            "strategy": "pick_skill",
                            "depends_on": [],
                            "strategy_params": {},
                        }
                    ],
                },
            },
        ],
    }
    tree = _arm_decomposer(_FixedBackend(json.dumps(plan))).decompose("g", "world")
    owner = {sg.name: sg for sg in tree.sub_goals}["act_each"]
    assert owner.foreach is not None
    # The loop owner is NOT marked invalid despite the hallucinated strategy on it.
    assert owner.cleared_strategy == ""
