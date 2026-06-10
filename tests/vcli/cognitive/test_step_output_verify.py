# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Backlog #3 — detect-verify close-the-loop via ``step_output()``.

A detect step's query-less verify (``len(detect_objects()) > 0``) reads a
SEPARATE oracle and FALSE-PASSES when the requested target is absent but the
scene holds other objects. The real fix (Rule 4): the verify consumes the
detect STEP's OWN alias-aware structured output through a kernel-provided
``step_output(path)`` function, injected per-evaluation — no CN/EN alias
table, no second oracle call.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal


class _DirectSelector:
    def select(self, sub_goal: SubGoal) -> StrategyResult:
        return StrategyResult(
            "primitive", sub_goal.strategy, dict(sub_goal.strategy_params)
        )


# ---------------------------------------------------------------------------
# GoalVerifier — per-evaluation extra namespace
# ---------------------------------------------------------------------------


class TestVerifierExtraNamespace:
    def test_evaluate_with_extra_ns(self) -> None:
        v = GoalVerifier({})
        ok, val = v.evaluate(
            "len(step_output('objects')) > 0",
            extra_ns={"step_output": lambda path="": {"objects": [1, 2]}[path]},
        )
        assert ok is True and val is True

    def test_without_extra_ns_fails_safe(self) -> None:
        v = GoalVerifier({})
        ok, val = v.evaluate("len(step_output('objects')) > 0")
        assert ok is False and val is None

    def test_extra_ns_does_not_leak_between_calls(self) -> None:
        v = GoalVerifier({})
        v.evaluate("step_output() is not None", extra_ns={"step_output": lambda: {}})
        ok, _ = v.evaluate("step_output() is not None")
        assert ok is False  # the injection was per-call only


# ---------------------------------------------------------------------------
# Executor — the step's own output drives its verify
# ---------------------------------------------------------------------------


def _executor_with_detect(found: list[dict[str, Any]], scene_objects: list[str]):
    """A detect primitive that found *found*, in a world whose ORACLE sees
    *scene_objects* — the decoupling backlog #3 is about."""

    def detect(**_: Any) -> dict[str, Any]:
        return {"objects": list(found), "count": len(found)}

    namespace = {
        # The query-less oracle predicate: counts EVERYTHING in the scene.
        "detect_objects": lambda query="": list(scene_objects),
    }
    ex = GoalExecutor(
        strategy_selector=_DirectSelector(),
        verifier=GoalVerifier(namespace),
        primitives={"detect": detect},
    )
    ex.blackboard = Blackboard()
    return ex


def _tree(verify: str) -> GoalTree:
    return GoalTree(
        goal="find the apple",
        sub_goals=(
            SubGoal(
                name="find_apple",
                description="detect the apple",
                verify=verify,
                strategy="detect",
            ),
        ),
    )


class TestStepOutputClosesTheLoop:
    def test_old_oracle_verify_false_passes(self) -> None:
        """Sanity of the trap: step found NOTHING, scene has other objects —
        the query-less oracle verify passes anyway (the bug)."""
        ex = _executor_with_detect(found=[], scene_objects=["cup", "pear"])
        trace = ex.execute(_tree("len(detect_objects()) > 0"))
        assert trace.steps[0].verify_result is True  # FALSE PASS

    def test_step_output_verify_fails_honestly_when_step_found_nothing(self) -> None:
        ex = _executor_with_detect(found=[], scene_objects=["cup", "pear"])
        trace = ex.execute(_tree("len(step_output('objects')) > 0"))
        rec = trace.steps[0]
        assert rec.verify_result is False
        assert rec.failure_class == "verify_fail"

    def test_step_output_verify_passes_when_step_found_target(self) -> None:
        ex = _executor_with_detect(
            found=[{"name": "apple_0", "label": "苹果"}], scene_objects=[]
        )
        trace = ex.execute(_tree("len(step_output('objects')) > 0"))
        assert trace.steps[0].verify_result is True

    def test_step_output_path_semantics(self) -> None:
        ex = _executor_with_detect(
            found=[{"name": "apple_0"}], scene_objects=[]
        )
        # whole-dict access and a missing key (fail-safe None -> False)
        trace = ex.execute(_tree("step_output() is not None and step_output('nope') is None"))
        assert trace.steps[0].verify_result is True


# ---------------------------------------------------------------------------
# Decomposer allowlist + skill hint
# ---------------------------------------------------------------------------


class _MockBackend:
    def call(self, *a, **k):  # pragma: no cover
        raise AssertionError("backend must not be called")


class TestKernelAllowlist:
    def test_step_output_always_in_verify_functions(self) -> None:
        from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

        # Even when a world vocab OMITS it, the kernel unions it in.
        d = GoalDecomposer(
            _MockBackend(),
            verify_functions={"holding_object"},
            strategies={"detect_skill"},
        )
        assert "step_output" in d.VERIFY_FUNCTIONS

    def test_validate_accepts_step_output_verify(self) -> None:
        from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

        d = GoalDecomposer(
            _MockBackend(),
            verify_functions={"detect_objects"},
            strategies={"detect_skill"},
        )
        sg = d._validate_sub_goal(
            {
                "name": "find_apple",
                "description": "detect the apple",
                "verify": "len(step_output('objects')) > 0",
                "strategy": "detect_skill",
                "strategy_params": {"query": "苹果"},
            },
            valid_names={"find_apple"},
        )
        assert sg is not None
        assert sg.verify == "len(step_output('objects')) > 0"


class TestDetectSkillHint:
    def test_detect_verify_hint_uses_step_output(self) -> None:
        from vector_os_nano.skills.detect import DetectSkill

        assert DetectSkill.verify_hint == "len(step_output('objects')) > 0"
