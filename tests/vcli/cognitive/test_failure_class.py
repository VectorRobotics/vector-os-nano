# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""W2.4 — DETERMINISTIC typed ``failure_class`` into the replan context.

A failed StepRecord carries a bounded, world-agnostic ``failure_class`` derived
ONLY from already-available execution evidence (timeout vs verify-miss vs
execution failure, plus the step's machine-readable ``diagnosis`` and the
resolved executor_type) — NO new model call. The harness threads that typed
class through ``FailureRecord`` into the re-decompose context so the LLM re-plan
can branch on the CLASS instead of parsing the opaque error string.

These tests construct a failure of EACH class through the real GoalExecutor and
assert the resulting ``StepRecord.failure_class``; then assert the class reaches
the replan context (the FailureRecord field AND the decompose context string the
decomposer receives). A SUCCESS step must carry ``failure_class == ""``.

Hermetic: no real LLM, no mujoco — stub selector/verifier/registry/skills.
"""
from __future__ import annotations

import time
from typing import Any

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.types import (
    FAILURE_CLASSES,
    GoalTree,
    StepRecord,
    SubGoal,
    classify_exec_failure,
)
from vector_os_nano.vcli.cognitive.vgg_harness import (
    FailureRecord,
    HarnessConfig,
    VGGHarness,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _SkillResult:
    def __init__(self, success: bool, result_data: dict, error: str = "") -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = error


class _Skill:
    """A skill returning a fixed (success, result_data, error)."""

    def __init__(self, name: str, success: bool, result_data: dict, error: str = "") -> None:
        self.name = name
        self._success = success
        self._result_data = result_data
        self._error = error

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        return _SkillResult(self._success, dict(self._result_data), self._error)


class _SlowSkill:
    """A skill that sleeps long enough to blow a tiny timeout_sec."""

    name = "slow_skill"

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        time.sleep(0.2)
        return _SkillResult(True, {})


class _Registry:
    def __init__(self, skills: dict[str, Any]) -> None:
        self._skills = skills

    def get(self, name: str) -> Any | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return sorted(self._skills)


class _StrategyResult:
    def __init__(self, executor_type: str, name: str, params: dict | None = None) -> None:
        self.executor_type = executor_type
        self.name = name
        self.params = params or {}


class _Selector:
    """Returns a fixed StrategyResult; or raises if configured to."""

    def __init__(self, result: _StrategyResult | None = None, raises: bool = False) -> None:
        self._result = result
        self._raises = raises

    def select(self, sub_goal: Any) -> _StrategyResult:
        if self._raises:
            raise RuntimeError("selector boom")
        return self._result


class _Verifier:
    def __init__(self, value: bool) -> None:
        self._value = value

    def verify(self, expression: str) -> bool:
        return self._value


class _ToolDispatcher:
    """Dispatches a tool as a permission/allowlist failure."""

    def dispatch(self, tool_name: str, args: dict) -> tuple[bool, str]:
        return False, f"tool '{tool_name}' denied by allowlist"


def _plan(strategy: str = "", timeout_sec: float = 30.0, verify: str = "True") -> GoalTree:
    return GoalTree(
        goal="g",
        sub_goals=(
            SubGoal(
                name="step",
                description="a step",
                verify=verify,
                timeout_sec=timeout_sec,
                strategy=strategy,
            ),
        ),
    )


# ===========================================================================
# 1. classify_exec_failure helper — direct unit coverage of the mapping
# ===========================================================================


def test_classify_ik_from_diagnosis() -> None:
    assert classify_exec_failure(diagnosis="ik_unreachable") == "ik_fail"
    # Substring + case-insensitive; embodiment-neutral 'unreachable'.
    assert classify_exec_failure(diagnosis="Target UNREACHABLE") == "ik_fail"


def test_classify_tool_from_executor_type() -> None:
    assert classify_exec_failure(executor_type="tool") == "tool_error"
    # An IK diagnosis on a tool branch is still IK (most specific wins).
    assert classify_exec_failure(executor_type="tool", diagnosis="ik_unreachable") == "ik_fail"


def test_classify_defaults_to_exec_error() -> None:
    assert classify_exec_failure() == "exec_error"
    assert classify_exec_failure(executor_type="skill", diagnosis="object_not_found") == "exec_error"


def test_every_class_is_in_the_closed_set() -> None:
    for c in ("timeout", "verify_fail", "ik_fail", "tool_error", "exec_error"):
        assert c in FAILURE_CLASSES
    assert "" not in FAILURE_CLASSES  # success sentinel excluded


# ===========================================================================
# 2. Each failure class produced through the real GoalExecutor
# ===========================================================================


def test_success_step_has_empty_failure_class() -> None:
    reg = _Registry({"ok": _Skill("ok", True, {})})
    executor = GoalExecutor(
        strategy_selector=_Selector(_StrategyResult("skill", "ok")),
        verifier=_Verifier(True),
        skill_registry=reg,
    )
    trace = executor.execute(_plan(strategy="ok_skill"))
    step = trace.steps[0]
    assert step.success is True
    assert step.failure_class == ""


def test_timeout_failure_class() -> None:
    # Campaign #4 #17: a timed-out step that VERIFIES is an honest PASS with a
    # timing warning — the timeout CLASS is for blown budget + verify miss.
    reg = _Registry({"slow_skill": _SlowSkill()})
    executor = GoalExecutor(
        strategy_selector=_Selector(_StrategyResult("skill", "slow_skill")),
        verifier=_Verifier(False),
        skill_registry=reg,
    )
    # timeout_sec=0.01 << 0.2s sleep, no typical_duration_sec floor -> timeout.
    trace = executor.execute(_plan(strategy="slow_skill_skill", timeout_sec=0.01))
    step = trace.steps[0]
    assert step.success is False
    assert step.failure_class == "timeout"
    assert "timeout" in step.error.lower()


def test_verify_fail_failure_class() -> None:
    # Skill executes fine but the deterministic verify is False -> verify-miss.
    reg = _Registry({"ok": _Skill("ok", True, {})})
    executor = GoalExecutor(
        strategy_selector=_Selector(_StrategyResult("skill", "ok")),
        verifier=_Verifier(False),
        skill_registry=reg,
    )
    trace = executor.execute(_plan(strategy="ok_skill", verify="False"))
    step = trace.steps[0]
    assert step.success is False
    assert step.verify_result is False
    assert step.failure_class == "verify_fail"


def test_ik_fail_failure_class_from_diagnosis() -> None:
    # Skill returns success=False with a machine-readable ik_unreachable diagnosis.
    reg = _Registry({
        "pick": _Skill("pick", False, {"diagnosis": "ik_unreachable"}, error="IK unreachable for grasp"),
    })
    executor = GoalExecutor(
        strategy_selector=_Selector(_StrategyResult("skill", "pick")),
        verifier=_Verifier(True),
        skill_registry=reg,
    )
    trace = executor.execute(_plan(strategy="pick_skill"))
    step = trace.steps[0]
    assert step.success is False
    assert step.failure_class == "ik_fail"


def test_tool_error_failure_class() -> None:
    # The tool branch fails (permission/allowlist) -> tool_error.
    executor = GoalExecutor(
        strategy_selector=_Selector(
            _StrategyResult("tool", "bash", {"tool": "bash", "args": {}})
        ),
        verifier=_Verifier(True),
        tool_dispatcher=_ToolDispatcher(),
    )
    trace = executor.execute(_plan(strategy="bash"))
    step = trace.steps[0]
    assert step.success is False
    assert step.failure_class == "tool_error"


def test_exec_error_failure_class_generic_skill_failure() -> None:
    # Skill returns success=False with a non-IK diagnosis -> generic exec_error.
    reg = _Registry({
        "pick": _Skill("pick", False, {"diagnosis": "object_not_found"}, error="no such object"),
    })
    executor = GoalExecutor(
        strategy_selector=_Selector(_StrategyResult("skill", "pick")),
        verifier=_Verifier(True),
        skill_registry=reg,
    )
    trace = executor.execute(_plan(strategy="pick_skill"))
    step = trace.steps[0]
    assert step.success is False
    assert step.failure_class == "exec_error"


def test_exec_error_failure_class_selector_raises() -> None:
    # The selector raises before any strategy runs -> exec_error (via _make_step).
    executor = GoalExecutor(
        strategy_selector=_Selector(raises=True),
        verifier=_Verifier(True),
    )
    trace = executor.execute(_plan(strategy="anything_skill"))
    step = trace.steps[0]
    assert step.success is False
    assert step.failure_class == "exec_error"
    assert "selector boom" in step.error


# ===========================================================================
# 3. failure_class reaches the replan context
# ===========================================================================


class _RecordingDecomposer:
    """Captures the (task, context) it is asked to decompose."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def decompose(self, task: str, context: str) -> GoalTree | None:
        self.calls.append((task, context))
        return None  # stop the harness after capturing the replan context


def test_failure_class_threads_into_failure_record_and_context() -> None:
    # An ik_unreachable skill failure -> ik_fail on the StepRecord -> FailureRecord
    # -> the decompose context the decomposer receives on replan.
    reg = _Registry({
        "pick": _Skill("pick", False, {"diagnosis": "ik_unreachable"}, error="IK unreachable"),
    })
    selector = _Selector(_StrategyResult("skill", "pick"))
    executor = GoalExecutor(
        strategy_selector=selector, verifier=_Verifier(True), skill_registry=reg
    )
    decomposer = _RecordingDecomposer()
    harness = VGGHarness(
        decomposer=decomposer,
        executor=executor,
        selector=selector,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=1),
    )
    tree = _plan(strategy="pick_skill")
    harness.run("pick the mug", "world", goal_tree=tree)

    # The leaf failure is recorded with the typed class threaded through.
    # (run() routes failures into _decompose_with_context on pipeline retry.)
    assert decomposer.calls, "decomposer should have been asked to re-plan"
    _, replan_context = decomposer.calls[-1]
    # The typed tag appears in the human-readable failure line ...
    assert "[ik_fail]" in replan_context
    # ... and the deterministic adaptation hint block names the class.
    assert "ik_fail:" in replan_context
    assert "alternate grasp" in replan_context


def test_format_failure_line_includes_typed_tag() -> None:
    line = VGGHarness._format_failure_line(
        FailureRecord(
            sub_goal_name="grab", strategy_tried="pick", error="boom",
            step_index=0, failure_class="ik_fail",
        )
    )
    assert "grab" in line and "pick" in line and "boom" in line
    assert "[ik_fail]" in line


def test_format_failure_line_no_tag_when_untyped() -> None:
    # A failure with no typed class (e.g. an abort) renders the legacy line shape.
    line = VGGHarness._format_failure_line(
        FailureRecord(sub_goal_name="x", strategy_tried="s", error="e", step_index=0)
    )
    assert "[" not in line  # no empty tag
    assert line == "  - x: s failed (e)"


def test_failure_class_hints_distinct_classes_only() -> None:
    failures = [
        FailureRecord("a", "s1", "e", 0, failure_class="timeout"),
        FailureRecord("b", "s2", "e", 1, failure_class="ik_fail"),
        FailureRecord("c", "s3", "e", 2, failure_class="timeout"),  # dup -> once
        FailureRecord("d", "s4", "e", 3, failure_class=""),          # untyped -> skip
    ]
    block = VGGHarness._failure_class_hints(failures)
    assert block.count("timeout:") == 1
    assert "ik_fail:" in block
    # First-seen order preserved.
    assert block.index("timeout:") < block.index("ik_fail:")


def test_failure_class_hints_empty_when_all_untyped() -> None:
    failures = [FailureRecord("a", "s", "e", 0)]  # default failure_class == ""
    assert VGGHarness._failure_class_hints(failures) == ""


def test_failure_record_in_harness_carries_class_end_to_end() -> None:
    # Drive the harness's _execute_with_retry directly and inspect the failures
    # list it populates — the FailureRecord must carry the StepRecord's class.
    reg = _Registry({
        "pick": _Skill("pick", False, {"diagnosis": "ik_unreachable"}, error="IK unreachable"),
    })
    selector = _Selector(_StrategyResult("skill", "pick"))
    executor = GoalExecutor(
        strategy_selector=selector, verifier=_Verifier(True), skill_registry=reg
    )
    harness = VGGHarness(
        decomposer=object(),
        executor=executor,
        selector=selector,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=0),
    )
    failures: list[FailureRecord] = []
    harness._execute_with_retry(_plan(strategy="pick_skill"), failures)
    assert failures, "the failed step should have been recorded"
    assert failures[-1].failure_class == "ik_fail"
