# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Regression — arm VGG step with a VALID skill strategy must NOT fall to 'unmatched'.

Live symptom (SO-101 arm sim, user typed "扫一眼看看"):
  - the decomposer produced a 1-step plan whose step shows ``strategy='scan_skill'``
    (the REPL plan UI printed ``via scan_skill``), yet
  - EXECUTION reported ``no strategy matched for 'unmatched'`` and the run FAILED 0/1,
  - a replan ("scan_to_look") failed the SAME way.

Root cause: ``VGGHarness._execute_step_with_retry`` cleared the sub-goal's explicit
strategy to ``""`` on every retry after attempt 0 ("force selector to pick fresh",
vgg_harness.py ~L489-500). On a BASELESS arm world the empty-strategy selector path has
nothing to route on — the GO2 keyword ladder is gated off (``has_base=False``) and
``skill_registry.match(description)`` returns None for an arm description with no alias —
so it fell straight to StrategySelector.select() Priority 4 =>
StrategyResult("fallback", "unmatched", ...). The retry therefore DESTROYED a perfectly
valid explicit strategy and masked the real attempt-0 failure (here a verify miss, e.g.
``certainty()`` returning 0.0 on a fresh world) behind the opaque ``unmatched`` error.

Fix (vgg_harness ``_retry_strategy``): the retry now PROBES what the cleared strategy
would resolve to. It clears only when the empty-strategy selector path re-derives a
real, actionable executor; when clearing would resolve to ``fallback``/``unmatched`` or a
fail-loud ``invalid`` (a baseless arm world with no keyword ladder / no alias match) it
KEEPS the original explicit strategy, so the step routes to the real skill and surfaces
the honest failure. World-agnostic: driven solely by the selector's resolution.

This is hermetic: no LLM, no mujoco. A stub arm registry whose ``scan`` skill SUCCEEDS
but whose verify fails on attempt 0 exercises the retry path exactly.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness


# ---------------------------------------------------------------------------
# Stub arm registry: a real 'scan' skill that SUCCEEDS, no alias match (an arm
# description never alias-matches), no mobile base.
# ---------------------------------------------------------------------------


class _SkillResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.result_data = {"scanned": True}
        self.error_message = ""


class _ScanSkill:
    name = "scan"

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        # The skill itself always succeeds — the failure on attempt 0 is the
        # VERIFY predicate, exactly like a fresh-world ``certainty()`` returning 0.
        return _SkillResult(success=True)


class _ArmRegistry:
    """Baseless arm registry: only 'scan'; ``match`` never aliases."""

    def __init__(self) -> None:
        self._skills = {"scan": _ScanSkill()}

    def list_skills(self) -> list[str]:
        return ["scan"]

    def get(self, name: str) -> _ScanSkill | None:
        return self._skills.get(name)

    def match(self, _description: str) -> None:
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        return [{"name": "scan", "description": "sweep the workspace", "parameters": {}}]


def _arm_harness() -> tuple[VGGHarness, StrategySelector]:
    reg = _ArmRegistry()
    selector = StrategySelector(skill_registry=reg, has_base=False)
    executor = GoalExecutor(
        strategy_selector=selector,
        verifier=GoalVerifier({}),
        skill_registry=reg,
    )
    harness = VGGHarness(
        decomposer=object(),
        executor=executor,
        selector=selector,
        config=HarnessConfig(max_step_retries=2, max_pipeline_retries=0),
    )
    return harness, selector


# ---------------------------------------------------------------------------
# Sanity: the SAME explicit strategy resolves correctly when NOT cleared.
# (Proves the bug is the retry's strategy-clearing, not the vocab/selector.)
# ---------------------------------------------------------------------------


def test_attempt0_resolves_scan_skill_correctly() -> None:
    """Attempt 0 (explicit ``scan_skill`` kept) routes to the real ``scan`` skill;
    its only failure is the verify miss — NEVER 'unmatched'."""
    harness, _ = _arm_harness()
    sg = SubGoal(
        name="step_1_scan",
        description="扫一眼工作台",
        verify="False",  # fresh-world certainty() == 0 -> verify fails on attempt 0
        strategy="scan_skill",
    )
    step0 = harness._executor._execute_sub_goal(sg)
    assert step0.strategy == "scan"
    assert step0.success is False
    assert "unmatched" not in step0.error  # the honest cause is a verify miss


# ---------------------------------------------------------------------------
# The regression: the retry must NOT turn a valid explicit strategy into
# 'unmatched' on a baseless arm world.
# ---------------------------------------------------------------------------


def test_arm_retry_keeps_valid_strategy_not_unmatched() -> None:
    harness, _ = _arm_harness()
    sg = SubGoal(
        name="step_1_scan",
        description="扫一眼工作台",
        verify="False",  # forces the retry path
        strategy="scan_skill",  # VALID — 'scan' is a registered arm skill
    )

    step = harness._execute_step_with_retry(sg, 0, max_retries=2)

    # The step legitimately fails (verify miss), but it must keep routing to the
    # real skill and surface the REAL cause — never the opaque 'unmatched'
    # produced by throwing the explicit strategy away on a world with no
    # keyword-ladder / alias fallback.
    assert step.strategy != "unmatched", (
        "retry cleared a valid explicit strategy and fell to 'unmatched' on a "
        "baseless arm world"
    )
    assert "unmatched" not in step.error
    assert step.strategy == "scan"


# ---------------------------------------------------------------------------
# Fail-loud is NOT regressed: an UNKNOWN '<x>_skill' strategy still surfaces a
# clear 'invalid' error (never a fabricated phantom skill, never 'unmatched').
# The retry must keep failing loud on the unknown strategy, not paper over it.
# ---------------------------------------------------------------------------


def test_arm_retry_unknown_skill_still_fails_loud() -> None:
    harness, _ = _arm_harness()
    sg = SubGoal(
        name="step_1_bogus",
        description="扫一眼工作台",
        verify="False",
        strategy="bogus_skill",  # NOT a registered arm skill
    )

    step = harness._execute_step_with_retry(sg, 0, max_retries=2)

    assert step.success is False
    # Fail-loud: a clear, named error including the valid set — never the opaque
    # 'unmatched', never a fabricated phantom skill.
    assert "unmatched" not in step.error
    assert "bogus_skill" in step.error
    assert "scan" in step.error  # the valid set is surfaced
