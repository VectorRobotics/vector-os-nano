# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC8) — the verified loop is VISIBLE through the CLI/engine path.

INC5 built the pure EXPORT VIEW (step_view / run_snapshot); INC6/INC7 proved the
verified tabletop chain runs (static, then NL-decomposed). INC8 closes the
visibility gap: the live CLI must DRIVE the observation surface — fan each
sub-goal's per-step view out through the engine's ``on_vgg_step_view`` sink, and
on run-complete RENDER the goal tree + per-step PASS/FAIL + replan notes + the
overall outcome as readable text (no GUI).

This test exercises the production wiring deterministically (mock backend — NO
live LLM):

  - A tabletop arm chain runs through the REAL ``GoalExecutor`` whose ``on_step``
    is the engine's ``_on_vgg_step`` (exactly how the engine drives it). The
    engine's ``on_vgg_step_view`` sink therefore receives one JSON-safe EXPORT
    VIEW per step, live.
  - ``render_step_view`` of each emitted view carries the sub-goal, its strategy,
    the verify predicate, and a stable PASS/FAIL marker.
  - ``render_run_snapshot`` of the run-complete snapshot CONTAINS the goal tree
    (every sub-goal + its verify predicate), a PASS/FAIL marker per step, any
    validation note, and the outcome. The snapshot round-trips through
    ``json.dumps``.

Assertions are on STRUCTURE and the stable markers, not on exact prose. Hermetic:
no MuJoCo, no network — deterministic stubs sharing the sim-oracle surface the
playground predicates read.
"""

from __future__ import annotations

import json
from typing import Any

from vector_os_nano.playground import PlaygroundWorld
from tests.harness.fake_gt import bind_fake_sim
from vector_os_nano.embodiments import registry

# The SO-101 home pose (its manifest stance; the arm_at_home oracle reads it there).
_HOME_JOINTS = tuple(registry.load("so101").stance.values())
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.oracle_registry import evidence_names
from vector_os_nano.vcli.cognitive.observation import (
    render_run_snapshot,
    render_step_view,
    run_snapshot,
)
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.engine import VectorEngine

_EE = [0.20, 0.10, 0.25]
_REGION = (0.0, 0.0, 0.5, 0.5)
_NOT_HOME_JOINTS: tuple[float, ...] = tuple(j + 0.5 for j in _HOME_JOINTS)


# ---------------------------------------------------------------------------
# Deterministic stub arm + gripper + skill registry — the mutable sim oracle
# the chain advances (same surface the playground predicates read).
# ---------------------------------------------------------------------------


class _StubArm:
    def __init__(self) -> None:
        self._joints: list[float] = list(_NOT_HOME_JOINTS)
        self._objects: dict[str, list[float]] = {"mug": [0.21, 0.10, 0.06]}

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def fk(self, joint_positions: list[float]):
        return list(_EE), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def go_home(self) -> None:
        self._joints = list(_HOME_JOINTS)

    def lift(self, name: str) -> None:
        self._objects[name] = [_EE[0] + 0.01, _EE[1], _EE[2]]

    def place(self, name: str, xy: tuple[float, float]) -> None:
        self._objects[name] = [xy[0], xy[1], 0.06]


class _StubGripper:
    def __init__(self) -> None:
        self._holding = False

    def is_holding(self) -> bool:
        return self._holding

    def close(self) -> None:
        self._holding = True

    def open(self) -> None:
        self._holding = False


_ARM_SKILL_NAMES = frozenset({"home", "detect", "pick", "place"})


class _SkillResult:
    def __init__(self, success: bool, result_data: dict) -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = ""


class _OracleSkill:
    def __init__(self, name: str, arm: _StubArm, gripper: _StubGripper) -> None:
        self.name = name
        self._arm = arm
        self._gripper = gripper

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        if self.name == "home":
            self._arm.go_home()
        elif self.name == "pick":
            self._gripper.close()
            self._arm.lift("mug")
        elif self.name == "place":
            self._arm.place("mug", (_REGION[0] + 0.1, _REGION[1] + 0.1))
            self._gripper.open()
        return _SkillResult(success=True, result_data={"ran": self.name})


class _ArmRegistry:
    def __init__(self, arm: _StubArm, gripper: _StubGripper) -> None:
        self._skills = {n: _OracleSkill(n, arm, gripper) for n in _ARM_SKILL_NAMES}

    def list_skills(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str):
        return self._skills.get(name)

    def match(self, _description: str):
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        return []


# A tabletop arm plan whose verify predicates flip False -> True as the chain
# advances the shared oracle (real evidence, deterministic).
_TREE = GoalTree(
    goal="pick up the mug",
    sub_goals=(
        SubGoal(name="home_arm", description="home the arm", verify="arm_at_home()", strategy="home_skill"),
        SubGoal(
            name="detect_mug",
            description="detect the mug",
            verify="object_on('mug', 'table')",
            strategy="detect_skill",
            depends_on=("home_arm",),
        ),
        SubGoal(
            name="grasp_mug",
            description="pick up the mug",
            verify="holding_object()",
            strategy="pick_skill",
            depends_on=("detect_mug",),
        ),
        SubGoal(
            name="place_mug",
            description="place the mug in the target region",
            verify="object_at('mug', 0.1, 0.1)",
            strategy="place_skill",
            depends_on=("grasp_mug",),
        ),
    ),
    # A replan note as the harness would thread it in — must reach the render.
    validation_notes=("strategy 'scan_360' is not valid; cleared (valid: pick_skill, ...)",),
)


def _engine_with_view_sink(arm: _StubArm, gripper: _StubGripper, view_sink) -> VectorEngine:
    """A VGG engine whose _on_vgg_step fans steps out to the view sink.

    Skips the heavy __init__ (matching test_observation_surface) and wires only
    what _on_vgg_step needs: a raw callback and the observation view sink.
    """
    eng = VectorEngine.__new__(VectorEngine)
    eng._vgg_step_callback = None
    eng._vgg_step_view_callback = view_sink
    return eng


def test_cli_engine_path_emits_step_views_and_renders_verified_loop() -> None:
    arm, gripper = _StubArm(), _StubGripper()
    agent = type("Agent", (), {"_arm": arm, "_gripper": gripper})()
    bind_fake_sim(agent, "so101")
    registry = _ArmRegistry(arm, gripper)

    # The observation surface sink — exactly what cli.py's on_vgg_step_view feeds.
    emitted_views: list[dict[str, Any]] = []
    engine = _engine_with_view_sink(arm, gripper, emitted_views.append)

    # Real executor + real verifier over the MERGED playground verify namespace,
    # driven by the engine's _on_vgg_step (how the engine wires the executor).
    namespace = PlaygroundWorld().build_verify_namespace(agent)
    executor = GoalExecutor(
        strategy_selector=StrategySelector(skill_registry=registry, has_base=False),
        verifier=GoalVerifier(namespace),
        skill_registry=registry,
    )
    trace = executor.execute(_TREE, on_step=engine._on_vgg_step)

    # --- The engine fanned one JSON-safe EXPORT VIEW per step out, live. ---
    assert [v["sub_goal_name"] for v in emitted_views] == [
        "home_arm",
        "detect_mug",
        "grasp_mug",
        "place_mug",
    ]
    for v in emitted_views:
        assert v["success"] is True
        assert v["verify_result"] is True
        json.dumps(v)  # each view round-trips

    # --- Per-step rendering carries sub-goal + strategy + verify. A LIVE view is
    #     pre-grading, so its marker is RAN (never a false PASS). ---
    verify_by_name = {sg.name: sg.verify for sg in _TREE.sub_goals}
    for v in emitted_views:
        line = render_step_view(v, verify_by_name[v["sub_goal_name"]])
        assert "[RAN]" in line
        assert v["sub_goal_name"] in line
        assert v["strategy"] in line
        assert verify_by_name[v["sub_goal_name"]] in line

    # --- Run-complete snapshot renders the FULL verified loop, graded against the
    #     playground's tagged GT oracles. ---
    oracle_names = evidence_names(namespace)
    snapshot = run_snapshot(trace, oracle_names)
    json.dumps(snapshot)  # the snapshot the CLI renders round-trips

    rendered = render_run_snapshot(snapshot)

    # Goal + goal tree: every sub-goal AND its verify predicate appear.
    assert _TREE.goal in rendered
    for sg in _TREE.sub_goals:
        assert sg.name in rendered
        assert sg.strategy in rendered
        assert sg.verify in rendered

    # One GROUNDED marker per step (all four proven here via GT oracles).
    assert rendered.count("[GROUNDED]") >= len(_TREE.sub_goals)
    assert "[FAILED]" not in rendered

    # The replan / validation note surfaces in the render.
    assert "scan_360" in rendered
    assert "not valid" in rendered

    # Overall outcome line present and marked GROUNDED.
    assert "Outcome:" in rendered
    assert "4/4 steps grounded" in rendered


def test_render_run_snapshot_marks_failed_steps_and_partial_outcome() -> None:
    # A run where one step's predicate fails -> the render must mark that step
    # FAIL and the overall outcome FAIL (structure/markers, not prose).
    trace = ExecutionTrace(
        goal_tree=GoalTree(
            goal="tidy",
            sub_goals=(
                SubGoal(name="a", description="do a", verify="arm_at_home()", strategy="home_skill"),
                SubGoal(name="b", description="do b", verify="holding_object()", strategy="pick_skill"),
            ),
            validation_notes=("strategy 'look_skill' is not valid; cleared",),
        ),
        steps=(
            StepRecord(
                sub_goal_name="a", strategy="home_skill", success=True,
                verify_result=True, duration_sec=0.1,
            ),
            StepRecord(
                sub_goal_name="b", strategy="pick_skill", success=False,
                verify_result=False, duration_sec=0.2, error="grasp failed",
            ),
        ),
        success=False,
        total_duration_sec=0.3,
    )
    snapshot = run_snapshot(trace, frozenset({"arm_at_home", "holding_object"}))

    rendered = render_run_snapshot(snapshot)
    json.dumps(snapshot)

    # Mixed markers: step a is GROUNDED (arm_at_home, verify True); step b FAILED.
    assert "[GROUNDED]" in rendered
    assert "[FAILED]" in rendered
    # The failing step's error surfaces.
    assert "grasp failed" in rendered
    # The replan note surfaces.
    assert "look_skill" in rendered
    # Outcome marked FAILED (the run did not succeed), 1/2 grounded.
    assert "Outcome:" in rendered
    assert "[FAILED]" in rendered.splitlines()[-1]
    assert "1/2 steps grounded" in rendered
