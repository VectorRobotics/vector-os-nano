# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Observation surface — the verified-loop's serializable view (SHARED PRELUDE).

This is the kernel-side, JSON-serializable snapshot a front-end (e.g. the
playground view) renders. It is a pure EXPORT VIEW over the frozen VGG types in
``cognitive/types.py`` — it never mutates them and adds no fields to them.

Two views:

- ``step_view(StepRecord)`` — the per-step payload for the ``on_vgg_step``
  callback: just the fields the UI needs (sub_goal_name, strategy, success,
  verify_result, error, result_data).
- ``run_snapshot(ExecutionTrace)`` — the run-complete snapshot: the GoalTree
  (goal + each SubGoal's name/description/verify/strategy/depends_on), every
  executed step's view, the run's ``validation_notes`` (replan feedback), and
  the overall success/duration.

Both round-trip through ``json.dumps`` (deterministic, no eval, no secrets).
``result_data`` is sanitized to JSON-safe primitives so an exotic strategy
output can never break serialization.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

# Snapshot shape version. Bump when the exported keys change so a front-end can
# tolerate older/newer producers (matches trace_store's _SCHEMA_VERSION idea).
SNAPSHOT_VERSION = 1


# ---------------------------------------------------------------------------
# JSON-safety
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Coerce *value* into a JSON-serializable primitive tree.

    Strategy outputs in ``result_data`` are arbitrary; this guarantees the
    snapshot always survives ``json.dumps`` by reducing unknown objects to their
    ``str`` form rather than raising. Deterministic and side-effect-free.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


# ---------------------------------------------------------------------------
# Per-step view
# ---------------------------------------------------------------------------


def step_view(step: StepRecord) -> dict[str, Any]:
    """Return the JSON-safe per-step payload for the ``on_vgg_step`` hook.

    Carries exactly the fields the UI renders for a step. ``result_data`` is the
    captured structured output (closing the loop) made JSON-safe.
    """
    return {
        "sub_goal_name": step.sub_goal_name,
        "strategy": step.strategy,
        "success": bool(step.success),
        "verify_result": bool(step.verify_result),
        "error": step.error,
        "fallback_used": bool(getattr(step, "fallback_used", False)),
        "visual_override": bool(getattr(step, "visual_override", False)),
        "pre_satisfied": bool(getattr(step, "pre_satisfied", False)),
        "result_data": _json_safe(getattr(step, "result_data", {}) or {}),
    }


# ---------------------------------------------------------------------------
# GoalTree view
# ---------------------------------------------------------------------------


def _sub_goal_view(sg: SubGoal) -> dict[str, Any]:
    return {
        "name": sg.name,
        "description": sg.description,
        "verify": sg.verify,
        "strategy": sg.strategy,
        "depends_on": list(sg.depends_on),
    }


def goal_tree_view(tree: GoalTree) -> dict[str, Any]:
    """Return the JSON-safe view of a GoalTree (goal + sub-goals + replan notes)."""
    return {
        "goal": tree.goal,
        "sub_goals": [_sub_goal_view(sg) for sg in tree.sub_goals],
        "validation_notes": list(getattr(tree, "validation_notes", ()) or ()),
    }


# ---------------------------------------------------------------------------
# Run-complete snapshot
# ---------------------------------------------------------------------------


def run_snapshot(trace: ExecutionTrace) -> dict[str, Any]:
    """Return the JSON-safe run-complete snapshot from an ExecutionTrace.

    Includes the goal tree, every executed step's view, the run's
    ``validation_notes`` (replan feedback, lifted from the tree for easy UI
    access), and the overall outcome. Guaranteed to ``json.dumps`` cleanly.
    """
    tree = trace.goal_tree
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "goal": tree.goal,
        "goal_tree": goal_tree_view(tree),
        "steps": [step_view(s) for s in trace.steps],
        "validation_notes": list(getattr(tree, "validation_notes", ()) or ()),
        "success": bool(trace.success),
        "total_duration_sec": float(trace.total_duration_sec),
    }


# ---------------------------------------------------------------------------
# Plain-text rendering of the verified loop (INC8)
# ---------------------------------------------------------------------------
#
# These turn the JSON-safe export views above into human-readable plain-text
# lines so a CLI front-end can SHOW the verified loop without re-deriving
# anything from the frozen VGG types. They are pure (no rich markup, no I/O,
# no side effects) — the caller decides how to colourize/print. The PASS/FAIL
# markers are stable tokens the UI (and tests) can key off.


# Stable, render-agnostic markers (assert on these, not on prose).
_PASS = "PASS"
_FAIL = "FAIL"


def render_step_view(view: dict[str, Any], verify: str | None = None) -> str:
    """Render one per-step EXPORT VIEW as a single readable line.

    Shows the sub-goal, its strategy, the verify predicate (passed in from the
    goal tree, since the per-step view carries only the result), and a stable
    ``PASS``/``FAIL`` marker. ``verify`` is optional: when the predicate is not
    available the line still renders without it.
    """
    name = view.get("sub_goal_name", "?")
    strategy = view.get("strategy") or "(none)"
    passed = bool(view.get("success")) and bool(view.get("verify_result"))
    marker = _PASS if passed else _FAIL
    parts = [f"[{marker}] {name}", f"via {strategy}"]
    if verify:
        parts.append(f"verify {verify}")
    if view.get("fallback_used"):
        parts.append("(fallback)")
    if view.get("visual_override"):
        parts.append("(visual override)")
    if view.get("pre_satisfied"):
        # Invariant I honesty: the predicate held BEFORE the action ran.
        parts.append("(already satisfied pre-exec)")
    err = view.get("error")
    if not passed and err:
        parts.append(f"-- {err}")
    return " | ".join(parts)


def render_run_snapshot(snapshot: dict[str, Any]) -> str:
    """Render a run-complete snapshot as a readable multi-line block.

    Lays out the goal, the goal tree (each sub-goal's strategy + verify
    predicate), each executed step's ``PASS``/``FAIL`` marker, any replan /
    validation notes, and the overall outcome. Pure: consumes only the
    JSON-safe snapshot (the ``run_snapshot`` shape), returns a string.
    """
    tree = snapshot.get("goal_tree", {}) or {}
    sub_goals = tree.get("sub_goals", []) or []
    # Map sub_goal name -> verify predicate so each step line can show it.
    verify_by_name = {sg.get("name"): sg.get("verify") for sg in sub_goals}

    lines: list[str] = []
    lines.append(f"Goal: {snapshot.get('goal', '?')}")

    lines.append("Goal tree:")
    if sub_goals:
        for sg in sub_goals:
            dep = sg.get("depends_on") or []
            dep_txt = f" (after {', '.join(dep)})" if dep else ""
            strat = sg.get("strategy") or "(none)"
            verify = sg.get("verify") or "(none)"
            lines.append(
                f"  - {sg.get('name', '?')}: {sg.get('description', '')}"
                f" | via {strat} | verify {verify}{dep_txt}"
            )
    else:
        lines.append("  (no sub-goals)")

    lines.append("Steps:")
    steps = snapshot.get("steps", []) or []
    if steps:
        for sv in steps:
            lines.append("  " + render_step_view(sv, verify_by_name.get(sv.get("sub_goal_name"))))
    else:
        lines.append("  (no steps executed)")

    notes = snapshot.get("validation_notes", []) or []
    if notes:
        lines.append("Validation notes (replan feedback):")
        for note in notes:
            lines.append(f"  - {note}")

    n_steps = len(steps)
    n_pass = sum(1 for sv in steps if bool(sv.get("success")) and bool(sv.get("verify_result")))
    outcome = _PASS if snapshot.get("success") else _FAIL
    dur = snapshot.get("total_duration_sec", 0.0)
    lines.append(f"Outcome: [{outcome}] {n_pass}/{n_steps} steps verified ({dur:.1f}s)")

    return "\n".join(lines)
