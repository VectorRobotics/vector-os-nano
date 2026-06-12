# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""trace_store — persist, reload, and replay VGG execution traces.

Turns a run's ``ExecutionTrace`` into a replayable, self-grading eval signal:

- ``save_trace`` / ``load_trace`` round-trip a trace to JSON under
  ``~/.vector/traces/`` (frozen dataclasses -> dicts and back; tuples survive).
- ``replay`` re-evaluates each sub-goal's *deterministic* verify predicate with
  a fresh ``GoalVerifier``. Visual overrides are NOT reproduced — only
  deterministic evidence counts, which is the point of replay.
- ``evidence_passed`` gates a "verified done": in the dev world a step backs the
  outcome with evidence only when its verify is a real predicate (not the
  sentinels ``""`` / ``"True"``) AND that predicate verified. Robot worlds bypass
  the gate so async motor skills that legitimately use ``verify="True"`` do not
  regress.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

# Verify strings that carry no deterministic evidence (empty, or the trivial
# fallback the decomposer/robot motor steps use).
_NO_EVIDENCE: frozenset[str] = frozenset({"", "True"})

_DEFAULT_TRACES_DIR = Path.home() / ".vector" / "traces"

# Bump when the on-disk shape changes; load_trace tolerates older/unknown keys.
# v2 adds StepRecord.visual_override; v3 adds SubGoal.answer_only (S5.2).
_SCHEMA_VERSION = 3

# The ONLY side-effect-free dispatch route (``GoalExecutor._execute_answer`` does
# no I/O and no model call). The evidence-gate exemption for a no-robot-evidence
# step is bound to it, never to the LLM-controlled ``answer_only`` flag alone.
_ANSWER_STRATEGY = "answer"


def _is_answer_only(sub_goal: SubGoal) -> bool:
    """True iff *sub_goal* is a legitimately-exempt answer-only step.

    The evidence-gate relaxation (skip the real-predicate requirement) applies
    ONLY to a step that is BOTH flagged ``answer_only`` AND routed through the
    side-effect-free ``answer`` strategy. Tying the exemption to the zero-I/O
    executor — not to the LLM-controlled ``answer_only`` bit alone — keeps the
    moat (rule 5) intact: an LLM that sets ``answer_only: true`` on a
    side-effecting strategy (``tool_call`` / a skill) does NOT get waived, because
    that step still runs a real executor and so must carry deterministic evidence.
    The decomposer additionally refuses the flag on non-``answer`` strategies, so
    this gate and the decomposer agree; this check is the belt to that suspenders.
    """
    return bool(getattr(sub_goal, "answer_only", False)) and sub_goal.strategy == _ANSWER_STRATEGY


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------


def _trace_to_dict(trace: ExecutionTrace) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "goal_tree": {
            "goal": trace.goal_tree.goal,
            "sub_goals": [
                {
                    "name": sg.name,
                    "description": sg.description,
                    "verify": sg.verify,
                    "timeout_sec": sg.timeout_sec,
                    "depends_on": list(sg.depends_on),
                    "strategy": sg.strategy,
                    "strategy_params": sg.strategy_params,
                    "fail_action": sg.fail_action,
                    "answer_only": getattr(sg, "answer_only", False),
                }
                for sg in trace.goal_tree.sub_goals
            ],
            "context_snapshot": trace.goal_tree.context_snapshot,
        },
        "steps": [
            {
                "sub_goal_name": s.sub_goal_name,
                "strategy": s.strategy,
                "success": s.success,
                "verify_result": s.verify_result,
                "duration_sec": s.duration_sec,
                "error": s.error,
                "fallback_used": s.fallback_used,
                "visual_override": getattr(s, "visual_override", False),
            }
            for s in trace.steps
        ],
        "success": trace.success,
        "total_duration_sec": trace.total_duration_sec,
    }


def _dict_to_trace(data: dict[str, Any]) -> ExecutionTrace:
    gt = data.get("goal_tree", {}) or {}
    sub_goals = tuple(
        SubGoal(
            name=str(sg.get("name", "")),
            description=str(sg.get("description", "")),
            verify=str(sg.get("verify", "")),
            timeout_sec=float(sg.get("timeout_sec", 30.0)),
            depends_on=tuple(sg.get("depends_on", []) or ()),
            strategy=str(sg.get("strategy", "")),
            strategy_params=dict(sg.get("strategy_params", {}) or {}),
            fail_action=str(sg.get("fail_action", "")),
            answer_only=bool(sg.get("answer_only", False)),
        )
        for sg in gt.get("sub_goals", []) or []
    )
    goal_tree = GoalTree(
        goal=str(gt.get("goal", "")),
        sub_goals=sub_goals,
        context_snapshot=str(gt.get("context_snapshot", "")),
    )
    steps = tuple(
        StepRecord(
            sub_goal_name=str(s.get("sub_goal_name", "")),
            strategy=str(s.get("strategy", "")),
            success=bool(s.get("success", False)),
            verify_result=bool(s.get("verify_result", False)),
            duration_sec=float(s.get("duration_sec", 0.0)),
            error=str(s.get("error", "")),
            fallback_used=bool(s.get("fallback_used", False)),
            visual_override=bool(s.get("visual_override", False)),
        )
        for s in data.get("steps", []) or []
    )
    return ExecutionTrace(
        goal_tree=goal_tree,
        steps=steps,
        success=bool(data.get("success", False)),
        total_duration_sec=float(data.get("total_duration_sec", 0.0)),
    )


def save_trace(trace: ExecutionTrace, path: str | Path | None = None) -> Path:
    """Serialise *trace* to JSON; return the written path.

    With no *path*, writes a uuid-named file under ``~/.vector/traces/``.
    Written atomically (temp file + ``os.replace``) so a concurrent reader never
    sees a half-written file.
    """
    import os

    if path is None:
        _DEFAULT_TRACES_DIR.mkdir(parents=True, exist_ok=True)
        path = _DEFAULT_TRACES_DIR / f"trace-{uuid.uuid4().hex}.json"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(_trace_to_dict(trace), ensure_ascii=False, indent=2)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)
    return path


def load_trace(path: str | Path) -> ExecutionTrace:
    """Reconstruct an ExecutionTrace from a JSON file written by ``save_trace``."""
    data = json.loads(Path(path).read_text())
    return _dict_to_trace(data)


# ---------------------------------------------------------------------------
# Replay + evidence
# ---------------------------------------------------------------------------


def replay(trace: ExecutionTrace, verifier: Any) -> bool:
    """Re-evaluate every sub-goal's deterministic verify predicate.

    Returns True iff every non-sentinel verify expression evaluates truthy under
    *verifier* (a ``GoalVerifier``). Sentinel verifies (``""`` / ``"True"``) are
    skipped — they carry no evidence — and a trace with *no* deterministic
    predicate to replay returns False (nothing was actually checked).

    Stage 5 (S5.2): an explicitly ``answer_only`` step carries no robot evidence
    BY DESIGN — it is skipped exactly like a sentinel and never counts toward
    ``checked``. This does NOT loosen the gate: the exemption is tied to the
    side-effect-free ``answer`` strategy (``answer_only`` alone is fully
    LLM-controlled, so it is NOT trusted to waive evidence for a step that runs a
    side-effecting executor). An action step with a sentinel verify is still
    non-evidence and a pure-answer trace still returns False (nothing deterministic
    was checked).
    """
    checked = 0
    for sg in trace.goal_tree.sub_goals:
        if _is_answer_only(sg):
            continue
        expr = (sg.verify or "").strip()
        if expr in _NO_EVIDENCE:
            continue
        checked += 1
        if not verifier.verify(sg.verify):
            return False
    return checked > 0


def evidence_passed(
    trace: ExecutionTrace,
    is_robot: bool = False,
    exempt_strategies: "frozenset[str]" = frozenset(),
) -> bool:
    """True when the trace's success is backed by deterministic evidence.

    Dev world (strict): every executed step must map to a sub-goal whose verify
    is a real predicate (not ``""`` / ``"True"``), whose ``verify_result`` is
    True, and whose pass was NOT a VLM visual override (a visual override is not
    deterministic evidence and cannot be replayed — this keeps ``evidence_passed``
    and ``replay`` in agreement). Strategies in *exempt_strategies* (world-
    declared: no symbolic post-condition exists, e.g. async explore) are exempt
    from the real-predicate requirement only — never from verify_result/override.

    Stage 5 (S5.2): a step whose sub-goal is explicitly ``answer_only`` is a
    pure-conversation step that carries no robot evidence BY DESIGN. It is exempt
    from the real-predicate requirement (it would otherwise be indistinguishable
    from an unverified action) but must still have ``verify_result`` True and not
    be a visual override.

    MOAT (rule 5): the exemption is keyed on ``answer_only`` AND tied to the
    side-effect-free ``answer`` strategy (see ``_is_answer_only``). The
    ``answer_only`` flag is fully LLM-controlled, so on its own it is no safer than
    the verify string; binding it to the only executor that performs zero I/O
    (``GoalExecutor._execute_answer``) means an LLM that sets ``answer_only: true``
    on a side-effecting strategy (``tool_call`` / a skill) does NOT launder that
    action past the gate — it still requires a real deterministic predicate. An
    action step with a sentinel ``""`` / ``"True"`` verify is therefore still NOT
    counted as verified. A trace with no non-answer step still requires at least
    one answer_only step to have passed (an empty trace fails).
    """
    # Invariant II (design review 2026-06-12 #4): the WORLD-level bypass is
    # gone — ``is_robot`` is accepted for signature compatibility but no
    # longer bypasses anything. Exemption is PER-STEP via *exempt_strategies*
    # (a world-declared, bounded set of strategies with no symbolic
    # post-condition — mechanism modeled on the answer_only exemption). An
    # exempt step still requires verify_result=True and no visual override:
    # the exemption never launders a failed or VLM-graded step.
    sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
    checked = [s for s in trace.steps if s.sub_goal_name in sg_by_name]
    if not checked:
        return False
    return all(
        (
            _is_answer_only(sg_by_name[s.sub_goal_name])
            or _strategy_exempt(s, sg_by_name[s.sub_goal_name], exempt_strategies)
            or (sg_by_name[s.sub_goal_name].verify or "").strip() not in _NO_EVIDENCE
        )
        and s.verify_result
        and not getattr(s, "visual_override", False)
        for s in checked
    )


def _strategy_exempt(
    step: StepRecord, sub_goal: SubGoal, exempt: "frozenset[str]"
) -> bool:
    """True when the step's RESOLVED strategy is world-exempt (invariant II).

    Matched on both the StepRecord's resolved name and the sub-goal's planned
    strategy (with/without the ``_skill`` suffix) so neither naming form
    escapes or fakes the set.
    """
    if not exempt:
        return False
    names = {
        (step.strategy or "").removesuffix("_skill"),
        (sub_goal.strategy or "").removesuffix("_skill"),
    }
    names.discard("")
    return bool(names) and names <= exempt


def step_evidence_ok(
    step: StepRecord,
    sub_goal: SubGoal,
    is_robot: bool = False,
    exempt_strategies: "frozenset[str]" = frozenset(),
) -> bool:
    """True when a SINGLE step's pass is backed by deterministic evidence.

    The per-step analogue of ``evidence_passed`` (it mirrors that gate's
    per-step inner clause EXACTLY): the step's sub-goal must either be a
    legitimately-exempt answer-only step OR carry a real predicate verify (not
    the sentinels ``""`` / ``"True"``), the step's ``verify_result`` must be
    True, and the pass must NOT be a VLM visual override (not replayable, so not
    deterministic evidence).

    Strategies in *exempt_strategies* keep their learning signal without a real
    predicate (async motor skills with no symbolic post-condition); everything
    else requires deterministic evidence in EVERY world (invariant II — the
    caller AND-composes with ``step.success``, so a FAILED step is never
    rewarded either way).
    """
    # Invariant II: no world-level bypass (``is_robot`` kept for signature
    # compatibility only); per-step exemption mirrors evidence_passed exactly.
    return bool(
        (
            _is_answer_only(sub_goal)
            or _strategy_exempt(step, sub_goal, exempt_strategies)
            or (sub_goal.verify or "").strip() not in _NO_EVIDENCE
        )
        and step.verify_result
        and not getattr(step, "visual_override", False)
    )
