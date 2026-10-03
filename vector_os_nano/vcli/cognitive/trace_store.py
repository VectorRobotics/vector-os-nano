# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""trace_store — persist and reload VGG execution traces, and grade their evidence.

- ``save_trace`` / ``load_trace`` round-trip a trace to JSON under
  ``~/.vector/traces/`` (frozen dataclasses -> dicts and back; tuples survive).
- ``evidence_passed`` gates a "verified done" in EVERY world (dev and robot
  alike — the old ``if is_robot: return True`` bypass is gone): a step backs the
  outcome with evidence only when its verify actually consumes a world oracle
  (single-sourced via ``classify_step_evidence`` -> ``classify_verify_expr``).
  A sentinel ``""`` / ``"True"`` verify, an absent oracle, or a tautology
  classify RAN and do NOT pass the gate.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Literal

from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused, from_name
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.oracle_registry import evidence_names
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

logger = logging.getLogger(__name__)

StepEvidence = Literal["GROUNDED", "RAN", "FAILED"]

_DEFAULT_TRACES_DIR = Path.home() / ".vector" / "traces"

# Bump when the on-disk shape changes; load_trace tolerates older/unknown keys.
# v2 adds StepRecord.visual_override; v3 adds SubGoal.answer_only (S5.2);
# v4 adds StepRecord.actor_caused (R2b actor-causation grade).
_SCHEMA_VERSION = 4

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
# Oracle names — single-sourced from the LIVE verify namespace
# ---------------------------------------------------------------------------


def verify_oracle_names(agent: Any, engine: Any = None) -> frozenset[str]:
    """The EVIDENCE oracle names in the LIVE verify namespace GoalVerifier uses.

    Single source (rule 3): read off the SAME dict
    ``engine._build_verifier_namespace(agent)`` builds for ``GoalVerifier`` — but
    only the callables tagged ``@oracle`` with the active world's scope
    (``"robot"`` in a robot world, ``"dev"`` in the dev world). Untagged callables
    (memory readers, helpers) and dev predicates inside a robot world stay callable
    but are not evidence, so a verify built on them classifies RAN.

    *engine* is required to build the namespace; when it is None (or namespace /
    world resolution raises) this FAILS CLOSED to ``frozenset()`` — with no oracle
    names every predicate classifies RAN, never a spurious GROUNDED (rule 5).
    """
    if engine is None:
        return frozenset()
    builder = getattr(engine, "_build_verifier_namespace", None)
    if builder is None:
        return frozenset()
    try:
        ns = builder(agent)
        world = getattr(engine, "_world", None)
        if world is None:
            from vector_os_nano.vcli.worlds.registry import resolve_world

            world = resolve_world(agent)
        scope = "robot" if world.is_robot() else "dev"
    except Exception as exc:  # noqa: BLE001
        logger.debug("verify_oracle_names: namespace build failed: %s", exc)
        return frozenset()
    return evidence_names(ns, scope)


# ---------------------------------------------------------------------------
# Per-step evidence classification (single source for BOTH gates — rule 3)
# ---------------------------------------------------------------------------


def _actor_uncaused(step: StepRecord) -> bool:
    """True iff the executor graded this step's actor-causation as UNCAUSED (R2b).

    Reads ``step.actor_caused`` defensively (``getattr``) so a legacy StepRecord
    built before the field existed reads as ``NOT_GRADED`` -> not uncaused -> no
    downgrade. Only the explicit ``UNCAUSED`` value downgrades a GROUNDED step.
    """
    return getattr(step, "actor_caused", ActorCaused.NOT_GRADED) == ActorCaused.UNCAUSED


def classify_step_evidence(
    step: StepRecord, sub_goal: SubGoal, oracle_names: frozenset[str],
    goal_text: str | None = None,
) -> StepEvidence:
    """Classify a single executed step's evidence: GROUNDED / RAN / FAILED.

    The ONE source of truth for both the done-gate (``evidence_passed``) and the
    per-step reward gate (``step_evidence_ok``) — rule 3 (no split-brain). It
    replaces the old ``if is_robot: return True`` short-circuit with the honest
    classifier: a robot step that merely RAN (``verify="True"``) is no longer
    auto-passed; only a step whose verify actually consumes a world oracle in a
    non-tautological way reaches GROUNDED.

    - FAILED — the step did not succeed (``not step.success``).
    - GROUNDED — the step succeeded AND either:
        * it is a legitimately-exempt answer-only step (zero-I/O ``answer``
          strategy) whose ``verify_result`` is True and not a visual override; OR
        * ``classify_verify_expr(sub_goal.verify, oracle_names) == "GROUNDED"``
          AND ``step.verify_result`` AND not a visual override.
      A VLM visual override is not deterministic (not replayable) evidence, so it
      can never reach GROUNDED.
    - RAN — the step succeeded but its verify carries no real, non-tautological
      world evidence (a sentinel ``""``/``"True"``, an absent oracle, or a
      tautology), or it succeeded with a visual override.

    R2b ACTOR-CAUSATION downgrade (single-sourced HERE so it flows to BOTH gates
    and the VECTOR_VERDICT report): a step that would otherwise classify GROUNDED is
    DOWNGRADED to RAN when ``step.actor_caused == ActorCaused.UNCAUSED`` — i.e. the
    R1 predicate is GROUNDED but the executor graded that the ACTOR did NOT cause the
    state change (a satisfied-at-baseline NO-OP, or a teleport with no commanded
    motion). The downgrade fires ONLY on the explicit ``UNCAUSED`` value; the default
    ``NOT_GRADED`` (legacy / hand-built / non-robot-predicate steps the executor never
    graded) classifies EXACTLY as before R2b — zero regression. ``CAUSED`` is a no-op
    here (the step keeps its GROUNDED classification).

    STEP-13 GOAL-AUTHENTICITY downgrade (also subtractive, fail-OPEN): when *goal_text*
    is provided (the turn's NL goal, single-sourced from ``trace.goal_tree.goal``) and
    BOTH it parses to a coordinate AND the verify is a literal ``at_position(x, y)``
    whose constant differs from the goal by more than the oracle tolerance, the
    otherwise-GROUNDED step DOWNGRADES to RAN — the model verified its OWN landing, not
    the commanded target. Non-coordinate goals/verifies fail OPEN (unchanged);
    ``goal_text=None`` (the per-step reward gate) is a no-op. Stricter-only (rule 5).
    """
    if not step.success:
        return "FAILED"
    visual_override = bool(getattr(step, "visual_override", False))
    if _is_answer_only(sub_goal):
        return "GROUNDED" if (step.verify_result and not visual_override) else "RAN"
    grounded = (
        classify_verify_expr(sub_goal.verify, oracle_names) == "GROUNDED"
        and step.verify_result
        and not visual_override
    )
    if grounded and _actor_uncaused(step):
        # R2b: GROUNDED predicate but the actor did not cause it -> downgrade to RAN.
        return "RAN"
    if grounded and goal_text is not None:
        # STEP-13 goal-authenticity: an actor-CAUSED at_position verify whose constant
        # does NOT match the user's parsed COORDINATE goal means the model verified its
        # OWN landing, not the commanded target -> downgrade to RAN. Fail-OPEN whenever
        # the goal or the verify is not a parseable coordinate. Stricter-only.
        from vector_os_nano.vcli.cognitive.coord_goal import coord_goal_mismatch

        if coord_goal_mismatch(goal_text, sub_goal.verify):
            return "RAN"
    return "GROUNDED" if grounded else "RAN"


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
                # R2b: serialize the actor-causation grade as its enum ``.value``.
                "actor_caused": getattr(
                    s, "actor_caused", ActorCaused.NOT_GRADED
                ).value,
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
            # R2b: deserialize the actor-causation grade; absent/unknown (older
            # traces) maps to NOT_GRADED -> legacy-equivalent (no downgrade).
            actor_caused=from_name(s.get("actor_caused")),
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
# Evidence
# ---------------------------------------------------------------------------


def evidence_passed(trace: ExecutionTrace, oracle_names: frozenset[str]) -> bool:
    """True when the trace's success is backed by deterministic evidence.

    Single-sourced through ``classify_step_evidence`` (rule 3) and applied to BOTH
    dev and robot worlds (the dishonest ``if is_robot: return True`` bypass is
    GONE): every executed step that maps to a sub-goal must classify GROUNDED, and
    there must be at least one such checked step (an empty trace fails). A step
    classifies GROUNDED only when its verify actually consumes a world oracle in
    *oracle_names* in a non-tautological way (or it is a legitimately-exempt
    answer-only step); a sentinel ``""`` / ``"True"`` verify (the robot motor
    default), an absent oracle, a tautology, or a VLM visual override classify RAN
    and so do NOT pass the gate.

    *oracle_names* MUST be the live verify-namespace callable names from
    ``verify_oracle_names(agent, engine)`` — single-sourced from the SAME namespace
    ``GoalVerifier`` uses; never a hand-authored copy. An empty set fails closed
    (everything classifies RAN), so the moat only ever gets stricter (rule 5).

    Stage 5 (S5.2): a step whose sub-goal is explicitly ``answer_only`` (and routed
    through the zero-I/O ``answer`` strategy) is a pure-conversation step that
    carries no robot evidence BY DESIGN; ``classify_step_evidence`` exempts it from
    the real-predicate requirement but still requires ``verify_result`` True and no
    visual override. The exemption is keyed on the side-effect-free ``answer``
    strategy, NOT the LLM-controlled ``answer_only`` flag alone, so an LLM cannot
    launder a side-effecting step past the gate.

    WHOLE-TRACE rule (stricter-only): a trace that did not succeed, a step with no
    sub-goal to grade it against (a foreach child ``owner[i].tmpl`` — its resolved
    verify is not in the tree), or a foreach node at all (zero iterations leave no
    step behind) can never verify. The gate grades every step or none.
    """
    if not trace.success:
        return False
    sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
    if any(getattr(sg, "foreach", None) is not None for sg in trace.goal_tree.sub_goals):
        return False
    if any(s.sub_goal_name not in sg_by_name for s in trace.steps):
        return False
    checked = list(trace.steps)
    if not checked:
        return False
    goal_text = trace.goal_tree.goal
    all_grounded = all(
        classify_step_evidence(s, sg_by_name[s.sub_goal_name], oracle_names, goal_text) == "GROUNDED"
        for s in checked
    )
    if not all_grounded:
        return False
    # STEP-15 COORDINATE-GOAL TURN GATE (stricter-only). A turn whose NL goal targets a
    # coordinate must actually VERIFY reaching it: >=1 checked step that BOTH classifies
    # GROUNDED and whose verify is a bounded-tol literal ``at_position`` matching the
    # commanded coord. Closes the wrong-predicate-type hole — a coordinate goal verified
    # ENTIRELY with non-``at_position`` predicates (``facing(...)``,
    # ``len(get_position())==3``): STEP-13's per-step mismatch downgrade fails OPEN on a
    # non-``at_position`` verify, so every step stays GROUNDED and the coordinate is never
    # asserted. The gate can only REJECT (all_grounded already held) and can't fake-pass
    # (the matching step must itself be GROUNDED = actor-CAUSED + honest constant).
    # SCOPED to a go2-coordinate world (``at_position`` in the live oracle set, rule 3),
    # so dev / arm / answer worlds — whose goal text may incidentally contain an (x,y)
    # pair — are never affected.
    if "at_position" not in oracle_names:
        return True
    from vector_os_nano.vcli.cognitive.coord_goal import (
        at_position_const_matches,
        goal_has_coordinate_intent,
        has_necessary_at_position,
        parse_goal_coord,
    )

    goal_xy = parse_goal_coord(goal_text)
    if goal_xy is not None:
        return any(
            classify_step_evidence(s, sg_by_name[s.sub_goal_name], oracle_names, goal_text) == "GROUNDED"
            and at_position_const_matches(sg_by_name[s.sub_goal_name].verify, goal_xy)
            for s in checked
        )
    # Parse-evasion: a coordinate-shaped goal the regex cannot pin to a single (x, y)
    # (paren-less "导航到 11,3", or >=2 distinct coords) must still not pass on a
    # non-``at_position`` verify — require SOME GROUNDED bounded-tol literal at_position
    # step (the coord can't be pinned, but the facing-only escape is closed). A go2 goal
    # with NO coordinate intent (room name / "explore" / "turn") fails OPEN (unchanged).
    if goal_has_coordinate_intent(goal_text):
        return any(
            classify_step_evidence(s, sg_by_name[s.sub_goal_name], oracle_names, goal_text) == "GROUNDED"
            and has_necessary_at_position(sg_by_name[s.sub_goal_name].verify)
            for s in checked
        )
    return _object_goal_turn_ok(checked, sg_by_name, oracle_names, goal_text)


def _object_goal_turn_ok(
    checked: list[StepRecord],
    sg_by_name: dict[str, SubGoal],
    oracle_names: frozenset[str],
    goal_text: str | None,
) -> bool:
    """D17 OBJECT-GOAL TURN GATE (stricter-only) — the grasp analogue of the STEP-15
    coordinate turn gate above.

    A turn whose NL goal commands a physical GRASP / PICK must actually PROVE physical
    possession via a GT MANIPULATION oracle (``holding_object`` / ``placed_count``) the
    actor cannot author — NOT via a self-authored file/scene/marker predicate. This
    closes the fakeable-grasp false-green (D17): deepseek satisfied "抓前面的东西" by
    ``file_write('grabbed.txt')`` then ``verify(file_exists('grabbed.txt'))`` — and
    ``file_exists`` is a PREDICATE oracle (classifies GROUNDED) AND a non-robot predicate
    (never actor-causation-graded), so every prior gate failed OPEN and the turn went
    GROUNDED. Mirrors the D15/D16 coordinate turn gate exactly — present-but-not-
    necessary decoys are rejected by the same necessity discipline (``object_goal``).

    Like the coordinate gate, this can ONLY REJECT (``all_grounded`` already held above)
    and can NEVER fake-pass: the satisfying step must ITSELF classify GROUNDED (actor-
    CAUSED + honest oracle) AND carry a NECESSARY manipulation-oracle conjunct. So a real
    grasp (``perception_grasp`` -> ``holding_object('banana')`` weld-CAUSED) stays GROUNDED;
    a fabricated grasp (file/scene/marker) downgrades the whole turn to RAN.

    SCOPED to a manipulation world (``holding_object`` in the live oracle set, rule 3),
    so go2-coordinate / dev / g1-detect worlds — whose goal text may incidentally contain
    a grasp word — never see this gate. FAIL-OPEN for any goal with no grasp/pick intent.
    """
    if "holding_object" not in oracle_names:
        return True
    from vector_os_nano.vcli.cognitive.object_goal import (
        goal_has_object_intent,
        has_necessary_manip_oracle,
    )

    if not goal_has_object_intent(goal_text):
        return True
    return any(
        classify_step_evidence(s, sg_by_name[s.sub_goal_name], oracle_names, goal_text) == "GROUNDED"
        and has_necessary_manip_oracle(sg_by_name[s.sub_goal_name].verify)
        for s in checked
    )


def step_evidence_ok(
    step: StepRecord, sub_goal: SubGoal, oracle_names: frozenset[str]
) -> bool:
    """True when a SINGLE step's pass is backed by deterministic evidence.

    The per-step analogue of ``evidence_passed``, single-sourced through the SAME
    ``classify_step_evidence`` (rule 3): True iff this step classifies GROUNDED.
    The robot bypass is GONE — a robot motor step with ``verify="True"`` now
    classifies RAN, not GROUNDED, so it no longer trains the bandit on unverified
    "success". Reward parity with the done-gate is intentional (W1.1 → R1):
    learning is rewarded only for steps that actually proved their post-condition.

    *oracle_names* MUST come from ``verify_oracle_names(agent, engine)`` (live
    namespace, single source). An empty set fails closed (classifies RAN).
    """
    return classify_step_evidence(step, sub_goal, oracle_names) == "GROUNDED"
