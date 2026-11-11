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
- WP07: only the latest attempt of each attempt group is graded (FAIL -> retry ->
  PASS on the same predicate verifies; the FAIL row stays as audit), a turn-level
  task contract compiled from the goal must go False -> True across the turn, and
  the coordinate and manipulation gates run independently (``turn_reject_reason``).
"""
from __future__ import annotations

import ast
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Literal

from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused, from_name
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.oracle_registry import OracleTag, oracle_tags
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
# v4 adds StepRecord.actor_caused (R2b actor-causation grade); v5 adds the WP07
# attempt fields (attempt_of / superseded / channels) and the trace's task contract.
_SCHEMA_VERSION = 5

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
    """The names of ``verify_oracles`` (the evidence oracles of the live namespace)."""
    return frozenset(verify_oracles(agent, engine))


def verify_oracles(agent: Any, engine: Any = None) -> dict[str, OracleTag]:
    """``{name: OracleTag}`` of the EVIDENCE oracles in the LIVE verify namespace.

    Single source (rule 3): read off the SAME dict
    ``engine._build_verifier_namespace(agent)`` builds for ``GoalVerifier`` — but
    only the callables tagged ``@oracle`` with the active world's scope
    (``"robot"`` in a robot world, ``"dev"`` in the dev world). Untagged callables
    (memory readers, helpers) and dev predicates inside a robot world stay callable
    but are not evidence, so a verify built on them classifies RAN. The tags carry
    each oracle's channel / transition for the causation grader.

    *engine* is required to build the namespace; when it is None (or namespace /
    world resolution raises) this FAILS CLOSED to ``frozenset()`` — with no oracle
    names every predicate classifies RAN, never a spurious GROUNDED (rule 5).
    """
    if engine is None:
        return {}
    builder = getattr(engine, "_build_verifier_namespace", None)
    if builder is None:
        return {}
    try:
        ns = builder(agent)
        world = getattr(engine, "_world", None)
        if world is None:
            from vector_os_nano.vcli.worlds.registry import resolve_world

            world = resolve_world(agent)
        scope = "robot" if world.is_robot() else "dev"
    except Exception as exc:  # noqa: BLE001
        logger.debug("verify_oracle_names: namespace build failed: %s", exc)
        return {}
    return oracle_tags(ns, scope)


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
    return _classify(step, sub_goal, oracle_names, goal_text)[0]


def _classify(
    step: StepRecord, sub_goal: SubGoal, oracle_names: frozenset[str], goal_text: str | None = None,
) -> tuple[StepEvidence, str]:
    """``(evidence, reason)`` for one step — the reason is "" when GROUNDED."""
    if not step.success:
        return "FAILED", f"the action failed: {step.error}" if step.error else "the action failed"
    visual_override = bool(getattr(step, "visual_override", False))
    if _is_answer_only(sub_goal):
        if step.verify_result and not visual_override:
            return "GROUNDED", ""
        return "RAN", "the answer was not confirmed"
    if visual_override:
        return "RAN", "a VLM visual override is not deterministic evidence"
    if classify_verify_expr(sub_goal.verify, oracle_names) != "GROUNDED":
        return "RAN", "no verify" if not (sub_goal.verify or "").strip() else "the verify is not ground-truth evidence"
    if not step.verify_result:
        return "RAN", "verify FAIL"
    if _actor_uncaused(step):
        # R2b: a GROUNDED predicate the actor did not cause (a no-op, a teleport, a
        # target already met at the baseline) is not evidence.
        detail = ", ".join(getattr(step, "channels", ()) or ())
        return "RAN", f"UNCAUSED ({detail})" if detail else "UNCAUSED"
    if goal_text is not None:
        # STEP-13 goal authenticity: an at_position verify whose constant is not the
        # user's parsed coordinate verified the actor's own landing. Fail-open when
        # the goal or the verify is not a parseable coordinate.
        from vector_os_nano.vcli.cognitive.coord_goal import coord_goal_mismatch

        if coord_goal_mismatch(goal_text, sub_goal.verify):
            return "RAN", "the verified coordinate is not the commanded one"
    return "GROUNDED", ""


def step_reject_reason(
    step: StepRecord, sub_goal: SubGoal, oracle_names: frozenset[str], goal_text: str | None = None,
) -> str:
    """Why *step* is not GROUNDED ("" when it is) — the same rules as
    ``classify_step_evidence``, for the verdict's audit row."""
    return _classify(step, sub_goal, oracle_names, goal_text)[1]


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
                "attempt_of": getattr(s, "attempt_of", ""),
                "superseded": getattr(s, "superseded", False),
                "channels": list(getattr(s, "channels", ()) or ()),
                "housekeeping": getattr(s, "housekeeping", False),
            }
            for s in trace.steps
        ],
        "success": trace.success,
        "total_duration_sec": trace.total_duration_sec,
        "contract": list(getattr(trace, "contract", ()) or ()),
        "contract_start": list(getattr(trace, "contract_start", ()) or ()),
        "contract_end": list(getattr(trace, "contract_end", ()) or ()),
        "intents": list(getattr(trace, "intents", ()) or ()),
        "world": str(getattr(trace, "world", "") or ""),
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
            attempt_of=str(s.get("attempt_of", "") or ""),
            superseded=bool(s.get("superseded", False)),
            channels=tuple(str(c) for c in s.get("channels", []) or ()),
            housekeeping=bool(s.get("housekeeping", False)),
        )
        for s in data.get("steps", []) or []
    )
    return ExecutionTrace(
        goal_tree=goal_tree,
        steps=steps,
        success=bool(data.get("success", False)),
        total_duration_sec=float(data.get("total_duration_sec", 0.0)),
        contract=tuple(str(c) for c in data.get("contract", []) or ()),
        contract_start=tuple(bool(v) for v in data.get("contract_start", []) or ()),
        contract_end=tuple(bool(v) for v in data.get("contract_end", []) or ()),
        intents=tuple(str(c) for c in data.get("intents", []) or ()),
        world=str(data.get("world", "") or ""),
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
# Evidence — attempt groups, the task contract, the turn gates
# ---------------------------------------------------------------------------


def attempt_key(expr: str | None) -> str:
    """The normalized verify AST an attempt group is keyed on: whitespace, quote
    style, parentheses and ``9`` vs ``9.0`` collapse; anything else (a different
    predicate, other arguments) is a different sub-goal."""
    s = (expr or "").strip()
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return s
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            node.value = float(node.value)
    return ast.dump(tree.body)


def attempt_groups(trace: ExecutionTrace) -> list[list[int]]:
    """The trace's attempt groups (step indices, in order), sorted by their latest.

    A step joins a group only when the producer opted it in (``attempt_of``) AND
    the group's verify normalizes the same (``attempt_key``), so a producer can never
    merge two predicates and a PASS of a different, easier predicate never supersedes
    a FAIL. Every other step is its own group.
    """
    verify_of = {sg.name: sg.verify for sg in trace.goal_tree.sub_goals}
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, s in enumerate(trace.steps):
        verify = verify_of.get(s.sub_goal_name, "")
        attempt = getattr(s, "attempt_of", "")
        key = ("attempt", attempt, attempt_key(verify)) if attempt and verify else ("own", str(i))
        groups.setdefault(key, []).append(i)
    return sorted(groups.values(), key=lambda g: g[-1])


def graded_indices(trace: ExecutionTrace) -> list[int]:
    """Indices of the steps the gate grades: the LATEST attempt of every group.
    Superseded rows stay in the trace as the audit of the attempts (FAIL -> retry ->
    PASS)."""
    return [g[-1] for g in attempt_groups(trace)]


def _contract_reason(
    trace: ExecutionTrace, graded: list[StepRecord], sg_by_name: dict[str, SubGoal],
    ev: dict[int, StepEvidence], oracle_names: frozenset[str],
) -> str:
    """Why the turn's task contract is not met ("" when it is): every predicate must
    be GT-provable here, False at turn start, True at turn end, and asserted by a
    GROUNDED (actor-caused) graded step."""
    from vector_os_nano.vcli.cognitive.object_goal import backs_contract, task_contract

    contract, world = tuple(trace.contract), str(getattr(trace, "world", "") or "")
    # The spine recompiles the contract: a producer can never swap in a weaker one.
    if contract != task_contract(trace.goal_tree.goal, oracle_names, world):
        return "the task contract does not match the goal"
    if len(trace.contract_start) != len(contract) or len(trace.contract_end) != len(contract):
        return "task contract was not evaluated at turn start and end"
    for item, was, now in zip(contract, trace.contract_start, trace.contract_end):
        if classify_verify_expr(item, oracle_names) != "GROUNDED":
            return (f"task contract {item}: the goal names no object the world resolves"
                    if "None" in item else f"task contract {item} is not provable in this world")
        if was:
            return f"already satisfied at turn start: {item}"
        if not now:
            return f"task contract not met at turn end: {item}"
        if not any(ev[id(s)] == "GROUNDED" and backs_contract(item, sg_by_name[s.sub_goal_name].verify, world)
                   for s in graded):
            return f"no actor-caused step proves {item}"
    return ""


def _coordinate_reason(
    graded: list[StepRecord], sg_by_name: dict[str, SubGoal], ev: dict[int, StepEvidence],
    oracle_names: frozenset[str], goal_text: str,
) -> str:
    """STEP-15 coordinate gate: a goal naming a coordinate needs a GROUNDED graded step
    whose verify necessarily asserts it (a facing-only / decoy turn never passes).
    Scoped to a world with ``at_position``."""
    from vector_os_nano.vcli.cognitive.coord_goal import (
        at_position_const_matches,
        goal_has_coordinate_intent,
        has_necessary_at_position,
        parse_goal_coord,
    )

    if "at_position" not in oracle_names:
        return ""
    goal_xy = parse_goal_coord(goal_text)
    if goal_xy is not None:
        ok = any(ev[id(s)] == "GROUNDED" and at_position_const_matches(sg_by_name[s.sub_goal_name].verify, goal_xy)
                 for s in graded)
        return "" if ok else f"no actor-caused step verifies at_position{goal_xy}"
    if goal_has_coordinate_intent(goal_text):
        ok = any(ev[id(s)] == "GROUNDED" and has_necessary_at_position(sg_by_name[s.sub_goal_name].verify)
                 for s in graded)
        return "" if ok else "no actor-caused step verifies the commanded coordinate"
    return ""


def _manipulation_reason(
    trace: ExecutionTrace, graded: list[StepRecord], sg_by_name: dict[str, SubGoal],
    ev: dict[int, StepEvidence], oracle_names: frozenset[str],
) -> str:
    """D17/D69 manipulation gate, independent of the coordinate gate: a goal (or a
    dispatched gripper skill) with a manipulation intent needs a GROUNDED graded step
    that necessarily asserts its proof oracle — ``holding_object`` for a grasp /
    fetch (``placed_count`` never proves a grasp), a placement or release for a pure
    place. Scoped to a world with ``holding_object``."""
    from vector_os_nano.vcli.cognitive.object_goal import (
        goal_intents,
        has_necessary_manip_oracle,
        proof_oracles,
    )

    if "holding_object" not in oracle_names:
        return ""
    intents = set(goal_intents(trace.goal_tree.goal)) | set(getattr(trace, "intents", ()) or ())
    names = proof_oracles(intents)
    if not names:
        return ""
    ok = any(ev[id(s)] == "GROUNDED" and has_necessary_manip_oracle(sg_by_name[s.sub_goal_name].verify, names)
             for s in graded)
    return "" if ok else f"a {'/'.join(sorted(intents))} goal needs an actor-caused {'/'.join(sorted(names))} step"


def turn_reject_reason(trace: ExecutionTrace, oracle_names: frozenset[str]) -> str:
    """Why the turn is NOT verified, or "" when it is — the ONE done-gate.

    Whole-trace rules: the trace succeeded, every step maps to a sub-goal, no foreach
    node, and at least one graded step. Graded = the latest attempt of each attempt
    group (``graded_indices``); every graded action succeeded.

    - Every graded step must classify GROUNDED, with ONE exception: under a task
      contract (the producer compiled one from the goal), a row the producer marked
      ``housekeeping`` (its whole chain read-only or a declared housekeeping action:
      a trailing stop, where_am_i) may stay RAN unless its verify FAILED — and only
      when EVERY row of its attempt group is housekeeping and succeeded. An
      actuating row is never exempt, and a housekeeping row that superseded one
      (stop -> the same re-verify) must GROUND on its own, so a sub-goal the
      contract does not cover ("... then turn left") needs its own GROUNDED step.
    - With a contract, each predicate must also be False at turn start, True at turn
      end, and asserted by a GROUNDED graded step.

    Then the coordinate and the manipulation gates run INDEPENDENTLY (both apply
    when both intents are present). Every rule only rejects (rule 5).
    """
    if not trace.success:
        return "the turn did not succeed"
    sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
    if any(getattr(sg, "foreach", None) is not None for sg in trace.goal_tree.sub_goals):
        return "a foreach plan cannot be graded step by step"
    if any(s.sub_goal_name not in sg_by_name for s in trace.steps):
        return "a step has no sub-goal to grade it against"
    groups = attempt_groups(trace)
    graded = [trace.steps[g[-1]] for g in groups]
    if not graded:
        return "no checked step"
    goal_text = trace.goal_tree.goal
    ev: dict[int, StepEvidence] = {}
    for s, group in zip(graded, groups):
        sg = sg_by_name[s.sub_goal_name]
        evidence, why = _classify(s, sg, oracle_names, goal_text)
        ev[id(s)] = evidence
        if evidence == "FAILED":
            return f"{s.sub_goal_name}: {why}"
        # A housekeeping row never stands in for an attempt it superseded: exempt
        # only when EVERY row of its group is housekeeping and succeeded.
        exempt = bool(trace.contract) and all(
            getattr(trace.steps[j], "housekeeping", False) is True and trace.steps[j].success
            for j in group
        )
        if exempt:
            if sg.verify and not s.verify_result:
                return f"{s.sub_goal_name}: {why}"
        elif evidence != "GROUNDED":
            return f"{s.sub_goal_name}: {why}"
    # A producer that grades attempts (``attempt_of``: the native runner) must carry
    # the contract the goal compiles to: dropping it (or compiling it against another
    # body's oracles) can never skip the check. The legacy executor (no attempt_of,
    # deleted in R1) predates contracts and keeps the coordinate / manipulation gates.
    from vector_os_nano.vcli.cognitive.object_goal import task_contract

    expected = task_contract(goal_text, oracle_names, str(getattr(trace, "world", "") or ""))
    if trace.contract or (expected and any(getattr(s, "attempt_of", "") for s in trace.steps)):
        why = _contract_reason(trace, graded, sg_by_name, ev, oracle_names)
        if why:
            return why
    return (
        _coordinate_reason(graded, sg_by_name, ev, oracle_names, goal_text)
        or _manipulation_reason(trace, graded, sg_by_name, ev, oracle_names)
    )


def evidence_passed(trace: ExecutionTrace, oracle_names: frozenset[str]) -> bool:
    """True when the turn's success is backed by deterministic evidence (see
    ``turn_reject_reason``, the single source). *oracle_names* MUST be the live
    ``verify_oracle_names``; an empty set fails closed (everything RAN)."""
    return not turn_reject_reason(trace, oracle_names)


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
