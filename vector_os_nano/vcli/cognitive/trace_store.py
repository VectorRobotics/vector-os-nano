# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""trace_store — grade the evidence of a native turn's execution trace.

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
import logging
from typing import Any, Literal

from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.oracle_registry import OracleTag, oracle_tags
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    StepRecord,
    SubGoal,
)

logger = logging.getLogger(__name__)

StepEvidence = Literal["GROUNDED", "RAN", "FAILED"]

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

    The ONE source of truth for the done-gate (``evidence_passed``) and the
    verdict's per-step rows — rule 3 (no split-brain): a robot step that merely RAN
    (``verify="True"``) is never auto-passed; only a step whose verify actually
    consumes a world oracle in a non-tautological way reaches GROUNDED.

    - FAILED — the step did not succeed (``not step.success``).
    - GROUNDED — the step succeeded AND
      ``classify_verify_expr(sub_goal.verify, oracle_names) == "GROUNDED"`` AND
      ``step.verify_result``.
    - RAN — the step succeeded but its verify carries no real, non-tautological
      world evidence (a sentinel ``""``/``"True"``, an absent oracle, or a
      tautology).

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
    ``goal_text=None`` skips this check. Stricter-only (rule 5).
    """
    return _classify(step, sub_goal, oracle_names, goal_text)[0]


def _classify(
    step: StepRecord, sub_goal: SubGoal, oracle_names: frozenset[str], goal_text: str | None = None,
) -> tuple[StepEvidence, str]:
    """``(evidence, reason)`` for one step — the reason is "" when GROUNDED."""
    if not step.success:
        return "FAILED", f"the action failed: {step.error}" if step.error else "the action failed"
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
    ev: dict[int, tuple[StepEvidence, str]], oracle_names: frozenset[str],
) -> str:
    """Why the turn's task contract is not met ("" when it is): every predicate must
    be GT-provable here, False at turn start, True at turn end, and asserted by a
    GROUNDED (actor-caused) graded step."""
    from vector_os_nano.vcli.cognitive.object_goal import _HOLD, backs_contract, task_contract, unbound_reason

    contract, world = tuple(trace.contract), str(getattr(trace, "world", "") or "")

    def proven_at(item: str) -> list[int]:  # graded positions of the GROUNDED steps asserting item
        return [i for i, s in enumerate(graded)
                if ev[id(s)][0] == "GROUNDED" and backs_contract(item, sg_by_name[s.sub_goal_name].verify, world)]

    # The spine recompiles the contract: a producer can never swap in a weaker one.
    if contract != task_contract(trace.goal_tree.goal, oracle_names, world):
        return "the task contract does not match the goal"
    if len(trace.contract_start) != len(contract) or len(trace.contract_end) != len(contract):
        return "task contract was not evaluated at turn start and end"
    for item, was, now in zip(contract, trace.contract_start, trace.contract_end):
        if classify_verify_expr(item, oracle_names) != "GROUNDED":
            why = unbound_reason(item, trace.goal_tree.goal, world)
            return f"task contract {item}: {why}" if why else f"task contract {item} is not provable in this world"
        proofs = proven_at(item)
        # A pick-and-place starts released: that is no evidence unless a grounded grasp
        # of the same object precedes the release proof (I4F1-D3).
        obj = item[len("released("):-1] if item.startswith("released(") else None
        picked = obj is not None and proofs and any(
            i < proofs[-1] for n in _HOLD for i in proven_at(f"{n}({obj})"))
        if was and not picked:
            return f"already satisfied at turn start: {item}"
        if not now:
            return f"task contract not met at turn end: {item}"
        if not proofs:
            return f"no actor-caused step proves {item}"
    return ""


def _coordinate_reason(
    graded: list[StepRecord], sg_by_name: dict[str, SubGoal], ev: dict[int, tuple[StepEvidence, str]],
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
        ok = any(ev[id(s)][0] == "GROUNDED" and at_position_const_matches(sg_by_name[s.sub_goal_name].verify, goal_xy)
                 for s in graded)
        return "" if ok else f"no actor-caused step verifies at_position{goal_xy}"
    if goal_has_coordinate_intent(goal_text):
        ok = any(ev[id(s)][0] == "GROUNDED" and has_necessary_at_position(sg_by_name[s.sub_goal_name].verify)
                 for s in graded)
        return "" if ok else "no actor-caused step verifies the commanded coordinate"
    return ""


def _manipulation_reason(
    trace: ExecutionTrace, graded: list[StepRecord], sg_by_name: dict[str, SubGoal],
    ev: dict[int, tuple[StepEvidence, str]], oracle_names: frozenset[str],
) -> str:
    """D17/D69 manipulation gate, independent of the coordinate gate: a manipulation
    intent -- in the goal's words, in a dispatched gripper skill's hint, or in a GROUNDED
    graded step whose verify asserts a possession or a placement (it was performed,
    whatever skill ran: gripper_close's driver is where the world grasps, REF1c-V-01) --
    needs a GROUNDED graded step that necessarily asserts EACH proof it needs
    (``proof_oracles``) — a possession oracle for a grasp / fetch (``placed_count`` never
    proves a grasp), a placement or release for a place, both for a pick-and-place. Scoped to a world with
    ``holding_object``. With a world manifest (charter 3.5) the goal must be a closed
    form, and every proof names the object its contract bound."""
    from vector_os_nano.vcli.cognitive.object_goal import (
        _manifest,
        _sentence,
        backs_contract,
        checkable_goals,
        goal_intents,
        has_necessary_manip_oracle,
        proof_oracles,
        verify_intents,
    )

    if "holding_object" not in oracle_names:
        return ""
    goal = goal_intents(trace.goal_tree.goal)
    dispatched = set(getattr(trace, "intents", ()) or ()).union(
        *(verify_intents(sg_by_name[s.sub_goal_name].verify) for s in graded if ev[id(s)][0] == "GROUNDED"))
    # I4F1-D2: a hold of the red can never proves "pick up the green bottle"; an unbound
    # object fails closed, and a goal outside the forms binds none at all.
    world_id = str(getattr(trace, "world", "") or "")
    world = _manifest(world_id)
    form = _sentence(world, trace.goal_tree.goal or "") if world is not None else None
    want = form[1] if form is not None else None

    def proves(verify: str | None, names: frozenset[str]) -> bool:
        return has_necessary_manip_oracle(verify, names) and (
            world is None or any(backs_contract(f"{n}({want!r})", verify, world_id) for n in names))

    for names in proof_oracles(goal, dispatched):
        if world is not None and form is None:
            return (f"the goal {trace.goal_tree.goal!r} is outside the forms the spine can check, so its "
                    f"manipulation is not verified; checkable goals: {'; '.join(checkable_goals(world, oracle_names))}")
        if world is not None and want is None:
            return "a manipulation goal that names no object the world resolves is unprovable"
        if not any(ev[id(s)][0] == "GROUNDED" and proves(sg_by_name[s.sub_goal_name].verify, names)
                   for s in graded):
            return f"a {'/'.join(sorted(goal | dispatched))} goal needs an actor-caused {'/'.join(sorted(names))} step"
    return ""


def _graded(trace: ExecutionTrace, oracle_names: frozenset[str]) -> Any:
    """``(attempt groups, graded steps, sub-goal by name, {id(step): (evidence, why)})``
    for the latest attempt of every group, or why the trace cannot be graded step by step."""
    sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
    if any(s.sub_goal_name not in sg_by_name for s in trace.steps):
        return "a step has no sub-goal to grade it against"
    groups = attempt_groups(trace)
    graded = [trace.steps[g[-1]] for g in groups]
    if not graded:
        return "no checked step"
    ev = {id(s): _classify(s, sg_by_name[s.sub_goal_name], oracle_names, trace.goal_tree.goal) for s in graded}
    return groups, graded, sg_by_name, ev


def _checks_contract(trace: ExecutionTrace, oracle_names: frozenset[str]) -> bool:
    """Whether the turn has a task contract: one the trace carries, or the one its goal
    compiles to when the producer grades attempts (``attempt_of``: the native runner), so
    dropping it (or compiling it against another body's oracles) never skips the check.
    The legacy executor (no attempt_of, deleted in R1) predates contracts."""
    from vector_os_nano.vcli.cognitive.object_goal import task_contract

    if trace.contract:
        return True
    expected = task_contract(trace.goal_tree.goal, oracle_names, str(getattr(trace, "world", "") or ""))
    return bool(expected) and any(getattr(s, "attempt_of", "") for s in trace.steps)


GoalCheck = Literal["proven", "failed", "unchecked"]


def goal_check(trace: ExecutionTrace, oracle_names: frozenset[str]) -> tuple[GoalCheck, str]:
    """The turn's goal check (charter 3.5) and why it is not proven -- the ONE source
    the verdict and the REPL read. ``proven``: the task contract was compiled and holds
    (every item False at turn start, True at turn end, backed by a GROUNDED actor-caused
    step); ``failed``: it was compiled and does not hold, or a slot is unbound;
    ``unchecked``: the goal compiles no contract (no closed sentence form). Never a
    gate: it only says whether the user's end state was machine-checked."""
    if not _checks_contract(trace, oracle_names):
        return "unchecked", ""
    graded = _graded(trace, oracle_names)
    if isinstance(graded, str):
        return "failed", graded
    _groups, steps, sg_by_name, ev = graded
    why = _contract_reason(trace, steps, sg_by_name, ev, oracle_names)
    return ("failed", why) if why else ("proven", "")


def turn_reject_reason(trace: ExecutionTrace, oracle_names: frozenset[str]) -> str:
    """Why the turn is NOT verified, or "" when it is — the ONE done-gate.

    Whole-trace rules: the trace succeeded, every step maps to a sub-goal, and at
    least one graded step. Graded = the latest attempt of each attempt
    group (``graded_indices``); every graded action succeeded.

    - Every graded step must classify GROUNDED, with ONE exception: a row the
      producer marked ``housekeeping`` (its whole chain read-only or a declared
      housekeeping action: a trailing stop, where_am_i) may stay RAN unless its
      verify FAILED — and only when EVERY row of its attempt group is housekeeping
      and succeeded. An actuating row is never exempt, and a housekeeping row that
      superseded one (stop -> the same re-verify) must GROUND on its own, so a
      sub-goal the contract does not cover ("... then turn left") needs its own
      GROUNDED step. At least one graded row must be GROUNDED: a turn that only
      read or halted has no checked step.
    - With a contract (``goal_check``), the goal must be proven.

    Then the coordinate and the manipulation gates run INDEPENDENTLY (both apply
    when both intents are present). Every rule only rejects (rule 5).
    """
    if not trace.success:
        return "the turn did not succeed"
    graded = _graded(trace, oracle_names)
    if isinstance(graded, str):
        return graded
    groups, steps, sg_by_name, ev = graded
    proven = False
    for s, group in zip(steps, groups):
        sg = sg_by_name[s.sub_goal_name]
        evidence, why = ev[id(s)]
        if evidence == "FAILED":
            return f"{s.sub_goal_name}: {why}"
        # A housekeeping row never stands in for an attempt it superseded: exempt
        # only when EVERY row of its group is housekeeping and succeeded.
        exempt = all(
            getattr(trace.steps[j], "housekeeping", False) is True and trace.steps[j].success
            for j in group
        )
        if exempt:
            if sg.verify and not s.verify_result:
                return f"{s.sub_goal_name}: {why}"
        elif evidence != "GROUNDED":
            return f"{s.sub_goal_name}: {why}"
        proven = proven or evidence == "GROUNDED"
    if not proven:
        return "no checked step"  # every graded row only read / halted: nothing proven
    state, why = goal_check(trace, oracle_names)
    if state == "failed":
        return why
    return (
        _coordinate_reason(steps, sg_by_name, ev, oracle_names, trace.goal_tree.goal)
        or _manipulation_reason(trace, steps, sg_by_name, ev, oracle_names)
    )


def evidence_passed(trace: ExecutionTrace, oracle_names: frozenset[str]) -> bool:
    """True when the turn's success is backed by deterministic evidence (see
    ``turn_reject_reason``, the single source). *oracle_names* MUST be the live
    ``verify_oracle_names``; an empty set fails closed (everything RAN)."""
    return not turn_reject_reason(trace, oracle_names)

