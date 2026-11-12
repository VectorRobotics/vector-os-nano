# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""verdict — the machine-checkable acceptance signal for a single cli.main turn.

This is the acceptance INSTRUMENT for the orchestration redesign. The honest
verdict the engine already computes for a native run
(``evidence_passed(trace, verify_oracle_names(agent, engine))``) previously NEVER
escaped cli.main as a machine signal — the REPL only ``console.print``ed Rich
prose. ``VerdictReport`` turns that exact same computation into a frozen,
JSON-serializable record so a non-interactive ``-p/--json`` turn can emit ONE
stdout line a harness asserts against.

Honesty by construction (CLAUDE.md rule 5 — verify is the moat):
``VerdictReport.from_trace`` re-uses the EXISTING ``classify_step_evidence`` /
``evidence_passed`` from ``trace_store`` for EVERY field — it NEVER re-derives a
verdict with its own logic. The contract test pins this:

    VerdictReport.from_trace(trace, oracle_names).verified
        == evidence_passed(trace, oracle_names)

so the machine signal can only ever AGREE with the gate the engine itself uses.
An empty oracle set, a sentinel ``""``/``"True"`` verify, an absent oracle or a
tautology all classify RAN (not GROUNDED), so the signal fails CLOSED — the moat
only ever gets stricter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from vector_os_nano.vcli.cognitive.trace_store import (
    classify_step_evidence,
    goal_check,
    graded_indices,
    step_reject_reason,
    turn_reject_reason,
)

# Fixed stdout sentinel a harness scans for. NEVER changed lightly — it is the
# machine contract between cli.main and the PTY harness / CI gate.
VERDICT_SENTINEL = "VECTOR_VERDICT"

# The top-level evidence verdict for a whole turn.
#   GROUNDED — verified: success backed by deterministic, oracle-consuming evidence.
#   RAN      — the turn ran (some/all steps succeeded) but carries NO grounded evidence.
#   FAILED   — the trace did not succeed (>=1 step failed / aborted).
#   NO_TRACE — no trace was produced (a chat turn, or an error before the turn ran).
EVIDENCE_GROUNDED = "GROUNDED"
EVIDENCE_RAN = "RAN"
EVIDENCE_FAILED = "FAILED"
EVIDENCE_NO_TRACE = "NO_TRACE"

# Exit codes the harness asserts against (verified == (exit == 0)).
EXIT_VERIFIED = 0
EXIT_ERROR = 1
EXIT_RAN_NOT_VERIFIED = 2


@dataclass(frozen=True)
class StepVerdict:
    """Per-step evidence row — a pure projection of one StepRecord+SubGoal."""

    name: str
    strategy: str
    success: bool
    verify: str
    verify_result: bool
    evidence: str  # GROUNDED | RAN | FAILED (from classify_step_evidence)
    # WP07 audit (additive): the causation grade, why the step is not GROUNDED ("" when
    # it is; trace_store.step_reject_reason), and whether a later attempt of the same
    # sub-goal superseded it (then the gate does not grade it; the row stays as audit).
    actor_caused: str = "NOT_GRADED"
    reject_reason: str = ""
    superseded: bool = False
    # Charter 3.5 (additive): the row is a failed attempt (evidence FAILED, its own
    # reject_reason; it rejects the turn unless a later attempt of its predicate
    # superseded it), and
    # the calls refused in its step ("<tool>: <message>": tried, not an action).
    failed_attempt: bool = False
    refused: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerdictReport:
    """A frozen, JSON-serializable verdict for ONE executed turn.

    Built ONLY from ``trace_store.evidence_passed`` /
    ``trace_store.classify_step_evidence`` — never re-derived (see module docstring
    + the contract test ``test_verdict_matches_evidence_passed``).
    """

    verified: bool
    success: bool
    evidence: str  # GROUNDED | RAN | FAILED | NO_TRACE
    goal: str
    n_steps: int
    n_grounded: int
    oracle_names: tuple[str, ...]
    per_step: tuple[StepVerdict, ...] = ()
    error: str = ""
    # Display metadata (never part of the gate): the locomotion backend the sim
    # body actually ran ('mpc' / 'sinusoidal'), '' when there is no sim base.
    gait_backend: str = ""
    # The LLM backend failed (after its own bounded retries): an INFRASTRUCTURE
    # outcome, not an agent failure — a bench excludes it, and exit_code() says error.
    llm_unavailable: bool = False
    model: str = ""  # backend model id that drove the turn ("" when unknown)
    # WP07 (additive): why the turn is not verified ("" when it is; the gate's own
    # trace_store.turn_reject_reason), whether any side-effecting tool ran, the sim the
    # turn ran in, and the task contract the goal compiled to.
    reject_reason: str = ""
    acted: bool = False
    scene: str = ""
    contract: tuple[str, ...] = ()
    # Charter 3.5 (additive): whether the user's end state was machine-checked --
    # "proven" | "failed" | "unchecked" (trace_store.goal_check), never merged into
    # ``verified``. A consumer that needs "the task is done" reads goal_check == "proven".
    goal_check: str = "unchecked"

    # ------------------------------------------------------------------
    # Construction — the ONLY supported way to build a verdict from a run.
    # ------------------------------------------------------------------

    @classmethod
    def from_trace(
        cls, trace: Any, oracle_names: frozenset[str], *, gait_backend: str = ""
    ) -> "VerdictReport":
        """Build a verdict from an ExecutionTrace using the EXISTING gate.

        ``verified`` is delegated VERBATIM to the gate (``turn_reject_reason`` is
        ``evidence_passed``'s single source; no second opinion). Per-step
        ``evidence`` is delegated to ``classify_step_evidence``. ``n_steps`` /
        ``n_grounded`` count the GRADED steps (the latest attempt of each sub-goal);
        superseded attempts stay in ``per_step`` as audit. The top-level
        ``evidence`` summarizes: GROUNDED iff verified, else FAILED if the trace did
        not succeed, else RAN.
        """
        sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
        reject_reason = turn_reject_reason(trace, oracle_names)
        verified = not reject_reason
        graded = set(graded_indices(trace))

        per_step: list[StepVerdict] = []
        n_grounded = 0
        for i, s in enumerate(trace.steps):
            sg = sg_by_name.get(s.sub_goal_name)
            if sg is None:
                # A step with no matching sub-goal cannot be classified by the
                # gate — record it as RAN-shaped metadata only (it never counts
                # toward GROUNDED, mirroring evidence_passed which ignores it).
                ev = EVIDENCE_RAN if s.success else EVIDENCE_FAILED
                verify_str = ""
                why = "no sub-goal to grade it against"
            else:
                ev = classify_step_evidence(s, sg, oracle_names, trace.goal_tree.goal)
                verify_str = sg.verify
                why = step_reject_reason(s, sg, oracle_names, trace.goal_tree.goal)
            if ev == EVIDENCE_GROUNDED and i in graded:
                n_grounded += 1
            per_step.append(
                StepVerdict(
                    name=s.sub_goal_name,
                    strategy=s.strategy,
                    success=bool(s.success),
                    verify=verify_str,
                    verify_result=bool(s.verify_result),
                    evidence=ev,
                    actor_caused=getattr(getattr(s, "actor_caused", None), "value", "NOT_GRADED"),
                    reject_reason=why,
                    superseded=i not in graded,
                    failed_attempt=getattr(s, "failed_attempt", False) is True,
                    refused=tuple((getattr(s, "result_data", None) or {}).get("refused", ())),
                )
            )

        llm_error = str(getattr(trace, "llm_error", "") or "")
        if verified:
            top_evidence = EVIDENCE_GROUNDED
        elif llm_error and not trace.steps:
            top_evidence = EVIDENCE_NO_TRACE  # the LLM never got the turn going
        elif not trace.success:
            top_evidence = EVIDENCE_FAILED
        else:
            top_evidence = EVIDENCE_RAN

        return cls(
            verified=verified,
            success=bool(trace.success),
            evidence=top_evidence,
            goal=trace.goal_tree.goal,
            n_steps=len(graded),
            n_grounded=n_grounded,
            oracle_names=tuple(sorted(oracle_names)),
            per_step=tuple(per_step),
            gait_backend=str(gait_backend or ""),
            error=f"LLM unavailable: {llm_error}" if llm_error else "",
            llm_unavailable=bool(llm_error),
            model=str(getattr(trace, "model", "") or ""),
            reject_reason=reject_reason,
            acted=bool(getattr(trace, "actions_dispatched", 0)),
            scene=str(getattr(trace, "scene", "") or ""),
            contract=tuple(getattr(trace, "contract", ()) or ()),
            goal_check=goal_check(trace, oracle_names)[0],
        )

    @classmethod
    def no_trace(cls, goal: str = "", error: str = "") -> "VerdictReport":
        """A fail-closed verdict for a turn that produced NO trace.

        A chat-only / tool_use turn (or an error before any trace) has no
        deterministic per-step evidence to grade, so it can NEVER be verified.
        """
        return cls(
            verified=False,
            success=False,
            evidence=EVIDENCE_NO_TRACE,
            goal=goal,
            n_steps=0,
            n_grounded=0,
            oracle_names=(),
            per_step=(),
            error=error,
        )

    # ------------------------------------------------------------------
    # Serialization + exit-code contract.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A plain JSON-safe dict (frozen dataclasses -> dicts, tuples -> lists)."""
        d = asdict(self)
        d["oracle_names"] = list(self.oracle_names)
        d["contract"] = list(self.contract)
        d["per_step"] = [asdict(s) for s in self.per_step]
        return d

    def to_sentinel_line(self) -> str:
        """The single stdout line a harness scans for: ``VECTOR_VERDICT {<json>}``."""
        return f"{VERDICT_SENTINEL} {json.dumps(self.to_dict(), ensure_ascii=False)}"

    def exit_code(self) -> int:
        """0 = verified, 2 = ran-not-verified, 1 = error / no trace / LLM unavailable.

        ``verified == (exit_code() == 0)`` is the harness's invariant; this method
        is the single source of that mapping.
        """
        if self.verified:
            return EXIT_VERIFIED
        if self.evidence == EVIDENCE_NO_TRACE or self.llm_unavailable:
            return EXIT_ERROR
        return EXIT_RAN_NOT_VERIFIED
