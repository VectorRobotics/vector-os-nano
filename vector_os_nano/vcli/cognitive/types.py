# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""VGG cognitive layer — frozen dataclasses.

All types are immutable (frozen=True) to ensure safe sharing across
async executor threads without defensive copying.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------
# Failure taxonomy (W2.4) — a small CLOSED set of lowercase, world-agnostic
# class strings, plus "" for success / no-failure. This is a DETERMINISTIC
# typed signal derived ONLY from already-available execution evidence
# (exception type, the step's diagnosis, the verify-vs-execute distinction) —
# NO new model call. It lets observation-driven replan branch on the failure
# CLASS instead of an opaque stringified error (e.g. timeout -> a shorter
# strategy; ik_fail -> an alternate grasp pose); the LLM does the adapting.
#
# Kept embodiment-neutral on purpose: ``ik_fail`` is named after the GENERIC
# unreachable-target failure mode, not the arm — any world that surfaces an
# ``ik_unreachable``/``unreachable`` diagnosis maps to it.
# --------------------------------------------------------------------------

FailureClass = Literal[
    "",            # success / no failure
    "timeout",     # the step exceeded its (floored) wall-clock limit
    "verify_fail", # executed without error but the deterministic verify was False
    "ik_fail",     # an unreachable-target / IK diagnosis (e.g. ik_unreachable)
    "tool_error",  # a kernel-tool dispatch / permission / allowlist failure
    "exec_error",  # any other execution failure (selector/skill raised, generic)
]

# The closed set, for validation + tests. Excludes "" (the success sentinel).
FAILURE_CLASSES: frozenset[str] = frozenset(
    {"timeout", "verify_fail", "ik_fail", "tool_error", "exec_error",
     # Batch 1 campaign #4 (#11): the step never RAN — a (transitive)
     # depends_on ancestor failed, so executing it would act from an
     # unestablished world state (the hardware-dangerous case).
     "dep_skipped"}
)

# Diagnosis substrings (lowercased) that indicate an unreachable-target / IK
# failure, regardless of embodiment. Matched as substrings so "ik_unreachable",
# "unreachable", etc. all classify as ``ik_fail``.
_IK_DIAGNOSIS_MARKERS: tuple[str, ...] = ("ik_unreachable", "unreachable", "ik_fail")


def classify_exec_failure(
    *,
    executor_type: str = "",
    diagnosis: str = "",
) -> str:
    """Classify a NON-timeout, NON-verify execution failure deterministically.

    Used for the ``exec_success is False`` path (the strategy ran but reported a
    failure, e.g. a skill returned ``success=False`` with a ``diagnosis``, or the
    tool/capability branch failed). Derived ONLY from already-available signals —
    the executor_type of the resolved strategy and the step's machine-readable
    ``diagnosis`` — so there is NO new model call.

    Mapping (most specific wins):
      - a diagnosis naming an unreachable-target / IK failure -> ``ik_fail``
      - the kernel-``tool`` executor branch                   -> ``tool_error``
      - anything else                                         -> ``exec_error``

    The caller owns the timeout (-> ``timeout``) and verify-miss
    (-> ``verify_fail``) paths; this helper never returns those or ``""``.
    """
    diag = (diagnosis or "").strip().lower()
    if diag and any(marker in diag for marker in _IK_DIAGNOSIS_MARKERS):
        return "ik_fail"
    if executor_type == "tool":
        return "tool_error"
    return "exec_error"


@dataclass(frozen=True)
class ForEachSpec:
    """Control-flow FOREACH spec attached to a SubGoal (Stage 4, S4-1).

    Declares that a sub-goal's *body* is instantiated once per item of a list
    produced by an EARLIER step. The list is reached through the SAME pure
    ``${step.path}`` Blackboard convention every other step reference uses — e.g.
    ``detect_all.objects`` resolves to the detect step's ``result_data["objects"]``
    list. Each iteration binds the item to ``var`` (default ``"item"``); body
    sub-goal templates reference the item's fields with that var name (e.g.
    ``${obj.name}`` in strategy_params / verify), which the per-item binder
    resolves by PURE dict/list traversal — never string eval/format.

    S4-1 only models + parses + validates this spec; expansion at execution time
    is deferred to S4-2. All fields are frozen so the spec is safe to share.
    """

    # Producing step name (the step whose result_data holds the iterable).
    source_step: str
    # Dotted path INTO that step's result_data reaching the list to iterate, e.g.
    # ``"objects"`` or ``"objects.detections"``. Resolved by the Blackboard's pure
    # dict/list traversal; never evaluated.
    source_path: str
    # Iteration variable name bound to each item (referenced as ``${<var>.<field>}``
    # inside body templates). Defaults to ``"item"``.
    var: str = "item"
    # Body templates instantiated once per item. Empty tuple is a no-op loop.
    body: tuple["SubGoal", ...] = ()


@dataclass(frozen=True)
class SubGoal:
    """A single verifiable step in a goal decomposition tree."""

    name: str
    description: str
    verify: str  # Python expression evaluated by GoalVerifier
    timeout_sec: float = 30.0
    depends_on: tuple[str, ...] = ()
    strategy: str = ""
    strategy_params: dict = field(default_factory=dict)
    fail_action: str = ""
    # Stage 4 (S4-1) control flow: an optional FOREACH spec. When present, this
    # sub-goal is a loop node whose ``body`` is instantiated once per item of the
    # list produced by ``foreach.source_step``. None (the default) means a plain
    # leaf step — preserving every existing positional/keyword constructor and
    # keeping no-foreach plans byte-unaffected. Expansion is S4-2; S4-1 only
    # models + parses + validates the spec. Field is LAST + defaulted (frozen-safe).
    foreach: "ForEachSpec | None" = None
    # Stage 5 (S5.2) answer-only marker. True when this step is a pure-conversation
    # "answer" step that produces text and CARRIES NO ROBOT EVIDENCE BY DESIGN —
    # NOT an action step whose evidence is merely missing. This flag (not a
    # verify-string trick) is what lets the evidence gate DISTINGUISH a legitimate
    # answer-only step from an action step that produced no evidence, so the moat
    # (rule 5) is never weakened: an action step with a sentinel verify still fails
    # the gate; only an explicitly-flagged answer step is exempt. Additive + LAST +
    # defaulted False so every existing constructor and non-answer plan is
    # byte-unaffected.
    answer_only: bool = False
    # Fail-loud marker for a HALLUCINATED strategy (rule 8). When the decomposer
    # validates a step whose explicit ``strategy`` is not in the world's known set,
    # it clears ``strategy`` to "" (so existing pure-check routing is unaffected)
    # AND records the offending name here. A non-empty value tells the selector to
    # resolve this step to the LOUD ``invalid`` route (clear, named error + valid
    # set) instead of silently re-routing the cleared step through keyword/registry
    # matching to a phantom skill or the opaque ``unmatched`` fallback. Empty (the
    # default) means "no hallucination" — every existing constructor and every
    # legitimately strategy-less (pure-check / foreach owner) plan is byte-
    # unaffected. Additive + LAST + defaulted (frozen-safe, rule 6).
    cleared_strategy: str = ""


@dataclass(frozen=True)
class GoalTree:
    """Full decomposition of a high-level task into ordered SubGoals."""

    goal: str
    sub_goals: tuple[SubGoal, ...]
    context_snapshot: str = ""
    # Validator feedback (Stage 2b): human-readable notes about strategies or
    # sub_goals the decomposer DROPPED during validation (e.g. an unknown
    # strategy cleared to ""). The harness threads these into the next replan's
    # decompose context so the LLM stops repeating the hallucination. Additive +
    # keyword-default so existing positional constructors are unaffected.
    validation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepRecord:
    """Execution record for a single SubGoal attempt."""

    sub_goal_name: str
    strategy: str
    success: bool
    verify_result: bool
    duration_sec: float
    error: str = ""
    fallback_used: bool = False
    # True when verify_result was set by a VLM visual-verification override
    # rather than the deterministic predicate — NOT deterministic evidence
    # (see trace_store.evidence_passed).
    visual_override: bool = False
    # Captured structured output of the step (Stage 1a). Typically
    # {"output": <strategy output dict>, "verify_value": <raw verify value>}.
    # Additive + keyword-default so existing positional constructors are unaffected.
    result_data: dict = field(default_factory=dict)
    # W2.4 — DETERMINISTIC typed failure class for a FAILED step (one of
    # FAILURE_CLASSES; "" for a success/no-failure step). Derived ONLY from
    # already-available evidence (timeout vs verify-miss vs execution failure,
    # plus the step's machine-readable ``diagnosis`` and executor_type) — NO new
    # model call. The harness threads this into the replan/re-decompose context so
    # the re-plan (LLM) can branch on the failure CLASS instead of parsing the
    # opaque ``error`` string. SECURITY (rule 10): a bounded enum string ONLY —
    # never raw exception detail, file paths, or secrets. Additive + LAST +
    # defaulted "" so every existing positional/keyword constructor is
    # byte-unaffected (rule 6).
    failure_class: str = ""
    # Invariant I (design review 2026-06-12 #1): the NON-TRIVIAL predicate's
    # value BEFORE the strategy ran. True means the goal state was already
    # satisfied at dispatch time — 'became true' and 'was already true' are no
    # longer the same True (the '走到厨房' zero-motion-PASS family). Always
    # False for trivial/sentinel verifies. Additive + LAST + defaulted (rule 6).
    pre_satisfied: bool = False


@dataclass(frozen=True)
class ExecutionTrace:
    """Complete record of a goal execution run."""

    goal_tree: GoalTree
    steps: tuple[StepRecord, ...]
    success: bool
    total_duration_sec: float
