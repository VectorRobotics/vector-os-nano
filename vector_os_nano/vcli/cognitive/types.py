# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The trace the native producer records and the spine grades — frozen dataclasses.

All types are immutable (frozen=True) to ensure safe sharing across threads
without defensive copying.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused


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


@dataclass(frozen=True)
class GoalTree:
    """Full decomposition of a high-level task into ordered SubGoals."""

    goal: str
    sub_goals: tuple[SubGoal, ...]
    context_snapshot: str = ""


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
    # R2b — TRI-STATE actor-causation grade (NOT bool|None). Whether the ACTOR
    # caused this step's GROUNDED state change: CAUSED (commanded motion via the
    # instrumented ctrl path AND the pose changed), UNCAUSED (a satisfied-at-baseline
    # NO-OP or a teleport — no commanded motion), or NOT_GRADED (the executor never
    # evaluated causation for this step — a non-robot-predicate step or a legacy/
    # hand-built trace). ``classify_step_evidence`` downgrades a GROUNDED robot step
    # to RAN ONLY when this is ``UNCAUSED``; ``NOT_GRADED`` (the default) classifies
    # EXACTLY as before R2b, so every existing constructor and legacy trace is
    # byte-unaffected (rule 6). Stored as the ENUM (serialized via its ``.value``).
    # Additive + LAST + defaulted ``NOT_GRADED`` (frozen-safe).
    actor_caused: "ActorCaused" = field(default_factory=lambda: ActorCaused.NOT_GRADED)
    # WP07 attempt semantics (additive, rule 6). ``attempt_of`` opts the step into an
    # attempt group: steps with the same ``attempt_of`` AND the same normalized verify
    # are attempts at one sub-goal, and only the LATEST is graded (the spine computes
    # this itself — ``trace_store.graded_steps``; "" = its own group, never
    # superseded). ``superseded`` is that computation recorded for display/audit (a
    # FAIL row stays in the trace). ``channels`` is the per-call causation detail
    # (``"base:at_position:CAUSED"``, ``"auto-verify"``) for the audit row.
    attempt_of: str = ""
    superseded: bool = False
    channels: tuple[str, ...] = ()
    # The producer recorded that every tool of the step's chain was read-only or a
    # declared housekeeping action (``stop``). Only such a row may stay un-GROUNDED
    # (when another graded row proved the turn), and only when its whole attempt group
    # is housekeeping; the default (False) is an actuating row (fail closed).
    housekeeping: bool = False


@dataclass(frozen=True)
class ExecutionTrace:
    """Complete record of a goal execution run."""

    goal_tree: GoalTree
    steps: tuple[StepRecord, ...]
    success: bool
    total_duration_sec: float
    # Producer bookkeeping (additive + defaulted, rule 6). ``actions_dispatched``
    # counts the side-effecting tool calls the producer ran — nonzero means the
    # turn acted and must never be re-planned by another producer. ``llm_error``
    # marks an LLM-backend failure (infrastructure, not an agent failure);
    # ``model`` is the backend model id that drove the turn; ``refused`` lists the
    # tool calls the permission gate refused ("<tool>: <why>"); ``cancelled`` marks a
    # turn the owner stopped (no other producer may run it after the stop).
    actions_dispatched: int = 0
    llm_error: str = ""
    model: str = ""
    refused: tuple[str, ...] = ()
    cancelled: bool = False
    # WP07 turn-level task contract (additive): the GT predicates the NL goal compiles
    # to (``object_goal.task_contract``) and their values at turn start / turn end.
    # Each must be False at the start (else "already satisfied", never GROUNDED) and
    # True at the end. ``intents`` adds manipulation intents the producer derived from
    # the skills it dispatched ("grasp" / "place") to the ones the goal text names.
    contract: tuple[str, ...] = ()
    contract_start: tuple[bool, ...] = ()
    contract_end: tuple[bool, ...] = ()
    intents: tuple[str, ...] = ()
    scene: str = ""  # the sim the turn ran in ("go2_piper@go2_room"), display only
    # The world manifest id the contract bound its object / surface through
    # (``config/worlds/<id>.yaml``); the spine recompiles the contract against it.
    world: str = ""
