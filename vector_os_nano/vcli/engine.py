# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""VectorEngine — core tool_use agent loop for Vector CLI.

Mirrors Claude Code's query.ts / toolOrchestration.ts pattern.
Backend-agnostic: works with any LLMBackend (Anthropic, OpenRouter, local).

Public exports:
    ToolCall     — frozen record of a single tool execution
    TurnResult   — frozen result of one full user turn (may span N API calls)
    ToolBatch    — internal grouping for concurrent vs sequential execution
    VectorEngine — the stateful agent loop
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from vector_os_nano.vcli.backends import LLMBackend
from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session, TokenUsage
from vector_os_nano.vcli.tool_execution import (
    DECISION_ASK_ALLOW,
    DECISION_ASK_DENY,
    DECISION_DENY,
    execute_resolved_tool,
    resolve_permission,
)
from vector_os_nano.vcli.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
)

# Lazy import guard — VGG components may not be installed in all deployments
try:
    from vector_os_nano.vcli.cognitive import (
        GoalDecomposer,
        GoalExecutor,
        GoalVerifier,
        StrategySelector,
        StrategyStats,
    )
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, SubGoal, StepRecord
    from vector_os_nano.vcli.cognitive.vgg_harness import VGGHarness, HarnessConfig
    # Single-source the generic target-binding / prefer-suggested-verify guidance
    # so the engine's neutral intro carries exactly what the registry vocab does.
    from vector_os_nano.vcli.cognitive.vocab_from_registry import (
        _TARGET_BINDING_GUIDANCE,
    )
    _VGG_AVAILABLE = True
except ImportError:
    _VGG_AVAILABLE = False
    # Fallback copy kept in lockstep with vocab_from_registry._TARGET_BINDING_GUIDANCE.
    _TARGET_BINDING_GUIDANCE = (
        "When a step acts on a SPECIFIC object/target named in the task, copy "
        "that target into the chosen strategy's object/object_label/query/target "
        "parameter — never leave a known target blank. "
        "When the world context carries an 'Objects (live):' list, RESOLVE the "
        "user's reference (any language, attributes like color/size included) to "
        "the EXACT listed name whose name/attributes match, and bind THAT name — "
        "the scene knows objects by those names, not by the user's wording. If "
        "nothing listed matches, bind the user's wording as-is; the step will "
        "fail loudly and you can re-bind on replan from the fresh list. "
        "For an instruction that targets a NAMED object's location, bind the object's LABEL into params when the strategy supports it (the skill resolves live coordinates and fails loudly if the object is unknown) — NEVER invent x/y for an object. "
        "Use each strategy's 'suggested verify' predicate EXACTLY as written for "
        "that step's verify expression: put the target ONLY in strategy_params, "
        "never as an argument inside the verify expression. The verifier checks "
        "deterministic ground-truth state (or the step's own recorded output via "
        "step_output()), not your target string."
    )

logger = logging.getLogger(__name__)


def _world_exempt_strategies(world: "Any | None") -> "frozenset[str]":
    """World-declared per-step evidence exemptions (invariant II), fail-safe.

    A world without the method (or one that raises) declares NOTHING exempt —
    the gate can only get stricter on the failure path, never looser (rule 5).
    """
    getter = getattr(world, "evidence_exempt_strategies", None)
    if not callable(getter):
        return frozenset()
    try:
        return frozenset(str(n) for n in getter())
    except Exception:  # noqa: BLE001
        return frozenset()


# ---------------------------------------------------------------------------
# Message parameter extraction (keyword-based, no LLM)
# ---------------------------------------------------------------------------

import re as _re

_DIR_MAP: list[tuple[tuple[str, ...], str]] = [
    (("后退", "往后", "向后", "倒退", "backward", "back", "retreat", "reverse"), "backward"),
    (("往左", "向左", "左走", "left"), "left"),
    (("往右", "向右", "右走", "right"), "right"),
    # forward is default — checked last or as fallback
    (("往前", "向前", "前进", "forward", "ahead"), "forward"),
]


def _extract_direction(msg: str) -> str:
    """Extract movement direction from user message. Default: forward."""
    msg_lower = msg.lower()
    for keywords, direction in _DIR_MAP:
        for kw in keywords:
            if kw in msg_lower:
                return direction
    return "forward"


def _extract_number(msg: str, default: float = 1.0) -> float:
    """Extract first number from message. Supports Chinese numerals."""
    _CN_NUMS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}
    # Try Arabic numerals first
    m = _re.search(r'(\d+(?:\.\d+)?)', msg)
    if m:
        return float(m.group(1))
    # Try Chinese numerals
    for cn, val in _CN_NUMS.items():
        if cn in msg:
            return float(val)
    return default


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """Immutable record of a single tool invocation within a turn."""

    tool_name: str
    params: dict[str, Any]
    result: ToolResult
    duration_sec: float
    permission_action: str  # "allowed" | "denied" | "asked_allowed" | "asked_denied"


@dataclass(frozen=True)
class TurnResult:
    """Immutable result of one full user turn (may include multiple API round-trips)."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "max_tokens" | "tool_use"
    usage: TokenUsage


@dataclass(frozen=True)
class IntentDecision:
    """Immutable, inspectable record of the planning-path routing decision.

    Stage 5 scout: today the engine forks between two planning paths — the VGG
    closed loop (decompose -> execute -> verify -> replan) and the tool_use ReAct
    loop (``run_turn``) — via the keyword intent gate (``should_use_vgg``). That
    fork was previously buried as inline booleans inside ``vgg_decompose``. This
    value object surfaces the SAME decision as a single observable artefact so the
    eventual unification (one controller, gate dropped) has a stable, testable seam
    and the decision can be logged / rendered without re-deriving it.

    Fields:
        route:  ``"vgg"`` (the cognitive closed loop owns this turn) or
                ``"tool_use"`` (fall back to the ReAct tool loop).
        reason: short, stable, human-readable explanation of why — for logs and
                the observation surface. NOT parsed for control flow.
        complex: the gate's complexity classification (multi-step / conditional /
                scope), surfaced so a caller can tell a 1-step fast path from an
                LLM-decomposed plan without re-running the keyword heuristic.
    """

    route: str  # "vgg" | "tool_use"
    reason: str
    complex: bool = False

    @property
    def use_vgg(self) -> bool:
        """True when this turn should go through the VGG closed loop."""
        return self.route == "vgg"


@dataclass(frozen=True)
class UnifiedTurnResult:
    """Immutable result of ONE closed-loop turn (Stage 5, S5.3).

    The single result type the unified controller (``run_turn_unified``) returns
    for EVERY interaction shape — answer-only chat, a 1-step skill/tool plan, or an
    N-step DAG. Unlike the two legacy surfaces (``TurnResult`` for ReAct,
    ``ExecutionTrace`` for VGG), this carries BOTH the assistant text AND the
    verified-loop trace so a front-end can render the answer and the per-step
    PASS/FAIL the same way regardless of shape.

    Fields:
        text:    the assistant's answer text (the chat answer for an answer-only
                 turn; the empty string or a status line for an action plan whose
                 surface is the trace/snapshot).
        trace:   the ``ExecutionTrace`` the harness produced — ALWAYS present
                 (answer-only turns produce a 1-step trivially-verified trace, so
                 verify is never bypassed; rule 5). ``None`` only when VGG is not
                 available at all.
        snapshot: the JSON-safe run-complete EXPORT VIEW for *trace* (goal tree +
                 per-step views + replan notes + outcome), or an empty dict when
                 there is no trace. Pure observation surface — never re-derived.
        intent:  the observable :class:`IntentDecision` that shaped this turn (the
                 cheap pre-classifier hint), surfaced for logs / inspection.
        tool_calls: the tool calls executed while producing the answer (the
                 answer-only path may legitimately call read-only tools through the
                 ReAct loop before answering). Empty for a pure plan turn.
        verified: convenience — True when *trace* succeeded AND passed the evidence
                 gate (answer-only steps are evidence-backed by design; an action
                 step with no deterministic predicate is NOT). Computed by the
                 controller so a caller need not re-run the gate.
        usage:   cumulative token usage across every backend round-trip in the turn.
    """

    text: str
    trace: "ExecutionTrace | None"
    snapshot: dict[str, Any]
    intent: IntentDecision
    tool_calls: list[ToolCall] = field(default_factory=list)
    verified: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)


# ---------------------------------------------------------------------------
# Internal batching type
# ---------------------------------------------------------------------------


@dataclass
class ToolBatch:
    """A group of tool calls to execute together."""

    concurrent: bool
    tool_calls: list[LLMToolCall]


# ---------------------------------------------------------------------------
# VectorEngine
# ---------------------------------------------------------------------------


class VectorEngine:
    """Core agent loop: user message -> backend call -> tool execution -> repeat.

    Backend-agnostic: accepts any LLMBackend implementation.
    Thread-safety: a single VectorEngine instance should not be shared across
    concurrent threads. Create one instance per agent session.
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        permissions: PermissionContext | None = None,
        max_turns: int = 50,
        max_tokens: int = 16384,
        intent_router: Any = None,
        hooks: Any = None,
    ) -> None:
        self._backend: LLMBackend = backend
        self._registry: ToolRegistry = registry or ToolRegistry()
        self._system_prompt: list[dict[str, Any]] = system_prompt or []
        self._permissions: PermissionContext = permissions or PermissionContext()
        self._max_turns: int = max_turns
        self._max_tokens: int = max_tokens
        self._intent_router = intent_router  # IntentRouter or None
        self._hooks = hooks                  # ToolHookRegistry or None

        # VGG cognitive layer (optional — disabled by default)
        self._vgg_enabled: bool = False
        self._goal_decomposer: Any = None
        self._goal_executor: Any = None

        # Experience-compilation tier (wired only when init_vgg is given a
        # persist_dir; in-memory / off otherwise so tests don't touch ~/.vector).
        self._template_library: Any = None
        self._experience_compiler: Any = None
        self._successful_traces: list[Any] = []
        self._max_successful_traces: int = 50

        # World context cache — avoids repeated sensor/graph queries when robot is static
        _WORLD_CONTEXT_TTL: float = 5.0  # seconds
        self._world_context_ttl: float = _WORLD_CONTEXT_TTL
        self._world_context_cache: str | None = None
        self._world_context_ts: float = 0.0

    # ------------------------------------------------------------------
    # VGG — optional cognitive pipeline
    # ------------------------------------------------------------------

    def init_vgg(
        self,
        backend: Any = None,
        agent: Any = None,
        skill_registry: Any = None,
        on_vgg_step: "Callable[[StepRecord], None] | None" = None,
        on_vgg_step_view: "Callable[[dict[str, Any]], None] | None" = None,
        world: Any = None,
        tool_permission_resolver: "Callable[[str, dict[str, Any]], str] | None" = None,
        persist_dir: "str | Path | None" = None,
    ) -> None:
        """Initialise the VGG cognitive pipeline components.

        Safe to call at any time. If initialisation fails for any reason
        (missing dependencies, bad backend), _vgg_enabled stays False and
        the engine continues to work through the normal tool_use path.

        ``world`` selects the decompose vocabulary: a world returning a
        DecomposeVocab injects it into the GoalDecomposer; a robot world (or
        None) keeps the decomposer's robot defaults.

        ``on_vgg_step_view`` is the observation surface (INC5): an optional sink
        for the JSON-serializable per-step EXPORT VIEW (a front-end renders it).
        It runs alongside ``on_vgg_step`` (the raw StepRecord) and is best-effort
        — a failure there never affects execution. The run-complete snapshot is
        available via ``vgg_run_snapshot(trace)``.

        ``tool_permission_resolver`` resolves ``ask``-level tool permissions for
        dev-world ``tool_call`` sub-goals (called with ``(tool_name, params) ->
        "y"|"a"|"n"``). None auto-denies — the safe headless default; the CLI
        passes its interactive prompt.

        ``persist_dir`` enables the learning tier: when set, StrategyStats and the
        TemplateLibrary persist under it (e.g. ``~/.vector``) and successful runs
        are compiled into reusable templates. None keeps everything in memory
        (the default, so tests never touch the home dir).
        """
        if not _VGG_AVAILABLE:
            logger.warning("VGG components not available — VGG disabled")
            return

        _backend = backend or self._backend
        self._vgg_agent = agent
        self._vgg_step_callback = on_vgg_step
        # Observation surface (INC5): an optional sink for the per-step EXPORT
        # VIEW (JSON-safe dict the front-end renders). Independent of the raw
        # StepRecord callback above; either, both, or neither may be set.
        self._vgg_step_view_callback = on_vgg_step_view
        self._world = world

        # ObjectMemory — sync from SceneGraph if available.
        # Isolated try/except: failure here must not block the rest of VGG init.
        try:
            from vector_os_nano.vcli.cognitive.object_memory import ObjectMemory
            _sg_ref = getattr(agent, "_spatial_memory", None)
            if _sg_ref is not None:
                self._object_memory = ObjectMemory()
                self._object_memory.sync_from_scene_graph(_sg_ref)
                logger.debug(
                    "ObjectMemory initialized with %d objects",
                    len(self._object_memory._objects),
                )
            else:
                self._object_memory = None
        except ImportError:
            logger.info("VGG: ObjectMemory not available (missing import)")
            self._object_memory = None
        except Exception as exc:
            logger.warning("VGG: ObjectMemory init failed: %s", exc)
            self._object_memory = None

        # Cognitive layer (GoalDecomposer, GoalVerifier, GoalExecutor, VGGHarness).
        # Any failure here disables VGG — the engine falls back to tool_use path.
        # Persistence is opt-in: a persist_dir wires StrategyStats + TemplateLibrary
        # to disk for cross-session learning; None keeps them in memory so tests
        # (and headless evals) never write to the home dir.
        _persist = Path(persist_dir) if persist_dir is not None else None
        _stats_path = str(_persist / "strategy_stats.json") if _persist else None

        try:
            # Build primitives namespace for GoalVerifier
            ns = self._build_verifier_namespace(agent)
            stats = StrategyStats(persist_path=_stats_path)
        except Exception as exc:
            logger.warning("VGG: verifier namespace build failed: %s", exc)
            self._vgg_enabled = False
            return

        # Experience-compilation tier (only when persisting). A persistent
        # TemplateLibrary activates the decomposer's no-LLM fast path; an
        # ExperienceCompiler turns successful runs into reusable templates.
        template_library = None
        experience_compiler = None
        if _persist is not None:
            try:
                from vector_os_nano.vcli.cognitive.experience_compiler import ExperienceCompiler
                from vector_os_nano.vcli.cognitive.template_library import TemplateLibrary
                template_library = TemplateLibrary(
                    persist_path=str(_persist / "goal_templates.json")
                )
                experience_compiler = ExperienceCompiler()
            except Exception as exc:  # noqa: BLE001
                logger.debug("VGG: experience tier unavailable: %s", exc)
                template_library = None
                experience_compiler = None

        # Routable-capability registry (Phase C). The world registers its
        # capabilities (dev: the chat LLM; robot: detectors/planners in C.3).
        # Empty for a world that registers none, so capability routing is inert
        # and the path stays byte-identical.
        capability_registry: Any = None
        try:
            from vector_os_nano.vcli.cognitive.capabilities import CapabilityRegistry
            capability_registry = CapabilityRegistry()
            if world is not None and hasattr(world, "register_capabilities"):
                world.register_capabilities(capability_registry, agent, _backend)
        except Exception as exc:  # noqa: BLE001
            logger.debug("VGG: capability registry unavailable: %s", exc)
            capability_registry = None
        _capability_names = (
            capability_registry.names() if capability_registry is not None else frozenset()
        )

        # Per-world decompose vocabulary. A world either injects an explicit
        # DecomposeVocab (dev), or opts into engine-side derivation from the live
        # skill registry (robot/arm) so the prompt, the validator allowlist, and
        # the params-help are single-sourced and can never drift. Otherwise the
        # decomposer keeps its class defaults.
        _vocab_kwargs: dict = {}
        _has_base = getattr(agent, "_base", None) is not None
        try:
            _requires_registry_vocab = bool(
                world is not None
                and getattr(world, "derive_vocab_from_registry", None) is not None
                and world.derive_vocab_from_registry()
            )
        except Exception:  # noqa: BLE001
            _requires_registry_vocab = False
        if world is not None:
            try:
                _vocab = world.decompose_vocab()
                if _vocab is not None:
                    _vocab_kwargs = _vocab.as_kwargs()
                elif _requires_registry_vocab and skill_registry is not None:
                    _vocab_kwargs = self._build_registry_vocab_kwargs(
                        skill_registry, agent, _has_base
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("VGG: registry vocab derivation failed: %s", exc)

        # Invariant (single-source vocab): a world that REQUIRES registry-derived
        # vocab must never silently fall back to the hardcoded class defaults — that
        # re-opens the split-brain (e.g. teaching go2 navigate / "去厨房" to an arm
        # while the validator allowlist is arm-only). If derivation produced nothing
        # (registry missing, or it raised), use an explicit neutral vocab instead.
        if _requires_registry_vocab and not _vocab_kwargs:
            logger.warning(
                "VGG: registry vocab unavailable for a derivation-required world; "
                "using a neutral vocab (NOT class defaults)"
            )
            _vocab_kwargs = self._neutral_vocab_kwargs(agent, _has_base)

        try:
            decomposer = GoalDecomposer(
                _backend,
                template_library=template_library,
                skill_registry=skill_registry,
                has_base=_has_base,
                **_vocab_kwargs,
            )
        except ImportError as exc:
            logger.warning("VGG: GoalDecomposer not available: %s", exc)
            self._vgg_enabled = False
            return
        except Exception as exc:
            logger.warning("VGG: GoalDecomposer init failed: %s", exc)
            self._vgg_enabled = False
            return

        try:
            verifier = GoalVerifier(ns)
            selector = StrategySelector(
                skill_registry=skill_registry,
                stats=stats,
                capability_names=_capability_names,
                has_base=_has_base,
            )
        except ImportError as exc:
            logger.warning("VGG: cognitive layer not available: %s", exc)
            self._vgg_enabled = False
            return
        except Exception as exc:
            logger.warning("VGG: GoalVerifier/StrategySelector init failed: %s", exc)
            self._vgg_enabled = False
            return

        # Build a SkillContext factory so GoalExecutor can execute skills.
        # Skills need context.base, context.services etc. — wire from agent.
        _agent_ref = agent
        _skill_registry_ref = skill_registry

        def _build_context() -> Any:
            from vector_os_nano.core.skill import SkillContext
            _base = getattr(_agent_ref, "_base", None)
            _arm = getattr(_agent_ref, "_arm", None)
            _gripper = getattr(_agent_ref, "_gripper", None)
            _perception = getattr(_agent_ref, "_perception", None)
            _sg = getattr(_agent_ref, "_spatial_memory", None)
            _vlm = getattr(_agent_ref, "_vlm", None)
            _wm = getattr(_agent_ref, "_world_model", None)
            _config = getattr(_agent_ref, "_config", None) or {}
            _cal = getattr(_agent_ref, "_calibration", None)
            services: dict = {}
            if _sg is not None:
                services["spatial_memory"] = _sg
            if _skill_registry_ref is not None:
                services["skill_registry"] = _skill_registry_ref
            if _vlm is not None:
                services["vlm"] = _vlm
            # Populate arms / grippers / perception_sources too — manipulation
            # skills (pick_top_down, PickSkill, HomeSkill, etc.) read
            # context.arm / context.gripper and previously got None because
            # this builder only wired base + services.
            return SkillContext(
                arms={"default": _arm} if _arm is not None else {},
                grippers={"default": _gripper} if _gripper is not None else {},
                bases={"go2": _base} if _base is not None else {},
                perception_sources=(
                    {"default": _perception} if _perception is not None else {}
                ),
                services=services,
                world_model=_wm,
                calibration=_cal,
                config=_config,
            )

        # Phase B execution wiring. Both default None so the robot path is
        # behaviourally identical when no code/tool sub-goal is produced;
        # construction failures here never disable VGG (decompose + verify still
        # work) — they just leave the corresponding branch unavailable.
        code_executor: Any = None
        tool_dispatcher: Any = None
        try:
            from vector_os_nano.vcli.cognitive.code_executor import CodeExecutor
            # The code-as-policy sandbox must NOT receive side-effecting verifier
            # predicates (e.g. tests_pass spawns subprocesses, bypassing the
            # permission gate). The GoalVerifier keeps the full namespace; the
            # executor gets a filtered copy.
            _SANDBOX_DENY = {"tests_pass"}
            code_ns = {k: v for k, v in ns.items() if k not in _SANDBOX_DENY}
            code_executor = CodeExecutor(code_ns)
        except Exception as exc:  # noqa: BLE001
            logger.debug("VGG: CodeExecutor unavailable: %s", exc)
        # Tool-backed execution is dev-world only; the robot world keeps its
        # skill/primitive path (Phase C migrates robot tools behind this seam).
        if world is not None and not world.is_robot():
            try:
                from vector_os_nano.vcli.cognitive.tool_dispatcher import ToolDispatcher
                from vector_os_nano.vcli.worlds.dev import DEV_TOOL_ALLOWLIST
                tool_dispatcher = ToolDispatcher(
                    self._registry,
                    self._permissions,
                    allowlist=DEV_TOOL_ALLOWLIST,
                    ask_permission=tool_permission_resolver,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("VGG: ToolDispatcher unavailable: %s", exc)

        # W1.4 — world producing-step primitives. A world (e.g. PlaygroundWorld)
        # may expose ``build_step_primitives(agent)`` returning the per-step PRODUCER
        # callables (keyed by the strategy name a plan emits, e.g.
        # ``detect_objects_skill``) whose structured output a downstream ``foreach``
        # iterates over the Blackboard. Inject them into the executor so the producers
        # the tabletop/foreach tests exercise are the ones that actually run live (the
        # executor consults this dict before the importlib primitive fallback, and
        # dispatches the producer-only strategy name to it — see
        # GoalExecutor._world_primitive_strategy).
        # Built defensively: a world without the method, or any failure, leaves it
        # None so the path stays byte-identical (importlib fallback only). NOTHING
        # ELSE passes ``primitives`` to this construction, so reuse cannot clobber
        # another injection.
        _world_primitives: Any = None
        if world is not None and hasattr(world, "build_step_primitives"):
            try:
                _world_primitives = world.build_step_primitives(agent)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VGG: world.build_step_primitives failed (%s); "
                    "falling back to importlib primitive resolution",
                    exc,
                )
                _world_primitives = None

        try:
            executor = GoalExecutor(
                strategy_selector=selector,
                verifier=verifier,
                skill_registry=skill_registry,
                primitives=_world_primitives,
                build_context=_build_context,
                stats=stats,
                visual_verifier_agent=agent,
                code_executor=code_executor,
                tool_dispatcher=tool_dispatcher,
                capability_registry=capability_registry,
                is_robot=bool(world.is_robot()) if world is not None else False,
                evidence_exempt=_world_exempt_strategies(world),
            )
        except ImportError as exc:
            logger.warning("VGG: GoalExecutor not available: %s", exc)
            self._vgg_enabled = False
            return
        except Exception as exc:
            logger.warning("VGG: GoalExecutor init failed: %s", exc)
            self._vgg_enabled = False
            return

        # W1.2 — Pre-flight validator: surface vocab/route/verify-namespace drift
        # at init time with an actionable error instead of an opaque mid-plan failure.
        # The validator is wrapped so a bug in the validator itself never crashes a
        # healthy startup; a genuine WorldRegistrationError IS re-raised.
        _world_name = getattr(world, "name", repr(world)) if world is not None else "<no-world>"
        try:
            self._preflight_validate_world(
                _vocab_kwargs, selector, ns, _world_name, _has_base
            )
        except ValueError:
            raise  # genuine drift — fail loud
        except Exception as exc:  # noqa: BLE001
            logger.warning("VGG: preflight validator internal error (ignored): %s", exc)

        try:
            self._goal_decomposer = decomposer
            self._goal_executor = executor
            self._template_library = template_library
            self._experience_compiler = experience_compiler
            self._successful_traces = []
            self._vgg_harness = VGGHarness(
                decomposer=decomposer,
                executor=executor,
                selector=selector,
                config=HarnessConfig(
                    max_step_retries=2,
                    max_pipeline_retries=1,
                ),
                on_step=self._on_vgg_step,
            )
            self._vgg_enabled = True
            logger.debug("VGG pipeline initialised successfully")
        except ImportError as exc:
            logger.warning("VGG: VGGHarness not available: %s", exc)
            self._vgg_enabled = False
        except Exception as exc:
            logger.warning("VGG: harness init failed: %s", exc)
            self._vgg_enabled = False

    _NEUTRAL_PLANNER_INTRO: str = (
        "You are a robot task planner. Decompose the user's task into verifiable "
        "sub-goals, each with a deterministic verify predicate over the robot's "
        "world state. Choose a strategy for steps that must act; leave strategy "
        "empty for pure checks. " + _TARGET_BINDING_GUIDANCE
    )

    def _build_registry_vocab_kwargs(
        self, skill_registry: Any, agent: Any, has_base: bool
    ) -> dict[str, Any]:
        """Build GoalDecomposer vocab kwargs from the live skill registry.

        Single-sources the decompose vocabulary: schemas come from
        ``skill_registry.to_schemas()`` and the verify-function allowlist +
        signatures are derived from the engine's verify namespace (so the prompt
        and the validator can never drift). Falls back to the GoalDecomposer
        class-default verify signatures if the namespace can't be built here, so
        the decomposer is never left without a verify allowlist.
        """
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            build_decompose_vocab,
        )

        from vector_os_nano.vcli.primitives import primitives_ready

        schemas = skill_registry.to_schemas()
        verify_signatures = self._verify_signatures_from_namespace(agent)
        vocab = build_decompose_vocab(
            schemas,
            verify_signatures,
            has_base=has_base,
            planner_intro=self._NEUTRAL_PLANNER_INTRO,
            # Executable truth: never teach base primitives a world can't run
            # (owner finding (c) — scan_360 ghost in the habitat replan).
            teach_base_primitives=primitives_ready(),
        )
        return vocab.as_kwargs()

    def _verify_signatures_from_namespace(self, agent: Any) -> dict[str, str]:
        """Derive name -> readable signature for the verify namespace callables.

        Reads the same namespace GoalVerifier uses (``_build_verifier_namespace``)
        and renders each callable's signature via ``inspect.signature`` (e.g.
        ``"detect_objects(query='')"``). Falls back to the GoalDecomposer
        class-default signatures if the namespace can't be built or yields
        nothing, so the validator always has an allowlist.
        """
        import inspect

        try:
            ns = self._build_verifier_namespace(agent)
        except Exception as exc:  # noqa: BLE001
            logger.debug("verify namespace for vocab unavailable: %s", exc)
            ns = {}

        signatures: dict[str, str] = {}
        for name, fn in ns.items():
            if not callable(fn):
                continue
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(...)"
            signatures[name] = f"{name}{sig}"

        if not signatures:
            # Class-default robot verify signatures — never leave the validator
            # without an allowlist.
            from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

            return dict(GoalDecomposer._VERIFY_FN_SIGNATURES)
        return signatures

    def _neutral_vocab_kwargs(self, agent: Any, has_base: bool) -> dict[str, Any]:
        """Neutral fallback vocab for a derivation-required world with no registry.

        Used only when registry derivation could not run. Produces an empty
        strategy set (so the decomposer teaches/validates NO domain strategies,
        rather than the wrong domain's class defaults) while still exposing the
        real verify-function allowlist, so the single-source invariant holds even
        on the failure path.
        """
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            build_decompose_vocab,
        )

        from vector_os_nano.vcli.primitives import primitives_ready

        verify_signatures = self._verify_signatures_from_namespace(agent)
        vocab = build_decompose_vocab(
            [],
            verify_signatures,
            has_base=has_base,
            planner_intro=self._NEUTRAL_PLANNER_INTRO,
            teach_base_primitives=primitives_ready(),
        )
        return vocab.as_kwargs()

    # W1.2 — special strategy names that are ALWAYS valid routes (built-in to
    # _resolve_explicit; never need a registry entry).
    _ALWAYS_VALID_STRATEGIES: frozenset[str] = frozenset(
        {"code_as_policy", "tool_call", "answer"}
    )

    def _preflight_validate_world(
        self,
        vocab_kwargs: dict,
        selector: Any,
        verify_ns: dict,
        world_name: str,
        has_base: bool,
    ) -> None:
        """Validate that vocab, selector routes, and verify namespace are consistent.

        Called during ``init_vgg`` after all three components are built but before
        the VGG harness is wired up. Raises ``ValueError`` with a multi-line,
        actionable message when drift is detected between:
        - the strategies the vocab teaches and the routes the selector can resolve, OR
        - the verify-function names the vocab exposes and the names present in the
          verify namespace.

        SCOPE GUARD: only validates statically-knowable names.
        - When ``selector._registered_skill_names()`` returns None (undeterminable,
          e.g. a registry-less world), skill-route validation is SKIPPED with a
          WARNING rather than a hard fail. This prevents false positives on any world
          that does not expose a skill registry.
        - Strategies that are always-valid built-ins (``code_as_policy``,
          ``tool_call``, ``answer``), registered capabilities, or base primitives
          (when ``has_base`` is True) are exempted from registry checks.
        - Non-``_skill``-suffix strategies that are not in the above categories
          resolve by convention (the selector treats them as skills). These are
          skipped: the runtime won't produce ``executor_type="invalid"`` for them.
        """
        # --- collect the static vocab surfaces ---
        strategy_names: frozenset[str] = frozenset(
            vocab_kwargs.get("strategy_descriptions", {}).keys()
        )
        verify_fn_names: frozenset[str] = frozenset(
            vocab_kwargs.get("verify_functions", frozenset())
        )

        bad_strategies: list[str] = []
        bad_verify_fns: list[str] = []

        # --- validate strategy routes ---
        # Only strategies ending with "_skill" can produce executor_type="invalid"
        # at runtime (via _resolve_explicit). All other strategies either hit an
        # explicit built-in route (code_as_policy/tool_call/answer), match a
        # capability, match a base primitive, or fall through to the convention
        # "treat as skill" path (which never returns "invalid").
        try:
            from vector_os_nano.vcli.cognitive.strategy_selector import _PRIMITIVE_NAMES
            _SKILL_SUFFIX = "_skill"
        except Exception as _imp_exc:  # noqa: BLE001
            logger.debug("preflight: strategy_selector import failed (%s) — skipping route check", _imp_exc)
            _PRIMITIVE_NAMES = frozenset()
            _SKILL_SUFFIX = "_skill"

        registered_names = selector._registered_skill_names()
        capability_names: frozenset[str] = getattr(selector, "_capability_names", frozenset())

        if registered_names is None:
            # Undeterminable — skip the route check with a warning only.
            logger.warning(
                "VGG preflight: world %r — skill registry is not available; "
                "strategy-route validation skipped (undeterminable)",
                world_name,
            )
        else:
            for name in sorted(strategy_names):
                # Always-valid built-in routes.
                if name in self._ALWAYS_VALID_STRATEGIES:
                    continue
                # Registered capability — always routable.
                if name in capability_names:
                    continue
                # Base primitive — valid only when the selector would actually
                # route it (has_base AND an executable primitive layer; mirrors
                # StrategySelector._primitive_routable, owner finding (c)).
                if name in _PRIMITIVE_NAMES and getattr(
                    selector, "_primitive_routable", lambda: has_base
                )():
                    continue
                # Strategies ending in "_skill" are the only ones the selector can
                # reject at runtime with executor_type="invalid".
                if name.endswith(_SKILL_SUFFIX):
                    bare = name[: -len(_SKILL_SUFFIX)]
                    # Also valid if the bare name itself is a registered skill or a
                    # base primitive (the selector accepts both forms).
                    if bare not in registered_names and not (
                        bare in _PRIMITIVE_NAMES
                        and getattr(
                            selector, "_primitive_routable", lambda: has_base
                        )()
                    ):
                        bad_strategies.append(name)
                # Non-_skill strategies not in any of the above categories are
                # convention-routed (selector returns executor_type="skill"); they
                # can never produce "invalid" at runtime, so skip them.

        # --- validate verify-function names ---
        # A name is "documented-lazy" (opt-in, conditionally bound) when its
        # signature string contains "opt-in" or "may be disabled" — this is the
        # convention used for verify predicates that are available only under
        # specific conditions (e.g. tests_pass requires VECTOR_DEV_ALLOW_TESTS).
        # Lazy names that are absent from verify_ns generate a WARNING only, not
        # a hard fail, because their absence is a designed opt-in posture, not drift.
        verify_fn_signatures: dict[str, str] = vocab_kwargs.get("verify_fn_signatures", {})
        for fn_name in sorted(verify_fn_names):
            if fn_name not in verify_ns:
                sig = verify_fn_signatures.get(fn_name, "")
                if "opt-in" in sig or "may be disabled" in sig:
                    logger.debug(
                        "VGG preflight: world %r — verify-fn %r absent from namespace "
                        "(documented opt-in/lazy binding; not a drift error)",
                        world_name,
                        fn_name,
                    )
                else:
                    bad_verify_fns.append(fn_name)

        if not bad_strategies and not bad_verify_fns:
            return  # all clear — silent

        # --- build a multi-line actionable error message ---
        lines: list[str] = [
            f"World registration drift detected for world {world_name!r}.",
            "The decompose vocabulary, strategy selector, and verify namespace are"
            " out of sync. Fix the world registration before this world can start.",
        ]
        if bad_strategies:
            valid_set = sorted(registered_names) if registered_names else []
            lines.append("")
            lines.append("Unresolvable strategy names (not a registered skill, capability, or base primitive):")
            for s in bad_strategies:
                lines.append(f"  - {s!r}")
            lines.append(f"Valid registered skill names: {valid_set!r}")
        if bad_verify_fns:
            valid_fns = sorted(verify_ns.keys())
            lines.append("")
            lines.append("Verify-function names with no provider in the verify namespace:")
            for f in bad_verify_fns:
                lines.append(f"  - {f!r}")
            lines.append(f"Valid verify-namespace names: {valid_fns!r}")

        raise ValueError("\n".join(lines))

    def _build_verifier_namespace(self, agent: Any) -> dict[str, Any]:
        """Build function namespace for GoalVerifier from agent state.

        Dev-world predicates (file_exists/grep_count/path_contains, and opt-in
        tests_pass) are always included so the verifier works without a robot;
        robot bindings are added on top when an agent is connected.
        """
        ns: dict[str, Any] = {}
        # Domain-general dev predicates — always available (no robot required).
        try:
            from vector_os_nano.vcli.worlds.dev import dev_verify_namespace
            ns.update(dev_verify_namespace())
        except Exception as exc:  # noqa: BLE001
            logger.debug("dev verify namespace unavailable: %s", exc)
        if agent is None:
            return self._merge_world_verify_namespace(ns, agent)
        base = getattr(agent, "_base", None)
        sg = getattr(agent, "_spatial_memory", None)
        if base:
            ns["get_position"] = lambda: tuple(base.get_position())
            ns["get_heading"] = base.get_heading
        if sg:
            ns["nearest_room"] = lambda: sg.nearest_room(
                *base.get_position()[:2]
            ) if base else None
            ns["get_visited_rooms"] = sg.get_visited_rooms
            ns["query_rooms"] = lambda: [
                {"id": r.room_id, "x": r.center_x, "y": r.center_y}
                for r in sg.get_all_rooms()
            ]
            ns["world_stats"] = sg.stats
        # Safe stubs for perception (require camera — may not be available)
        ns.setdefault("describe_scene", lambda: "")
        ns.setdefault("detect_objects", lambda query="": [])

        # --- Phase 3: Active World Model functions ---
        # ObjectMemory functions (if ObjectMemory available on engine)
        _obj_mem = getattr(self, "_object_memory", None)
        if _obj_mem is not None:
            ns["last_seen"] = _obj_mem.last_seen
            ns["certainty"] = _obj_mem.certainty
            ns["objects_in_room"] = _obj_mem.objects_in_room
            ns["find_object"] = _obj_mem.find_object

        # Room coverage (from SceneGraph)
        if sg:
            ns["room_coverage"] = sg.get_room_coverage

        # predict_navigation (from predict module)
        if sg:
            from vector_os_nano.vcli.cognitive.predict import predict_navigation
            _current_room_fn = ns.get("nearest_room")
            def _predict_nav(target: str) -> dict:
                current = _current_room_fn() if _current_room_fn else ""
                return predict_navigation(sg, current or "", target)
            ns["predict_navigation"] = _predict_nav

        # Safe stubs for Phase 3 functions when dependencies unavailable
        ns.setdefault("last_seen", lambda category="": None)
        ns.setdefault("certainty", lambda fact="": 0.0)
        ns.setdefault("objects_in_room", lambda room_id="": [])
        ns.setdefault("find_object", lambda category="": [])
        ns.setdefault("room_coverage", lambda room_id="": 0.0)
        ns.setdefault(
            "predict_navigation",
            lambda target="": {
                "reachable": False,
                "door_count": 0,
                "estimated_steps": 0,
                "rooms_on_path": [],
                "confidence": 0.0,
            },
        )
        return self._merge_world_verify_namespace(ns, agent)

    def _merge_world_verify_namespace(
        self, ns: dict[str, Any], agent: Any
    ) -> dict[str, Any]:
        """Merge the active world's verify-namespace contribution into *ns*.

        Shared-prelude (1): a world may OWN verify predicates. The engine builds
        its existing dev/robot bindings and stubs first, then merges in
        ``world.build_verify_namespace(agent)`` on top — additively. World-
        provided predicates take precedence over the engine's empty perception
        stubs (``describe_scene`` -> "", ``detect_objects`` -> []); a world that
        contributes ``{}`` (RobotWorld/DevWorld today) leaves *ns* byte-identical.

        The active world is the one wired into this engine (``self._world``); when
        none is wired (older/headless call paths) it is resolved from the agent via
        the world registry, so the behaviour is identical to the kernel's default
        world selection. Resolution and the world hook are isolated so a failing or
        absent world never breaks namespace construction.
        """
        world = getattr(self, "_world", None)
        if world is None:
            try:
                from vector_os_nano.vcli.worlds.registry import resolve_world
                world = resolve_world(agent)
            except Exception as exc:  # noqa: BLE001
                logger.debug("world resolution for verify namespace failed: %s", exc)
                return ns
        builder = getattr(world, "build_verify_namespace", None)
        if builder is None:
            return ns
        try:
            world_ns = builder(agent)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "world %r build_verify_namespace failed: %s",
                getattr(world, "name", world),
                exc,
            )
            return ns
        if world_ns:
            ns.update(world_ns)
        return ns

    def classify_intent(self, user_message: str) -> IntentDecision:
        """Classify which planning path *user_message* should take (observable).

        Stage 5 scout — a PURE, side-effect-free read of the same conditions
        ``vgg_decompose`` uses to fork between the VGG closed loop and the
        tool_use ReAct loop, returned as one inspectable :class:`IntentDecision`
        instead of inline booleans. This does NOT change routing: ``vgg_decompose``
        now consults this method, so the decision is single-sourced and can be
        logged / rendered. The keyword intent gate (``should_use_vgg``) still owns
        the call; unification (dropping the gate) is the future Stage-5 work.

        Routing (in order):
          * VGG not ready / no intent router            -> tool_use
          * robot world without a connected base+arm    -> tool_use (sim not up)
          * gate says not a VGG task                    -> tool_use
          * otherwise                                   -> vgg
        The ``complex`` flag mirrors ``IntentRouter.is_complex`` so a caller can
        distinguish the deterministic 1-step fast path from an LLM decomposition.
        """
        if not self._vgg_enabled:
            return IntentDecision(route="tool_use", reason="vgg-disabled")
        if self._intent_router is None:
            return IntentDecision(route="tool_use", reason="no-intent-router")

        _agent = getattr(self, "_vgg_agent", None)
        _world = getattr(self, "_world", None)
        if _world is not None:
            _is_robot = bool(_world.is_robot())
        else:
            _is_robot = _agent is not None  # back-compat: agent present => robot
        if _is_robot and (
            _agent is None
            or (
                getattr(_agent, "_base", None) is None
                and getattr(_agent, "_arm", None) is None
            )
        ):
            return IntentDecision(
                route="tool_use", reason="robot-world-not-ready"
            )

        _sr = getattr(_agent, "_skill_registry", None) if _agent is not None else None
        if not self._intent_router.should_use_vgg(user_message, skill_registry=_sr):
            return IntentDecision(
                route="tool_use", reason="gate-not-a-vgg-task"
            )

        is_complex = False
        try:
            is_complex = bool(self._intent_router.is_complex(user_message))
        except Exception:  # noqa: BLE001 — classification only, never fatal
            is_complex = False
        reason = "vgg-complex" if is_complex else "vgg-actionable"
        return IntentDecision(route="vgg", reason=reason, complex=is_complex)

    def try_vgg(self, user_message: str) -> "ExecutionTrace | None":
        """Attempt VGG pipeline for complex tasks (decompose + execute).

        Returns an ExecutionTrace when VGG is enabled and the message is
        classified as complex. Returns None in all other cases so the caller
        can fall back to the normal tool_use path.
        """
        tree = self.vgg_decompose(user_message)
        if tree is None:
            return None
        try:
            return self.vgg_execute(tree)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VGG execution failed (%s) — falling back to tool_use", exc)
            return None

    def vgg_decompose(self, user_message: str) -> "GoalTree | None":
        """Decompose task into GoalTree. All actionable commands go through VGG.

        Fast path: if the message matches a single skill, create a 1-step
        GoalTree directly (no LLM call). This handles "探索", "去厨房", "站起来".

        Slow path: for complex tasks, call LLM GoalDecomposer for multi-step
        decomposition.

        Returns None when VGG is not ready (no agent/base connected).
        """
        # Clear abort flag at the start of every new VGG task.
        # Without this, a prior "stop" command leaves the flag set and
        # all subsequent VGG tasks are immediately aborted.
        try:
            from vector_os_nano.vcli.cognitive.abort import clear_abort
            clear_abort()
        except ImportError:
            pass

        # Single-source the routing decision (Stage 5 scout): classify_intent is a
        # pure read of the same gate conditions, returned as an inspectable
        # IntentDecision. Behaviour is unchanged — a non-vgg route here is exactly
        # the set of conditions that previously returned None inline.
        decision = self.classify_intent(user_message)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "intent route=%s reason=%s complex=%s",
                decision.route, decision.reason, decision.complex,
            )
        if not decision.use_vgg:
            return None

        _agent = getattr(self, "_vgg_agent", None)
        _sr = getattr(_agent, "_skill_registry", None) if _agent is not None else None

        # Fast path: single skill match → 1-step GoalTree, no LLM
        if _sr is not None and not decision.complex:
            tree = self._try_skill_goal_tree(user_message, _sr)
            if tree is not None:
                return tree

        # Slow path: LLM decomposition for complex tasks
        world_context = self._build_world_context()
        try:
            return self._goal_decomposer.decompose(user_message, world_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VGG decompose failed (%s)", exc)
            return None

    def _try_skill_goal_tree(self, user_message: str, skill_registry: Any) -> "GoalTree | None":
        """Create a 1-step GoalTree from a direct skill match.

        Returns None if no skill matches the message.
        """
        if not _VGG_AVAILABLE:
            return None
        try:
            match = skill_registry.match(user_message)
        except Exception:
            return None
        if match is None:
            return None

        skill_name = match.skill_name
        extracted = match.extracted_arg or ""

        # Skills with auto_steps (e.g. pick = scan->detect->pick) must run through
        # the agent's multi-step expansion, not a bare 1-step GoalTree. Let them fall
        # through to the tool_use path (SkillWrapperTool -> agent.execute_skill).
        _skill_for_steps = skill_registry.get(skill_name) if skill_registry else None
        if getattr(_skill_for_steps, "__skill_auto_steps__", None):
            return None

        # NavigateToPointSkill (N4): the fast path can neither bind numeric
        # x/y nor resolve a semantic label, and its verify must carry the SAME
        # coordinates as the params (the anti-false-pass contract) — always
        # the LLM planner's job.
        if skill_name == "navigate_to":
            return None

        # Resolve room alias to canonical ID (e.g. "客房" → "guest_bedroom")
        # so verify expressions and params use the same IDs as SceneGraph.
        resolved_room = ""
        if skill_name == "navigate" and extracted:
            resolved_room = self._resolve_room_alias(extracted)
            if not resolved_room:
                return None  # unknown room — let LLM handle

        # Build verify expression using resolved canonical ID — single-sourced
        # from the SKILL's own verify_template (invariant III; the old
        # hand-written second vocabulary in this method drifted and is gone).
        skill_obj = skill_registry.get(skill_name) if skill_registry else None
        verify_arg = resolved_room if resolved_room else extracted
        verify = self._verify_for_skill(skill_name, verify_arg, skill_obj)

        # Build strategy params — extract from user message text
        params: dict = {}
        skill_params = getattr(skill_obj, "parameters", {}) if skill_obj else {}

        # Generic extraction: match param names to user message content
        if "direction" in skill_params:
            params["direction"] = _extract_direction(user_message)
        if "distance" in skill_params:
            params["distance"] = _extract_number(user_message, default=1.0)
        if "angle" in skill_params:
            params["angle"] = _extract_number(user_message, default=90.0)
        if "speed" in skill_params:
            speed = _extract_number(user_message, default=0.0)
            if speed > 0:
                params["speed"] = speed
        if "room" in skill_params:
            if skill_name == "navigate" and resolved_room:
                params["room"] = resolved_room
            elif extracted:
                params["room"] = extracted
            elif skill_name == "navigate":
                # Navigate without room → skip fast path, let LLM handle
                return None
        if "object_label" in skill_params and extracted:
            params["object_label"] = extracted
        if "query" in skill_params and extracted:
            params["query"] = extracted

        sub_goal = SubGoal(
            name=f"{skill_name}_goal",
            description=user_message,
            verify=verify,
            strategy=f"{skill_name}_skill",
            strategy_params=params,
            timeout_sec=60.0 if skill_name in ("navigate", "explore", "patrol") else 30.0,
        )
        return GoalTree(goal=user_message, sub_goals=(sub_goal,))

    @staticmethod
    def _verify_for_skill(skill_name: str, arg: str, skill_obj: Any = None) -> str:
        """Verify expression for a fast-path skill step — single source.

        Invariant III (design review #5, rule 3): the SKILL's own
        ``verify_template`` is the only origin (``{arg}`` substituted with the
        extracted/resolved target). A skill without a template — or an
        ``{arg}`` template with no arg to bind — gets the honest ``"True"``
        sentinel, which the evidence gate reports as unverified (invariant
        II), never as deterministic evidence. No second hand-written map.
        """
        template = str(getattr(skill_obj, "verify_template", "") or "").strip()
        if not template:
            verify = "True"
        elif "{arg}" in template:
            verify = template.replace("{arg}", arg) if arg else "True"
        else:
            verify = template
        try:
            from vector_os_nano.vcli.cognitive.verify_strengthen import (
                strengthen_target_verify,
            )
            return strengthen_target_verify(
                verify, {"object_label": arg} if arg else {}
            )
        except ImportError:  # cognitive layer absent — engine degrades gracefully
            return verify

    def _resolve_room_alias(self, room_input: str) -> str:
        """Resolve a room name/alias to canonical SceneGraph ID.

        Uses NavigateSkill's alias table + SceneGraph fuzzy match.
        Returns empty string if unresolvable.
        """
        try:
            from vector_os_nano.skills.navigate import _resolve_room
        except ImportError:
            return ""
        agent = getattr(self, "_vgg_agent", None)
        sg = getattr(agent, "_spatial_memory", None) if agent else None
        return _resolve_room(room_input, sg=sg) or ""

    def vgg_execute(self, goal_tree: "GoalTree") -> "ExecutionTrace":
        """Execute GoalTree with feedback harness (retry + re-plan on failure)."""
        # Clear abort flag before every execute. Direct callers (e.g. tests) that
        # skip vgg_decompose() would otherwise inherit a stale abort from a prior stop.
        try:
            from vector_os_nano.vcli.cognitive.abort import clear_abort
            clear_abort()
        except ImportError:
            pass
        if hasattr(self, "_vgg_harness") and self._vgg_harness is not None:
            world_context = self._build_world_context()
            trace = self._vgg_harness.run(
                task=goal_tree.goal,
                world_context=world_context,
                goal_tree=goal_tree,
                # Stage 1b: rebuild world context fresh on every re-decompose so
                # replans see current robot/world state (bypass the TTL cache).
                context_provider=lambda: self._build_world_context(force=True),
            )
        else:
            # Fallback: raw executor (no harness)
            trace = self._goal_executor.execute(goal_tree, on_step=self._on_vgg_step)
        self._maybe_compile_experience(trace)
        return trace

    def _maybe_compile_experience(self, trace: Any) -> None:
        """Compile a successful trace into reusable templates (best-effort).

        No-op unless the experience tier was wired (init_vgg given a persist_dir).
        Bounded, never raises, and never blocks the caller — a failure here must
        not affect the execution result.
        """
        lib = getattr(self, "_template_library", None)
        comp = getattr(self, "_experience_compiler", None)
        # W1.1: gate template compilation on the SAME deterministic-evidence notion
        # the engine uses for its per-trace 'verified' flag. A visual-override or
        # sentinel verify="True" "success" must NOT compile into a reusable
        # 'verified' template (it would dilute the moat). Robot world bypasses the
        # evidence gate (_evidence_ok), so robot traces still compile as today.
        if (
            lib is None
            or comp is None
            or not getattr(trace, "success", False)
            or not self._evidence_ok(trace)
        ):
            return
        try:
            self._successful_traces.append(trace)
            if len(self._successful_traces) > self._max_successful_traces:
                self._successful_traces = self._successful_traces[-self._max_successful_traces:]
            for template in comp.compile(self._successful_traces):
                lib.add(template)
            lib.save()
        except Exception as exc:  # noqa: BLE001
            logger.debug("VGG: experience compilation skipped: %s", exc)

    def vgg_execute_async(
        self,
        goal_tree: "GoalTree",
        on_complete: "Callable[[ExecutionTrace], None] | None" = None,
    ) -> None:
        """Execute a GoalTree, off the caller's thread WHEN that is safe.

        Uses VGGHarness (with retry logic) when available, otherwise falls
        back to raw GoalExecutor.

        Threading: by default this runs on a background daemon thread so the
        CLI stays responsive. BUT when a GUI MuJoCo viewer is live
        (``_has_live_viewer()``) AND we are running under mjpython (macOS),
        it runs SYNCHRONOUSLY on the caller's thread instead — there GLFW is
        main-thread-only and the caller's thread owns the viewer, so it must
        be the only thread touching the sim; the caller blocks until execution
        finishes (acceptable: the user is watching the arm move, not typing).
        On Linux/Windows the viewer+physics run on their own daemon threads,
        so even with a live viewer execution stays on the background thread
        (the REPL keeps accepting input, e.g. "stop").

        Args:
            goal_tree: The goal tree to execute.
            on_complete: Called when execution finishes (on the background
                thread, or inline on the caller's thread in the
                viewer-live-under-mjpython case).
        """
        import threading

        self._vgg_cancel = threading.Event()

        def _run() -> None:
            # Catch Exception (not BaseException) so KeyboardInterrupt still
            # propagates to the REPL's try/except KeyboardInterrupt (cli.py),
            # preserving Ctrl-C abort of an in-flight goal.
            try:
                trace = self.vgg_execute(goal_tree)
                if on_complete:
                    on_complete(trace)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VGG async execution failed: %s", exc)

        # WHY: GLFW is main-thread-ONLY on macOS/mjpython, where the viewer is
        # driven by the caller's (main) thread pump — there a background skill
        # thread touching mjData/GLFW concurrently with the main-thread render
        # would segfault, so skills MUST run synchronously on the viewer-owning
        # thread. On Linux/Windows the viewer runs in MuJoCo's own background
        # render thread and physics+viewer.sync() run on the mujoco_*_physics
        # daemon (viewer_mode.BACKGROUND_DAEMON); the skill thread never touches
        # the viewer, so forcing sync-on-main there only freezes the REPL during
        # a motion (laggy Ctrl-C / "stop") with no safety benefit. Gate the
        # synchronous path on mjpython, mirroring viewer_mode's platform seam.
        from vector_os_nano.hardware.sim.viewer_mode import (  # noqa: PLC0415
            running_under_mjpython,
        )
        if self._has_live_viewer() and running_under_mjpython():
            self._vgg_thread = None
            _run()
            return

        t = threading.Thread(target=_run, name="vgg-executor", daemon=True)
        t.start()
        self._vgg_thread = t

    def _has_live_viewer(self) -> bool:
        """True when a connected sim has an OPEN GUI viewer.

        Used (together with ``running_under_mjpython()``) to gate synchronous
        skill execution: ONLY under mjpython/macOS is GLFW main-thread-only and
        the caller's thread the viewer owner — there a background skill thread
        racing the render segfaults. On other platforms a live viewer runs on
        its own daemon threads and this signal alone forces nothing.
        World-agnostic: duck-types the connected agent's arm/base hardware for
        a live viewer handle (no world import, no embodiment-specific branch —
        kernel rule 7).
        """
        agent = getattr(self, "_vgg_agent", None)
        if agent is None:
            return False
        for attr in ("_arm", "_base"):
            hw = getattr(agent, attr, None)
            if hw is None:
                continue
            if getattr(hw, "_viewer", None) is not None or getattr(hw, "viewer", None) is not None:
                return True
        return False

    def _build_world_context(self, force: bool = False) -> str:
        """Build a brief world context string for the GoalDecomposer.

        Results are cached for _world_context_ttl seconds to avoid repeated
        sensor/graph queries when the robot has not moved.

        Args:
            force: When True, bypass the TTL cache and rebuild from current
                state (Stage 1b — used so re-decompose sees fresh world state).
                The freshly built result still refreshes the cache.
        """
        now = time.monotonic()
        if (
            not force
            and self._world_context_cache is not None
            and now - self._world_context_ts < self._world_context_ttl
        ):
            return self._world_context_cache

        parts: list[str] = []
        agent = getattr(self, "_vgg_agent", None)
        if agent is None:
            result = ""
            self._world_context_cache = result
            self._world_context_ts = now
            return result
        # W3.3: resolve the base by PROTOCOL (narrow read-only spec), not by
        # private attribute name. Context building is a fail-safe surface, so
        # a missing/non-conforming base resolves to None rather than raising.
        from vector_os_nano.vcli.providers import BaseStateProvider, resolve_provider
        base = resolve_provider(
            agent, BaseStateProvider, what="base", required=False
        )
        sg = getattr(agent, "_spatial_memory", None)
        if base:
            try:
                pos = base.get_position()
                heading = base.get_heading()
                parts.append(f"Position: ({pos[0]:.1f}, {pos[1]:.1f})")
                parts.append(f"Heading: {heading:.1f} rad")
            except Exception:
                pass
        if sg:
            try:
                stats = sg.stats()
                parts.append(
                    f"SceneGraph: {stats.get('rooms', 0)} rooms, "
                    f"{stats.get('visited_rooms', 0)} visited"
                )
                if base:
                    pos = base.get_position()
                    room = sg.nearest_room(pos[0], pos[1])
                    if room:
                        parts.append(f"Current room: {room}")
                rooms = sg.get_visited_rooms()
                if rooms:
                    parts.append(f"Known rooms: {', '.join(rooms)}")
            except Exception:
                pass
        # Stage 3 (referring-expression grounding): surface the LIVE object set
        # so the decomposer can resolve a reference ("the red cup" / "红色杯子")
        # to an EXACT scene name at plan/replan time. Deterministic plumbing
        # only — the LLM is the language component; the kernel just reports
        # names (+ attribute hints) from the world model, falling back to the
        # sim oracle's ground-truth body names. Capped to bound the prompt.
        objects_line = self._live_objects_line(agent)
        if objects_line:
            parts.append(objects_line)
        result = "\n".join(parts) if parts else ""
        self._world_context_cache = result
        self._world_context_ts = now
        return result

    _LIVE_OBJECTS_CAP = 20

    @staticmethod
    def _live_objects_line(agent: Any) -> str:
        """Return 'Objects (live): ...' from the freshest available source.

        Priority: world-model objects (label + property hints — what detect
        populated) > sim-oracle body names (ground truth when no detect ran).
        Fail-safe '' on any error; never raises into context building.
        """
        entries: list[str] = []
        try:
            wm = getattr(agent, "_world_model", None)
            if wm is not None:
                # Collapse to the highest-confidence instance per label and
                # carry its COORDINATES — a referring expression is only
                # resolvable into an actionable binding (navigate_to(x, y) +
                # geodesic verify) when the planner can read real positions.
                best: dict[str, Any] = {}
                for obj in wm.get_objects():
                    # rule 5 (moat): GT-seeded objects exist ONLY as the verify
                    # anchor (boot writes them with source '*_ground_truth'). They
                    # must NOT reach the planner-facing world context — else the
                    # decomposer reads their ground-truth (x, y) and navigate_to's
                    # them, bypassing VLN (the chair is "found" without ever being
                    # perceived). Only PERCEIVED objects (detect/VLM-populated)
                    # belong here; the verify still binds to the GT anchor directly
                    # via the world model. Filtering here only tightens the moat.
                    try:
                        _src = str((getattr(obj, "properties", None) or {}).get("source", ""))
                    except Exception:  # noqa: BLE001
                        _src = ""
                    if _src.endswith("ground_truth"):
                        continue
                    label = obj.label or obj.object_id
                    cur = best.get(label)
                    if cur is None or obj.confidence > cur.confidence:
                        best[label] = obj
                for label in sorted(best):
                    obj = best[label]
                    parts_o = [f"x={obj.x:.2f}", f"y={obj.y:.2f}"]
                    props = getattr(obj, "properties", None) or {}
                    parts_o += [
                        f"{k}={v}" for k, v in sorted(props.items())
                        if isinstance(v, (str, int, float))
                    ]
                    entries.append(f"{label}({','.join(parts_o)})")
        except Exception:  # noqa: BLE001
            entries = []
        if not entries:
            try:
                arm = getattr(agent, "_arm", None) or getattr(agent, "arm", None)
                if arm is not None and hasattr(arm, "get_object_positions"):
                    entries = sorted(arm.get_object_positions().keys())
            except Exception:  # noqa: BLE001
                entries = []
        if not entries:
            return ""
        cap = VectorEngine._LIVE_OBJECTS_CAP
        shown, extra = entries[:cap], len(entries) - cap
        suffix = f" (+{extra} more)" if extra > 0 else ""
        return "Objects (live): " + ", ".join(shown) + suffix

    def _emergency_stop(
        self,
        user_message: str,
        session: Session,
        agent: Any = None,
        app_state: dict[str, Any] | None = None,
    ) -> TurnResult:
        """P0 stop bypass — execute StopSkill directly, no LLM call."""
        from vector_os_nano.vcli.cognitive.abort import request_abort
        request_abort()

        # Invalidate world context cache — robot state changes after a stop
        self._world_context_cache = None

        # Kill any running VGG thread
        cancel_ev = getattr(self, "_vgg_cancel", None)
        if cancel_ev is not None:
            cancel_ev.set()

        # Execute stop skill if available
        _agent = agent or getattr(self, "_vgg_agent", None)
        if _agent is not None:
            try:
                _agent.execute_skill("stop", {})
            except Exception:
                pass

        session.append_user(user_message)
        session.append_assistant("Stopped.", None)
        return TurnResult(text="Stopped.", tool_calls=[], stop_reason="end_turn", usage=TokenUsage())

    def _on_vgg_step(self, step: Any) -> None:
        """Callback invoked by GoalExecutor after each sub-goal completes.

        Fans out to (a) the raw ``StepRecord`` callback and (b) the observation
        surface's per-step EXPORT VIEW callback (INC5), if wired. The view sink
        is best-effort and isolated: a failure building/emitting it must never
        abort execution or affect the raw callback.
        """
        cb = getattr(self, "_vgg_step_callback", None)
        if cb:
            cb(step)
        view_cb = getattr(self, "_vgg_step_view_callback", None)
        if view_cb:
            try:
                from vector_os_nano.vcli.cognitive.observation import step_view
                view_cb(step_view(step))
            except Exception as exc:  # noqa: BLE001
                logger.debug("VGG: step_view emit skipped: %s", exc)

    def vgg_run_snapshot(self, trace: "ExecutionTrace") -> dict[str, Any]:
        """Return the run-complete observation snapshot for *trace* (INC5).

        A pure, JSON-serializable EXPORT VIEW (goal tree + per-step views +
        replan ``validation_notes`` + outcome) for the front-end to render. Does
        not mutate the trace and adds no fields to the frozen VGG types.
        """
        from vector_os_nano.vcli.cognitive.observation import run_snapshot
        return run_snapshot(trace)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_turn(
        self,
        user_message: str,
        session: Session,
        agent: Any = None,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[str, ToolResult], None] | None = None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
        app_state: dict[str, Any] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Run one user turn through the tool_use agent loop.

        Algorithm:
        1. Append user message to session
        2. Call backend (handles streaming + format conversion)
        3. If tool_calls present: execute tools, append results, loop
        4. If no tool_calls: return TurnResult

        Args:
            user_message:   The user's input text for this turn.
            session:        Mutable session object; updated in-place.
            agent:          Optional back-reference to the outer Agent (passed to ToolContext).
            on_text:        Called with each text chunk as it streams.
            on_tool_start:  Called before each tool execution with (tool_name, params).
            on_tool_end:    Called after each tool execution with (tool_name, result).
            ask_permission: For "ask"-level permissions, called with (tool_name, params).
                            Returns "y" (allow once), "a" (always allow), or "n" (deny).
            on_reasoning:   Called with each hidden reasoning chunk (reasoning models
                            only). Used purely for a live "thinking…" heartbeat during
                            the no-text gap; never part of the returned text.

        Returns:
            TurnResult with the final assistant text, all tool calls, stop reason, and
            cumulative token usage across all API round-trips in this turn.
        """
        # --- P0 stop bypass: hardcoded match, no LLM, <100ms ---
        _stop_words = {"stop", "停", "停下", "halt", "freeze", "别动", "停止"}
        if user_message.strip().lower() in _stop_words:
            return self._emergency_stop(user_message, session, agent, app_state)

        # --- Clear abort flag at start of each new task ---
        try:
            from vector_os_nano.vcli.cognitive.abort import clear_abort
            clear_abort()
        except ImportError:
            pass

        session.append_user(user_message)

        all_tool_calls: list[ToolCall] = []
        total_usage = TokenUsage()
        final_text = ""
        stop_reason = "end_turn"
        turns = 0
        abort_event = threading.Event()

        tool_context = ToolContext(
            agent=agent,
            cwd=Path.cwd(),
            session=session,
            permissions=self._permissions,
            abort=abort_event,
            app_state=app_state,
        )

        while turns < self._max_turns:
            if abort_event.is_set():
                break

            messages = session.to_messages()

            # Intent routing: select relevant tool categories
            if self._intent_router is not None and hasattr(self._registry, "to_anthropic_schemas"):
                categories = self._intent_router.route(user_message)
                if categories is not None:
                    tools = self._registry.to_anthropic_schemas(categories=categories)
                else:
                    tools = self._registry.to_anthropic_schemas()
            else:
                tools = self._registry.to_anthropic_schemas()

            # Backend handles streaming, format conversion, and retry
            # Only forward on_reasoning when a caller actually wants the heartbeat,
            # so backends (and test mocks) that predate the kwarg keep working.
            _call_kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": tools,
                "system": self._system_prompt,
                "max_tokens": self._max_tokens,
                "on_text": on_text,
            }
            if on_reasoning is not None:
                _call_kwargs["on_reasoning"] = on_reasoning
            response: LLMResponse = self._backend.call(**_call_kwargs)

            final_text = response.text
            stop_reason = response.stop_reason
            total_usage = total_usage.add(response.usage)

            # Append assistant message to session
            tool_use_dicts: list[dict[str, Any]] | None = None
            if response.tool_calls:
                tool_use_dicts = [
                    {"id": tc.id, "name": tc.name, "input": tc.input, "type": "tool_use"}
                    for tc in response.tool_calls
                ]
            session.append_assistant(response.text, tool_use_dicts)

            if not response.tool_calls:
                break  # end_turn — no tools called, conversation complete

            # Execute tools and collect results
            raw_results = self._dispatch_tools(
                response.tool_calls, tool_context, on_tool_start, on_tool_end, ask_permission
            )

            result_dicts: list[dict[str, Any]] = []
            for result_dict, tool_call in raw_results:
                result_dicts.append(result_dict)
                all_tool_calls.append(tool_call)

            session.append_tool_results(result_dicts)

            turns += 1

        session.add_usage(total_usage)

        return TurnResult(
            text=final_text,
            tool_calls=all_tool_calls,
            stop_reason=stop_reason,
            usage=total_usage,
        )

    # ------------------------------------------------------------------
    # Stage 5 — unified closed-loop controller (LIVE since S5.4)
    # ------------------------------------------------------------------

    def run_turn_unified(
        self,
        user_message: str,
        session: Session,
        agent: Any = None,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[str, ToolResult], None] | None = None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
        app_state: dict[str, Any] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> UnifiedTurnResult:
        """Run ONE user turn through the unified closed loop (Stage 5).

        LIVE since the S5.4 cut-over: BOTH frontends (``cli.py`` and
        ``mcp/tools.py``) call this by DEFAULT for every turn — chat included.
        ``VECTOR_LEGACY_TURN=1`` restores the pre-cut-over open ``run_turn``
        ReAct fork for one release as the fallback.

        North star (rule 1): every interaction is a CLOSED loop. This method ALWAYS
        produces a :class:`GoalTree` and runs the SAME harness loop (topo-execute
        via the one tool-dispatch seam -> capture to Blackboard -> deterministic
        verify -> replan on fail/divergence), then returns ONE
        :class:`UnifiedTurnResult`:

        - **Answer-only / conversation** (greeting, question, non-actionable chat):
          ``classify_intent`` routes it to ``tool_use`` — the cheap pre-classifier
          (the intent-router conversational guard) keeps trivial chat off full
          decomposition. The ReAct ``run_turn`` loop produces the answer (preserving
          streaming ``on_text``, the 7-layer permission gate + interactive ask, tool
          hooks, and any read-only tool the chat turn legitimately calls), and its
          text is wrapped into a 0-action ``answer_only`` GoalTree (the S5.2 shape)
          that the harness then runs: verify is trivially true, and the evidence
          gate recognises a legitimate evidence-free answer step. The moat is NOT
          bypassed — an action step with no predicate still fails the gate.

        - **Plan** (single skill / tool, or a complex N-step DAG): ``classify_intent``
          routes it to ``vgg``; ``vgg_decompose`` builds the GoalTree (1-step fast
          path or LLM decomposition) and the harness verifies + replans as today.

        When VGG is unavailable, this degrades to a plain ``run_turn`` wrapped with a
        ``None`` trace (no closed loop is possible without the cognitive layer) — the
        text is still returned so the turn is never silently dropped.
        """
        # M4-B (ADR-012 Option A): a compound command that BOTH switches embodiment
        # and acts is split at switch boundaries and run segment-by-segment — the
        # switch leg rebinds the active agent (ADR-011) via the proven top-level
        # path, then later segments run on the NEW embodiment. This composes
        # switch+action WITHOUT making switch_embodiment (a @tool) reachable inside a
        # single VGG decompose (which only plans @skills, rule 3). Guarded on
        # app_state (needed to rebind between segments) and the router's presence;
        # split returns None for a pure switch / non-switch / unsplittable command,
        # so the normal route below is unchanged for every non-compound-switch turn.
        router = getattr(self, "_intent_router", None)
        if app_state is not None and router is not None:
            segments = router.split_switch_segments(user_message)
            if segments is not None:
                return self._run_switch_segmented(
                    segments, session, agent, on_text, on_tool_start,
                    on_tool_end, ask_permission, app_state, on_reasoning,
                )

        intent = self.classify_intent(user_message)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "unified turn route=%s reason=%s complex=%s",
                intent.route, intent.reason, intent.complex,
            )

        if intent.use_vgg:
            return self._unified_plan_turn(user_message, intent)
        return self._unified_answer_turn(
            user_message, session, agent, on_text,
            on_tool_start, on_tool_end, ask_permission, app_state, on_reasoning,
            intent,
        )

    def _run_switch_segmented(
        self,
        segments: list[str],
        session: Session,
        agent: Any,
        on_text: Callable[[str], None] | None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
        app_state: dict[str, Any],
        on_reasoning: Callable[[str], None] | None,
    ) -> UnifiedTurnResult:
        """Run a switch-segmented compound command (M4-B / ADR-012 Option A).

        Each segment is run as its own ``run_turn_unified`` turn IN ORDER. The active
        agent is re-read from ``app_state`` before every segment, so a switch segment
        that rebinds ``app_state['agent']`` (via switch_embodiment's ``_rebind_agent``)
        makes every subsequent segment run on the NEW embodiment — eliminating the
        stale-agent hazard by construction (the switch completes before the next plan
        begins). Each segment splits to None (a pure switch or a non-switch action),
        so there is no re-entry into segmentation. Returns the LAST segment's result
        (the residual action's verified trace); switch text streams live via on_text.
        """
        result: UnifiedTurnResult | None = None
        all_tool_calls: list[Any] = []
        texts: list[str] = []
        for seg in segments:
            current = app_state.get("agent", agent)
            result = self.run_turn_unified(
                seg,
                session,
                agent=current,
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                ask_permission=ask_permission,
                app_state=app_state,
                on_reasoning=on_reasoning,
            )
            all_tool_calls.extend(getattr(result, "tool_calls", None) or [])
            seg_text = (getattr(result, "text", "") or "").strip()
            if seg_text:
                texts.append(seg_text)
        # Surface the compound turn faithfully: the UNION of tool_calls across all
        # segments (so the switch the command performed is observable, not only the
        # last leg's), plus the joined text. The trace/verify/snapshot stay the LAST
        # (residual action) segment's — the action whose success the user asked for.
        if result is not None and all_tool_calls:
            try:
                result = replace(
                    result,
                    tool_calls=all_tool_calls,
                    text="\n".join(texts) if texts else result.text,
                )
            except Exception:  # noqa: BLE001 — non-dataclass stub (tests) → as-is
                pass
        return result  # type: ignore[return-value]  # segments is non-empty by construction

    def _unified_answer_turn(
        self,
        user_message: str,
        session: Session,
        agent: Any,
        on_text: Callable[[str], None] | None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
        app_state: dict[str, Any] | None,
        on_reasoning: Callable[[str], None] | None,
        intent: IntentDecision,
    ) -> UnifiedTurnResult:
        """Conversation turn: ReAct produces the answer, harness verifies it.

        The answer text comes from the ReAct ``run_turn`` loop (so streaming,
        permissions, tool hooks, and read-only tool calls are all preserved exactly
        as the tool_use path owns them). That text is then wrapped as a 0-action
        ``answer_only`` GoalTree and run through the SAME harness loop so the turn
        produces a verified-loop trace (trivially-true verify) and one observation
        surface — closing the loop for chat too, without weakening the moat.
        """
        turn = self.run_turn(
            user_message,
            session,
            agent=agent,
            on_text=on_text,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            ask_permission=ask_permission,
            app_state=app_state,
            on_reasoning=on_reasoning,
        )

        trace, snapshot, verified = self._verify_answer_plan(user_message, turn.text)
        return UnifiedTurnResult(
            text=turn.text,
            trace=trace,
            snapshot=snapshot,
            intent=intent,
            tool_calls=list(turn.tool_calls),
            verified=verified,
            usage=turn.usage,
        )

    def _verify_answer_plan(
        self, user_message: str, answer_text: str
    ) -> "tuple[ExecutionTrace | None, dict[str, Any], bool]":
        """Wrap *answer_text* in an answer-only GoalTree and run the harness on it.

        Returns ``(trace, snapshot, verified)``. When VGG is unavailable there is no
        closed loop to run, so the trace/snapshot are empty and ``verified`` is
        False (the answer text is still surfaced by the caller). The evidence gate
        is consulted in the DEV world (strict) — an answer-only step is exempt by
        design; a robot world bypasses the gate as elsewhere.
        """
        if not (_VGG_AVAILABLE and self._vgg_enabled and self._goal_executor is not None):
            return None, {}, False

        tree = GoalDecomposer.answer_plan(user_message, answer_text)
        try:
            trace = self.vgg_execute(tree)
        except Exception as exc:  # noqa: BLE001
            logger.warning("unified: answer-plan execution failed: %s", exc)
            return None, {}, False

        snapshot = self._snapshot_safe(trace)
        verified = bool(trace.success) and self._evidence_ok(trace)
        return trace, snapshot, verified

    def _unified_plan_turn(
        self, user_message: str, intent: IntentDecision
    ) -> UnifiedTurnResult:
        """Action/plan turn: decompose to a GoalTree and run the harness loop.

        Reuses the existing VGG path (``vgg_decompose`` -> ``vgg_execute``) so the
        deterministic per-step verify, Blackboard binding, foreach expansion, and
        the replan layers are all unchanged — only the RESULT is re-surfaced as the
        unified type. If decomposition unexpectedly yields no tree (the classifier
        said vgg but the decomposer declined), there is nothing to verify; surface
        an empty trace rather than silently dropping the turn (fail loud, rule 8).
        """
        tree = self.vgg_decompose(user_message)
        if tree is None:
            logger.warning(
                "unified: classify_intent routed to vgg but vgg_decompose "
                "returned no GoalTree for %r — empty trace", user_message
            )
            return UnifiedTurnResult(
                text="", trace=None, snapshot={}, intent=intent,
            )

        trace = self.vgg_execute(tree)
        snapshot = self._snapshot_safe(trace)
        verified = bool(trace.success) and self._evidence_ok(trace)
        # The plan surface is the trace/snapshot; the answer text of an answer-only
        # step (if the decomposer produced one) is surfaced as the turn text.
        text = self._answer_text_from_trace(trace)
        return UnifiedTurnResult(
            text=text,
            trace=trace,
            snapshot=snapshot,
            intent=intent,
            verified=verified,
        )

    @staticmethod
    def _world_evidence_exemptions(world: Any) -> "frozenset[str]":
        """The world-declared per-step evidence exemptions (invariant II)."""
        return _world_exempt_strategies(world)

    def _evidence_ok(self, trace: "ExecutionTrace") -> bool:
        """Run the evidence gate for *trace* under the active world (fail-safe).

        Mirrors the CLI's gate call. Invariant II: no world-level bypass — the
        world may declare a bounded per-step exemption set
        (``evidence_exempt_strategies``); everything else needs deterministic
        evidence. Any failure reading the gate is treated as "not verified" so
        the moat never silently passes.
        """
        try:
            from vector_os_nano.vcli.cognitive.trace_store import evidence_passed
            world = getattr(self, "_world", None)
            return bool(evidence_passed(
                trace, exempt_strategies=_world_exempt_strategies(world)
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("unified: evidence gate read failed: %s", exc)
            return False

    def _snapshot_safe(self, trace: "ExecutionTrace") -> dict[str, Any]:
        """Return the run snapshot EXPORT VIEW for *trace*, or ``{}`` on failure."""
        try:
            return self.vgg_run_snapshot(trace)
        except Exception as exc:  # noqa: BLE001
            logger.debug("unified: run_snapshot failed: %s", exc)
            return {}

    @staticmethod
    def _answer_text_from_trace(trace: "ExecutionTrace") -> str:
        """Extract an answer-only step's text from *trace*, or ``""``.

        A plan may legitimately be (or include) an ``answer_only`` step; surface its
        captured text as the turn's assistant text. Pure read over the trace —
        never re-derives or evaluates anything.
        """
        sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
        for step in trace.steps:
            sg = sg_by_name.get(step.sub_goal_name)
            if sg is None or not getattr(sg, "answer_only", False):
                continue
            data = getattr(step, "result_data", {}) or {}
            text = (data.get("output") or {}).get("text") if isinstance(data, dict) else None
            if isinstance(text, str) and text:
                return text
        return ""

    # ------------------------------------------------------------------
    # Internal: tool partitioning and dispatch
    # ------------------------------------------------------------------

    def _partition_tools(self, tool_calls: list[LLMToolCall]) -> list[ToolBatch]:
        """Partition tool calls into concurrent (read-only) and sequential batches."""
        batches: list[ToolBatch] = []
        for tc in tool_calls:
            tool = self._registry.get(tc.name)
            is_safe = bool(
                tool is not None
                and hasattr(tool, "is_concurrency_safe")
                and tool.is_concurrency_safe(tc.input)
            )
            if is_safe and batches and batches[-1].concurrent:
                batches[-1].tool_calls.append(tc)
            else:
                batches.append(ToolBatch(concurrent=is_safe, tool_calls=[tc]))
        return batches

    def _dispatch_tools(
        self,
        tool_calls: list[LLMToolCall],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Dispatch all tool calls, respecting concurrency partitioning."""
        results: list[tuple[dict[str, Any], ToolCall]] = []
        batches = self._partition_tools(tool_calls)

        for batch in batches:
            if batch.concurrent and len(batch.tool_calls) > 1:
                batch_results = self._run_concurrent(
                    batch.tool_calls, tool_context, on_tool_start, on_tool_end, ask_permission
                )
            else:
                batch_results = self._run_sequential(
                    batch.tool_calls, tool_context, on_tool_start, on_tool_end, ask_permission
                )
            results.extend(batch_results)

        return results

    def _execute_single_tool(
        self,
        tc: LLMToolCall,
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> tuple[dict[str, Any], ToolCall]:
        """Execute one tool call with full permission checking."""
        tool_name = tc.name
        params = tc.input
        tool = self._registry.get(tool_name)

        if tool is None:
            result = ToolResult(content=f"Unknown tool: {tool_name}", is_error=True)
            logger.warning("Tool %r not found in registry", tool_name)
            return (
                {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="denied"),
            )

        # Permission gate — shared with the VGG tool path via the S5.1 seam. The
        # ReAct path does NOT swallow check/always-allow errors (a buggy
        # permission object or resolver is a real bug it surfaces), matching its
        # pre-seam behaviour.
        decision = resolve_permission(
            tool, params, tool_context, self._permissions, ask_permission,
        )

        if decision.kind == DECISION_DENY:
            reason = decision.reason or f"Permission denied for {tool_name}"
            result = ToolResult(content=f"Permission denied: {reason}", is_error=True)
            logger.info("Permission denied for tool %r: %s", tool_name, reason)
            return (
                {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="denied"),
            )

        if decision.kind == DECISION_ASK_DENY:
            denial = f"Permission denied by user for {tool_name}"
            result = ToolResult(content=denial, is_error=True)
            logger.info("User denied permission for tool %r", tool_name)
            return (
                {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="asked_denied"),
            )

        perm_action = "asked_allowed" if decision.kind == DECISION_ASK_ALLOW else "allowed"

        # Execute the tool
        if on_tool_start is not None:
            on_tool_start(tool_name, params)

        # Pre-hook
        if self._hooks is not None:
            from vector_os_nano.vcli.hooks import ToolHookContext
            self._hooks.fire_pre(ToolHookContext(tool_name=tool_name, params=params))

        start = time.monotonic()
        result = execute_resolved_tool(
            tool, params, tool_context,
            error_prefix="Tool error",
            on_error=lambda _name, exc: logger.error(
                "Tool %r raised %r", tool_name, exc, exc_info=True
            ),
        )
        duration = time.monotonic() - start

        # Post-hook
        if self._hooks is not None:
            from vector_os_nano.vcli.hooks import ToolHookContext
            self._hooks.fire_post(ToolHookContext(
                tool_name=tool_name, params=params, result=result, duration=duration,
            ))

        if on_tool_end is not None:
            on_tool_end(tool_name, result)

        return (
            {"tool_use_id": tc.id, "content": result.content, "is_error": result.is_error},
            ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=duration, permission_action=perm_action),
        )

    def _run_sequential(
        self,
        tool_calls: list[LLMToolCall],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Execute tool calls one-by-one in order."""
        return [
            self._execute_single_tool(tc, tool_context, on_tool_start, on_tool_end, ask_permission)
            for tc in tool_calls
        ]

    def _run_concurrent(
        self,
        tool_calls: list[LLMToolCall],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Execute read-only tool calls concurrently using a thread pool."""
        max_workers = min(len(tool_calls), 10)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._execute_single_tool, tc, tool_context, on_tool_start, on_tool_end, ask_permission
                )
                for tc in tool_calls
            ]
            return [f.result() for f in futures]
