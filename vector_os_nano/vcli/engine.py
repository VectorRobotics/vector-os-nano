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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vector_os_nano.vcli.backends import LLMBackend
from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session, TokenUsage
from vector_os_nano.vcli.tools.base import (
    PermissionResult,
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
    _VGG_AVAILABLE = True
except ImportError:
    _VGG_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Initialise the VGG cognitive pipeline components.

        Safe to call at any time. If initialisation fails for any reason
        (missing dependencies, bad backend), _vgg_enabled stays False and
        the engine continues to work through the normal tool_use path.
        """
        if not _VGG_AVAILABLE:
            logger.warning("VGG components not available — VGG disabled")
            return

        _backend = backend or self._backend
        self._vgg_agent = agent
        self._vgg_step_callback = on_vgg_step

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
        try:
            # Build primitives namespace for GoalVerifier
            ns = self._build_verifier_namespace(agent)
            stats = StrategyStats()
        except Exception as exc:
            logger.warning("VGG: verifier namespace build failed: %s", exc)
            self._vgg_enabled = False
            return

        try:
            decomposer = GoalDecomposer(_backend, skill_registry=skill_registry)
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
            selector = StrategySelector(skill_registry=skill_registry, stats=stats)
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
            _sg = getattr(_agent_ref, "_spatial_memory", None)
            _vlm = getattr(_agent_ref, "_vlm", None)
            services: dict = {}
            if _sg is not None:
                services["spatial_memory"] = _sg
            if _skill_registry_ref is not None:
                services["skill_registry"] = _skill_registry_ref
            if _vlm is not None:
                services["vlm"] = _vlm
            return SkillContext(
                bases={"go2": _base} if _base is not None else {},
                services=services,
            )

        try:
            executor = GoalExecutor(
                strategy_selector=selector,
                verifier=verifier,
                skill_registry=skill_registry,
                build_context=_build_context,
                stats=stats,
                visual_verifier_agent=agent,
            )
        except ImportError as exc:
            logger.warning("VGG: GoalExecutor not available: %s", exc)
            self._vgg_enabled = False
            return
        except Exception as exc:
            logger.warning("VGG: GoalExecutor init failed: %s", exc)
            self._vgg_enabled = False
            return

        try:
            self._goal_decomposer = decomposer
            self._goal_executor = executor
            self._vgg_harness = VGGHarness(
                decomposer=decomposer,
                executor=executor,
                selector=selector,
                config=HarnessConfig(
                    max_step_retries=2,
                    max_redecompose=1,
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

    def _build_verifier_namespace(self, agent: Any) -> dict[str, Any]:
        """Build function namespace for GoalVerifier from agent state."""
        ns: dict[str, Any] = {}
        if agent is None:
            return ns
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
        return ns

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

        if not self._vgg_enabled:
            return None
        if self._intent_router is None:
            return None
        # VGG needs a functioning robot — don't decompose before sim starts
        _agent = getattr(self, "_vgg_agent", None)
        if _agent is None or getattr(_agent, "_base", None) is None:
            return None
        _sr = getattr(_agent, "_skill_registry", None)
        if not self._intent_router.should_use_vgg(user_message, skill_registry=_sr):
            return None

        # Fast path: single skill match → 1-step GoalTree, no LLM
        if _sr is not None and not self._intent_router.is_complex(user_message):
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

        # Resolve room alias to canonical ID (e.g. "客房" → "guest_bedroom")
        # so verify expressions and params use the same IDs as SceneGraph.
        resolved_room = ""
        if skill_name == "navigate" and extracted:
            resolved_room = self._resolve_room_alias(extracted)
            if not resolved_room:
                return None  # unknown room — let LLM handle

        # Build verify expression using resolved canonical ID
        verify_arg = resolved_room if resolved_room else extracted
        verify = self._verify_for_skill(skill_name, verify_arg)

        # Build strategy params — extract from user message text
        params: dict = {}
        skill_obj = skill_registry.get(skill_name) if skill_registry else None
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
    def _verify_for_skill(skill_name: str, arg: str) -> str:
        """Generate a verify expression for a known skill."""
        _VERIFY_MAP: dict[str, str] = {
            "navigate": "nearest_room() == '{arg}'" if arg else "True",
            "explore": "True",  # async skill — launched = success, progress via events
            "patrol": "True",   # async skill — launched = success
            "look": "len(describe_scene()) > 0",
            "describe_scene": "len(describe_scene()) > 0",
            "where_am_i": "True",
            "stand": "True",
            "sit": "True",
            "stop": "True",
            "walk": "True",
            "turn": "True",
        }
        template = _VERIFY_MAP.get(skill_name, "True")
        return template.replace("{arg}", arg) if "{arg}" in template else template

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
            return self._vgg_harness.run(
                task=goal_tree.goal,
                world_context=world_context,
                goal_tree=goal_tree,
            )
        # Fallback: raw executor (no harness)
        return self._goal_executor.execute(goal_tree, on_step=self._on_vgg_step)

    def vgg_execute_async(
        self,
        goal_tree: "GoalTree",
        on_complete: "Callable[[ExecutionTrace], None] | None" = None,
    ) -> None:
        """Execute GoalTree in background thread. CLI remains responsive.

        Uses VGGHarness (with retry logic) when available, otherwise falls
        back to raw GoalExecutor.

        Args:
            goal_tree: The goal tree to execute.
            on_complete: Called when execution finishes (in background thread).
        """
        import threading

        self._vgg_cancel = threading.Event()

        def _run() -> None:
            try:
                trace = self.vgg_execute(goal_tree)
                if on_complete:
                    on_complete(trace)
            except Exception as exc:  # noqa: BLE001
                logger.warning("VGG async execution failed: %s", exc)

        t = threading.Thread(target=_run, name="vgg-executor", daemon=True)
        t.start()
        self._vgg_thread = t

    def _build_world_context(self) -> str:
        """Build a brief world context string for the GoalDecomposer.

        Results are cached for _world_context_ttl seconds to avoid repeated
        sensor/graph queries when the robot has not moved.
        """
        now = time.monotonic()
        if (
            self._world_context_cache is not None
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
        base = getattr(agent, "_base", None)
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
        result = "\n".join(parts) if parts else ""
        self._world_context_cache = result
        self._world_context_ts = now
        return result

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
        """Callback invoked by GoalExecutor after each sub-goal completes."""
        cb = getattr(self, "_vgg_step_callback", None)
        if cb:
            cb(step)

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
            response: LLMResponse = self._backend.call(
                messages=messages,
                tools=tools,
                system=self._system_prompt,
                max_tokens=self._max_tokens,
                on_text=on_text,
            )

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

        # Permission check
        perm: PermissionResult = self._permissions.check(tool, params, tool_context)

        if perm.behavior == "deny":
            reason = perm.reason or f"Permission denied for {tool_name}"
            result = ToolResult(content=f"Permission denied: {reason}", is_error=True)
            logger.info("Permission denied for tool %r: %s", tool_name, reason)
            return (
                {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="denied"),
            )

        if perm.behavior == "ask":
            response = ask_permission(tool_name, params) if ask_permission else "n"
            if response == "n":
                denial = f"Permission denied by user for {tool_name}"
                result = ToolResult(content=denial, is_error=True)
                logger.info("User denied permission for tool %r", tool_name)
                return (
                    {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                    ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="asked_denied"),
                )
            if response == "a":
                self._permissions.add_always_allow(tool_name)
            perm_action = "asked_allowed"
        else:
            perm_action = "allowed"

        # Execute the tool
        if on_tool_start is not None:
            on_tool_start(tool_name, params)

        # Pre-hook
        if self._hooks is not None:
            from vector_os_nano.vcli.hooks import ToolHookContext
            self._hooks.fire_pre(ToolHookContext(tool_name=tool_name, params=params))

        start = time.monotonic()
        try:
            result = tool.execute(params, tool_context)
        except Exception as exc:
            result = ToolResult(content=f"Tool error: {exc}", is_error=True)
            logger.error("Tool %r raised %r", tool_name, exc, exc_info=True)
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
