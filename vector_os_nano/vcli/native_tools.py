# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""native_tools — the native producer's own tools and the tool schemas the model is offered.

The loop's action surface (``_build_motor_tools``): the engine registry's code / sim /
robot tools, the coordinate ``navigate`` route, every registered capability and the
world's skills; plus the synthetic ``verify`` / ``finish`` schemas, whose predicate list
is read off the live verify namespace. Nothing here dispatches, verifies or grades
(``native_runner``) or talks to the model (``native_loop``).
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult
from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

logger = logging.getLogger(__name__)

# Synthetic tool names the runner OWNS (never wrapped from a skill).
VERIFY_TOOL = "verify"
FINISH_TOOL = "finish"

# The registry category whose tools are the kernel's domain-general ACTION surface
# (file_read/file_write/file_edit/bash/glob/grep — each tool declares its category).
# The native loop offers
# these as motor tools alongside the world's skills, so the dev world (no robot agent,
# hence no wrapped skills) can still ACT — and the robot world gains the same code
# tools. World-agnostic BY CONSTRUCTION: native asks the ENGINE'S registry for its
# registered action tools; there is NO "if dev" branch.
_CODE_TOOL_CATEGORY = "code"
_SIM_TOOL_CATEGORY = "sim"
_ROBOT_TOOL_CATEGORY = "robot"

# The MUTATING code tools — the ones whose effect is a file/shell WRITE the model can
# author at will (so it can "accomplish" a task by writing a marker file or running a
# shell command), NEVER a read-only diagnostic. In a ROBOT world these must NOT be an
# action path to a PHYSICAL goal: the fakeable-grasp defect was deepseek satisfying
# "抓前面的东西" by ``file_write('grabbed.txt')`` then ``verify(file_exists('grabbed.txt'))``
# (D17). A physical robot task is accomplished by a robot SKILL and proven by a GT
# oracle the actor cannot author — never by touching a file. So a robot world drops
# these from the loop's ACTION toolset (Prong 1) while KEEPING the read-only diagnostics
# (file_read / glob / grep) the persona uses to inspect code. The DEV world is untouched
# — it has no robot agent, so the gate below never fires there. Prong 2 (the goal-
# authenticity gate in the frozen spine) is the un-fakeable backstop if a model ever
# reaches a file oracle another way; this prong removes the easy path by construction.
_MUTATING_CODE_TOOLS: frozenset[str] = frozenset({"file_write", "file_edit", "bash"})


# D9 #1 — the avoidance NAVIGATION route.
_NAV_TIMEOUT_S = 60.0


def _agent_base(context: ToolContext) -> Any:
    """The connected mobile base from the tool context's agent (the WIRED accessor).

    The ``walk`` skill, the verify oracles, and actor-causation all reach the live base
    via ``agent._base``, so native does too.
    """
    agent = getattr(context, "agent", None)
    return getattr(agent, "_base", None)


class _NativeBaseNavigateTool:
    """Coordinate navigation through the nav-stack AVOIDANCE route (D9 #1).

    ``execute`` calls the connected base's ``navigate_to(x, y)``: the in-process
    planner (vgraph + the one sim-time follower) plans around the scene's obstacles and
    blocks until arrival, a stop, or timeout. This is the obstacle avoidance the
    open-loop ``walk`` lacks. The follower streams velocity inside the producer's
    ``ACTOR_EPOCH`` (the dispatch is open), so its commanded motion counts and a
    ``navigate`` step's ``at_position`` verify grades CAUSED when the base really moved
    there (closes D14's "nav RAN-only"); an external TARE/FAR follower driving the base
    between tools is never counted. ``verify_hint`` auto-verifies the target after it
    returns; it and the sentence about it are read off the world's arrival oracle
    (``ARRIVAL_ORACLE``): its own parameters bind the call's ``x`` / ``y``, its own
    signature and docstring are what the model reads. A world without that oracle gets
    no hint. Duck-typed like a ``Tool`` so ``dispatch_skill`` runs it uniformly.
    """

    name = "navigate"
    ARRIVAL_ORACLE = "at_position"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "Target x (world frame, metres)."},
            "y": {"type": "number", "description": "Target y (world frame, metres)."},
        },
        "required": ["x", "y"],
    }

    def __init__(self, arrived: Any = None, world: Any = None) -> None:
        self._world = world  # the bound world: is the body a live sim
        self.description = (
            "Go to a world coordinate (x, y) using the navigation PLANNER, which AVOIDS "
            "obstacles (lidar + local planner). Use this to REACH a place/coordinate."
        )
        self.verify_hint = ""
        try:
            params = list(inspect.signature(arrived).parameters) if callable(arrived) else []
        except (TypeError, ValueError):
            params = []
        bound = [p for p in params if p in self.input_schema["required"]]
        if bound:
            self.verify_hint = f"{self.ARRIVAL_ORACLE}({', '.join('{%s}' % p for p in bound)})"
            self.description += (f" It is then verified with {_oracle_line(self.ARRIVAL_ORACLE, arrived)}")

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        # It drives the base: confirm unless the body is a live in-process sim.
        live = self._world is not None and self._world.session(getattr(context, "agent", None)) is not None
        return PermissionResult("allow" if live else "ask")

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return False

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            x = float(params["x"])
            y = float(params["y"])
        except (KeyError, TypeError, ValueError):
            return ToolResult(content="navigate requires numeric x and y.", is_error=True)

        base = _agent_base(context)
        navigate_to = getattr(base, "navigate_to", None)
        if not callable(navigate_to):
            return ToolResult(
                content="navigate: the connected base has no navigate_to (no nav stack).",
                is_error=True,
            )

        res = None
        # Pass a timeout only to a base that takes one: retrying on ANY TypeError
        # would re-run a navigation that raised one mid-way.
        try:
            params_ = inspect.signature(navigate_to).parameters.values()
            takes_timeout = any(p.name == "timeout" or p.kind is p.VAR_KEYWORD for p in params_)
        except (TypeError, ValueError):
            takes_timeout = False
        try:
            res = navigate_to(x, y, timeout=_NAV_TIMEOUT_S) if takes_timeout else navigate_to(x, y)
            ok = bool(res)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"navigate({x}, {y}) failed: {exc}", is_error=True)

        pos = None
        getter = getattr(base, "get_position", None)
        if callable(getter):
            try:
                pos = getter()
            except Exception as exc:  # noqa: BLE001
                logger.debug("native_loop: navigate get_position failed: %s", exc)
        where = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos else "unknown"
        reason = getattr(res, "reason", None)  # the in-process planner's NavResult (Rule 4)
        if ok:
            status = "arrived"
            if reason is not None and res.get("snapped"):
                status += f" at {tuple(res['target'])}, the nearest free point (the target is inside an obstacle)"
        elif getattr(getattr(context, "abort", None), "is_set", lambda: False)():
            status = "was STOPPED before arrival"
        elif reason is not None:
            status = f"did NOT arrive ({reason})"
        else:
            status = "did NOT confirm arrival (FAR graph unavailable or timeout)"
        return ToolResult(
            content=f"navigate: {status}; now at {where} (target ({x}, {y}))."
        )


class _NativeDetectTool:
    """Route a NAMED/COLOUR query to the learned grounding-dino DETECTOR capability.

    R4 — composes the two North-Star axes on ONE turn: this surfaces the registered
    ``DetectorCapability`` (the learned SECOND model family, grounding-dino-tiny) into
    the native loop so a bare-cli detect command reaches it on whatever camera-bearing
    embodiment is connected — go2+arm OR g1 (the humanoid, camera but NO arm). It does
    NOT re-implement detection: ``execute`` pulls the SAME capability instance the world
    registered into the engine's ``CapabilityRegistry`` (agent-bound for the live
    perception) and invokes it — single-sourced, no second model load (rule 3).

    Read-only PERCEPTION: the detector only localizes, it does not act on the world.
    The capability never self-certifies: the suggested verify is the GT ``in_view``
    oracle (segmentation of the robot's own camera, which the detector never sees), and
    it grades on the base channel, so a detect without a base move grades RAN — the
    honest grade for a perceive-only route. ``localized`` (the scene graph vs GT) holds
    once look/explore have put the object on the map.

    Duck-typed like a ``Tool`` so ``dispatch_skill`` runs it uniformly. Only surfaced
    when a ``detect`` capability is registered AND the agent has a camera (see
    ``_build_motor_tools``) — sensorless/dev paths never see it.
    """

    name = "detect"
    description = (
        "Localize a named or coloured object in the robot's CAMERA view with a learned "
        "detector. Pass the target as 'query' (e.g. 'red object', 'a red stool', "
        "'红色的东西'). 'backend' picks the model: 'grounding-dino' (default: local, "
        "~0.1 s, scored boxes, best with an English noun phrase) or 'rynnbrain' (an "
        "external embodied VLM: ~1 s, one unscored box, reads Chinese and relations like "
        "'the bottle left of the red can'; use it when the user asks for RynnBrain or "
        "the target is described by relation). Returns box(es) + label(s) + score(s). "
        "PERCEPTION only (read-only). After it returns, call the verify expression the "
        "tool result tells you to use."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language target to localize (named or coloured).",
            },
            "backend": {
                "type": "string",
                "enum": ["grounding-dino", "rynnbrain"],
                "description": "Detector model (default grounding-dino).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, capability: Any) -> None:
        # The SAME registered DetectorCapability the world bound to the agent's live
        # perception — never a fresh instance, never a second model load.
        self._capability = capability
        self.name = str(getattr(capability, "name", "") or "detect")

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return PermissionResult("allow")  # perception only

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str((params or {}).get("query", "")).strip()
        if not query:
            return ToolResult(content="detect requires a non-empty 'query'.", is_error=True)
        hint = f"verify(in_view({query!r}))"  # the GT verify to make after detect (the model authors it)
        payload = {"query": query}
        if (params or {}).get("backend"):
            payload["backend"] = str(params["backend"])
        try:
            result = self._capability.invoke(payload, context)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"detect({query!r}) raised: {exc}", is_error=True)

        output = dict(getattr(result, "output", {}) or {})
        boxes = output.get("boxes") or []
        labels = output.get("labels") or []
        scores = output.get("scores") or []
        model = output.get("backend") or payload.get("backend") or "grounding-dino"

        # MOAT DISCIPLINE (R6 audit): the detector's OWN output is never stashed on
        # the agent for a verify to read back (the D61 self-certifying tautology); it
        # flows only to the human-readable tool output below.
        if not getattr(result, "success", False) and not boxes:
            err = getattr(result, "error", "")
            if err:  # the model did not run on this query (down, ungroundable, bad backend)
                return ToolResult(content=f"detect({query!r}) via {model} failed: {err}", is_error=True)
            return ToolResult(
                content=(
                    f"detect({query!r}): {model} RAN but localized nothing. "
                    f"{hint} checks whether it is in the camera view at all."
                )
            )
        first = (
            f"label={labels[0]!r} box={[round(float(v), 1) for v in boxes[0]]} "
            f"score={float(scores[0]):.2f}"
            if boxes and labels and scores
            else "(box)"
        )
        return ToolResult(
            content=(
                f"detect({query!r}): {model} localized {len(boxes)} object(s); "
                f"top {first}. Call {hint} to confirm."
            )
        )


class _NativeCapabilityTool:
    """Any registered Capability as a native tool (S6): the model routes to it by name.

    A ``side_effecting`` capability (a VLA, a planner that moves the robot) goes
    through the permission gate as ``ask``; a read-only one is allowed. The
    capability never self-certifies — its output is tool text, the verify is the moat.
    """

    def __init__(self, capability: Any) -> None:
        self._capability = capability
        self.name = str(capability.name)
        self.description = str(
            getattr(capability, "description", "")
            or f"Routable {getattr(capability, 'kind', 'model')} capability '{self.name}'."
        )
        self.input_schema = dict(
            getattr(capability, "input_schema", None) or {"type": "object", "properties": {}}
        )
        self._side_effecting = bool(getattr(capability, "side_effecting", True))

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return PermissionResult("ask" if self._side_effecting else "allow")

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return not self._side_effecting

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        result = self._capability.invoke(dict(params or {}), context)
        if not getattr(result, "success", False):
            return ToolResult(
                content=f"{self.name} failed: {getattr(result, 'error', '') or 'no output'}",
                is_error=True,
            )
        output = json.dumps(getattr(result, "output", {}) or {}, ensure_ascii=False, default=str)
        return ToolResult(content=f"{self.name}: {output[:2000]}")


# ---------------------------------------------------------------------------
# Synthetic tool schemas (the verify/finish/motor tool-set the model is offered)
# ---------------------------------------------------------------------------


def _oracle_line(name: str, fn: Any) -> str:
    """``name(args): <first paragraph of the oracle's docstring>`` — what the model reads."""
    try:
        params = inspect.signature(fn).parameters.values()
        sig = str(inspect.Signature([p.replace(annotation=inspect.Parameter.empty) for p in params]))
    except (TypeError, ValueError):
        sig = "(...)"
    doc = " ".join((inspect.getdoc(fn) or "").split("\n\n")[0].split())
    return f"{name}{sig}: {doc}" if doc else f"{name}{sig}"


def _verify_tool_schema(namespace: dict[str, Any], oracle_names: frozenset[str]) -> dict[str, Any]:
    """Anthropic-shaped schema for the synthetic ``verify(expr)`` tool.

    The predicates are listed from ONE source: for every evidence oracle of the
    active world (``verify_oracle_names``), its signature and the first paragraph of
    its own docstring, read off the SAME namespace the verifier evaluates.
    """
    lines = "\n".join(f"- {_oracle_line(n, namespace[n])}" for n in sorted(oracle_names) if n in namespace)
    desc = (
        "Check a deterministic post-condition predicate against real world state, "
        "then bind it as this step's verification. Call this AFTER the action skill(s) "
        "that should have achieved the goal. The predicate is evaluated by an "
        "independent verifier over real sensor/sim ground truth — it cannot be faked. "
        "Pass the FULL predicate expression as 'expr' (combine predicates with and / "
        "or / not). After a retry, verify the SAME predicate again: the latest attempt "
        "counts, and a different, easier predicate never replaces a FAIL. Each verify "
        "must follow an action. Predicates:\n" + (lines or "(none)")
    )
    return {
        "name": VERIFY_TOOL,
        "description": desc,
        "input_schema": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "The deterministic verify predicate to evaluate.",
                },
                "goal_id": {
                    "type": "string",
                    "description": "Optional: a short id of the sub-goal this verify proves.",
                },
            },
            "required": ["expr"],
        },
    }


def _finish_tool_schema() -> dict[str, Any]:
    return {
        "name": FINISH_TOOL,
        "description": (
            "Signal the task is complete. Call this once every step has been "
            "executed and verified."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


def _build_motor_tools(agent: Any, engine: Any) -> dict[str, Any]:
    """Assemble the loop's ACTION surface: the world's skills + the engine's code tools.

    Two sources, both world-agnostic by construction:

    1. ``wrap_skills(agent)`` — the world's robot skills (``walk`` etc.). The world's
       room ``navigate`` skill (door-chain walk) is replaced by the coordinate
       ``navigate`` route below. When ``agent`` is None (the dev world), this source is
       empty.
    2. The engine registry's ``code``-category tools (file_read/file_write/file_edit/
       bash/glob/grep) — the kernel's domain-general action surface. These
       are real ``Tool`` objects whose ``.execute(params, ctx)`` is the interface
       ``dispatch_skill`` already calls, so they slot in directly. This is what lets the
       dev world (no agent -> no skills) ACT, and adds the code tools to the robot world
       too. Native asks the ENGINE for its registered action tools — there is NO
       embodiment/"if dev" branch in this module (rule 7).

    The synthetic ``verify``/``finish`` names are loop-owned and never collide with a
    registry tool; a code tool can never shadow them. A wrapped skill that happens to
    share a code-tool name (none do today) would take precedence — skills are layered
    AFTER the code tools so the world's own skill wins if a clash ever arises.
    """
    tools: dict[str, Any] = {}
    # Source 2: the engine registry's code tools (present in every world). In a ROBOT
    # world (a connected robot ``agent``), the MUTATING code tools (file_write/file_edit/
    # bash) are DROPPED from the action surface — a physical robot goal must be achieved
    # by a robot SKILL and proven by a GT oracle, never by writing a marker file or
    # running a shell command (the D17 fakeable-grasp path). The READ-ONLY diagnostics
    # (file_read/glob/grep) are kept so the persona can still inspect code. The dev world
    # (no robot agent) keeps the full set unchanged.
    is_robot_world = agent is not None
    for name, code_tool in _registry_tools(engine, _CODE_TOOL_CATEGORY).items():
        if is_robot_world and name in _MUTATING_CODE_TOOLS:
            continue  # Prong 1: no file/shell WRITE as a path to a physical robot task
        tools[name] = code_tool
    # The sim lifecycle (start/stop_simulation) in every world: a dev-world turn can
    # bring a body up and use it in the same turn (the runner rebinds after it).
    tools.update(_registry_tools(engine, _SIM_TOOL_CATEGORY))
    # A robot world's registry tools that are not skill wrappers: scene_graph_query
    # (read-only memory), go_to_room. Skills come fresh from wrap_skills below.
    if is_robot_world:
        tools.update({n: t for n, t in _registry_tools(engine, _ROBOT_TOOL_CATEGORY).items()
                      if not getattr(t, "_is_skill_wrapper", False)})
    # Source 3 (D9 #1): the avoidance NAVIGATION route, only for a world with a mobile
    # base: the base's planner routes around obstacles; its follower's commands count as
    # the actor's (ACTOR_EPOCH), so a navigate that arrived grades CAUSED. The arm/dev
    # worlds (no base) do not get it.
    # The base gate is single-sourced (Rule 11) onto the capability resolver.
    world = getattr(engine, "_world", None)
    if world is not None and world.has_base(agent):
        namespace = getattr(engine, "_build_verifier_namespace", lambda a: {})(agent)
        tools["navigate"] = _NativeBaseNavigateTool(namespace.get(_NativeBaseNavigateTool.ARRIVAL_ORACLE), world)
    # Source 4 (S6): EVERY capability the world registered into the engine's
    # CapabilityRegistry, as a tool the model routes to by name — the SAME instances
    # the world bound (agent-bound for live perception), never a second model load.
    # A detector keeps its verify-hint adapter (R4); any other capability is generic,
    # and a side-effecting one asks permission first. Layered before the world skills
    # so a learned route WINS a name clash (the bare-cli 'detect' is the MODEL route,
    # not the classical DetectSkill).
    cap_names: set[str] = set()
    for cap in _registered_capabilities(engine):
        tool = _NativeDetectTool(cap) if getattr(cap, "kind", "") == "detector" else _NativeCapabilityTool(cap)
        tools[tool.name] = tool
        cap_names.add(tool.name)
    # Source 1: the world's skills (robot worlds only; dev world has no agent). The
    # world's own ``navigate`` skill (door-chain walk, NOT lidar avoidance) stays
    # excluded; our coordinate ``navigate`` above is the planner/avoidance route.
    if agent is not None:
        try:
            for skill_tool in wrap_skills(agent):
                if skill_tool.name == "navigate":
                    continue  # world room-navigate is door-chain walk; ours is the avoidance route
                if skill_tool.name in cap_names:
                    continue  # a registered capability (Source 4) wins over a same-named skill
                tools[skill_tool.name] = skill_tool
        except Exception as exc:  # noqa: BLE001
            logger.debug("native_loop: wrap_skills failed: %s", exc)
    return tools


def _registered_capabilities(engine: Any) -> list[Any]:
    """Every Capability in the ENGINE's CapabilityRegistry (``engine._capability_registry``).

    The engine owns the registry the world populated at bind, so the native loop
    surfaces the EXACT instances —
    never a fresh construction (rule 3). No registry / a lookup failure -> none.
    """
    registry = getattr(engine, "_capability_registry", None)
    if registry is None:
        return []
    try:
        return [c for c in (registry.get(n) for n in sorted(registry.names())) if c is not None]
    except Exception as exc:  # noqa: BLE001
        logger.debug("native_loop: capability discovery failed: %s", exc)
        return []


def _registry_tools(engine: Any, category: str) -> dict[str, Any]:
    """Return the engine registry's *category* Tool objects, keyed by name.

    Pulls the live, instantiated tools out of the engine's ``CategorizedToolRegistry``
    so the native loop offers the EXACT action surface the kernel registered (no
    duplicate construction, no hand-authored allowlist here — single-sourced from the
    registry, rule 3). Defensive: a registry without the categorized API, a missing
    category, or any lookup failure yields an empty set (the caller gates robot tools
    on a connected agent).
    """
    registry = getattr(engine, "_registry", None)
    if registry is None:
        return {}
    list_categories = getattr(registry, "list_categories", None)
    get = getattr(registry, "get", None)
    if list_categories is None or get is None:
        return {}  # a plain ToolRegistry (no categories) -> no code tools to surface
    tools: dict[str, Any] = {}
    try:
        names = list_categories().get(category, [])
        for name in names:
            code_tool = get(name)
            if code_tool is not None:
                tools[name] = code_tool
    except Exception as exc:  # noqa: BLE001
        logger.debug("native_loop: %s-tool discovery failed: %s", category, exc)
    return tools


def _native_tool_schemas(
    motor_tools: dict[str, Any], namespace: dict[str, Any], oracle_names: frozenset[str]
) -> list[dict[str, Any]]:
    """The Anthropic-shaped tool schemas the model is offered for a native turn."""
    schemas: list[dict[str, Any]] = []
    for name, tool in motor_tools.items():
        schemas.append(
            {
                "name": name,
                "description": getattr(tool, "description", name),
                "input_schema": getattr(tool, "input_schema", {"type": "object", "properties": {}}),
            }
        )
    schemas.append(_verify_tool_schema(namespace, oracle_names))
    schemas.append(_finish_tool_schema())
    return schemas
