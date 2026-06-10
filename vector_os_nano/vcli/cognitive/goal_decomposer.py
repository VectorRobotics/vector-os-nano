# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GoalDecomposer — LLM-backed natural language task decomposition.

Converts a natural language task string into a structured GoalTree by:
1. Building a system prompt that describes the JSON schema, known strategies,
   and allowed verify-expression functions.
2. Calling the injected LLMBackend.
3. Extracting JSON from the response (handles markdown fences).
4. Validating every SubGoal:
   - verify: valid Python expression (ast.parse), only VERIFY_FUNCTIONS called
   - strategy: in KNOWN_STRATEGIES or cleared to ""
   - depends_on: all referenced names must exist in the tree
5. Truncating to MAX_SUB_GOALS.
6. Returning a single-step fallback GoalTree on any JSON parse failure.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from vector_os_nano.vcli.cognitive.types import ForEachSpec, GoalTree, SubGoal

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST visitor — collect all function-call names in an expression
# ---------------------------------------------------------------------------

class _CallNameCollector(ast.NodeVisitor):
    """Collect the set of base function names called in an AST."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name):
            self.names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # e.g. obj.method() — collect the root Name if present
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value  # type: ignore[assignment]
            if isinstance(root, ast.Name):
                self.names.add(root.id)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# GoalDecomposer
# ---------------------------------------------------------------------------

class GoalDecomposer:
    """Decomposes natural language tasks into structured GoalTrees via LLM."""

    # Max sub-goals to prevent over-decomposition
    MAX_SUB_GOALS: int = 8

    # Token budget for the decompose LLM call. A REASONING model (deepseek-v4-flash)
    # emits a hidden reasoning_content trace BEFORE the final JSON content; that trace
    # spends part of the completion budget, so a too-small max_tokens truncates the
    # FINAL JSON (the only part the backend keeps), producing the "no JSON found" /
    # "Expecting ',' delimiter" failures seen live. 2048 was tuned for non-reasoning
    # models. 8192 gives ample headroom for the reasoning trace PLUS a full
    # MAX_SUB_GOALS plan (worst-case plan JSON is well under ~2k tokens), so the final
    # JSON is never the thing that gets cut. Configurable per instance/world.
    DEFAULT_DECOMPOSE_MAX_TOKENS: int = 8192

    # Bounded re-asks on a JSON extraction/parse failure (the model gets ONE terse
    # "return ONLY valid JSON" nudge before we fall back loud). Keeps cost bounded.
    DECOMPOSE_MAX_RETRIES: int = 1

    # Default strategies — overridden at runtime from actual SkillRegistry
    KNOWN_STRATEGIES: frozenset[str] = frozenset({
        "navigate_skill",
        "look_skill",
        "describe_scene_skill",
        "stand_skill",
        "sit_skill",
        "stop_skill",
        "explore_skill",
        "walk_forward",
        "turn",
        "scan_360",
    })

    # Functions available in verify expressions
    VERIFY_FUNCTIONS: frozenset[str] = frozenset({
        "nearest_room",
        "get_position",
        "get_heading",
        "get_visited_rooms",
        "query_rooms",
        "describe_scene",
        "detect_objects",
        "world_stats",
        # Phase 3: Active World Model functions
        "last_seen",
        "certainty",
        "objects_in_room",
        "find_object",
        "room_coverage",
        "predict_navigation",
    })

    # Safe Python builtins that may appear in verify expressions alongside
    # VERIFY_FUNCTIONS (mirrors GoalVerifier._SAFE_BUILTINS).
    _ALLOWED_BUILTINS: frozenset[str] = frozenset({
        "len", "str", "int", "float", "bool", "list", "tuple",
        "abs", "min", "max", "isinstance", "any", "all",
    })

    # ------------------------------------------------------------------
    # Strategy descriptions for the system prompt
    # ------------------------------------------------------------------

    _STRATEGY_DESCRIPTIONS: dict[str, str] = {
        "navigate_skill": "Navigate to a named room or pose",
        "look_skill": "Point camera and observe the environment",
        "describe_scene_skill": "Ask VLM to describe what is in view",
        "stand_skill": "Command robot to stand upright",
        "sit_skill": "Command robot to sit down",
        "stop_skill": "Emergency stop — halt all motion",
        "explore_skill": "Explore unknown area autonomously",
        "walk_forward": "Walk straight forward a set distance",
        "turn": "Rotate in place by given angle",
        "scan_360": "Rotate 360° while recording observations",
        "patrol_skill": "Visit multiple rooms in sequence",
    }

    # Verify function signatures for the system prompt
    _VERIFY_FN_SIGNATURES: dict[str, str] = {
        "nearest_room": "nearest_room() -> str  # room id closest to current position",
        "get_position": "get_position() -> tuple[float,float,float]  # (x, y, z) in metres",
        "get_heading": "get_heading() -> float  # heading in radians",
        "get_visited_rooms": "get_visited_rooms() -> list[str]  # list of visited room ids",
        "query_rooms": "query_rooms() -> list[dict]  # all known rooms",
        "describe_scene": "describe_scene() -> str  # VLM description of current view",
        "detect_objects": "detect_objects(query: str = '') -> list[dict]  # object detections",
        "world_stats": "world_stats() -> dict  # {'rooms': int, 'objects': int, 'visited': int}",
        # Phase 3: Active World Model functions
        "last_seen": "last_seen(category: str) -> dict | None  # most recent observation of category",
        "certainty": "certainty(fact: str) -> float  # time-decayed confidence, e.g. certainty('cup在kitchen')",
        "objects_in_room": "objects_in_room(room_id: str) -> list[dict]  # objects with confidence in room",
        "find_object": "find_object(category: str) -> list[dict]  # all known locations of category",
        "room_coverage": "room_coverage(room_id: str) -> float  # exploration coverage 0.0~1.0",
        "predict_navigation": "predict_navigation(target: str) -> dict  # {reachable, door_count, rooms_on_path}",
    }

    # ------------------------------------------------------------------
    # JSON schema description embedded in the system prompt
    # ------------------------------------------------------------------

    _JSON_SCHEMA = """\
{
  "goal": "<original task string>",
  "sub_goals": [
    {
      "name": "<unique snake_case identifier>",
      "description": "<human-readable step description>",
      "verify": "<Python expression using ONLY the verify functions listed below>",
      "timeout_sec": <float, default 30.0 — use 45+ for slow motor / long-running skill steps under a live sim, short values for pure checks>,
      "depends_on": ["<name of preceding sub_goal>"],
      "strategy": "<one of KNOWN_STRATEGIES or empty string>",
      "strategy_params": {},
      "fail_action": "<optional: what to do on failure>",
      "foreach": <optional loop block — OMIT for a plain step; see below>
    }
  ],
  "context_snapshot": "<optional: brief summary of world context used>"
}

## Loops (foreach) — repeat a body once per item of a list
To do the same action over EVERY item a prior step found (e.g. "grab everything,
one by one"), add a "foreach" block to a sub_goal INSTEAD of writing the steps
out by hand. Leave that sub_goal's own "strategy" empty (""); the body does the
work. The loop reads its list from an EARLIER step's captured result:
{
  "foreach": {
    "source_step": "<name of an earlier sub_goal that produced the list>",
    "source_path": "<dotted path into that step's result to the list>",
    "var": "<iteration variable name, e.g. 'item'>",
    "body": [ <sub_goal templates, same shape as above, run once per item> ]
  }
}
Rules:
  - "source_step" MUST name an earlier sub_goal; the loop auto-depends on it.
  - Inside body templates, reference the current item's fields as a string
    "${<var>.<field>}" (e.g. "${item.name}") in strategy_params or verify. This
    is resolved per item by safe path lookup — never code, never eval.
  - body templates use the SAME strategies/verify functions as top-level steps.

## Singular vs. ALL intent — CRITICAL
Use a foreach loop ONLY when the task explicitly means EVERY / ALL items
(keywords like "all", "every", "each", "所有", "每个", "全部", "一个个", "一遍").
When the task asks to act on ONE / ANY single unspecified item
(e.g. "a thing", "one object", "something", "一个", "个", "随便", "某个"),
use a SINGLE action step — NO foreach — and leave the object target param
BLANK (empty string or omit it entirely); the skill will resolve the nearest
object autonomously. Never iterate over all objects for a singular request."""

    # Example decomposition
    _EXAMPLE = """\
Task: "去厨房看看有没有杯子"
Response:
{
  "goal": "去厨房看看有没有杯子",
  "sub_goals": [
    {
      "name": "reach_kitchen",
      "description": "导航到厨房",
      "verify": "nearest_room() == 'kitchen'",
      "strategy": "navigate_skill",
      "timeout_sec": 60,
      "depends_on": [],
      "strategy_params": {"room": "kitchen"},
      "fail_action": ""
    },
    {
      "name": "observe_table",
      "description": "观察厨房桌面",
      "verify": "'table' in describe_scene()",
      "strategy": "look_skill",
      "timeout_sec": 15,
      "depends_on": ["reach_kitchen"],
      "strategy_params": {},
      "fail_action": ""
    },
    {
      "name": "detect_cup",
      "description": "检测杯子是否存在",
      "verify": "len(detect_objects('cup')) > 0",
      "strategy": "detect_skill",
      "timeout_sec": 10,
      "depends_on": ["observe_table"],
      "strategy_params": {"query": "cup"},
      "fail_action": ""
    }
  ],
  "context_snapshot": "Robot is in hallway, kitchen is adjacent."
}

Task: "向前走2米然后右转90度"
Response:
{
  "goal": "向前走2米然后右转90度",
  "sub_goals": [
    {
      "name": "walk_forward_2m",
      "description": "向前走2米",
      "verify": "True",
      "strategy": "walk_forward",
      "timeout_sec": 30,
      "depends_on": [],
      "strategy_params": {"distance": 2.0, "speed": 0.3},
      "fail_action": ""
    },
    {
      "name": "turn_right_90",
      "description": "右转90度",
      "verify": "True",
      "strategy": "turn",
      "timeout_sec": 15,
      "depends_on": ["walk_forward_2m"],
      "strategy_params": {"angle": -90},
      "fail_action": ""
    }
  ],
  "context_snapshot": ""
}"""

    # World-neutral worked FOREACH example. Always appended to the prompt (NOT
    # part of the per-world ``_EXAMPLE`` override), so every world — arm, go2, dev
    # — is taught the loop shape with a concrete detect -> foreach(act) plan.
    # Uses placeholder strategy names (``<detect>_skill`` / ``<act>_skill``) and
    # the always-safe ``detect_objects()`` verify so it never hardcodes a specific
    # world's vocabulary; the LLM substitutes its own real strategies.
    _FOREACH_EXAMPLE: str = """\
Loop example — "do <something> to every detected object, one by one":
{
  "goal": "do <something> to every detected object",
  "sub_goals": [
    {
      "name": "detect_items",
      "description": "detect every target object",
      "verify": "len(detect_objects()) > 0",
      "strategy": "<detect>_skill",
      "timeout_sec": 10,
      "depends_on": [],
      "strategy_params": {},
      "fail_action": ""
    },
    {
      "name": "act_on_each",
      "description": "act on each detected object, one by one",
      "verify": "True",
      "strategy": "",
      "timeout_sec": 30,
      "depends_on": ["detect_items"],
      "strategy_params": {},
      "fail_action": "",
      "foreach": {
        "source_step": "detect_items",
        "source_path": "objects",
        "var": "item",
        "body": [
          {
            "name": "act_item",
            "description": "act on the current object",
            "verify": "True",
            "strategy": "<act>_skill",
            "timeout_sec": 45,
            "depends_on": [],
            "strategy_params": {"target": "${item.name}"},
            "fail_action": ""
          }
        ]
      }
    }
  ],
  "context_snapshot": ""
}"""

    # Default planner intro + strategy-params help (robot world). These are
    # injectable so a non-robot "world" can supply its own decompose vocabulary.
    _PLANNER_INTRO: str = (
        "You are a robot task planner. Decompose the user's task into verifiable sub-goals."
    )
    _STRATEGY_PARAMS_HELP: str = """\
  - navigate_skill: {"room": "<room_name>"}
  - walk_forward: {"distance": <meters float>, "speed": <m/s float>}
  - turn: {"angle": <degrees int, positive=left, negative=right>}
  - detect_skill: {"query": "<object_name>"}
  - stand_skill: {}
  - sit_skill: {}
  - stop_skill: {}
  - explore_skill: {}
  - look_skill: {}
  - scan_360: {}"""

    # Default fallback verify (robot world). Dev world overrides to "True".
    _FALLBACK_VERIFY: str = "world_stats() is not None"

    def __init__(
        self,
        backend: Any,
        template_library: Any = None,
        skill_registry: Any = None,
        *,
        verify_functions: "frozenset[str] | set[str] | None" = None,
        verify_fn_signatures: "dict[str, str] | None" = None,
        strategy_descriptions: "dict[str, str] | None" = None,
        strategies: "frozenset[str] | set[str] | None" = None,
        strategy_params_help: "str | None" = None,
        examples: "str | None" = None,
        fallback_verify: "str | None" = None,
        planner_intro: "str | None" = None,
        has_base: bool = True,
        decompose_max_tokens: "int | None" = None,
    ) -> None:
        """Initialise with an LLMBackend (must implement .call()).

        Args:
            backend: Any object implementing the LLMBackend Protocol.
            template_library: Optional TemplateLibrary for template matching.
            skill_registry: Optional SkillRegistry — when provided (and ``strategies``
                           is not given), KNOWN_STRATEGIES is built from skill names.
            verify_functions / verify_fn_signatures / strategy_descriptions /
            strategies / strategy_params_help / examples / fallback_verify /
            planner_intro: optional per-world decompose vocabulary overrides. When
                           omitted, the robot defaults (class attributes) are used,
                           preserving existing behaviour exactly.
            has_base: whether the connected agent has a mobile base. Gates the
                     registry-derived fallback's base-primitive union
                     (walk_forward/turn/scan_360); an arm-only agent (False) is
                     never taught the base primitives. Ignored when an explicit
                     ``strategies`` set is injected. Defaults True (robot/go2),
                     preserving existing behaviour.
            decompose_max_tokens: completion-token budget for the decompose LLM
                     call. Defaults to ``DEFAULT_DECOMPOSE_MAX_TOKENS`` (sized for a
                     reasoning model whose hidden reasoning trace shares the budget).
        """
        self._backend = backend
        self._template_library = template_library
        self._skill_registry = skill_registry
        # Token budget for the decompose call (reasoning-model-aware default).
        self._decompose_max_tokens = (
            decompose_max_tokens
            if decompose_max_tokens is not None
            else self.DEFAULT_DECOMPOSE_MAX_TOKENS
        )
        # Cached system prompt — built once per instance, reused across decompose() calls.
        self._cached_system_prompt: list[dict[str, Any]] | None = None

        # --- Per-world vocabulary injection (override instance attrs; class
        #     attributes remain the robot defaults when nothing is injected) ---
        if verify_functions is not None:
            self.VERIFY_FUNCTIONS = frozenset(verify_functions)
        if verify_fn_signatures is not None:
            self._VERIFY_FN_SIGNATURES = dict(verify_fn_signatures)
        # ``step_output`` is KERNEL-provided (the executor injects it per-step,
        # bound to that step's own structured output — backlog #3 / Rule 4), so
        # it is valid in EVERY world regardless of the injected vocab — union it
        # in unconditionally, like ``answer`` on the strategy side.
        self.VERIFY_FUNCTIONS = frozenset(self.VERIFY_FUNCTIONS | {"step_output"})
        self._VERIFY_FN_SIGNATURES = {
            **self._VERIFY_FN_SIGNATURES,
            "step_output": (
                "step_output(path='') -> Any  # THIS step's own structured output; "
                "e.g. len(step_output('objects')) > 0 verifies what THIS detect "
                "step itself found (not a separate scene query)"
            ),
        }
        if strategy_descriptions is not None:
            self._STRATEGY_DESCRIPTIONS = dict(strategy_descriptions)
        if strategy_params_help is not None:
            self._STRATEGY_PARAMS_HELP = strategy_params_help
        if examples is not None:
            self._EXAMPLE = examples
        if fallback_verify is not None:
            self._FALLBACK_VERIFY = fallback_verify
        if planner_intro is not None:
            self._PLANNER_INTRO = planner_intro

        # Strategies: explicit injection wins; else derive from skill registry;
        # else keep the robot defaults.
        if strategies is not None:
            self.KNOWN_STRATEGIES = frozenset(strategies)
        elif skill_registry is not None:
            try:
                skill_names = set(skill_registry.list_skills())
                real_strategies = {f"{n}_skill" for n in skill_names}
                # Base locomotion primitives only when the agent has a base; an
                # arm-only agent must never be taught walk_forward/turn/scan_360.
                if has_base:
                    real_strategies |= {"walk_forward", "turn", "scan_360"}
                self.KNOWN_STRATEGIES = frozenset(real_strategies)
            except Exception:
                pass  # keep defaults

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, task: str, world_context: str) -> GoalTree:
        """Decompose *task* into a GoalTree using the LLM backend.

        If a template_library is injected and a template matches *task*,
        the instantiated GoalTree is returned immediately — no LLM call.

        Args:
            task: Natural language instruction (may be empty).
            world_context: Current world model summary.

        Returns:
            Validated GoalTree. Never raises — falls back to a single-step
            GoalTree on any parsing or communication failure.
        """
        # Template check — skip LLM when a reusable template matches
        if self._template_library is not None:
            try:
                match_result = self._template_library.match(task)
                if match_result is not None:
                    template, params = match_result
                    return self._template_library.instantiate(template, params)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("GoalDecomposer: template_library match/instantiate failed: %s", exc)

        if self._cached_system_prompt is None:
            self._cached_system_prompt = self._build_system_prompt()
        system = self._cached_system_prompt
        messages = self._build_messages(task, world_context)

        # Decompose call + BOUNDED retry. A reasoning model can leak its trace,
        # truncate, or wrap the JSON in fences/prose; on an extraction/parse miss
        # we re-ask ONCE with a terse "return ONLY valid JSON" nudge before falling
        # back. Each attempt is independent + idempotent (no state carried).
        attempts = 1 + max(0, self.DECOMPOSE_MAX_RETRIES)
        for attempt in range(attempts):
            attempt_messages = (
                messages if attempt == 0 else self._retry_messages(messages)
            )
            try:
                response = self._backend.call(
                    messages=attempt_messages,
                    tools=[],
                    system=system,
                    max_tokens=self._decompose_max_tokens,
                )
                raw_text = response.text
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("GoalDecomposer: backend call failed: %s", exc)
                return self._fallback_goal_tree(task)

            json_str = self._extract_json(raw_text)
            if json_str is not None:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    _LOG.warning(
                        "GoalDecomposer: JSON parse error (attempt %d/%d): %s",
                        attempt + 1,
                        attempts,
                        exc,
                    )
                else:
                    return self._build_goal_tree(task, data)
            else:
                _LOG.warning(
                    "GoalDecomposer: no JSON found in response (attempt %d/%d)",
                    attempt + 1,
                    attempts,
                )

            if attempt + 1 < attempts:
                _LOG.info("GoalDecomposer: re-asking with a JSON-only nudge")

        # Every attempt failed to yield parseable JSON — fail loud, then fall back
        # to a single-step plan (never fabricate a multi-step plan out of garbage).
        _LOG.warning(
            "GoalDecomposer: no valid JSON after %d attempt(s) — using fallback plan",
            attempts,
        )
        return self._fallback_goal_tree(task)

    def _retry_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append a terse JSON-only nudge to the user turn for a bounded re-ask.

        Additive + idempotent: builds a NEW list (never mutates the cached prompt)
        and only strengthens the existing instruction — no new schema, no relaxed
        validation. The nudge targets the exact reasoning-model failure mode:
        leaked prose / code fences / a truncated final answer.
        """
        nudge = (
            "\n\nIMPORTANT: Your previous reply could not be parsed. "
            "Respond with ONLY the raw JSON object — no reasoning, no explanation, "
            "no markdown code fences, no text before or after it."
        )
        retried = [dict(m) for m in messages]
        if retried and retried[-1].get("role") == "user":
            last = retried[-1]
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = content + nudge
            else:
                retried.append({"role": "user", "content": nudge.strip()})
        else:
            retried.append({"role": "user", "content": nudge.strip()})
        return retried

    @staticmethod
    def answer_plan(task: str, answer: str) -> GoalTree:
        """Build a 0-action / answer-only GoalTree for pure conversation (S5.2).

        Returns a single-``SubGoal`` tree whose step is explicitly ``answer_only``
        (the evidence gate's marker), routes through the side-effect-free
        ``answer`` strategy carrying the answer text, and verifies ``"True"`` (no
        deterministic predicate exists for free-form chat). Because the step is
        flagged ``answer_only``, the evidence gate treats it as a legitimate
        no-robot-evidence step rather than an unverified action — the moat (rule 5)
        is unaffected for real action steps.

        Deterministic + no LLM call: the answer text is supplied by the caller.
        Nothing routes to this yet (S5.2 is additive — no cut-over).
        """
        sg = SubGoal(
            name="answer",
            description=task,
            verify="True",
            timeout_sec=30.0,
            strategy="answer",
            strategy_params={"answer": answer},
            answer_only=True,
        )
        return GoalTree(goal=task, sub_goals=(sg,), context_snapshot="")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> list[dict[str, Any]]:
        """Build the system prompt block list for the LLM."""
        strategies_block = "\n".join(
            f"  - {name}: {desc}"
            for name, desc in sorted(self._STRATEGY_DESCRIPTIONS.items())
        )
        verify_fns_block = "\n".join(
            f"  - {sig}"
            for sig in sorted(self._VERIFY_FN_SIGNATURES.values())
        )

        text = f"""\
{self._PLANNER_INTRO}

## Output Format
Respond with ONLY valid JSON matching this schema — no prose, no markdown fences:
{self._JSON_SCHEMA}

## Rules
1. Each sub_goal MUST have a verify expression using ONLY the verify functions listed below.
2. Maximum {self.MAX_SUB_GOALS} sub_goals — prefer fewer.
3. Simple tasks should have 1-2 sub_goals.
4. depends_on must reference sub_goal names defined earlier in the same list.
5. strategy must be one of the KNOWN_STRATEGIES below, or an empty string "".
6. Do NOT call any function not in the verify list. Do NOT use import, exec, or eval.
7. verify expressions must be syntactically valid Python.
8. strategy_params MUST contain the required parameters for the chosen strategy (see STRATEGY_PARAMS below).

## STRATEGY_PARAMS (required keys per strategy)
{self._STRATEGY_PARAMS_HELP}

## KNOWN_STRATEGIES
{strategies_block}

## VERIFY_FUNCTIONS (the ONLY functions allowed in verify expressions)
{verify_fns_block}

## Example
{self._EXAMPLE}

## Loop Example
{self._FOREACH_EXAMPLE}
"""
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_messages(self, task: str, world_context: str) -> list[dict[str, Any]]:
        """Build the user messages list."""
        content = f"Task: {task}\n\nWorld context:\n{world_context}"
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Parsing + validation
    # ------------------------------------------------------------------

    def _parse_and_validate(self, task: str, raw_text: str) -> GoalTree:
        """Extract GoalTree from LLM response text. Falls back on any error."""
        json_str = self._extract_json(raw_text)
        if json_str is None:
            _LOG.warning("GoalDecomposer: no JSON found in response")
            return self._fallback_goal_tree(task)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            _LOG.warning("GoalDecomposer: JSON parse error: %s", exc)
            return self._fallback_goal_tree(task)

        return self._build_goal_tree(task, data)

    def _extract_json(self, text: str) -> str | None:
        """Extract the JSON object from a (possibly noisy) LLM response.

        Robust against a REASONING model's output: a leaked reasoning preamble,
        markdown code fences (```json ... ```), and trailing prose after the JSON
        — including prose that itself contains stray braces. Strategy:

        1. Prefer the contents of a fenced ```json``` / ``` block (the model was
           told to emit raw JSON, but reasoning models often fence it anyway).
        2. Otherwise scan for the OUTERMOST balanced ``{ ... }`` object, tracking
           string literals + escapes so braces inside strings never miscount and
           a trailing ``{curly}`` in prose can't over-capture (the old greedy
           ``\\{.*\\}`` regex bug that produced "Extra data" / "Expecting ','").

        Pure string scanning — NEVER eval/exec (rule: never execute model output).
        Returns the JSON substring, or None if no balanced object is present.
        """
        if not text:
            return None

        # 1. Fenced block — extract its body, then balance-scan it. Using the
        #    balanced scanner on the fence body (instead of a non-greedy regex)
        #    means a multi-step plan with nested objects survives intact.
        fence_match = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            inner = self._first_balanced_object(fence_match.group(1))
            if inner is not None:
                return inner

        # 2. No usable fence — balance-scan the whole text for the outermost {...}.
        return self._first_balanced_object(text)

    @staticmethod
    def _first_balanced_object(text: str) -> str | None:
        """Return the first top-level balanced ``{...}`` substring, or None.

        Skips any leading prose/reasoning before the first ``{`` and stops at the
        matching close brace (ignoring braces inside JSON string literals), so any
        trailing prose is discarded. String-aware: handles ``"`` quotes and ``\\``
        escapes. Does not parse JSON — only delimits a candidate for ``json.loads``.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        # Unbalanced (e.g. truncated by max_tokens): no complete object.
        return None

    def _build_goal_tree(self, task: str, data: dict) -> GoalTree:
        """Build and validate a GoalTree from parsed JSON data."""
        if not isinstance(data, dict):
            return self._fallback_goal_tree(task)

        goal = str(data.get("goal", task))
        raw_sub_goals = data.get("sub_goals", [])
        context_snapshot = str(data.get("context_snapshot", ""))

        if not isinstance(raw_sub_goals, list):
            return self._fallback_goal_tree(task)

        # Truncate to MAX_SUB_GOALS before validation
        raw_sub_goals = raw_sub_goals[: self.MAX_SUB_GOALS]

        # Collect valid names (pre-pass) for depends_on validation
        valid_names: set[str] = {
            str(sg.get("name", ""))
            for sg in raw_sub_goals
            if isinstance(sg, dict) and sg.get("name")
        }

        # Collect validator feedback (Stage 2b): notes about dropped/unknown
        # strategies and dropped sub_goals, surfaced on GoalTree.validation_notes
        # so the harness can warn the next replan off the same hallucination.
        notes: list[str] = []

        validated: list[SubGoal] = []
        for raw in raw_sub_goals:
            sg = self._validate_sub_goal(raw, valid_names, notes)
            if sg is not None:
                validated.append(sg)

        if not validated:
            return self._fallback_goal_tree(task)

        return GoalTree(
            goal=goal,
            sub_goals=tuple(validated),
            context_snapshot=context_snapshot,
            validation_notes=tuple(notes),
        )

    def _validate_sub_goal(
        self,
        raw: Any,
        valid_names: set[str],
        notes: "list[str] | None" = None,
    ) -> SubGoal | None:
        """Validate and normalise a raw sub_goal dict.

        Returns a SubGoal on success, or None to discard. When *notes* is given,
        appends human-readable validator feedback for any dropped sub_goal or
        cleared (unknown) strategy (Stage 2b).
        """
        if not isinstance(raw, dict):
            return None

        name = str(raw.get("name", "")).strip()
        if not name:
            return None

        description = str(raw.get("description", ""))
        verify = str(raw.get("verify", ""))
        timeout_sec = float(raw.get("timeout_sec", 30.0))
        strategy = str(raw.get("strategy", ""))
        strategy_params = raw.get("strategy_params", {})
        if not isinstance(strategy_params, dict):
            strategy_params = {}
        fail_action = str(raw.get("fail_action", ""))

        # Named-target strengthening (backlog #2b): a step whose params bind a
        # named target must verify against THAT target, not "holding anything".
        # Runs BEFORE _validate_verify so the rewritten expression still passes
        # the AST allowlist like any LLM-authored one. Stricter-only.
        from vector_os_nano.vcli.cognitive.verify_strengthen import (
            strengthen_target_verify,
        )
        verify = strengthen_target_verify(verify, strategy_params)

        # Validate verify expression
        verify = self._validate_verify(verify)
        if verify is None:
            # Discard sub_goal whose verify is non-parseable / malicious
            _LOG.warning("GoalDecomposer: dropping sub_goal %r — invalid verify", name)
            if notes is not None:
                notes.append(
                    f"dropped sub_goal {name!r}: its verify expression was invalid"
                )
            return None

        # Stage 5 (S5.2): an answer-only step is a pure-conversation leaf that
        # carries no robot evidence by design. The decomposer marks it explicitly
        # (``answer_only: true``) so the evidence gate can DISTINGUISH it from an
        # action step that merely produced no evidence (rule 5 — the gate keys on
        # this flag, never on the verify string). The dedicated ``answer`` strategy
        # is the kernel-level dispatch route for such a step (mirrors how
        # ``tool_call`` is recognized regardless of per-world vocab), so it is
        # exempt from the unknown-strategy clearing below.
        #
        # MOAT (rule 5): ``answer_only`` is fully LLM-controlled, so it must NEVER
        # waive the evidence gate for a step that runs a side-effecting executor.
        # Bind the flag to the side-effect-free ``answer`` strategy (the only route
        # that performs zero I/O — ``GoalExecutor._execute_answer``). An LLM that
        # sets ``answer_only: true`` on an action strategy (``tool_call`` / a
        # skill) has the flag REFUSED here (fail-loud note), so the step stays a
        # real action that the gate still requires a deterministic predicate for.
        # This is belt-and-suspenders with the gate, which also keys on
        # ``strategy == 'answer'`` (trace_store.evidence_passed / replay).
        answer_only = bool(raw.get("answer_only", False)) or strategy == "answer"
        if answer_only and strategy != "answer":
            _LOG.warning(
                "GoalDecomposer: refusing answer_only on non-answer strategy %r "
                "in sub_goal %r — answer_only is reserved for the side-effect-free "
                "'answer' route",
                strategy,
                name,
            )
            if notes is not None:
                notes.append(
                    f"answer_only refused on sub_goal {name!r}: it is reserved for "
                    f"the side-effect-free 'answer' strategy, not {strategy!r}"
                )
            answer_only = False

        # Validate strategy. ``answer`` is a kernel dispatch route (like
        # ``tool_call``), valid in every world — never cleared.
        #
        # Fail-loud (rule 8): an unknown explicit strategy is a HALLUCINATION. We
        # clear ``strategy`` to "" (so the validated tree never carries a phantom
        # strategy name and existing pure-check / foreach routing is unaffected)
        # AND stamp the offending name on ``cleared_strategy``. The latter routes
        # this step to the selector's LOUD ``invalid`` path at execution, so a
        # cleared hallucination surfaces a clear, named error with the valid set
        # instead of silently re-routing through keyword/registry matching to a
        # phantom skill (base world) or the opaque ``unmatched`` fallback (baseless
        # world). This applies on EVERY decompose, including replan, because the
        # harness re-decomposes through this same validator.
        cleared_strategy = ""
        if strategy and strategy != "answer" and strategy not in self.KNOWN_STRATEGIES:
            _LOG.warning(
                "GoalDecomposer: unknown strategy %r in sub_goal %r — clearing "
                "(will fail loud at execution)",
                strategy,
                name,
            )
            if notes is not None:
                valid = ", ".join(sorted(self.KNOWN_STRATEGIES)) or "(none)"
                notes.append(
                    f"strategy {strategy!r} is not valid; valid strategies: {valid}"
                )
            cleared_strategy = strategy
            strategy = ""

        # Validate depends_on
        raw_deps = raw.get("depends_on", [])
        if not isinstance(raw_deps, list):
            raw_deps = []
        depends_on = tuple(
            dep for dep in raw_deps
            if isinstance(dep, str) and dep in valid_names and dep != name
        )

        # Stage 4 (S4-1): optional FOREACH control-flow spec. Parsed + validated
        # here; not yet expanded at execution time (S4-2). Body strategies are
        # validated against the SAME world vocab as top-level steps; an unknown
        # body strategy is cleared with the same fail-loud note.
        foreach = self._validate_foreach(raw.get("foreach"), name, valid_names, notes)

        # An answer-only step is a conversation leaf — it never loops.
        if answer_only:
            foreach = None

        # A foreach loop OWNER legitimately carries an empty strategy (the body
        # does the work). If the LLM additionally named a hallucinated strategy on
        # the loop owner, the cleared name must NOT mark the owner ``invalid`` —
        # that would fail the (otherwise valid) loop. The body templates are
        # validated independently with the same fail-loud rules, so the
        # hallucination is still caught where it actually executes.
        if foreach is not None:
            cleared_strategy = ""

        # Stage 4 (H-1b): a foreach node iterates a list produced by its
        # source_step, so it MUST be ordered AFTER that producer. If the author
        # omitted the ordering edge, the topological sort could place the loop
        # before its producer and it would iterate zero times while still
        # reporting success. Auto-inject source_step into depends_on (idempotent —
        # a plan that already lists it is unchanged).
        if foreach is not None and foreach.source_step not in depends_on:
            depends_on = depends_on + (foreach.source_step,)

        return SubGoal(
            name=name,
            description=description,
            verify=verify,
            timeout_sec=timeout_sec,
            depends_on=depends_on,
            strategy=strategy,
            strategy_params=strategy_params,
            fail_action=fail_action,
            foreach=foreach,
            answer_only=answer_only,
            cleared_strategy=cleared_strategy,
        )

    def _validate_foreach(
        self,
        raw: Any,
        owner_name: str,
        valid_names: set[str],
        notes: "list[str] | None" = None,
    ) -> ForEachSpec | None:
        """Validate a raw ``foreach`` block into a ForEachSpec, or None.

        Shape (all parsed by pure dict access — never evaluated)::

            "foreach": {
              "source_step": "<name of an earlier step producing the list>",
              "source_path": "<dotted path INTO that step's result_data list>",
              "var": "<iteration variable name, default 'item'>",
              "body": [ <sub_goal template>, ... ]
            }

        Returns None (a plain leaf step) when *raw* is absent or malformed —
        fail-safe, never raising into decomposition. ``source_step`` must name an
        earlier sub_goal in the tree; an unknown reference drops the foreach with a
        fail-loud note. Body templates are validated with the SAME world-vocab
        rules as top-level steps (unknown body strategies cleared + noted). The
        per-item binding ``${var.field}`` is left as data on the body templates —
        S4-1 does NOT expand it; the Blackboard's pure path traversal resolves it
        at execution time (S4-2).
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            if notes is not None:
                notes.append(
                    f"sub_goal {owner_name!r}: foreach must be an object — ignored"
                )
            return None

        source_step = str(raw.get("source_step", "")).strip()
        source_path = str(raw.get("source_path", "")).strip()
        if not source_step or not source_path:
            if notes is not None:
                notes.append(
                    f"sub_goal {owner_name!r}: foreach requires 'source_step' and "
                    "'source_path' — ignored"
                )
            return None

        # The producing step must be a real, earlier sub_goal in this tree.
        if source_step not in valid_names or source_step == owner_name:
            if notes is not None:
                valid = ", ".join(sorted(valid_names)) or "(none)"
                notes.append(
                    f"sub_goal {owner_name!r}: foreach.source_step {source_step!r} "
                    f"is not a known step; known steps: {valid}"
                )
            return None

        var = str(raw.get("var", "item")).strip() or "item"

        raw_body = raw.get("body", [])
        if not isinstance(raw_body, list):
            raw_body = []
        # Body templates may depend on each other within the body; build the set
        # of body names so intra-body depends_on validates. They are validated
        # against the same strategy/verify vocab as top-level steps.
        body_names: set[str] = {
            str(t.get("name", ""))
            for t in raw_body
            if isinstance(t, dict) and t.get("name")
        }
        body: list[SubGoal] = []
        for raw_t in raw_body:
            template = self._validate_sub_goal(raw_t, body_names, notes)
            if template is not None:
                body.append(template)

        return ForEachSpec(
            source_step=source_step,
            source_path=source_path,
            var=var,
            body=tuple(body),
        )

    def _validate_verify(self, verify: str) -> str | None:
        """Check verify expression safety. Returns cleaned expression or None.

        Rules:
        - Must be syntactically valid Python (ast.parse in eval mode)
        - May not contain dunder names
        - May only call functions from VERIFY_FUNCTIONS
        - No Import/ImportFrom/Assign/FunctionDef/ClassDef nodes
        """
        if not verify or not verify.strip():
            # Empty verify is acceptable (truthy fallback in executor)
            return verify

        # Dunder check
        if "__" in verify:
            _LOG.warning("GoalDecomposer: dunder in verify expression — rejecting")
            return None

        # AST parse check. Suppress SyntaxWarning from sloppy LLM escape sequences
        # (e.g. '\.') — they parse fine but would spam '<unknown>' to the console.
        import warnings  # noqa: PLC0415
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(verify, mode="eval")
        except SyntaxError:
            _LOG.warning("GoalDecomposer: SyntaxError in verify: %r", verify)
            return None

        # Blocked node types
        _BLOCKED = (
            ast.Import,
            ast.ImportFrom,
            ast.Assign,
            ast.AugAssign,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        )
        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED):
                _LOG.warning(
                    "GoalDecomposer: blocked AST node %s in verify: %r",
                    type(node).__name__,
                    verify,
                )
                return None

        # Function whitelist check — VERIFY_FUNCTIONS union safe builtins
        collector = _CallNameCollector()
        collector.visit(tree)
        allowed = self.VERIFY_FUNCTIONS | self._ALLOWED_BUILTINS
        disallowed = collector.names - allowed
        if disallowed:
            _LOG.warning(
                "GoalDecomposer: disallowed function(s) %s in verify: %r",
                disallowed,
                verify,
            )
            return None

        return verify

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_goal_tree(self, task: str) -> GoalTree:
        """Return a minimal single-step GoalTree when decomposition fails."""
        fallback_sg = SubGoal(
            name="execute_task",
            description=task,
            verify=self._FALLBACK_VERIFY,
            timeout_sec=60.0,
        )
        return GoalTree(
            goal=task,
            sub_goals=(fallback_sg,),
            context_snapshot="",
        )
