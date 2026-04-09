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
from dataclasses import field
from typing import Any

from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

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

    # Available strategies the LLM can reference
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
        "door_chain_fallback",
        "scan_room_360_then_report",
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
        "door_chain_fallback": "Follow door chain to reach target room",
        "scan_room_360_then_report": "Rotate full 360° then summarise scene",
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
      "timeout_sec": <float, default 30.0>,
      "depends_on": ["<name of preceding sub_goal>"],
      "strategy": "<one of KNOWN_STRATEGIES or empty string>",
      "strategy_params": {},
      "fail_action": "<optional: what to do on failure>"
    }
  ],
  "context_snapshot": "<optional: brief summary of world context used>"
}"""

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
      "strategy_params": {},
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
      "strategy": "",
      "timeout_sec": 10,
      "depends_on": ["observe_table"],
      "strategy_params": {},
      "fail_action": ""
    }
  ],
  "context_snapshot": "Robot is in hallway, kitchen is adjacent."
}"""

    def __init__(self, backend: Any, template_library: Any = None) -> None:
        """Initialise with an LLMBackend (must implement .call()).

        Args:
            backend: Any object implementing the LLMBackend Protocol
                     (has .call(messages, tools, system, max_tokens)).
            template_library: Optional TemplateLibrary — checked before calling
                              the LLM.  When a template matches the task, the
                              instantiated GoalTree is returned immediately and
                              the LLM backend is not called.
        """
        self._backend = backend
        self._template_library = template_library

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

        system = self._build_system_prompt()
        messages = self._build_messages(task, world_context)

        try:
            response = self._backend.call(
                messages=messages,
                tools=[],
                system=system,
                max_tokens=2048,
            )
            raw_text = response.text
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("GoalDecomposer: backend call failed: %s", exc)
            return self._fallback_goal_tree(task)

        return self._parse_and_validate(task, raw_text)

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
You are a robot task planner. Decompose the user's task into verifiable sub-goals.

## Output Format
Respond with ONLY valid JSON matching this schema — no prose, no markdown fences:
{self._JSON_SCHEMA}

## Rules
1. Each sub_goal MUST have a verify expression using ONLY the verify functions listed below.
2. Maximum {self.MAX_SUB_GOALS} sub_goals — prefer fewer.
3. Simple tasks (stand, sit, go to X) should have 1-2 sub_goals.
4. depends_on must reference sub_goal names defined earlier in the same list.
5. strategy must be one of the KNOWN_STRATEGIES below, or an empty string "".
6. Do NOT call any function not in the verify list. Do NOT use import, exec, or eval.
7. verify expressions must be syntactically valid Python.

## KNOWN_STRATEGIES
{strategies_block}

## VERIFY_FUNCTIONS (the ONLY functions allowed in verify expressions)
{verify_fns_block}

## Example
{self._EXAMPLE}
"""
        return [{"type": "text", "text": text}]

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
        """Extract JSON from text, stripping markdown fences if present."""
        # Try ```json ... ``` first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        # Try to find a bare { ... } block
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1)

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

        validated: list[SubGoal] = []
        for raw in raw_sub_goals:
            sg = self._validate_sub_goal(raw, valid_names)
            if sg is not None:
                validated.append(sg)

        if not validated:
            return self._fallback_goal_tree(task)

        return GoalTree(
            goal=goal,
            sub_goals=tuple(validated),
            context_snapshot=context_snapshot,
        )

    def _validate_sub_goal(
        self,
        raw: Any,
        valid_names: set[str],
    ) -> SubGoal | None:
        """Validate and normalise a raw sub_goal dict.

        Returns a SubGoal on success, or None to discard.
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

        # Validate verify expression
        verify = self._validate_verify(verify)
        if verify is None:
            # Discard sub_goal whose verify is non-parseable / malicious
            _LOG.warning("GoalDecomposer: dropping sub_goal %r — invalid verify", name)
            return None

        # Validate strategy
        if strategy and strategy not in self.KNOWN_STRATEGIES:
            _LOG.warning(
                "GoalDecomposer: unknown strategy %r in sub_goal %r — clearing",
                strategy,
                name,
            )
            strategy = ""

        # Validate depends_on
        raw_deps = raw.get("depends_on", [])
        if not isinstance(raw_deps, list):
            raw_deps = []
        depends_on = tuple(
            dep for dep in raw_deps
            if isinstance(dep, str) and dep in valid_names and dep != name
        )

        return SubGoal(
            name=name,
            description=description,
            verify=verify,
            timeout_sec=timeout_sec,
            depends_on=depends_on,
            strategy=strategy,
            strategy_params=strategy_params,
            fail_action=fail_action,
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

        # AST parse check
        try:
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
            verify="world_stats() is not None",
            timeout_sec=60.0,
        )
        return GoalTree(
            goal=task,
            sub_goals=(fallback_sg,),
            context_snapshot="",
        )
