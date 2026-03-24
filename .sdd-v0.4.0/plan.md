# Plan: v0.4.0 — Built-in Agent Loop

**Status**: approved
**Author**: Alpha (Sonnet)
**Date**: 2026-03-24
**Spec**: `.sdd-v0.4.0/spec.md`

---

## 1. Scope Summary

One new file (~200 lines) + six small modifications. No existing public APIs changed. No new dependencies.

| Change | File | Size |
|--------|------|------|
| New | `core/agent_loop.py` | ~200 lines |
| Modify | `core/types.py` | +40 lines (2 new dataclasses) |
| Modify | `llm/prompts.py` | +50 lines (new prompt + builder) |
| Modify | `llm/claude.py` | +60 lines (2 new methods) |
| Modify | `core/agent.py` | +10 lines (run_goal() method) |
| Modify | `mcp/tools.py` | +20 lines (tool def + handler branch) |
| Modify | `config/default.yaml` | +6 lines (agent_loop section) |

---

## 2. Dependency Graph

```
T1 (types + skeleton)  ──┐
                          ├──> T3 (AgentLoop impl) ──┐
T2 (LLM decide)       ──┘                             ├──> T5 (integration test)
                                                       │
T4 (Agent + MCP)      ────────────────────────────────┘
```

Wave 1 (parallel): T1, T2
Wave 2 (parallel): T3, T4
Wave 3: T5 (depends on T3 + T4)

---

## 3. File-by-File Changes

### 3.1 `core/types.py` — Add GoalResult + ActionRecord

**Insert after line 385** (end of ExecutionResult.from_dict), before the final blank line.

```python
# ---------------------------------------------------------------------------
# Agent loop types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """Execution record for one iteration of the agent loop."""

    iteration: int
    action: str
    params: dict
    skill_success: bool
    verified: bool
    reasoning: str
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action": self.action,
            "params": self.params,
            "skill_success": self.skill_success,
            "verified": self.verified,
            "reasoning": self.reasoning,
            "duration_sec": self.duration_sec,
        }


@dataclass(frozen=True)
class GoalResult:
    """Final result of an agent loop run_goal() call."""

    success: bool
    goal: str
    iterations: int
    total_duration_sec: float
    actions: list[ActionRecord]
    summary: str
    final_world_state: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "iterations": self.iterations,
            "total_duration_sec": self.total_duration_sec,
            "actions": [a.to_dict() for a in self.actions],
            "summary": self.summary,
            "final_world_state": self.final_world_state,
        }
```

---

### 3.2 `llm/prompts.py` — Add AGENT_LOOP_SYSTEM_PROMPT + build_agent_loop_prompt()

**Insert before line 214** (before `build_tool_definitions()`).

```python
# ---------------------------------------------------------------------------
# Agent loop prompt
# ---------------------------------------------------------------------------

AGENT_LOOP_SYSTEM_PROMPT = """\
You are a robot action planner executing an iterative goal.

GOAL: {goal}

AVAILABLE SKILLS:
{skills_json}

CURRENT OBSERVATION:
{observation_json}

EXECUTION HISTORY:
{history_json}

RULES:
1. Return EXACTLY ONE JSON object. No markdown fences. No explanation outside JSON.
2. If the goal is achieved, return: {{"done": true, "summary": "..."}}
3. If more work is needed, return: {{"action": "skill_name", "params": {{...}}, "reasoning": "..."}}
4. ONLY use skill names from AVAILABLE SKILLS. Parameters must match the schema.
5. If a previous action failed, try a different approach — do NOT repeat the same action.
6. If you need to see the workspace, use "scan" then "detect" as your action.
7. Maximum {max_iterations} iterations — be efficient.
"""


def build_agent_loop_prompt(
    goal: str,
    observation: dict[str, Any],
    skill_schemas: list[dict[str, Any]],
    history: list[dict[str, Any]],
    max_iterations: int,
) -> str:
    """Build the system prompt for one agent loop iteration."""
    return AGENT_LOOP_SYSTEM_PROMPT.format(
        goal=goal,
        skills_json=json.dumps(skill_schemas, indent=2, ensure_ascii=False),
        observation_json=json.dumps(observation, indent=2, ensure_ascii=False),
        history_json=json.dumps(history, indent=2, ensure_ascii=False),
        max_iterations=max_iterations,
    )
```

**Update the `__all__`-equivalent imports** in `llm/claude.py` line 19:
```python
# Before:
from vector_os_nano.llm.prompts import build_planning_prompt
# After:
from vector_os_nano.llm.prompts import build_planning_prompt, build_agent_loop_prompt
```

---

### 3.3 `llm/claude.py` — Add decide_next_action() + parse_action_response()

**Insert `parse_action_response()` after `parse_plan_response()` (after line 107):**

```python
def parse_action_response(raw_text: str) -> dict:
    """Parse a single-action or done response from the agent loop LLM.

    Returns:
        {"action": str, "params": dict, "reasoning": str}  — next action
        {"done": True, "summary": str}                      — goal complete
        {"action": "scan", "params": {}, "reasoning": "parse fallback"} — on failure
    """
    if not raw_text or not raw_text.strip():
        return {"action": "scan", "params": {}, "reasoning": "empty response fallback"}

    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if data.get("done"):
                return {"done": True, "summary": data.get("summary", "Goal complete.")}
            if "action" in data:
                return {
                    "action": str(data["action"]),
                    "params": dict(data.get("params", {})),
                    "reasoning": str(data.get("reasoning", "")),
                }
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract action name even from broken JSON
    match = re.search(r'"action"\s*:\s*"(\w+)"', raw_text)
    if match:
        log.warning("parse_action_response: JSON parse failed, extracted action via regex")
        return {"action": match.group(1), "params": {}, "reasoning": "regex fallback"}

    log.warning("parse_action_response: all parse strategies failed, defaulting to scan")
    return {"action": "scan", "params": {}, "reasoning": "parse failure fallback"}
```

**Insert `decide_next_action()` on `ClaudeProvider` after `summarize()` (after line 244):**

```python
def decide_next_action(
    self,
    goal: str,
    observation: dict,
    skill_schemas: list[dict],
    history: list[dict],
    model_override: str | None = None,
) -> dict:
    """Call the LLM for one agent loop iteration.

    Returns one of:
        {"action": str, "params": dict, "reasoning": str}
        {"done": True, "summary": str}
        {"action": "scan", "params": {}, "reasoning": "<fallback reason>"}  # on error
    """
    system_prompt = build_agent_loop_prompt(
        goal=goal,
        observation=observation,
        skill_schemas=skill_schemas,
        history=history,
        max_iterations=observation.get("max_iterations", 10),
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Decide the next action."}
    ]
    raw = self._chat_completion(system_prompt, messages, model_override)
    return parse_action_response(raw)
```

---

### 3.4 `core/agent_loop.py` — New file (~200 lines)

**Full pseudocode / structure:**

```python
"""AgentLoop — observe-decide-act-verify engine for Agent.run_goal()."""

_VERIFY_AFTER: frozenset[str] = frozenset({"pick", "place"})

class AgentLoop:
    def __init__(self, agent_ref, config):
        self._agent = agent_ref
        self._config = config  # agent.agent_loop sub-dict
        self._is_sim = hasattr(agent_ref._arm, "get_object_positions")

    def run(self, goal, max_iterations, verify, on_step, on_message) -> GoalResult:
        history = []
        actions = []
        t_start = time.monotonic()

        for i in range(max_iterations):
            obs = self._observe(goal, i, history, max_iterations)
            decision = self._decide(goal, obs, history)

            if decision.get("done"):
                return GoalResult(success=True, ..., summary=decision["summary"])

            record = self._act_and_record(i, decision, verify)
            actions.append(record)
            history = self._trim_history(history + [record.to_dict()])

            if on_step:
                on_step(record.action, i, max_iterations)

        # max_iterations reached
        return GoalResult(success=False, ..., summary="Max iterations reached")

    def _observe(self, goal, iteration, history, max_iterations) -> dict:
        if self._is_sim:
            self._agent._refresh_objects()
        return {
            "world_state": self._agent._world_model.to_dict(),
            "iteration": iteration,
            "goal": goal,
            "max_iterations": max_iterations,
        }

    def _decide(self, goal, observation, history) -> dict:
        loop_cfg = self._config.get("agent", {}).get("agent_loop", {})
        model = loop_cfg.get("model") or None
        skill_schemas = self._agent._skill_registry.to_schemas()
        return self._agent._llm.decide_next_action(
            goal=goal,
            observation=observation,
            skill_schemas=skill_schemas,
            history=history,
            model_override=model,
        )

    def _act_and_record(self, iteration, decision, verify) -> ActionRecord:
        action = decision["action"]
        params = decision.get("params", {})
        reasoning = decision.get("reasoning", "")
        t0 = time.monotonic()

        skill = self._agent._skill_registry.get(action)
        if skill is None:
            return ActionRecord(iteration=iteration, action=action, params=params,
                                skill_success=False, verified=False,
                                reasoning=reasoning, duration_sec=0.0)

        context = self._agent._build_context()
        result = skill.execute(params, context)
        self._agent._world_model.apply_skill_effects(action, params, result)

        did_verify = False
        if verify and self._should_verify(action):
            self._verify()
            did_verify = True

        return ActionRecord(
            iteration=iteration, action=action, params=params,
            skill_success=result.success, verified=did_verify,
            reasoning=reasoning, duration_sec=time.monotonic() - t0,
        )

    def _verify(self) -> None:
        if self._is_sim:
            self._agent._refresh_objects()
            return
        # Hardware: run scan + detect to refresh world model
        context = self._agent._build_context()
        for skill_name in ("scan", "detect"):
            skill = self._agent._skill_registry.get(skill_name)
            if skill:
                params = {} if skill_name == "scan" else {"query": "all objects"}
                skill.execute(params, context)

    @staticmethod
    def _should_verify(action: str) -> bool:
        return action in _VERIFY_AFTER

    def _trim_history(self, history: list[dict]) -> list[dict]:
        loop_cfg = self._config.get("agent", {}).get("agent_loop", {})
        max_h = loop_cfg.get("history_max_actions", 6)
        return history[-max_h:]
```

---

### 3.5 `core/agent.py` — Add run_goal() method

**Insert after the `execute()` method (after line 275), before `_execute_matched()` at line 277.**

Current line 277:
```python
    def _execute_matched(self, match: SkillMatch, instruction: str) -> ExecutionResult:
```

Insert before it:
```python
    def run_goal(
        self,
        goal: str,
        max_iterations: int = 10,
        verify: bool = True,
        on_step: Any = None,
        on_message: Any = None,
    ) -> "GoalResult":
        """Execute an iterative observe-decide-act-verify goal loop.

        Delegates to AgentLoop. Agent.execute() is unchanged.
        """
        from vector_os_nano.core.agent_loop import AgentLoop
        from vector_os_nano.core.types import GoalResult

        loop = AgentLoop(agent_ref=self, config=self._config)
        return loop.run(
            goal=goal,
            max_iterations=max_iterations,
            verify=verify,
            on_step=on_step,
            on_message=on_message,
        )
```

Also update the import at line 20:
```python
# Before:
from vector_os_nano.core.types import ExecutionResult
# After:
from vector_os_nano.core.types import ExecutionResult, GoalResult
```

---

### 3.6 `mcp/tools.py` — Add run_goal tool

**In `skills_to_mcp_tools()` (lines 12-29), add `build_run_goal_tool()` call:**

```python
# After line 27 (tools.append(build_diagnostics_tool())):
    tools.append(build_run_goal_tool())
```

**New function, insert after `build_natural_language_tool()` (after line 80):**

```python
def build_run_goal_tool() -> dict:
    """Build the run_goal MCP tool for iterative goal execution."""
    return {
        "name": "run_goal",
        "description": (
            "Execute an iterative goal using the observe-decide-act-verify loop. "
            "Use for multi-step goals like 'clean the table', 'sort objects by color'. "
            "Returns a full trace of actions taken."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Natural language goal to achieve iteratively",
                },
                "max_iterations": {
                    "type": "integer",
                    "default": 10,
                    "description": "Safety cap on loop iterations",
                },
                "verify": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run perception verification after each pick/place",
                },
            },
            "required": ["goal"],
        },
    }
```

**In `handle_tool_call()` (lines 158-192), add handler before the direct-skill fallback:**

```python
# After the natural_language handler block (after line 186), before the final fallback:
    if tool_name == "run_goal":
        goal = arguments.get("goal", "")
        max_iterations = int(arguments.get("max_iterations", 10))
        verify = bool(arguments.get("verify", True))
        result = await asyncio.to_thread(
            agent.run_goal, goal, max_iterations, verify
        )
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
```

Note: add `import json` — already present in `_format_execution_result`, but `handle_tool_call` needs it in scope. Verify the import is at module level or inside the branch.

---

### 3.7 `config/default.yaml` — Add agent_loop section

**After line 9** (`planning_timeout_sec: 10.0`):

```yaml
  agent_loop:
    max_iterations: 10
    verify: true              # post-action perception verification
    verify_in_sim: false      # skip perception verify in sim (use ground truth)
    history_max_actions: 6    # max actions in LLM context
    model: null               # null = use plan_complex model
```

---

## 4. Execution Waves

### Wave 1 (parallel — no dependencies)

| Task | Agent | Files touched |
|------|-------|---------------|
| T1: types + skeleton | Alpha | `core/types.py`, `core/agent_loop.py` |
| T2: LLM decide | Beta | `llm/prompts.py`, `llm/claude.py` |

### Wave 2 (parallel — T1+T2 must be done)

| Task | Agent | Files touched |
|------|-------|---------------|
| T3: AgentLoop impl | Alpha | `core/agent_loop.py` (fill stubs from T1) |
| T4: Agent + MCP | Beta | `core/agent.py`, `mcp/tools.py`, `config/default.yaml` |

### Wave 3 (sequential — T3+T4 must be done)

| Task | Agent | Files touched |
|------|-------|---------------|
| T5: Integration test | Gamma | `tests/test_agent_loop_integration.py` |

---

## 5. Key Invariants

- `Agent.execute()` is **not modified**. Existing tests must pass unchanged.
- `_handle_task()` is **not modified**.
- `LLMProvider` protocol is **not modified**. `decide_next_action()` is a concrete method on `ClaudeProvider` only.
- `WorldModel`, `SessionMemory`, `TaskExecutor` are **not modified**.
- `AgentLoop` accesses Agent private attributes (`_arm`, `_llm`, `_world_model`, `_skill_registry`, `_config`) directly — this is acceptable since `AgentLoop` is in the same package and is only instantiated by `Agent.run_goal()`.
