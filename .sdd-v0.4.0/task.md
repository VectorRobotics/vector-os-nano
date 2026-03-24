# Tasks: v0.4.0 — Built-in Agent Loop

**Status**: approved
**Author**: Alpha (Sonnet)
**Date**: 2026-03-24
**Plan**: `.sdd-v0.4.0/plan.md`

---

## Execution Waves

```
Wave 1 (parallel): T1 [alpha], T2 [beta]
Wave 2 (parallel): T3 [alpha], T4 [beta]   — after Wave 1 complete
Wave 3:            T5 [gamma]               — after Wave 2 complete
```

---

## T1 — Core Types + AgentLoop Skeleton

**Agent**: [alpha]
**Estimate**: 30 min
**Dependencies**: none
**Branch**: `feat/alpha-agent-loop-types`

### Files
- `vector_os_nano/core/types.py` — add `ActionRecord`, `GoalResult`
- `vector_os_nano/core/agent_loop.py` — create with `AgentLoop` skeleton

### TDD

**RED**: Write `tests/test_agent_loop_types.py` first.

```python
# tests/test_agent_loop_types.py
from vector_os_nano.core.types import ActionRecord, GoalResult

def test_action_record_frozen():
    r = ActionRecord(iteration=0, action="pick", params={"object_label": "banana"},
                     skill_success=True, verified=True, reasoning="closest", duration_sec=1.2)
    with pytest.raises(Exception):
        r.action = "place"  # frozen

def test_action_record_to_dict():
    r = ActionRecord(iteration=0, action="scan", params={},
                     skill_success=True, verified=False, reasoning="", duration_sec=0.5)
    d = r.to_dict()
    assert d["action"] == "scan"
    assert d["skill_success"] is True

def test_goal_result_to_dict():
    gr = GoalResult(success=True, goal="clean table", iterations=3,
                    total_duration_sec=12.5, actions=[], summary="Done.", final_world_state={})
    d = gr.to_dict()
    assert d["success"] is True
    assert d["iterations"] == 3
```

Also write `tests/test_agent_loop_skeleton.py`:

```python
# tests/test_agent_loop_skeleton.py
from unittest.mock import MagicMock
from vector_os_nano.core.agent_loop import AgentLoop
from vector_os_nano.core.types import GoalResult

def test_agent_loop_instantiates():
    mock_agent = MagicMock()
    mock_agent._arm = None
    loop = AgentLoop(agent_ref=mock_agent, config={})
    assert loop is not None

def test_run_returns_goal_result_on_done():
    """LLM immediately returns done -> GoalResult(success=True, iterations=0+1)."""
    mock_agent = MagicMock()
    mock_agent._arm = None
    mock_agent._world_model.to_dict.return_value = {"objects": []}
    mock_agent._skill_registry.to_schemas.return_value = []
    mock_agent._llm.decide_next_action.return_value = {"done": True, "summary": "Already done."}

    loop = AgentLoop(agent_ref=mock_agent, config={})
    result = loop.run(goal="test", max_iterations=5, verify=False,
                      on_step=None, on_message=None)
    assert isinstance(result, GoalResult)
    assert result.success is True
    assert result.summary == "Already done."
```

**GREEN**: Implement `ActionRecord`, `GoalResult` in `core/types.py`. Create `core/agent_loop.py` with `AgentLoop.__init__()`, `run()` skeleton (calls `_observe`, `_decide`, returns `GoalResult`), `_observe()` stub, `_decide()` stub, `_act_and_record()` stub, `_verify()` stub, `_should_verify()`, `_trim_history()`.

**REFACTOR**: Confirm `GoalResult` and `ActionRecord` follow frozen dataclass pattern matching existing types in `core/types.py`.

### Acceptance Criteria
- [ ] `ActionRecord` and `GoalResult` are frozen dataclasses with `to_dict()`
- [ ] `AgentLoop(agent_ref=mock, config={})` constructs without error
- [ ] `run()` with mock LLM returning `{"done": True}` returns `GoalResult(success=True)`
- [ ] `run()` with `max_iterations=2` and LLM never returning done returns `GoalResult(success=False)`
- [ ] All tests pass: `pytest tests/test_agent_loop_types.py tests/test_agent_loop_skeleton.py`

---

## T2 — LLM decide_next_action

**Agent**: [beta]
**Estimate**: 30 min
**Dependencies**: none
**Branch**: `feat/beta-llm-decide`

### Files
- `vector_os_nano/llm/prompts.py` — add `AGENT_LOOP_SYSTEM_PROMPT`, `build_agent_loop_prompt()`
- `vector_os_nano/llm/claude.py` — add `parse_action_response()`, `decide_next_action()`

### TDD

**RED**: Write `tests/test_parse_action_response.py` first.

```python
# tests/test_parse_action_response.py
from vector_os_nano.llm.claude import parse_action_response

def test_parses_action():
    raw = '{"action": "pick", "params": {"object_label": "banana"}, "reasoning": "closest"}'
    result = parse_action_response(raw)
    assert result["action"] == "pick"
    assert result["params"]["object_label"] == "banana"
    assert result["reasoning"] == "closest"

def test_parses_done():
    raw = '{"done": true, "summary": "All 3 objects removed."}'
    result = parse_action_response(raw)
    assert result.get("done") is True
    assert "All 3 objects" in result["summary"]

def test_handles_markdown_fence():
    raw = '```json\n{"action": "scan", "params": {}}\n```'
    result = parse_action_response(raw)
    assert result["action"] == "scan"

def test_handles_broken_json_regex_fallback():
    raw = 'Here is my answer: {"action": "detect", broken...'
    result = parse_action_response(raw)
    assert result["action"] == "detect"

def test_handles_empty_response():
    result = parse_action_response("")
    assert result["action"] == "scan"  # safe fallback

def test_handles_unparseable():
    result = parse_action_response("I cannot decide what to do.")
    assert result["action"] == "scan"  # safe fallback
```

Also write `tests/test_decide_next_action.py`:

```python
# tests/test_decide_next_action.py
from unittest.mock import patch, MagicMock
from vector_os_nano.llm.claude import ClaudeProvider

def test_decide_next_action_returns_action():
    provider = ClaudeProvider(api_key="test-key")
    with patch.object(provider, '_chat_completion',
                      return_value='{"action": "pick", "params": {"object_label": "mug"}, "reasoning": "only object"}'):
        result = provider.decide_next_action(
            goal="pick all objects",
            observation={"world_state": {}, "iteration": 0},
            skill_schemas=[],
            history=[],
        )
    assert result["action"] == "pick"

def test_decide_next_action_returns_done():
    provider = ClaudeProvider(api_key="test-key")
    with patch.object(provider, '_chat_completion',
                      return_value='{"done": true, "summary": "Table is clear."}'):
        result = provider.decide_next_action(
            goal="clean table", observation={}, skill_schemas=[], history=[]
        )
    assert result.get("done") is True
```

Also write `tests/test_agent_loop_prompt.py`:

```python
# tests/test_agent_loop_prompt.py
from vector_os_nano.llm.prompts import build_agent_loop_prompt

def test_prompt_contains_goal():
    prompt = build_agent_loop_prompt(
        goal="clean the table",
        observation={"world_state": {}},
        skill_schemas=[],
        history=[],
        max_iterations=10,
    )
    assert "clean the table" in prompt
    assert "10" in prompt

def test_prompt_contains_skills():
    prompt = build_agent_loop_prompt(
        goal="test",
        observation={},
        skill_schemas=[{"name": "pick", "description": "picks objects"}],
        history=[],
        max_iterations=5,
    )
    assert "pick" in prompt
```

**GREEN**: Implement `parse_action_response()` at module level in `claude.py` (after `parse_plan_response()`). Implement `build_agent_loop_prompt()` in `prompts.py`. Add `decide_next_action()` method on `ClaudeProvider`.

Update import in `claude.py` line 19:
```python
from vector_os_nano.llm.prompts import build_planning_prompt, build_agent_loop_prompt
```

**REFACTOR**: Confirm `parse_action_response()` follows the same guard pattern as `parse_plan_response()` — never raises, always returns a dict.

### Acceptance Criteria
- [ ] `parse_action_response()` handles: valid JSON, markdown-fenced JSON, broken JSON (regex), empty string — all without raising
- [ ] `parse_action_response("")` returns `{"action": "scan", ...}` (safe fallback)
- [ ] `build_agent_loop_prompt()` returns a string containing goal, skills, observation, history
- [ ] `ClaudeProvider.decide_next_action()` calls `_chat_completion()` once and returns parsed dict
- [ ] All tests pass: `pytest tests/test_parse_action_response.py tests/test_decide_next_action.py tests/test_agent_loop_prompt.py`

---

## T3 — AgentLoop Implementation

**Agent**: [alpha]
**Estimate**: 45 min
**Dependencies**: T1 (skeleton), T2 (decide_next_action)
**Branch**: `feat/alpha-agent-loop-impl` (or continue on `feat/alpha-agent-loop-types`)

### Files
- `vector_os_nano/core/agent_loop.py` — implement all stub methods

### TDD

**RED**: Write `tests/test_agent_loop.py` first.

```python
# tests/test_agent_loop.py
from unittest.mock import MagicMock, patch
from vector_os_nano.core.agent_loop import AgentLoop
from vector_os_nano.core.types import GoalResult, SkillResult

def _make_agent(decide_responses, skill_result=None):
    """Factory for mock agent that returns pre-set LLM decisions."""
    agent = MagicMock()
    agent._arm = None  # not sim
    agent._world_model.to_dict.return_value = {"objects": [{"label": "banana"}]}
    agent._world_model.apply_skill_effects.return_value = None
    agent._skill_registry.to_schemas.return_value = [{"name": "pick"}, {"name": "scan"}]
    agent._llm.decide_next_action.side_effect = decide_responses
    if skill_result is None:
        skill_result = SkillResult(success=True)
    mock_skill = MagicMock()
    mock_skill.execute.return_value = skill_result
    agent._skill_registry.get.return_value = mock_skill
    agent._build_context.return_value = MagicMock()
    return agent

def test_full_loop_two_picks_then_done():
    responses = [
        {"action": "pick", "params": {"object_label": "banana"}, "reasoning": "first"},
        {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "second"},
        {"done": True, "summary": "All picked."},
    ]
    agent = _make_agent(responses)
    loop = AgentLoop(agent_ref=agent, config={})
    result = loop.run(goal="pick all", max_iterations=10, verify=False,
                      on_step=None, on_message=None)

    assert result.success is True
    assert result.iterations == 3
    assert len(result.actions) == 2
    assert result.summary == "All picked."

def test_max_iterations_cap():
    # LLM never returns done
    responses = [{"action": "scan", "params": {}, "reasoning": "loop"} for _ in range(20)]
    agent = _make_agent(responses)
    loop = AgentLoop(agent_ref=agent, config={})
    result = loop.run(goal="infinite", max_iterations=3, verify=False,
                      on_step=None, on_message=None)

    assert result.success is False
    assert result.iterations == 3
    assert len(result.actions) == 3

def test_verify_called_only_for_pick_place():
    responses = [
        {"action": "scan", "params": {}, "reasoning": "look"},
        {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "grab"},
        {"done": True, "summary": "Done."},
    ]
    agent = _make_agent(responses)
    loop = AgentLoop(agent_ref=agent, config={})

    verify_calls = []
    original_verify = loop._verify
    loop._verify = lambda: verify_calls.append(1) or original_verify()

    result = loop.run(goal="pick mug", max_iterations=10, verify=True,
                      on_step=None, on_message=None)

    # verify called once (for pick), not for scan
    assert len(verify_calls) == 1

def test_unknown_skill_records_failure():
    responses = [
        {"action": "nonexistent_skill", "params": {}, "reasoning": "bad"},
        {"done": True, "summary": "Gave up."},
    ]
    agent = _make_agent(responses)
    agent._skill_registry.get.return_value = None  # skill not found
    loop = AgentLoop(agent_ref=agent, config={})
    result = loop.run(goal="test", max_iterations=5, verify=False,
                      on_step=None, on_message=None)

    assert len(result.actions) == 1
    assert result.actions[0].skill_success is False

def test_history_trimmed_to_max():
    n = 10
    responses = [
        {"action": "scan", "params": {}, "reasoning": f"step {i}"} for i in range(n)
    ] + [{"done": True, "summary": "Done."}]
    agent = _make_agent(responses)
    loop = AgentLoop(agent_ref=agent, config={"agent": {"agent_loop": {"history_max_actions": 3}}})
    result = loop.run(goal="test", max_iterations=n + 2, verify=False,
                      on_step=None, on_message=None)

    # Verify decide_next_action was called with history <= 3 entries
    for call in agent._llm.decide_next_action.call_args_list:
        history_arg = call.kwargs.get("history") or call.args[3]
        assert len(history_arg) <= 3

def test_sim_mode_calls_refresh_objects():
    """In sim mode, _observe() calls _refresh_objects()."""
    responses = [{"done": True, "summary": "Done."}]
    agent = _make_agent(responses)
    # Make arm have get_object_positions -> sim mode
    agent._arm = MagicMock(spec=["get_object_positions"])
    loop = AgentLoop(agent_ref=agent, config={})
    loop.run(goal="test", max_iterations=5, verify=False, on_step=None, on_message=None)

    agent._refresh_objects.assert_called_once()

def test_on_step_callback_called():
    responses = [
        {"action": "scan", "params": {}, "reasoning": "look"},
        {"done": True, "summary": "Done."},
    ]
    agent = _make_agent(responses)
    loop = AgentLoop(agent_ref=agent, config={})
    calls = []
    loop.run(goal="test", max_iterations=5, verify=False,
             on_step=lambda action, i, total: calls.append((action, i, total)),
             on_message=None)
    assert calls == [("scan", 0, 5)]
```

**GREEN**: Fill in all stub methods in `core/agent_loop.py`:
- `_observe()`: builds observation dict, calls `_refresh_objects()` in sim
- `_decide()`: reads `agent_loop.model` from config, calls `_llm.decide_next_action()`
- `_act_and_record()`: gets skill, calls `skill.execute()`, calls `_world_model.apply_skill_effects()`, optionally calls `_verify()`
- `_verify()`: sim path calls `_refresh_objects()`; hardware path runs scan + detect skills
- `_should_verify()`: returns `action in _VERIFY_AFTER`
- `_trim_history()`: slices to `[-history_max_actions:]`

**REFACTOR**: No dynamic allocation in `_act_and_record()`. `ActionRecord` construction is the only allocation per iteration.

### Acceptance Criteria
- [ ] Full loop with 2 picks + done: `result.success=True`, `len(result.actions)==2`
- [ ] `max_iterations=3` with looping LLM: `result.success=False`, `result.iterations==3`
- [ ] Verify called once for pick, zero times for scan
- [ ] Unknown skill: `ActionRecord(skill_success=False)` recorded, loop continues
- [ ] History trimmed to `history_max_actions` — never exceeds limit
- [ ] Sim mode calls `_refresh_objects()` on each observe
- [ ] `on_step` callback invoked after each action
- [ ] All tests pass: `pytest tests/test_agent_loop.py`

---

## T4 — Agent Integration + MCP

**Agent**: [beta]
**Estimate**: 30 min
**Dependencies**: T1 (GoalResult type)
**Branch**: `feat/beta-agent-mcp-integration`

### Files
- `vector_os_nano/core/agent.py` — add `run_goal()` method, update GoalResult import
- `vector_os_nano/mcp/tools.py` — add `build_run_goal_tool()`, handler in `handle_tool_call()`
- `vector_os_nano/config/default.yaml` — add `agent.agent_loop` section

### TDD

**RED**: Write `tests/test_agent_run_goal.py` first.

```python
# tests/test_agent_run_goal.py
from unittest.mock import MagicMock, patch
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.types import GoalResult

def test_agent_run_goal_delegates_to_agent_loop():
    agent = Agent()  # no hardware
    mock_result = GoalResult(success=True, goal="test", iterations=1,
                             total_duration_sec=1.0, actions=[], summary="Done.", final_world_state={})
    with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
        instance = MockLoop.return_value
        instance.run.return_value = mock_result
        result = agent.run_goal("test goal", max_iterations=5, verify=False)

    MockLoop.assert_called_once_with(agent_ref=agent, config=agent._config)
    instance.run.assert_called_once_with(
        goal="test goal", max_iterations=5, verify=False,
        on_step=None, on_message=None
    )
    assert result.success is True

def test_agent_run_goal_returns_goal_result():
    agent = Agent()
    mock_result = GoalResult(success=False, goal="fail", iterations=10,
                             total_duration_sec=30.0, actions=[], summary="Max iterations.", final_world_state={})
    with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
        MockLoop.return_value.run.return_value = mock_result
        result = agent.run_goal("fail", max_iterations=10)
    assert isinstance(result, GoalResult)
    assert result.success is False
```

Also write `tests/test_mcp_run_goal.py`:

```python
# tests/test_mcp_run_goal.py
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from vector_os_nano.mcp.tools import skills_to_mcp_tools, handle_tool_call
from vector_os_nano.core.types import GoalResult

def test_run_goal_tool_in_mcp_list():
    registry = MagicMock()
    registry.to_schemas.return_value = []
    tools = skills_to_mcp_tools(registry)
    names = [t["name"] for t in tools]
    assert "run_goal" in names

def test_run_goal_tool_schema():
    registry = MagicMock()
    registry.to_schemas.return_value = []
    tools = skills_to_mcp_tools(registry)
    tool = next(t for t in tools if t["name"] == "run_goal")
    assert "goal" in tool["inputSchema"]["required"]
    assert "max_iterations" in tool["inputSchema"]["properties"]
    assert "verify" in tool["inputSchema"]["properties"]

def test_handle_tool_call_run_goal():
    agent = MagicMock()
    mock_result = GoalResult(success=True, goal="clean", iterations=3,
                             total_duration_sec=10.0, actions=[], summary="Done.", final_world_state={})
    agent.run_goal.return_value = mock_result

    result = asyncio.run(handle_tool_call(agent, "run_goal", {"goal": "clean the table"}))
    data = json.loads(result)
    assert data["success"] is True
    assert data["goal"] == "clean"
```

**GREEN**:
1. Add `run_goal()` to `Agent` (10 lines, lazy import of `AgentLoop`).
2. Update `agent.py` line 20: add `GoalResult` to the import.
3. Add `build_run_goal_tool()` to `mcp/tools.py`.
4. Update `skills_to_mcp_tools()` to call `build_run_goal_tool()`.
5. Add `run_goal` handler branch in `handle_tool_call()`.
6. Add `agent_loop` section to `config/default.yaml`.

**REFACTOR**: Confirm `run_goal()` in `agent.py` uses lazy import for `AgentLoop` (matches pattern of other lazy imports in `__init__`).

### Acceptance Criteria
- [ ] `agent.run_goal("goal")` delegates to `AgentLoop.run()` with correct args
- [ ] `skills_to_mcp_tools()` includes a tool with `name == "run_goal"`
- [ ] `run_goal` tool has `goal` in required, `max_iterations` and `verify` as optional
- [ ] `handle_tool_call(agent, "run_goal", {"goal": "..."})` calls `agent.run_goal()` and returns JSON
- [ ] `config/default.yaml` has `agent.agent_loop.max_iterations: 10`
- [ ] All tests pass: `pytest tests/test_agent_run_goal.py tests/test_mcp_run_goal.py`

---

## T5 — Integration Test

**Agent**: [gamma]
**Estimate**: 30 min
**Dependencies**: T3 (AgentLoop impl), T4 (Agent.run_goal + MCP)
**Branch**: `feat/gamma-agent-loop-integration-test`

### Files
- `tests/test_agent_loop_integration.py` — new integration test file

### TDD

Integration tests use a real `Agent` with a mock arm (sim mode) and mock LLM. No real hardware or network calls.

**RED**: Write the full test file first, confirm all tests fail (AgentLoop not yet complete when T5 starts — tests pass only after T3+T4 are done).

```python
# tests/test_agent_loop_integration.py
"""Integration tests for Agent.run_goal() with mock sim arm + mock LLM.

No real hardware. No network calls. Validates the full run_goal() path
from Agent through AgentLoop to skill execution.
"""
import json
from unittest.mock import MagicMock, patch
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.types import GoalResult, SkillResult


def _make_sim_arm(objects: dict[str, tuple]) -> MagicMock:
    """Mock arm with get_object_positions() -> sim mode."""
    arm = MagicMock()
    arm.get_object_positions.return_value = objects
    return arm


def _make_mock_skill(success: bool = True) -> MagicMock:
    skill = MagicMock()
    skill.execute.return_value = SkillResult(success=success)
    return skill


def test_run_goal_clean_table_sim():
    """Simulate picking 3 objects then receiving done."""
    objects = {"banana": (0.1, 0.0, 0.05), "mug": (0.2, 0.0, 0.05), "bottle": (0.15, 0.05, 0.05)}
    arm = _make_sim_arm(objects)

    llm = MagicMock()
    llm.decide_next_action.side_effect = [
        {"action": "pick", "params": {"object_label": "banana"}, "reasoning": "first"},
        {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "second"},
        {"action": "pick", "params": {"object_label": "bottle"}, "reasoning": "third"},
        {"done": True, "summary": "All 3 objects removed from table."},
    ]

    agent = Agent(arm=arm, llm=llm)

    # Patch pick skill to always succeed
    pick_skill = _make_mock_skill(success=True)
    agent._skill_registry._skills = {"pick": pick_skill, "scan": _make_mock_skill(), "detect": _make_mock_skill()}

    result = agent.run_goal("clean the table", max_iterations=10, verify=False)

    assert isinstance(result, GoalResult)
    assert result.success is True
    assert result.iterations == 4  # 3 picks + 1 done check
    assert len(result.actions) == 3
    assert result.summary == "All 3 objects removed from table."
    assert all(a.skill_success for a in result.actions)


def test_run_goal_max_iterations_termination():
    """Loop terminates at max_iterations when LLM never returns done."""
    arm = _make_sim_arm({"mug": (0.1, 0.0, 0.05)})
    llm = MagicMock()
    llm.decide_next_action.return_value = {"action": "scan", "params": {}, "reasoning": "stuck"}

    agent = Agent(arm=arm, llm=llm)
    agent._skill_registry._skills = {"scan": _make_mock_skill()}

    result = agent.run_goal("test", max_iterations=3, verify=False)

    assert result.success is False
    assert result.iterations == 3
    assert len(result.actions) == 3


def test_run_goal_via_mcp_tool():
    """End-to-end: MCP handle_tool_call -> Agent.run_goal -> GoalResult as JSON."""
    import asyncio
    from vector_os_nano.mcp.tools import handle_tool_call

    arm = _make_sim_arm({"banana": (0.1, 0.0, 0.05)})
    llm = MagicMock()
    llm.decide_next_action.side_effect = [
        {"action": "pick", "params": {"object_label": "banana"}, "reasoning": "only object"},
        {"done": True, "summary": "Table clear."},
    ]

    agent = Agent(arm=arm, llm=llm)
    pick_skill = _make_mock_skill(success=True)
    agent._skill_registry._skills = {"pick": pick_skill}

    raw = asyncio.run(handle_tool_call(
        agent, "run_goal", {"goal": "pick all objects", "max_iterations": 5, "verify": False}
    ))
    data = json.loads(raw)

    assert data["success"] is True
    assert data["goal"] == "pick all objects"
    assert len(data["actions"]) == 1
    assert data["actions"][0]["action"] == "pick"


def test_run_goal_parse_failure_fallback():
    """When LLM returns garbage, parse fallback to scan prevents crash."""
    arm = _make_sim_arm({})
    llm = MagicMock()
    llm.decide_next_action.side_effect = [
        {"action": "scan", "params": {}, "reasoning": "parse failure fallback"},  # fallback result
        {"done": True, "summary": "Recovered."},
    ]

    agent = Agent(arm=arm, llm=llm)
    agent._skill_registry._skills = {"scan": _make_mock_skill()}

    result = agent.run_goal("test", max_iterations=5, verify=False)

    assert result.success is True  # recovered after scan


def test_existing_execute_unaffected():
    """Agent.execute() still works after v0.4.0 changes — no regression."""
    from vector_os_nano.core.types import ExecutionResult
    agent = Agent()  # no hardware

    result = agent.execute("home")
    assert isinstance(result, ExecutionResult)
    # No arm connected — expected to fail gracefully, not crash
    assert isinstance(result.success, bool)
```

**GREEN**: Tests pass when T3 + T4 are complete. No new code to write beyond the test file itself.

**REFACTOR**: Confirm tests do not import private modules (only `Agent`, `GoalResult`, `handle_tool_call`). All mocking is at the boundary (arm, llm, skill registry).

### Acceptance Criteria
- [ ] `run_goal("clean the table")` with 3 mock picks + done: `GoalResult(success=True, iterations=4, len(actions)==3)`
- [ ] `run_goal` with `max_iterations=3` and looping LLM: `GoalResult(success=False, iterations=3)`
- [ ] MCP `handle_tool_call(agent, "run_goal", {...})` returns valid JSON with `GoalResult` structure
- [ ] Parse failure fallback (scan) does not crash the loop
- [ ] `agent.execute("home")` still returns `ExecutionResult` (no regression)
- [ ] All tests pass: `pytest tests/test_agent_loop_integration.py`

---

## Final Verification

After all tasks complete, run the full test suite:

```bash
cd /home/yusen/Desktop/vector_os_nano
pytest tests/ -v
```

All pre-existing tests must continue to pass (no regressions in `agent.execute()`, `parse_plan_response()`, MCP skill tools).
