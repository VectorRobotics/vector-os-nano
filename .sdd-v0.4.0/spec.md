# Spec: v0.4.0 — Built-in Agent Loop

**Status**: draft
**Author**: Lead Architect (Opus)
**Date**: 2026-03-24

---

## 1. Problem Statement

When the user says "clean the table" (three objects on table), the current system:

1. **Plans all steps once** via `_handle_task()` (agent.py:436). The LLM produces a flat plan: `[scan, detect, pick(banana), home]`. It picks ONE object and declares done.
2. **No post-execution verification**. `PickSkill.execute()` returns `SkillResult(success=True)` once the gripper close sequence completes (pick.py:371-373), regardless of whether the gripper actually grasped anything. On real hardware, servo drift means the gripper often misses.
3. **Retries are blind**. Pick's internal retry (pick.py:156-169) repeats the same grasp pose. If drift is systematic (always 1cm left), retrying the same pose fails identically.
4. **No goal-completion check**. After picking one object, the system has no mechanism to ask "is the table empty now?" and plan the next pick.

The root cause: the system has no observe-reason-act-verify cycle. The LLM plans once upfront, the executor blindly runs the plan, and there is no feedback loop.

## 2. Solution Overview

Add `Agent.run_goal(goal)` — an iterative observe-decide-act-verify loop where the LLM acts as the reasoning core at each iteration.

```
                       +---> OBSERVE (world model + perception)
                       |
run_goal("clean table")|
                       +---> DECIDE (LLM: given obs + history, pick ONE action)
                       |
                       +---> ACT (execute single skill via existing infrastructure)
                       |
                       +---> VERIFY (perception confirms outcome)
                       |
                       +---> DONE? (LLM judges goal completion) ---no---> loop
                       |
                      yes---> return GoalResult
```

### Design Principles

1. **Minimal new code**: Reuse `_build_context()`, `SkillRegistry.get()`, `Skill.execute()`, `WorldModel`, `SessionMemory`. The loop is ~150 lines in a new module.
2. **LLM-agnostic**: The loop calls `LLMProvider._chat_completion()` with a text prompt and parses a JSON response. No native tool_use API. Works with any OpenAI-compatible endpoint.
3. **Coexistence**: `Agent.execute()` is unchanged. `run_goal()` is a new public method. `_handle_task()` is NOT modified.
4. **Simplest viable version first**: v0.4.0 supports a single concurrent goal, synchronous execution, text-based LLM interaction.

## 3. Functional Requirements

### FR-1: Agent.run_goal(goal) Public API

```python
def run_goal(
    self,
    goal: str,
    max_iterations: int = 10,
    verify: bool = True,
    on_step: Callable | None = None,
    on_message: Callable | None = None,
) -> GoalResult:
```

- `goal`: Natural language goal string (e.g., "clean the table", "sort objects by color").
- `max_iterations`: Safety cap on loop iterations. Default 10.
- `verify`: Whether to run perception verification after each action. Default True. Set False in sim for speed.
- `on_step` / `on_message`: Existing callback pattern from `execute()`.
- Returns `GoalResult` (see FR-6).

### FR-2: Observe Phase

Before each LLM call, build an observation dict:

```python
observation = {
    "world_state": self._world_model.to_dict(),
    "iteration": i,
    "goal": goal,
    "history": [  # last N actions + outcomes
        {"action": "pick", "params": {"object_label": "banana"}, "success": True, "verification": "object no longer on table"},
        ...
    ],
}
```

Source data:
- `WorldModel.to_dict()` — already serializable (world_model.py:296-304).
- In sim: `_refresh_objects()` (agent.py:189-215) re-reads MuJoCo ground truth.
- On hardware with `verify=True`: run scan + detect to refresh world model.
- On hardware with `verify=False`: trust `SkillResult.result_data` only.

### FR-3: Decide Phase (LLM Call)

New method on LLMProvider protocol:

```python
def decide_next_action(
    self,
    goal: str,
    observation: dict,
    skill_schemas: list[dict],
    history: list[dict],
    model_override: str | None = None,
) -> dict:
```

**Returns a dict** with exactly one of:
```json
{"action": "pick", "params": {"object_label": "banana"}, "reasoning": "banana is closest"}
```
or:
```json
{"done": true, "summary": "All 3 objects removed from table."}
```

**Implementation**: This is a `_chat_completion()` call with a new system prompt (see Section 5). The response is parsed with `json.loads()` + `_strip_markdown_fences()` (already exists in claude.py:32-43).

**NOT a protocol change**: `decide_next_action()` is implemented as a concrete method on `ClaudeProvider` (like `classify()` and `chat()`), not added to the `LLMProvider` protocol. The protocol stays minimal. Other providers implement it when they need the agent loop.

### FR-4: Act Phase

Execute a single skill using existing infrastructure:

```python
skill = self._skill_registry.get(action["action"])
result = skill.execute(action["params"], context)
self._world_model.apply_skill_effects(action["action"], action["params"], result)
```

This is the same path as `TaskExecutor.execute()` lines 125-175, but for a single step. We skip `TaskExecutor` entirely — no `TaskPlan`, no topological sort, no precondition/postcondition checking. The LLM is responsible for ordering and preconditions.

Why skip TaskExecutor: The executor is designed for a pre-planned sequence. The agent loop generates one action at a time. Creating a single-step TaskPlan each iteration is unnecessary overhead.

### FR-5: Verify Phase

After each action, optionally verify the outcome using perception:

**Strategy 1 — Full verify (hardware, verify=True)**:
1. Execute `scan` skill (move arm to scan pose) — 3 seconds.
2. Execute `detect` skill with query "all objects" — 2-3 seconds (VLM + tracker).
3. World model is refreshed with new detections.
4. The updated world model feeds into the next OBSERVE phase.

**Strategy 2 — Sim verify (sim mode)**:
1. Call `_refresh_objects()` which reads MuJoCo ground truth — <1ms.
2. No perception call needed.

**Strategy 3 — Skip verify (verify=False)**:
1. Trust `SkillResult.success` and `result_data`.
2. Apply world model effects only.
3. Fastest, but vulnerable to false positives on hardware.

**Sim detection**: Check `hasattr(self._arm, "get_object_positions")` — already used at agent.py:199.

### FR-6: GoalResult Type

```python
@dataclass(frozen=True)
class GoalResult:
    success: bool
    goal: str
    iterations: int
    total_duration_sec: float
    actions: list[ActionRecord]  # full trace
    summary: str  # LLM-generated or default
    final_world_state: dict

@dataclass(frozen=True)
class ActionRecord:
    iteration: int
    action: str
    params: dict
    skill_success: bool
    verified: bool
    reasoning: str
    duration_sec: float
```

### FR-7: Prompt Template for decide_next_action

New constant in `prompts.py`:

```
AGENT_LOOP_SYSTEM_PROMPT = """
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
2. If the goal is achieved, return: {"done": true, "summary": "..."}
3. If more work is needed, return: {"action": "skill_name", "params": {...}, "reasoning": "..."}
4. ONLY use skill names from AVAILABLE SKILLS. Parameters must match the schema.
5. If a previous action failed, try a different approach — do NOT repeat the same action.
6. If you need to see the workspace, use "scan" then "detect" as your action.
7. Maximum {max_iterations} iterations — be efficient.
"""
```

### FR-8: MCP Integration

Add `run_goal` as an MCP tool in `mcp/tools.py`:

```python
{
    "name": "run_goal",
    "description": "Execute an iterative goal (observe-decide-act-verify loop). Use for multi-step goals like 'clean the table'.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Natural language goal"},
            "max_iterations": {"type": "integer", "default": 10},
            "verify": {"type": "boolean", "default": true},
        },
        "required": ["goal"],
    },
}
```

Handle in `handle_tool_call()` by calling `agent.run_goal()`.

### FR-9: Configuration

Add to `config/default.yaml`:

```yaml
agent:
  agent_loop:
    max_iterations: 10
    verify: true              # post-action perception verification
    verify_in_sim: false      # skip perception verify in sim (use ground truth)
    history_max_actions: 6    # max actions in LLM context (token management)
    model: null               # null = use plan_complex model
```

## 4. Non-Functional Requirements

### NFR-1: Latency Budget per Iteration

| Phase | Hardware | Sim |
|-------|----------|-----|
| Observe (world model serialize) | <10ms | <10ms |
| Decide (LLM call) | 1-3s (cloud) | 1-3s (cloud) |
| Act (single skill) | 3-10s | <1s |
| Verify (scan+detect) | 5-6s | <1ms |
| **Total per iteration** | **~10-20s** | **~2-5s** |

For "clean table" with 3 objects: ~6-8 iterations (3x pick + 3x verify + initial scan/detect + done check). Total: 60-160s on hardware, 12-40s in sim.

### NFR-2: Token Budget

Each `decide_next_action` call includes:
- System prompt: ~300 tokens
- Skill schemas: ~400 tokens (9 skills)
- Observation (world state): ~200-500 tokens (depends on objects)
- History: ~100 tokens per action record, capped at 6 = 600 tokens max
- **Total per call: ~1500-1800 tokens input, ~100 tokens output**

For 8 iterations: ~14,000 input tokens + 800 output tokens. At Haiku rates ($0.80/$4.00 per MTok): ~$0.015 per goal. At Sonnet rates ($3/$15): ~$0.054 per goal. Acceptable.

### NFR-3: Thread Safety

The agent loop runs synchronously on the caller's thread. The background tracking thread (pipeline.py:384-444) may be active. Interaction points:

- `pipeline.detect()` acquires `_lock` briefly (pipeline.py:293).
- `pipeline.track()` starts `start_continuous_tracking()` which spawns the background thread (pipeline.py:299).
- During the verify phase, calling `detect()` reinitializes the tracker, which is safe because `track()` calls `init_track()` which resets state.

No new threading required. No new locks needed.

## 5. Architecture

### 5.1 New Module: `vector_os_nano/core/agent_loop.py`

Contains the `AgentLoop` class — the observe-decide-act-verify engine.

```
Agent.run_goal(goal)
    |
    v
AgentLoop(agent_ref, config)
    .run(goal, max_iterations, verify, callbacks)
        |
        for i in range(max_iterations):
            obs = self._observe()
            decision = self._decide(obs)
            if decision.done: break
            result = self._act(decision)
            if verify: self._verify()
        |
        return GoalResult
```

Why a separate class (not inline in Agent):
- `Agent` is already 856 lines. Adding 150+ lines of loop logic pushes it past the 800-line cap.
- `AgentLoop` can be tested independently with mock LLM + mock skills.
- Clean separation: Agent = wiring, AgentLoop = loop logic.

### 5.2 Modified Files

| File | Change | Scope |
|------|--------|-------|
| `core/agent.py` | Add `run_goal()` method (5 lines, delegates to AgentLoop) | Small |
| `core/types.py` | Add `GoalResult`, `ActionRecord` dataclasses | Small |
| `llm/claude.py` | Add `decide_next_action()` method | Medium (~40 lines) |
| `llm/prompts.py` | Add `AGENT_LOOP_SYSTEM_PROMPT`, `build_agent_loop_prompt()` | Small |
| `mcp/tools.py` | Add `run_goal` tool definition + handler | Small |
| `config/default.yaml` | Add `agent.agent_loop` section | Trivial |

### 5.3 New Files

| File | Purpose | Lines |
|------|---------|-------|
| `core/agent_loop.py` | AgentLoop class | ~200 |

### 5.4 Unchanged Files

- `core/executor.py` — not used by agent loop (single-step execution is direct).
- `core/skill.py` — skills are called directly via `.execute()`.
- `core/world_model.py` — consumed as-is via `.to_dict()` and `.apply_skill_effects()`.
- `core/memory.py` — the agent loop maintains its own short history (list of ActionRecords), not SessionMemory. SessionMemory is for cross-execute() conversational context.
- `perception/pipeline.py` — called as-is during verify phase via detect skill.
- All skill files — unchanged.

## 6. Feasibility Analysis

### Q1: LLM Latency — Is 30-60s of LLM time acceptable?

**Analysis**: For "clean table" with 3 objects:
- Optimistic (Haiku, fast network): 8 iterations x 1.5s = 12s LLM time
- Pessimistic (Sonnet, slow network): 8 iterations x 3s = 24s LLM time

The LLM time is small relative to the physical execution time. Each pick takes 8-12 seconds (scan 3s + detect 2s + pick 5s). Total physical time for 3 picks: ~30-40s. LLM adds 30-50% overhead.

**Verdict**: Acceptable. The LLM calls overlap zero with the physical execution (synchronous pipeline), but the absolute time is within human patience for a "clean entire table" command.

**Mitigation**: Use Haiku for `decide_next_action` — the reasoning is simple (one skill at a time, constrained to skill schemas). Configure via `agent.agent_loop.model`.

**Local models (Ollama)**: Would reduce latency to <500ms per call but require Ollama setup. The architecture supports this via `model_override` in `_chat_completion()`. Not in scope for v0.4.0.

### Q2: Verification Cost — Is 5-6s per verify acceptable?

**Analysis**: Full verify = scan(3s) + detect(2-3s) = 5-6s. For 3 objects, that is 15-18s of pure verification time.

**Code evidence**: ScanSkill (scan.py:73) calls `move_joints(duration=3.0)`. DetectSkill (detect.py:50-187) calls `perception.detect()` then `perception.track()`. VLM detect takes ~1-2s (GPU inference). Track init takes ~0.5-1s.

**Cheaper alternatives considered**:
1. **Gripper position check**: `gripper.get_position()` returns servo angle. If gripper closed to minimum (no object), it is a miss. **Problem**: Not implemented in current `SO101Gripper` — would need hardware-level servo current reading. Out of scope.
2. **Quick camera check without VLM**: Grab a frame, compare object count at known positions. **Problem**: Requires VLM or classical CV to count objects. No shortcut.
3. **Skip verify for non-pick actions**: Only verify after `pick` and `place`. `scan`, `detect`, `home` don't need verification. This is a significant optimization.

**Verdict**: 5-6s per verify is acceptable for v0.4.0. Optimization: only verify after `pick` and `place` skills, not after every action. This cuts verify count from ~8 to ~3 for "clean table".

**Implementation**: In `_should_verify(action_name)`:
```python
_VERIFY_AFTER = {"pick", "place"}
```

### Q3: LLM Output Format — How to handle parse failures?

**Analysis**: `decide_next_action` returns ONE JSON object. Simpler than plan() which returns a list of steps. Current parsing infrastructure (claude.py:32-43 `_strip_markdown_fences`, claude.py:46-107 `parse_plan_response`) already handles:
- Markdown fences
- Malformed JSON
- Missing fields

For the agent loop, parsing is simpler: one JSON object with either `{"done": true}` or `{"action": "...", "params": {...}}`.

**Parse failure strategy**:
1. First attempt: `json.loads(_strip_markdown_fences(raw_text))`
2. Fallback: regex extract `"action"\s*:\s*"(\w+)"` — get action name even from broken JSON.
3. Final fallback: return a "scan" action (safe, non-destructive, gives the LLM fresh observation).

**Evidence that this is manageable**: `parse_plan_response()` (claude.py:46-107) already handles parse failures gracefully and returns empty plans. The agent loop variant is strictly simpler.

**Verdict**: LOW RISK. Single-action JSON is easier to parse than multi-step plans. Fallback to "scan" is safe.

### Q4: When to Use Loop vs One-Shot — How to decide?

**Option A: New classify intent "goal"**: Add a fourth intent type. `classify()` returns "goal" for iterative tasks. **Problem**: classify prompt would need to distinguish "pick the cup" (one-shot) from "clean the table" (goal). This is subtle and error-prone.

**Option B: Explicit API**: `run_goal()` is a separate method. The caller decides. **Problem**: The chatbot UI / MCP client must choose which to call.

**Option C: Subsume in execute()**: `execute()` detects iterative intent and delegates to `run_goal()`. **Problem**: Changes the core pipeline, risky.

**Recommended: Option B with heuristic routing in execute() (optional)**. For v0.4.0:
- `run_goal()` is a new public method — the explicit API.
- MCP exposes both `natural_language` (one-shot) and `run_goal` (iterative) tools.
- Optionally, `_handle_task()` can detect iterative patterns ("clean", "all", "every", "sort", "organize") and auto-delegate to `run_goal()`. This is a follow-up enhancement, NOT in v0.4.0 scope.

**Evidence**: `_needs_llm_planning()` (agent.py:302-316) already detects multi-object patterns (`"所有", "全部", "都", "all", "every", "each"`). This heuristic could be extended to trigger `run_goal()` in a future version.

**Verdict**: v0.4.0 ships `run_goal()` as explicit API only. Auto-detection is v0.4.1 scope.

### Q5: Token Cost — How to keep context manageable?

**Analysis**: The growing history is the main concern. Each iteration adds ~100 tokens of history. At 10 iterations, history is ~1000 tokens. Total context per call: ~2500 tokens.

**Mitigation**: `history_max_actions: 6` config. Only the last 6 action records are included. Older actions are summarized in a one-line string: "Previous: picked banana(ok), picked mug(ok)".

**Evidence**: SessionMemory (memory.py:60) already implements bounded history with `_trim()`. The agent loop uses a simpler approach: a Python list sliced to `[-history_max_actions:]`.

**Verdict**: LOW RISK. Even at 10 iterations, context stays under 3000 tokens total. Well within any model's context window.

### Q6: Sim vs Hardware — How to handle sim's determinism?

**Analysis**: In MuJoCo sim, `pick` is deterministic — no drift, no false positives. Verification is unnecessary. But `_refresh_objects()` (agent.py:189-215) provides ground truth via `arm.get_object_positions()` — instant, free, accurate.

**Strategy**:
- Detect sim: `is_sim = hasattr(self._arm, "get_object_positions")`
- In sim: always call `_refresh_objects()` for observation. Skip scan/detect verify. World model is ground truth.
- Config: `agent_loop.verify_in_sim: false` (default). User can override to `true` for testing the perception verify path.

**Verdict**: EASY. Sim detection already exists. Ground truth refresh is free.

### Q7: Concurrent Perception — Thread safety?

**Analysis**: The background tracking thread (pipeline.py:384-444) runs continuously after `track()` is called. During verify, the agent loop calls `detect()` (via DetectSkill), which calls `pipeline.detect()` then `pipeline.track()`.

- `pipeline.detect()` (pipeline.py:226-244): Calls `self._vlm.detect()`. No lock needed — VLM is single-call.
- `pipeline.track()` (pipeline.py:246-301): Acquires `self._lock` to update `_tracked_objects`. Calls `start_continuous_tracking()`.
- The background thread (pipeline.py:395-443) also acquires `self._lock` to update `_tracked_objects`.
- `track()` calls `self._tracker.init_track()` which resets the tracker state. The background thread checks `self._tracker.is_tracking()` and will pick up the new tracking session.

**Potential issue**: If the background thread is mid-`process_image()` when `track()` calls `init_track()`, the tracker's internal state could be inconsistent.

**Mitigation**: `stop_continuous_tracking()` before verify, `start_continuous_tracking()` after. But this adds complexity.

**Pragmatic decision**: The background thread is for UI overlay only (camera viewer). During the agent loop, no camera viewer is active (the arm is moving). The thread's stale output is harmless. No changes needed.

**Verdict**: LOW RISK. No new locks required. Background thread produces stale but harmless output during verify.

### Q8: MCP Integration — Two competing reasoning layers?

**Analysis**: If an external agent (Claude Code via MCP) calls `run_goal("clean table")`, the INTERNAL agent loop is doing the reasoning. The external agent gets back a `GoalResult` when done. There is no conflict because:
1. `run_goal` is blocking — the MCP call waits for completion.
2. The external agent does NOT make intermediate decisions — the internal loop does.

**However**, if the external agent is ALSO doing iterative reasoning (calling `pick` via MCP in a loop, checking `world://objects` resource between calls), then `run_goal` would conflict. Solution: they are different tools for different use cases.
- MCP `run_goal`: External agent delegates the ENTIRE iterative goal to the SDK.
- MCP `pick` / `natural_language`: External agent drives individual steps.

**Verdict**: NO CONFLICT. Expose both tools. Documentation makes the distinction clear.

## 7. Scope Boundaries

### In Scope (v0.4.0)

- `Agent.run_goal()` public API
- `AgentLoop` class in `core/agent_loop.py`
- `ClaudeProvider.decide_next_action()` method
- `AGENT_LOOP_SYSTEM_PROMPT` in prompts.py
- `GoalResult` / `ActionRecord` types
- `run_goal` MCP tool
- Config section `agent.agent_loop`
- Unit tests for AgentLoop with mock LLM

### Out of Scope (future versions)

- Auto-detection of iterative intent in `execute()` (v0.4.1)
- Gripper force/position feedback for cheap verification (hardware change)
- Parallel skill execution within the loop
- Async/non-blocking goal execution
- Multi-goal queuing
- LLMProvider protocol extension (decide_next_action stays on concrete providers)
- Changes to existing `_handle_task()` pipeline
- Local model (Ollama) integration

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| LLM returns unparseable JSON | Medium | Low | Triple fallback parser; "scan" as safe default |
| LLM loops without progress | Medium | Medium | `max_iterations` cap; history shows repetition |
| Verify adds too much latency on hardware | Low | Medium | Only verify after pick/place; configurable `verify: false` |
| PickSkill false positive persists after verify | Low | High | Verify via fresh detect overrides stale world model |
| Token cost spikes for complex goals | Low | Low | `history_max_actions` cap; Haiku model default |
| Background tracking thread interference | Low | Low | Thread-safe by design (lock-protected reads) |

## 9. Success Criteria

1. `agent.run_goal("clean the table")` in MuJoCo sim with 3 objects: picks all 3, returns `GoalResult(success=True, iterations<=8)`.
2. Each iteration produces exactly one skill call.
3. LLM correctly decides "done" when world model shows zero objects.
4. `run_goal` works with any LLMProvider that has `_chat_completion()` (not just Claude).
5. MCP `run_goal` tool returns structured result with per-action trace.
6. Unit test: mock LLM + mock skills, verify loop terminates correctly on "done".

## 10. Implementation Notes for Engineers

### Task Breakdown

**Task 1 [alpha]: Core types + AgentLoop skeleton** (~30min)
- Add `GoalResult`, `ActionRecord` to `core/types.py`
- Create `core/agent_loop.py` with `AgentLoop.__init__()`, `run()` skeleton, `_observe()`, `_act()`, `_verify()` stubs
- Write unit tests with mock decision function

**Task 2 [beta]: LLM decide_next_action** (~30min)
- Add `AGENT_LOOP_SYSTEM_PROMPT` and `build_agent_loop_prompt()` to `llm/prompts.py`
- Add `decide_next_action()` to `llm/claude.py`
- Add `parse_action_response()` to `llm/claude.py` (simpler than `parse_plan_response`)
- Write unit test for parse_action_response with various LLM outputs

**Task 3 [alpha]: AgentLoop implementation** (~45min)
- Implement `_observe()`: build observation dict from world model
- Implement `_decide()`: call LLM decide_next_action, parse response
- Implement `_act()`: look up skill, execute, apply effects
- Implement `_verify()`: detect sim vs hardware, run appropriate verification
- Implement `_should_verify()`: only verify after pick/place
- Wire `run()` loop with callbacks and GoalResult construction

**Task 4 [beta]: Agent integration + MCP** (~30min)
- Add `Agent.run_goal()` method (delegates to AgentLoop)
- Add `run_goal` MCP tool definition in `mcp/tools.py`
- Add handler in `handle_tool_call()`
- Add `agent.agent_loop` config section to `config/default.yaml`

**Task 5 [gamma]: Integration test** (~30min)
- Test `run_goal("pick all objects")` with MuJoCo sim
- Test `run_goal` via MCP tool call
- Test max_iterations termination
- Test parse failure fallback
