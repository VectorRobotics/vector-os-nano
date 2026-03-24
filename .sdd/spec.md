# v0.3.0 "Observable Symbolic Layer" Specification

## 1. Overview

Enrich the symbolic layer (SkillResult, StepTrace, WorldModel, MCP tools, @skill schema) so that both the internal LLM retry loop and external MCP agents receive structured, actionable diagnostics from every skill execution -- turning "IK failed" into a machine-parseable diagnostic with positions, world state, and recovery hints.

## 2. Background & Motivation

Vector OS Nano's neural-symbolic pipeline has two consumers of skill execution results:

1. **Internal Agent pipeline** -- the LLM planner's ADAPT stage (agent.py L517-527) receives failure context as a flat string and re-plans. It has no structured data to reason about why a step failed.
2. **External MCP agents** (Claude Code, GPT, etc.) -- receive flattened text from `_format_execution_result()` (tools.py L425-469). An external agent that sees `"pick(failed: IK failed for pre-grasp)"` cannot determine whether to re-detect, adjust coordinates, or give up.

Concurrently, the world model has correctness bugs that corrupt multi-step plans:
- `pick.py` clears ALL objects after pick (L326-328), destroying state needed for subsequent picks in multi-object plans.
- `detect.py` creates new `object_id`s on every call (L97: `f"{safe_label}_{idx}"`), never merging with existing objects, causing ID drift.

The `@skill` decorator's `parameters` dict carries type/description but no enum constraints, no source annotations, and no failure mode declarations -- meaning the JSON schema for MCP tools is incomplete, and the LLM prompt builder hardcodes constraint knowledge.

### Impact of Doing Nothing

- Multi-object plans ("pick all objects") fail silently after first pick because world model is cleared.
- External agents retry blindly without knowing whether failure is spatial (IK unreachable), perceptual (object not found), or mechanical (arm move failed).
- LLM occasionally generates invalid parameter values (e.g. `location: "left side"` instead of `"left"`) because schema lacks enum constraints.

## 3. Goals

### MUST (P0)

- **M1**: Every skill returns `result_data` on BOTH success AND failure, with a structured diagnostic payload including diagnosis code, relevant positions, and actionable hints.
- **M2**: `StepTrace` carries `result_data: dict` from the corresponding `SkillResult`, preserving per-step diagnostics through the execution trace.
- **M3**: MCP tool responses (`_format_execution_result`) return structured JSON with per-step trace, diagnostics, world model snapshot, and robot state -- while remaining backward compatible (text fallback for non-JSON consumers).
- **M4**: `pick.py` removes ONLY the picked object from world model (not all objects). `detect.py` merges detections with existing objects by label match instead of always creating new IDs.

### SHOULD (P1)

- **S1**: `@skill` decorator and `parameters` dict support `enum`, `source`, and `failure_modes` annotations. `SkillRegistry.to_schemas()` emits these in the JSON schema.
- **S2**: `build_planning_prompt()` dynamically injects enum constraints and world model object labels into the LLM prompt (replacing hardcoded PLACE LOCATIONS block).

### MAY (P2)

- **Y1**: `PlanValidator` class validates LLM-generated plans before execution: skill name existence, parameter type/enum conformance, dependency graph acyclicity, precondition satisfiability against world model.
- **Y2**: `PlanValidator` auto-repairs common LLM errors: fuzzy skill name match via alias map, fill missing defaults, normalize param names.

### FUTURE (P3, not in v0.3.0)

- **F1**: Symbolic-aware prompt builder that injects gripper state, available objects, and spatial context into LLM prompts dynamically.
- **F2**: Confidence-weighted object resolution (prefer recently-seen, high-confidence objects).

## 4. Non-Goals

- No changes to hardware interfaces (arm.py, gripper.py, serial_bus.py).
- No changes to perception pipeline internals (VLM, tracker, pointcloud).
- No changes to LLM provider implementations (claude.py, openai_compat.py, local.py).
- No new ROS2 nodes, topics, services, or actions.
- No new external dependencies (stdlib + existing deps only).
- No changes to CLI or web entry points (they consume ExecutionResult which gains fields but stays backward compatible).

## 5. User Scenarios

### Scenario 1: External Agent Self-Correction via MCP

- **Actor**: Claude Code connected via MCP
- **Trigger**: `call_tool("pick", {"object_label": "mug"})`
- **Current Behavior**: Returns `"pick(failed: IK failed for pre-grasp)"` -- agent retries blindly.
- **New Behavior**: Returns JSON:
  ```json
  {
    "success": false,
    "status": "failed",
    "steps": [
      {
        "step_id": "s1", "skill_name": "scan", "status": "success",
        "duration_sec": 3.1, "result_data": {"joint_values": [...]}
      },
      {
        "step_id": "s2", "skill_name": "detect", "status": "success",
        "duration_sec": 1.2, "result_data": {"objects": [{"object_id": "mug_0", "label": "mug", "position_cm": [25.1, 8.3, 1.2]}], "count": 1}
      },
      {
        "step_id": "s3", "skill_name": "pick", "status": "execution_failed",
        "duration_sec": 0.8,
        "result_data": {
          "diagnosis": "ik_unreachable",
          "target_base_cm": [25.1, 8.3, 11.2],
          "workspace_bounds_cm": [5, 35],
          "distance_cm": 26.4,
          "hint": "Object is near workspace boundary. Try detect with different query to get a closer grasp point, or move to a scan position with better view."
        }
      }
    ],
    "world_state": {
      "objects": [{"object_id": "mug_0", "label": "mug", "x": 0.251, "y": 0.083, "z": 0.012, "state": "on_table"}],
      "robot": {"gripper_state": "open", "held_object": null}
    },
    "failure_reason": "IK failed for pre-grasp"
  }
  ```
- **Success Criteria**: External agent can parse `diagnosis` field, decide to call `detect` with adjusted parameters or `scan` before retrying, without human intervention.

### Scenario 2: Multi-Object Pick-and-Place

- **Actor**: User says "pick all objects and put them on the left"
- **Trigger**: LLM generates plan: scan, detect, pick(banana, hold), place(left), pick(mug, hold), place(left), pick(bottle, hold), place(left), home
- **Current Behavior**: First pick clears ALL objects from world model. Second pick's detect creates new IDs (e.g. `mug_1` instead of `mug_0`). If LLM plan references `mug_0`, world model lookup fails.
- **New Behavior**: First pick removes only `banana_0`. World model still has `mug_0`, `bottle_0`. Second detect merges with existing objects (updates position, keeps `mug_0` ID). Plan references remain valid.
- **Success Criteria**: 3-object pick-and-place plan completes without world model ID mismatches.

### Scenario 3: LLM Plan Validation (P2)

- **Actor**: Internal agent pipeline
- **Trigger**: LLM generates plan with `skill_name: "pickup"` (typo) and `location: "left side"` (invalid enum)
- **Current Behavior**: Executor fails at runtime with "Skill not found: pickup".
- **New Behavior**: PlanValidator catches both errors before execution. Auto-repairs `"pickup"` to `"pick"` via alias map, normalizes `"left side"` to `"left"`. Plan executes successfully.
- **Success Criteria**: PlanValidator repairs at least: skill name typos/aliases, enum value normalization, missing default parameters.

## 6. Technical Constraints

- **Runtime**: Python 3.10+, no new pip dependencies.
- **Backward Compatibility**: All existing tests must pass unchanged. `SkillResult`, `StepTrace`, `ExecutionResult` gain new optional fields but existing code that constructs them without the new fields must still work.
- **Immutability**: All types remain `frozen=True` dataclasses.
- **Performance**: No measurable impact on skill execution latency (diagnostics are constructed from data already computed).
- **LLM-agnostic**: All changes work with any LLM provider. No Claude-specific, GPT-specific, or Llama-specific code.

## 7. Interface Definitions

### 7.1 Modified Types (core/types.py)

#### SkillResult (unchanged signature, enriched contract)

```python
@dataclass(frozen=True)
class SkillResult:
    success: bool
    result_data: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
```

No schema change. The CONTRACT changes: every skill MUST populate `result_data` on both success and failure paths. Failure `result_data` MUST include:

| Key | Type | Description |
|-----|------|-------------|
| `diagnosis` | `str` | Machine-readable diagnosis code (see Diagnosis Codes below) |
| `hint` | `str` | Human/LLM-readable recovery suggestion |

Additional keys are skill-specific (documented per skill in Section 8).

#### StepTrace (new field: result_data)

```python
@dataclass(frozen=True)
class StepTrace:
    step_id: str
    skill_name: str
    status: str
    duration_sec: float = 0.0
    error: str = ""
    result_data: dict[str, Any] = field(default_factory=dict)  # NEW
```

`result_data` is copied from `SkillResult.result_data` by the executor after each step. Existing code that constructs `StepTrace` without `result_data` continues to work (default empty dict).

#### ExecutionResult (unchanged)

No changes needed. The `trace: list[StepTrace]` field already carries per-step data; enriching `StepTrace` with `result_data` is sufficient.

### 7.2 Diagnosis Codes

Standardized string codes returned in `SkillResult.result_data["diagnosis"]`:

| Code | Meaning | Typical Skills |
|------|---------|---------------|
| `ok` | Skill succeeded | all |
| `no_arm` | No arm hardware connected | pick, place, scan, home |
| `no_perception` | No perception backend available | detect |
| `no_detections` | VLM returned zero detections for query | detect, pick |
| `track_failed` | Tracker failed to initialize or update | detect, pick |
| `no_3d_samples` | Depth data unavailable (zero valid samples) | pick |
| `out_of_workspace` | Target position outside workspace bounds | pick |
| `ik_unreachable` | IK solver returned no solution | pick, place |
| `move_failed` | Arm move_joints returned failure | pick, place, scan, home |
| `calibration_error` | Camera-to-base transform failed | detect, pick |
| `object_not_found` | Target object not in world model | pick |
| `gripper_error` | Gripper operation failed | pick, place |

Skills may define additional codes; the above are the mandatory base set.

### 7.3 Modified Executor (core/executor.py)

The executor's step loop changes at one point: after calling `skill.execute()`, it copies `SkillResult.result_data` into the `StepTrace`.

Current (L198-203):
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="success",
    duration_sec=duration,
))
```

New:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="success",
    duration_sec=duration,
    result_data=dict(skill_result.result_data),
))
```

This applies to ALL trace append points (success, precondition_failed, execution_failed, postcondition_failed). For failure cases where `skill_result` exists, copy its `result_data`. For failure cases where no `skill_result` exists (e.g., skill not found, precondition check), set `result_data` to `{"diagnosis": "<appropriate_code>"}`.

### 7.4 Modified MCP Tool Response (mcp/tools.py)

`_format_execution_result()` gains a new code path:

```python
def _format_execution_result(instruction: str, result: Any) -> str:
    """Format ExecutionResult as structured JSON for MCP consumers.

    Returns JSON string with:
    - success, status, failure_reason
    - steps: list of {step_id, skill_name, status, duration_sec, result_data}
    - world_state: current world model snapshot
    - robot_state: current robot state

    Falls back to plain text for non-ExecutionResult inputs.
    """
```

The function returns a JSON string (not a dict -- MCP tool responses are text). External agents parse the JSON to extract diagnostics.

### 7.5 Modified Skills

#### pick.py Changes

1. **Failure diagnostics**: Every `return SkillResult(success=False, ...)` also populates `result_data` with `diagnosis`, `hint`, and relevant positional data.

2. **World model fix**: Replace L326-328:
   ```python
   # CURRENT (broken): clears ALL objects
   for obj in list(context.world_model.get_objects()):
       context.world_model.remove_object(obj.object_id)
   ```
   With:
   ```python
   # NEW: remove only the picked object
   picked_id = params.get("object_id")
   if not picked_id:
       label = params.get("object_label", "")
       matches = context.world_model.get_objects_by_label(label)
       if matches:
           picked_id = matches[0].object_id
   if picked_id:
       context.world_model.remove_object(picked_id)
   ```

3. **Success diagnostics**: The success path already returns `result_data` with `position_cm`. Add `diagnosis: "ok"`.

4. **Specific failure result_data by failure point**:

| Failure Point | diagnosis | Extra Keys |
|--------------|-----------|------------|
| No arm | `no_arm` | -- |
| Cannot locate target | `object_not_found` | `query`, `world_model_objects` |
| Perception no detections | `no_detections` | `query` |
| Perception no 3D samples | `no_3d_samples` | `query`, `sample_count` |
| Out of workspace | `out_of_workspace` | `target_base_cm`, `distance_cm`, `workspace_bounds_cm` |
| IK failed pre-grasp | `ik_unreachable` | `target_base_cm`, `pre_grasp_cm`, `current_joints` |
| IK failed grasp | `ik_unreachable` | `target_base_cm`, `current_joints` |
| Move failed | `move_failed` | `phase` ("pre-grasp" / "descent" / "lift" / "home") |
| Max retries exhausted | (last attempt's diagnosis) | `attempts`, `last_diagnosis` |

#### place.py Changes

Same pattern as pick: populate `result_data` with `diagnosis` and positional data on every failure path. Specific additions:

| Failure Point | diagnosis | Extra Keys |
|--------------|-----------|------------|
| No arm | `no_arm` | -- |
| IK failed above | `ik_unreachable` | `target_cm`, `above_target_cm` |
| IK failed place | `ik_unreachable` | `target_cm` |
| Move failed | `move_failed` | `phase` ("approach" / "descend" / "lift") |
| Success | `ok` | `placed_at` (already present) |

#### detect.py Changes

1. **Merge with existing objects**: Replace the ID generation logic (L96-97):
   ```python
   # CURRENT: always creates new IDs
   obj_id = f"{safe_label}_{idx}"
   ```
   With a merge strategy:
   ```python
   # NEW: merge with existing world model objects by label
   existing = context.world_model.get_objects_by_label(label)
   if existing:
       obj_id = existing[0].object_id  # reuse existing ID
   else:
       # Generate new ID using a counter that avoids collisions
       existing_ids = {o.object_id for o in context.world_model.get_objects()}
       obj_id = _generate_unique_id(safe_label, existing_ids)
   ```

2. **Failure diagnostics**: Add `diagnosis` to all failure paths.

3. **Success diagnostics**: Already returns `result_data` with `objects` and `count`. Add `diagnosis: "ok"` and `merged_count: N` (number of objects that were updated rather than created).

#### scan.py, home.py, gripper.py, wave.py Changes

Add `diagnosis` code to `result_data` on all return paths. These are simple skills with few failure modes:

- `scan.py`: `no_arm`, `move_failed`, `ok`
- `home.py`: `no_arm`, `move_failed`, `ok`
- `gripper.py`: `no_arm` (mapped from no gripper), `ok`
- `wave.py`: `no_arm`, `ok`

### 7.6 @skill Schema Enhancement (P1: S1)

Extend the `parameters` dict format to support:

```python
parameters: dict = {
    "mode": {
        "type": "string",
        "enum": ["drop", "hold"],
        "default": "drop",
        "description": "...",
    },
    "location": {
        "type": "string",
        "enum": ["front", "front_left", "front_right", "center",
                 "left", "right", "back", "back_left", "back_right"],
        "default": "front",
        "description": "...",
        "source": "static",  # NEW: where valid values come from
    },
    "object_label": {
        "type": "string",
        "description": "...",
        "source": "world_model.objects.label",  # NEW: dynamic source
    },
}
```

New optional keys in parameter definitions:

| Key | Type | Description |
|-----|------|-------------|
| `enum` | `list[str]` | Already partially used; formalize across all skills |
| `source` | `str` | Where valid values come from: `"static"`, `"world_model.objects.label"`, `"world_model.objects.object_id"` |
| `failure_modes` | `list[str]` | Diagnosis codes this skill may return |

Add a `failure_modes` class attribute to the Skill protocol:

```python
@runtime_checkable
class Skill(Protocol):
    name: str
    description: str
    parameters: dict
    preconditions: list[str]
    postconditions: list[str]
    effects: dict
    failure_modes: list[str]  # NEW: diagnosis codes this skill may produce

    def execute(self, params: dict, context: "SkillContext") -> SkillResult: ...
```

`SkillRegistry.to_schemas()` emits `failure_modes` in the schema output. `skill_schema_to_mcp_tool()` in tools.py already passes through `enum` -- no change needed for MCP tool schema generation.

### 7.7 Dynamic Prompt Builder (P1: S2)

Modify `build_planning_prompt()` in `llm/prompts.py` to:

1. Extract enum values from skill schemas and inject them into the prompt as constraints.
2. Extract object labels from `world_state["objects"]` and inject them as `AVAILABLE OBJECTS: banana, mug, bottle`.
3. Extract robot gripper state from `world_state["robot"]` and inject context like `GRIPPER STATE: holding banana` or `GRIPPER STATE: empty`.

This replaces the hardcoded `PLACE LOCATIONS` block (L82-90) with dynamically generated constraint text.

### 7.8 PlanValidator (P2: Y1, Y2)

New file: `core/plan_validator.py`

```python
class PlanValidator:
    """Validate and auto-repair LLM-generated task plans.

    Pure symbolic validation -- no LLM calls. Runs between plan generation
    and execution in the agent pipeline.
    """

    def __init__(self, skill_registry: SkillRegistry, world_model: WorldModel) -> None: ...

    def validate(self, plan: TaskPlan) -> ValidationResult: ...

    def validate_and_repair(self, plan: TaskPlan) -> tuple[TaskPlan, list[Repair]]: ...
```

```python
@dataclass(frozen=True)
class ValidationError:
    step_id: str
    field: str          # "skill_name", "parameters.location", etc.
    code: str           # "unknown_skill", "invalid_enum", "missing_required", etc.
    message: str
    suggestion: str     # What to change

@dataclass(frozen=True)
class Repair:
    step_id: str
    field: str
    old_value: Any
    new_value: Any
    reason: str

@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
```

Validation checks:
1. **Skill existence**: `skill_name` must exist in registry. Auto-repair: fuzzy match against aliases + skill names using substring match (not Levenshtein -- no new deps).
2. **Required parameters**: All parameters without `default` must be present.
3. **Enum conformance**: Parameter values with `enum` constraint must be in the enum list. Auto-repair: case-insensitive match, strip whitespace, common synonyms (e.g. "left side" -> "left").
4. **Type conformance**: Parameter values match declared type (string, float, int, bool).
5. **Dependency graph**: `depends_on` references must exist, graph must be acyclic.
6. **Precondition satisfiability**: Static check against current world model (e.g., `gripper_holding_any` when gripper is empty).

### 7.9 WorldModel.apply_skill_effects() Fix

Current L350-353 (pick effect clears all objects):
```python
if _skill == "pick":
    self._objects.clear()
    self.update_robot_state(held_object=None, gripper_state="open")
```

New behavior branches on mode:
```python
if _skill == "pick":
    mode = params.get("mode", "drop")
    picked_id = params.get("object_id")
    if not picked_id:
        label = params.get("object_label", "")
        matches = self.get_objects_by_label(label)
        if matches:
            picked_id = matches[0].object_id

    if mode == "hold":
        # Mark object as grasped, update robot
        if picked_id and picked_id in self._objects:
            old = self._objects[picked_id]
            self._objects[picked_id] = ObjectState(
                object_id=old.object_id, label=old.label,
                x=old.x, y=old.y, z=old.z,
                confidence=old.confidence, state="grasped",
                last_seen=time.time(), properties=old.properties,
            )
        self.update_robot_state(held_object=picked_id, gripper_state="holding")
    else:
        # Drop mode: remove picked object, gripper is empty
        if picked_id:
            self.remove_object(picked_id)
        self.update_robot_state(held_object=None, gripper_state="open")
```

## 8. File-by-File Change Summary

| File | Priority | Change Type | Description |
|------|----------|-------------|-------------|
| `core/types.py` | P0 | Modify | Add `result_data` field to `StepTrace` |
| `core/executor.py` | P0 | Modify | Copy `SkillResult.result_data` into `StepTrace` at all trace append points |
| `skills/pick.py` | P0 | Modify | Add diagnostics to all return paths; fix world model clear to remove only picked object |
| `skills/place.py` | P0 | Modify | Add diagnostics to all return paths |
| `skills/detect.py` | P0 | Modify | Add diagnostics; merge with existing world model objects by label |
| `skills/scan.py` | P0 | Modify | Add diagnosis codes to result_data |
| `skills/home.py` | P0 | Modify | Add diagnosis codes to result_data |
| `skills/gripper.py` | P0 | Modify | Add diagnosis codes to result_data |
| `skills/wave.py` | P0 | Modify | Add diagnosis codes to result_data |
| `mcp/tools.py` | P0 | Modify | `_format_execution_result()` returns structured JSON |
| `core/world_model.py` | P0 | Modify | Fix `apply_skill_effects()` for pick: mode-aware, remove only picked object |
| `core/skill.py` | P1 | Modify | Add `failure_modes` to Skill protocol; update `to_schemas()` |
| `llm/prompts.py` | P1 | Modify | Dynamic constraint injection in `build_planning_prompt()` |
| `core/plan_validator.py` | P2 | New file | PlanValidator + ValidationResult + Repair types |
| `core/agent.py` | P2 | Modify | Insert PlanValidator call between plan() and execute() |

## 9. Test Contracts

All tests use the naming convention `test_<module>_<behavior>.py` and live under `tests/unit/` or `tests/integration/`.

### 9.1 Unit Tests: SkillResult Diagnostics (P0)

File: `tests/unit/test_skill_diagnostics.py`

| Test | Assertion |
|------|-----------|
| `test_pick_failure_ik_returns_diagnosis` | `PickSkill.execute()` with IK failure returns `result_data["diagnosis"] == "ik_unreachable"` and `"target_base_cm"` key exists |
| `test_pick_failure_workspace_returns_diagnosis` | Out-of-workspace pick returns `diagnosis == "out_of_workspace"` and `"distance_cm"` key exists |
| `test_pick_failure_no_arm_returns_diagnosis` | Pick with no arm returns `diagnosis == "no_arm"` |
| `test_pick_failure_no_detections_returns_diagnosis` | Pick with perception that returns [] returns `diagnosis == "no_detections"` |
| `test_pick_success_returns_diagnosis_ok` | Successful pick returns `result_data["diagnosis"] == "ok"` |
| `test_place_failure_ik_returns_diagnosis` | PlaceSkill IK failure returns `diagnosis == "ik_unreachable"` with `"target_cm"` key |
| `test_place_success_returns_diagnosis_ok` | Successful place returns `diagnosis == "ok"` |
| `test_detect_failure_no_perception_returns_diagnosis` | Detect with no perception returns `diagnosis == "no_perception"` |
| `test_detect_success_returns_diagnosis_ok` | Successful detect returns `diagnosis == "ok"` and `"merged_count"` key exists |
| `test_scan_failure_returns_diagnosis` | Scan with arm move failure returns `diagnosis == "move_failed"` |
| `test_home_failure_returns_diagnosis` | Home with arm move failure returns `diagnosis == "move_failed"` |
| `test_all_skills_have_failure_modes_attr` | Every registered skill has `failure_modes: list[str]` attribute (P1) |

### 9.2 Unit Tests: StepTrace result_data (P0)

File: `tests/unit/test_step_trace_diagnostics.py`

| Test | Assertion |
|------|-----------|
| `test_executor_copies_result_data_on_success` | After successful step, `trace[-1].result_data` equals `SkillResult.result_data` |
| `test_executor_copies_result_data_on_failure` | After failed step, `trace[-1].result_data` contains `"diagnosis"` key |
| `test_executor_result_data_default_empty` | `StepTrace()` with no `result_data` arg has `result_data == {}` |
| `test_step_trace_serialization_with_result_data` | `StepTrace.to_dict()` includes `"result_data"` key; `StepTrace.from_dict()` round-trips |
| `test_executor_skill_not_found_has_diagnosis` | When skill not found, trace has `result_data == {"diagnosis": "skill_not_found"}` |
| `test_executor_precondition_failed_has_diagnosis` | Precondition failure trace has `result_data == {"diagnosis": "precondition_failed"}` |

### 9.3 Unit Tests: MCP Structured Response (P0)

File: `tests/unit/test_mcp_structured_response.py`

| Test | Assertion |
|------|-----------|
| `test_format_execution_result_returns_json` | `_format_execution_result()` returns valid JSON string for ExecutionResult input |
| `test_format_result_json_has_steps_with_result_data` | Parsed JSON has `"steps"` list; each step has `"result_data"` dict |
| `test_format_result_json_has_world_state` | Parsed JSON has `"world_state"` with `"objects"` and `"robot"` keys |
| `test_format_result_string_passthrough` | String input returns string unchanged (backward compat) |
| `test_format_result_json_has_failure_reason` | Failed execution JSON has `"failure_reason"` string |
| `test_format_result_json_success_case` | Successful execution JSON has `"success": true` and complete step trace |

### 9.4 Unit Tests: World Model Fixes (P0)

File: `tests/unit/test_world_model_consistency.py`

| Test | Assertion |
|------|-----------|
| `test_pick_drop_removes_only_picked_object` | After pick(mode=drop) of "banana", world model still has "mug" and "bottle" |
| `test_pick_hold_marks_object_grasped` | After pick(mode=hold) of "banana", `world_model.get_object("banana_0").state == "grasped"` and `robot.held_object == "banana_0"` |
| `test_pick_drop_clears_held_object` | After pick(mode=drop), `robot.held_object is None` |
| `test_detect_merges_existing_objects` | After two detect calls, object with same label keeps original `object_id` |
| `test_detect_creates_new_for_unknown` | Detect with new label creates new object_id |
| `test_detect_updates_position_on_merge` | Merged object has updated position from latest detection |
| `test_apply_skill_effects_pick_hold` | `apply_skill_effects("pick", {"mode": "hold", "object_label": "banana"}, result)` sets `held_object` to the banana's object_id |
| `test_apply_skill_effects_pick_drop` | `apply_skill_effects("pick", {"mode": "drop", "object_label": "banana"}, result)` removes banana, leaves others |
| `test_multi_pick_preserves_remaining` | Execute pick(banana), pick(mug) -- after first pick, mug still in world model |

### 9.5 Unit Tests: @skill Schema Enhancement (P1)

File: `tests/unit/test_skill_schema_enhanced.py`

| Test | Assertion |
|------|-----------|
| `test_pick_schema_has_mode_enum` | `PickSkill.parameters["mode"]["enum"] == ["drop", "hold"]` |
| `test_place_schema_has_location_enum` | `PlaceSkill.parameters["location"]["enum"]` matches `_LOCATION_MAP.keys()` |
| `test_schema_source_annotation` | `PickSkill.parameters["object_label"]["source"] == "world_model.objects.label"` |
| `test_to_schemas_includes_failure_modes` | `registry.to_schemas()` output includes `"failure_modes"` list for each skill |
| `test_mcp_tool_schema_has_enum` | `skill_schema_to_mcp_tool()` output `inputSchema.properties.mode.enum` is `["drop", "hold"]` |
| `test_skill_protocol_has_failure_modes` | `isinstance(PickSkill(), Skill)` still True with `failure_modes` attribute |

### 9.6 Unit Tests: Dynamic Prompt Builder (P1)

File: `tests/unit/test_dynamic_prompts.py`

| Test | Assertion |
|------|-----------|
| `test_planning_prompt_includes_enum_constraints` | `build_planning_prompt(schemas, world)` output contains `"drop", "hold"` from pick mode enum |
| `test_planning_prompt_includes_available_objects` | Prompt contains `"AVAILABLE OBJECTS:"` with labels from world state |
| `test_planning_prompt_includes_gripper_state` | When `robot.held_object == "banana"`, prompt contains `"GRIPPER: holding banana"` |
| `test_planning_prompt_empty_world` | Empty world state produces prompt with `"AVAILABLE OBJECTS: none detected"` |

### 9.7 Unit Tests: PlanValidator (P2)

File: `tests/unit/test_plan_validator.py`

| Test | Assertion |
|------|-----------|
| `test_valid_plan_passes` | Well-formed plan returns `ValidationResult(valid=True, errors=[])` |
| `test_unknown_skill_detected` | Plan with `skill_name="pickup"` returns error with `code=="unknown_skill"` |
| `test_unknown_skill_auto_repaired` | `validate_and_repair()` fixes `"pickup"` to `"pick"` |
| `test_invalid_enum_detected` | `location="left side"` returns error with `code=="invalid_enum"` |
| `test_invalid_enum_auto_repaired` | `"left side"` repaired to `"left"` |
| `test_missing_required_param_detected` | pick step with no `object_label` and no `object_id` returns warning |
| `test_circular_dependency_detected` | Plan with circular `depends_on` returns error with `code=="circular_dependency"` |
| `test_precondition_unsatisfiable` | place step with `gripper_holding_any` when gripper is empty returns error |
| `test_type_mismatch_detected` | `x="hello"` for float param returns error with `code=="type_mismatch"` |
| `test_repair_returns_modified_plan` | Repaired plan is a NEW TaskPlan (immutable -- original unchanged) |

### 9.8 Integration Tests (P0)

File: `tests/integration/test_observable_pipeline.py`

| Test | Assertion |
|------|-----------|
| `test_mcp_pick_failure_returns_structured_json` | Full MCP pick call on sim with unreachable object returns parseable JSON with diagnosis |
| `test_multi_pick_world_model_consistent` | 3-object sim scene: pick first object, world model retains other 2 |
| `test_detect_merge_preserves_ids` | Run detect twice in sim, second run reuses first run's object IDs |
| `test_executor_trace_carries_diagnostics` | Full plan execution trace has `result_data` on every step |

## 10. Migration & Rollout

### Phase 1: P0 (Enriched Diagnostics + World Model Fixes)
- Modify `StepTrace`, executor, all skills, `_format_execution_result()`, `apply_skill_effects()`
- All existing tests pass; new tests pass
- Ship as v0.3.0-alpha

### Phase 2: P1 (Schema Enhancement + Dynamic Prompts)
- Modify `@skill` decorator, `Skill` protocol, `to_schemas()`, `build_planning_prompt()`
- Ship as v0.3.0-beta

### Phase 3: P2 (PlanValidator)
- New file `core/plan_validator.py`, modify `agent.py` to call it
- Ship as v0.3.0-rc

### Release: v0.3.0
- All phases complete, 80%+ test coverage on new code
- Bump `version.py` to `"0.3.0"`

## 11. Open Questions

None. All design decisions are made. The spec is ready for approval.
