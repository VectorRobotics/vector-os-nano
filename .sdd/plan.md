# v0.3.0 "Observable Symbolic Layer" Implementation Plan

## 0. Document Info

- **Spec**: `.sdd/spec.md` (approved)
- **Target**: v0.3.0
- **Phases**: P0 (foundation) -> P1 (schema + prompts) -> P2 (PlanValidator)
- **Estimated tasks**: 18 (8 P0, 5 P1, 5 P2)
- **Parallel agents**: Alpha, Beta, Gamma

---

## 1. Data Flow: Before / After

### Before (v0.2.x)

```
skill.execute() -> SkillResult(success, result_data, error_message)
                              |
executor loop ------> StepTrace(step_id, skill_name, status, duration, error)
                              |        # result_data is LOST here
                              v
ExecutionResult.trace: [StepTrace, ...]
                              |
_format_execution_result() -> flat text: "pick(failed: IK failed)"
                              |
MCP consumer -----> no structured diagnostics, blind retry
```

### After (v0.3.0)

```
skill.execute() -> SkillResult(success, result_data={diagnosis, hint, ...}, error_message)
                              |
executor loop ------> StepTrace(step_id, skill_name, status, duration, error, result_data)
                              |        # result_data PRESERVED
                              v
ExecutionResult.trace: [StepTrace, ...]
                              |
_format_execution_result() -> JSON: {success, status, steps: [{result_data}], world_state, failure_reason}
                              |
MCP consumer -----> parse diagnosis, decide recovery strategy
```

### World Model Fix: Before / After

```
BEFORE: pick(banana) -> _objects.clear() -> mug_0, bottle_0 GONE
AFTER:  pick(banana,drop) -> remove_object("banana_0") -> mug_0, bottle_0 PRESERVED
        pick(banana,hold) -> banana_0.state="grasped", robot.held_object="banana_0"
```

### Detect Merge: Before / After

```
BEFORE: detect() -> obj_id = f"{safe_label}_{idx}" -> banana_0, then banana_0 again (duplicate)
AFTER:  detect() -> existing = world_model.get_objects_by_label(label)
                    if existing: reuse existing[0].object_id, update position
                    else: generate new unique ID
```

---

## 2. Phase 1 (P0): Diagnostics + World Model Fixes

### Dependencies

```
                    +-------------------+
                    | T1: StepTrace     |
                    | (types.py)        |
                    +--------+----------+
                             |
                    +--------v----------+
                    | T2: Executor      |
                    | (executor.py)     |
                    +-------------------+
                             |
         +-------------------+-------------------+
         |                                       |
+--------v----------+              +-------------v--------+
| T3: WorldModel    |              | T4-T8: Skill         |
| (world_model.py)  |              | diagnostics          |
+-------------------+              | (pick,place,detect,  |
                                   |  scan,home,gripper,  |
                                   |  wave)               |
                                   +----------+-----------+
                                              |
                                   +----------v-----------+
                                   | T9: MCP structured   |
                                   | response (tools.py)  |
                                   +----------------------+
```

T1+T2 are serial (T2 depends on T1). T3 is independent. T4-T8 are independent of each other but depend on T1 (the diagnosis contract). T9 depends on T1+T2.

### Task T1: Add result_data to StepTrace [alpha]

**File**: `vector_os_nano/core/types.py`

**Change 1**: Add `result_data` field to `StepTrace` (L303-334)

Current `StepTrace`:
```python
@dataclass(frozen=True)
class StepTrace:
    step_id: str
    skill_name: str
    status: str
    duration_sec: float = 0.0
    error: str = ""
```

After:
```python
@dataclass(frozen=True)
class StepTrace:
    step_id: str
    skill_name: str
    status: str
    duration_sec: float = 0.0
    error: str = ""
    result_data: dict[str, Any] = field(default_factory=dict)
```

**Change 2**: Update `to_dict()` (L317-324) to include `result_data`:
```python
def to_dict(self) -> dict[str, Any]:
    return {
        "step_id": self.step_id,
        "skill_name": self.skill_name,
        "status": self.status,
        "duration_sec": self.duration_sec,
        "error": self.error,
        "result_data": self.result_data,
    }
```

**Change 3**: Update `from_dict()` (L326-334) to parse `result_data`:
```python
@classmethod
def from_dict(cls, d: dict[str, Any]) -> StepTrace:
    return cls(
        step_id=str(d["step_id"]),
        skill_name=str(d["skill_name"]),
        status=str(d["status"]),
        duration_sec=float(d.get("duration_sec", 0.0)),
        error=str(d.get("error", "")),
        result_data=dict(d.get("result_data", {})),
    )
```

**Backward compat**: `result_data` has a `default_factory`, so all existing code constructing `StepTrace` without it continues to work.

**Test file**: `tests/unit/test_step_trace_diagnostics.py`
- `test_executor_result_data_default_empty`: `StepTrace(step_id="s1", skill_name="x", status="success").result_data == {}`
- `test_step_trace_serialization_with_result_data`: round-trip through `to_dict()`/`from_dict()`

**Estimated effort**: Small (< 30 min)

---

### Task T2: Executor copies result_data into StepTrace [alpha]

**File**: `vector_os_nano/core/executor.py`

**Depends on**: T1

There are 5 places where `StepTrace` is constructed in `execute()`. Each must be updated.

**Change 1**: Skill not found (L82-88). No `skill_result` available, so inject synthetic diagnosis:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="skill_not_found",
    duration_sec=time.monotonic() - step_start,
    error=reason,
    result_data={"diagnosis": "skill_not_found"},
))
```

**Change 2**: Precondition failed (L105-111). No `skill_result` available:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="precondition_failed",
    duration_sec=time.monotonic() - step_start,
    error=reason,
    result_data={"diagnosis": "precondition_failed", "predicate": pred},
))
```

**Change 3**: Skill raised exception (L128-134). No `skill_result` available:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="execution_failed",
    duration_sec=time.monotonic() - step_start,
    error=reason,
    result_data={"diagnosis": "exception", "exception_type": type(exc).__name__},
))
```

**Change 4**: Skill returned failure (L153-159). `skill_result` exists:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="execution_failed",
    duration_sec=duration,
    error=reason,
    result_data=dict(skill_result.result_data),
))
```

**Change 5**: Postcondition failed (L181-187). `skill_result` exists:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="postcondition_failed",
    duration_sec=duration,
    error=reason,
    result_data={**dict(skill_result.result_data), "diagnosis": "postcondition_failed", "predicate": pred},
))
```

**Change 6**: Success (L198-203). `skill_result` exists:
```python
trace.append(StepTrace(
    step_id=step.step_id,
    skill_name=step.skill_name,
    status="success",
    duration_sec=duration,
    result_data=dict(skill_result.result_data),
))
```

**Test file**: `tests/unit/test_step_trace_diagnostics.py`
- `test_executor_copies_result_data_on_success`
- `test_executor_copies_result_data_on_failure`
- `test_executor_skill_not_found_has_diagnosis`
- `test_executor_precondition_failed_has_diagnosis`

**Estimated effort**: Small (< 45 min)

---

### Task T3: WorldModel.apply_skill_effects() fix [beta]

**File**: `vector_os_nano/core/world_model.py`

**Independent of T1/T2** -- can run in parallel.

**Change**: Replace L350-353 (`_skill == "pick"` branch):

Current (L350-353):
```python
if _skill == "pick":
    self._objects.clear()
    self.update_robot_state(held_object=None, gripper_state="open")
```

New:
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
        if picked_id and picked_id in self._objects:
            old = self._objects[picked_id]
            self._objects[picked_id] = ObjectState(
                object_id=old.object_id,
                label=old.label,
                x=old.x, y=old.y, z=old.z,
                confidence=old.confidence,
                state="grasped",
                last_seen=time.time(),
                properties=old.properties,
            )
        self.update_robot_state(held_object=picked_id, gripper_state="holding")
    else:
        if picked_id:
            self.remove_object(picked_id)
        self.update_robot_state(held_object=None, gripper_state="open")
```

**Test file**: `tests/unit/test_world_model_consistency.py` (NEW)
- `test_pick_drop_removes_only_picked_object`
- `test_pick_hold_marks_object_grasped`
- `test_pick_drop_clears_held_object`
- `test_apply_skill_effects_pick_hold`
- `test_apply_skill_effects_pick_drop`
- `test_multi_pick_preserves_remaining`

**Estimated effort**: Medium (< 1 hr)

---

### Task T4: PickSkill world model fix + diagnostics [beta]

**File**: `vector_os_nano/skills/pick.py`

**Depends on**: T3 (world model fix), and the diagnosis contract from spec (no code dependency on T1).

**Change 1**: Fix world model clear in `_single_pick_attempt()`. Replace L325-328:

Current (L325-328):
```python
# Clear world model -- object has moved, stale data is dangerous
for obj in list(context.world_model.get_objects()):
    context.world_model.remove_object(obj.object_id)
logger.info("[PICK] World model cleared")
```

New:
```python
# Remove only the picked object -- other objects' positions are still valid
picked_id = params.get("object_id")
if not picked_id:
    label = params.get("object_label", "")
    matches = context.world_model.get_objects_by_label(label)
    if matches:
        picked_id = matches[0].object_id
if picked_id:
    context.world_model.remove_object(picked_id)
    logger.info("[PICK] Removed picked object %s from world model", picked_id)
else:
    logger.info("[PICK] No specific object to remove from world model")
```

**Change 2**: Add `diagnosis` to every return path. Each failure return becomes:

| Location (approx line) | Current `error_message` | New `result_data` additions |
|---|---|---|
| L132 (no arm) | `"No arm connected"` | `{"diagnosis": "no_arm"}` |
| L155-158 (max retries) | `"Pick failed after N attempts: ..."` | `{"diagnosis": last_diagnosis, "attempts": max_retries, "last_diagnosis": last_diagnosis}` |
| L191-193 (no target) | `"Cannot locate target object"` | `{"diagnosis": "object_not_found", "query": label, "world_model_objects": [o.label for o in context.world_model.get_objects()]}` |
| L221-227 (workspace) | `"Object at ... outside workspace"` | `{"diagnosis": "out_of_workspace", "target_base_cm": [...], "distance_cm": ..., "workspace_bounds_cm": [5, 35]}` |
| L243-244 (IK pre-grasp) | `"IK failed for pre-grasp"` | `{"diagnosis": "ik_unreachable", "target_base_cm": [...], "pre_grasp_cm": [...]}` |
| L257-258 (IK grasp) | `"IK failed for grasp position"` | `{"diagnosis": "ik_unreachable", "target_base_cm": [...]}` |
| L276-277 (pre-grasp move) | `"Pre-grasp move failed"` | `{"diagnosis": "move_failed", "phase": "pre-grasp"}` |
| L281-282 (descent move) | `"Descent to grasp failed"` | `{"diagnosis": "move_failed", "phase": "descent"}` |
| L300-301 (home move) | `"Return home after pick failed"` | `{"diagnosis": "move_failed", "phase": "home"}` |
| L334-342 (success) | (none) | Add `"diagnosis": "ok"` to existing `result_data` |

Also update `_sample_from_perception()` failure paths to propagate diagnosis info up:
| L434-435 (no detections) | perception `_sample_from_perception` returns None | Will be caught at L191 as `object_not_found` -- fine |
| L476-478 (no 3D samples) | perception `_sample_from_perception` returns None | Will be caught at L191 -- needs refinement: return a tuple or set an instance var |

**Implementation note**: For perception sub-failures, refactor `_get_target_base_pos` to return `(None, diagnosis_data)` tuple OR keep returning `None` and have the caller set a generic `object_not_found` diagnosis. The spec says `_get_target_base_pos` returns None; we keep that simple and set the diagnosis at the caller level based on available context.

To propagate specific perception failure info, change `_single_pick_attempt` to:
1. Check `_get_target_base_pos` returns None.
2. Then check if perception was attempted: if context.perception is not None, set `diagnosis = "no_detections"` or `"no_3d_samples"`. Otherwise `"object_not_found"`.
3. This avoids changing the return type of `_get_target_base_pos`.

Actually, a cleaner approach: store the last perception failure reason on `self` as a transient attribute (set/cleared within `_single_pick_attempt`). This keeps the return type clean.

```python
# At top of _single_pick_attempt:
self._last_perception_diag: str = "object_not_found"

# In _sample_from_perception, set self._last_perception_diag before returning None:
#   "no_detections" when detect() returns []
#   "track_failed" when track() fails
#   "no_3d_samples" when samples list is empty
#   "calibration_error" when camera_to_base fails

# At L191 (target is None):
diag = self._last_perception_diag
result_data = {"diagnosis": diag, "query": label_used}
# ... add world_model_objects list if diag == "object_not_found"
```

**Change 3**: Track last_diagnosis through retry loop (L142-158):
```python
last_error: str = "unknown error"
last_diagnosis: str = "unknown"
for attempt in range(1, max_retries + 1):
    result = self._single_pick_attempt(params, context)
    if result.success:
        return result
    last_error = result.error_message
    last_diagnosis = result.result_data.get("diagnosis", "unknown")
    ...

return SkillResult(
    success=False,
    error_message=f"Pick failed after {max_retries} attempts: {last_error}",
    result_data={
        "diagnosis": last_diagnosis,
        "attempts": max_retries,
        "last_diagnosis": last_diagnosis,
        "hint": "All retry attempts exhausted. Consider re-detecting or adjusting position.",
    },
)
```

**Test file**: `tests/unit/test_skill_diagnostics.py` (NEW)
- `test_pick_failure_ik_returns_diagnosis`
- `test_pick_failure_workspace_returns_diagnosis`
- `test_pick_failure_no_arm_returns_diagnosis`
- `test_pick_failure_no_detections_returns_diagnosis`
- `test_pick_success_returns_diagnosis_ok`

**Estimated effort**: Large (1.5-2 hrs) -- most complex skill, most failure paths

---

### Task T5: PlaceSkill diagnostics [gamma]

**File**: `vector_os_nano/skills/place.py`

**Independent of T4**. Only needs the diagnosis contract from spec.

Add `result_data` with `diagnosis` to every return path:

| Location | Current | New `result_data` |
|---|---|---|
| L114 (no arm) | `error_message="No arm connected"` | `{"diagnosis": "no_arm"}` |
| L153-156 (IK above) | `error_message="IK failed for above-place position"` | `{"diagnosis": "ik_unreachable", "target_cm": [tx*100, ty*100, tz*100], "above_target_cm": [above*100...]}` |
| L162 (move above failed) | `error_message="Move to above-place failed"` | `{"diagnosis": "move_failed", "phase": "approach"}` |
| L169-173 (IK place) | `error_message="IK failed for place position"` | `{"diagnosis": "ik_unreachable", "target_cm": [tx*100, ty*100, tz*100]}` |
| L179 (descend failed) | `error_message="Place descent failed"` | `{"diagnosis": "move_failed", "phase": "descend"}` |
| L189 (lift failed) | `error_message="Place lift failed"` | `{"diagnosis": "move_failed", "phase": "lift"}` |
| L199-202 (success) | `result_data={"placed_at": [...]}` | Add `"diagnosis": "ok"` |

**Test file**: `tests/unit/test_skill_diagnostics.py`
- `test_place_failure_ik_returns_diagnosis`
- `test_place_success_returns_diagnosis_ok`

**Estimated effort**: Small (< 45 min)

---

### Task T6: DetectSkill merge + diagnostics [gamma]

**File**: `vector_os_nano/skills/detect.py`

**Independent of T4/T5**. This is a correctness fix (merge logic) plus diagnostics.

**Change 1**: Replace ID generation logic (L90-97). Current:
```python
for idx, det in enumerate(detections):
    label = det.label
    if label.lower() in ("all objects", "all", "objects", "everything"):
        label = f"object_{idx}"
    safe_label = label.replace(" ", "_").lower()
    obj_id = f"{safe_label}_{idx}"
```

New merge strategy:
```python
merged_count = 0
for idx, det in enumerate(detections):
    label = det.label
    if label.lower() in ("all objects", "all", "objects", "everything"):
        label = f"object_{idx}"
    safe_label = label.replace(" ", "_").lower()

    # Merge with existing world model objects by label
    existing = context.world_model.get_objects_by_label(label)
    if existing:
        obj_id = existing[0].object_id
        merged_count += 1
    else:
        # Generate unique ID avoiding collisions
        existing_ids = {o.object_id for o in context.world_model.get_objects()}
        counter = 0
        obj_id = f"{safe_label}_{counter}"
        while obj_id in existing_ids:
            counter += 1
            obj_id = f"{safe_label}_{counter}"
```

**Change 2**: Add `diagnosis` to all return paths:

| Location | Current | New `result_data` |
|---|---|---|
| L52-55 (no perception) | `error_message="No perception..."` | `{"diagnosis": "no_perception"}` |
| L65-68 (perception error) | `error_message=f"Perception error: {exc}"` | `{"diagnosis": "no_perception", "error_detail": str(exc)}` |
| L72-75 (no detections) | `result_data={"objects": [], "count": 0}` | Add `"diagnosis": "no_detections", "query": query` |
| L83-84 (tracking failed) | continues without 3D | Add `"track_warning": str(exc)` to final result_data |
| L156-162 (success) | `result_data={"objects": ..., "count": ...}` | Add `"diagnosis": "ok", "merged_count": merged_count` |

**Test file**: `tests/unit/test_world_model_consistency.py`
- `test_detect_merges_existing_objects`
- `test_detect_creates_new_for_unknown`
- `test_detect_updates_position_on_merge`

And `tests/unit/test_skill_diagnostics.py`:
- `test_detect_failure_no_perception_returns_diagnosis`
- `test_detect_success_returns_diagnosis_ok`

**Estimated effort**: Medium (< 1 hr)

---

### Task T7: Simple skills diagnostics (scan, home, gripper, wave) [alpha]

**Files**: `vector_os_nano/skills/scan.py`, `home.py`, `gripper.py`, `wave.py`

**Independent** -- can run parallel with T4-T6.

**scan.py** changes:
- L65: Add `result_data={"diagnosis": "no_arm"}` to no-arm failure
- L72: Add `result_data={"diagnosis": "move_failed"}` to move failure
- L75-78: Add `"diagnosis": "ok"` to success result_data

**home.py** changes:
- L62: Add `result_data={"diagnosis": "no_arm"}` to no-arm failure
- L69: Add `result_data={"diagnosis": "move_failed"}` to move failure
- L76-79: Add `"diagnosis": "ok"` to success result_data

**gripper.py** changes:
- GripperOpenSkill L31: Add `result_data={"diagnosis": "no_arm"}` to no-gripper failure
- GripperOpenSkill L34: Add `result_data={"diagnosis": "ok"}` to success
- GripperCloseSkill L56: Same pattern
- GripperCloseSkill L60: Same pattern

**wave.py** changes:
- L45: Add `result_data={"diagnosis": "no_arm"}` to no-arm failure
- L50: Add `result_data={"diagnosis": "move_failed"}` to raise failure
- L80: Add `result_data={"diagnosis": "ok"}` to success

**Test file**: `tests/unit/test_skill_diagnostics.py`
- `test_scan_failure_returns_diagnosis`
- `test_home_failure_returns_diagnosis`

**Estimated effort**: Small (< 45 min)

---

### Task T8: MCP structured JSON response [gamma]

**File**: `vector_os_nano/mcp/tools.py`

**Depends on**: T1 (StepTrace.result_data exists), T2 (executor populates it)

**Change**: Rewrite `_format_execution_result()` (L425-469):

```python
def _format_execution_result(instruction: str, result: Any) -> str:
    """Format ExecutionResult as structured JSON for MCP consumers.

    Returns JSON string with per-step diagnostics, world state, and robot state.
    Falls back to plain text for non-ExecutionResult inputs (backward compat).
    """
    if isinstance(result, str):
        return result

    from vector_os_nano.core.types import ExecutionResult
    import json

    if not isinstance(result, ExecutionResult):
        return str(result)

    # Build structured response
    steps = []
    for t in result.trace:
        steps.append({
            "step_id": t.step_id,
            "skill_name": t.skill_name,
            "status": t.status,
            "duration_sec": round(t.duration_sec, 3),
            "result_data": t.result_data,
        })
        if t.error:
            steps[-1]["error"] = t.error

    response: dict[str, Any] = {
        "success": result.success,
        "status": result.status,
        "steps_completed": result.steps_completed,
        "steps_total": result.steps_total,
        "steps": steps,
    }

    if result.failure_reason:
        response["failure_reason"] = result.failure_reason
    if result.message:
        response["message"] = result.message

    # Include world state snapshot if agent is accessible
    # (The world_model_diff is already on ExecutionResult)
    if result.world_model_diff:
        response["world_state"] = result.world_model_diff

    total_duration = sum(t.duration_sec for t in result.trace)
    response["total_duration_sec"] = round(total_duration, 3)

    return json.dumps(response, ensure_ascii=False, indent=2)
```

**Note on world_state**: The current `ExecutionResult.world_model_diff` is populated by `agent.py` only on success (L503, L512). For the full world snapshot, we need the `_format_execution_result` to receive the world model. Two options:
1. Pass world model to `_format_execution_result` (signature change).
2. Populate `world_model_diff` on the ExecutionResult in agent.py for all cases (success + failure).

Option 2 is cleaner -- it keeps the function signature simpler and the data flows through the existing type. We'll update `agent.py`'s `_handle_task` and `execute_skill` to always set `world_model_diff` on the returned `ExecutionResult`. This is a small addition to those methods.

**Additional change in `agent.py`**: After executor returns (both success and failure), snapshot world model:
```python
world_snapshot = self._world_model.to_dict()
# Then include world_model_diff=world_snapshot in the ExecutionResult construction
```

This applies at:
- `_handle_task()` success path (L505-513) -- already has `world_model_diff`
- `_handle_task()` failure paths (L543-552, L556-565) -- add `world_model_diff`
- `_execute_auto_steps()` (L365-370) -- add world_model_diff to result
- `execute_skill()` (L737-747) -- add world_model_diff to result

For `_execute_matched()` (L277-300), the result is constructed manually without going through executor trace, so we add world_model_diff there too.

Actually, the simplest approach: make `handle_tool_call` pass the world model snapshot alongside the result. But that couples MCP to agent internals. Better: add a helper in `_format_execution_result` that accepts an optional `world_state` dict parameter.

Final decision: Add optional `world_state` parameter to `_format_execution_result()`:

```python
def _format_execution_result(instruction: str, result: Any, world_state: dict | None = None) -> str:
```

And update `handle_tool_call` to pass `agent.world.to_dict()` as the world_state. This is backward-compatible since the parameter is optional.

**Test file**: `tests/unit/test_mcp_structured_response.py` (NEW)
- `test_format_execution_result_returns_json`
- `test_format_result_json_has_steps_with_result_data`
- `test_format_result_json_has_world_state`
- `test_format_result_string_passthrough`
- `test_format_result_json_has_failure_reason`
- `test_format_result_json_success_case`

**Estimated effort**: Medium (< 1 hr)

---

### P0 Execution Waves

```
Wave 1 (parallel):
  Alpha: T1 (StepTrace) + T2 (Executor) [serial, same agent]
  Beta:  T3 (WorldModel fix)
  Gamma: T5 (PlaceSkill diagnostics)

Wave 2 (parallel, after Wave 1):
  Alpha: T7 (simple skills diagnostics)
  Beta:  T4 (PickSkill fix + diagnostics)
  Gamma: T6 (DetectSkill merge + diagnostics)

Wave 3 (after Wave 2):
  Gamma: T8 (MCP structured response)
  Alpha: Run full test suite, fix integration issues
  Beta:  (available for P1 start)
```

**P0 total tasks**: 8 (T1-T8)

---

## 3. Phase 2 (P1): Schema Enhancement + Dynamic Prompts

### Dependencies

```
+-------------------+     +-------------------+
| T9: Skill proto   |     | T11: Skill param  |
| + to_schemas()    |     | annotations       |
| (skill.py)        |     | (all skill files)  |
+--------+----------+     +--------+----------+
         |                          |
         +----------+---------------+
                    |
           +--------v----------+
           | T12: Dynamic      |
           | prompts           |
           | (prompts.py)      |
           +-------------------+
                    |
           +--------v----------+
           | T13: MCP schema   |
           | pass-through      |
           | (tools.py)        |
           +-------------------+
```

### Task T9: Skill protocol + to_schemas() [alpha]

**File**: `vector_os_nano/core/skill.py`

**Change 1**: Add `failure_modes` to Skill protocol (L76-86):
```python
@runtime_checkable
class Skill(Protocol):
    name: str
    description: str
    parameters: dict
    preconditions: list[str]
    postconditions: list[str]
    effects: dict
    failure_modes: list[str]  # NEW

    def execute(self, params: dict, context: "SkillContext") -> SkillResult: ...
```

**Change 2**: Update `to_schemas()` (L198-220) to include `failure_modes`:
```python
def to_schemas(self) -> list[dict]:
    schemas: list[dict] = []
    for s in self._skills.values():
        aliases = getattr(s, '__skill_aliases__', [])
        auto = getattr(s, '__skill_auto_steps__', [])
        failure_modes = getattr(s, 'failure_modes', [])
        schema = {
            "name": s.name,
            "description": s.description,
            "parameters": s.parameters,
            "preconditions": list(s.preconditions),
            "postconditions": list(s.postconditions),
            "effects": dict(s.effects),
        }
        if aliases:
            schema["aliases"] = aliases
        if auto:
            schema["auto_steps"] = auto
        if failure_modes:
            schema["failure_modes"] = failure_modes
        schemas.append(schema)
    return schemas
```

**Backward compat**: `runtime_checkable` Protocol with `failure_modes` means existing skills without it won't pass `isinstance()` check. However, the codebase doesn't use `isinstance(x, Skill)` checks in production code -- it's just a typing hint. The `to_schemas()` uses `getattr` with default, so it's safe. The only risk is if any test checks `isinstance`. We'll verify and add `failure_modes = []` as a default to all existing skills in T11.

**Test file**: `tests/unit/test_skill_schema_enhanced.py` (NEW)
- `test_to_schemas_includes_failure_modes`
- `test_skill_protocol_has_failure_modes`

**Estimated effort**: Small (< 30 min)

---

### Task T10: (MERGED INTO T11 -- see below)

---

### Task T11: Skill parameter annotations (enum, source, failure_modes) [beta]

**Files**: ALL skill files in `vector_os_nano/skills/`

Add `enum`, `source`, and `failure_modes` to each skill class.

**pick.py**:
```python
parameters: dict = {
    "object_id": {
        "type": "string",
        "required": False,
        "description": "ID of the object in the world model",
        "source": "world_model.objects.object_id",
    },
    "object_label": {
        "type": "string",
        "required": False,
        "description": "Label of the object to pick (e.g. 'mug', 'banana')",
        "source": "world_model.objects.label",
    },
    "mode": {
        "type": "string",
        "required": False,
        "default": "drop",
        "enum": ["drop", "hold"],
        "description": "'hold' = grasp and hold at home (for subsequent place), 'drop' = grasp and discard to side",
    },
}
failure_modes: list[str] = [
    "no_arm", "object_not_found", "no_detections", "no_3d_samples",
    "out_of_workspace", "ik_unreachable", "move_failed", "track_failed",
    "calibration_error",
]
```

**place.py**:
```python
parameters: dict = {
    "location": {
        "type": "string",
        "required": False,
        "default": "front",
        "enum": list(_LOCATION_MAP.keys()),
        "description": "Named position: front, front_left, front_right, center, left, right, back, back_left, back_right",
        "source": "static",
    },
    "x": { ... },  # unchanged
    "y": { ... },  # unchanged
    "z": { ... },  # unchanged
}
failure_modes: list[str] = ["no_arm", "ik_unreachable", "move_failed"]
```

**detect.py**:
```python
failure_modes: list[str] = ["no_perception", "no_detections", "track_failed", "calibration_error"]
```

**scan.py**:
```python
failure_modes: list[str] = ["no_arm", "move_failed"]
```

**home.py**:
```python
failure_modes: list[str] = ["no_arm", "move_failed"]
```

**gripper.py** (both classes):
```python
failure_modes: list[str] = ["no_arm"]
```

**wave.py**:
```python
failure_modes: list[str] = ["no_arm", "move_failed"]
```

**Test file**: `tests/unit/test_skill_schema_enhanced.py`
- `test_pick_schema_has_mode_enum`
- `test_place_schema_has_location_enum`
- `test_schema_source_annotation`
- `test_mcp_tool_schema_has_enum`
- `test_all_skills_have_failure_modes_attr`

**Estimated effort**: Medium (< 1 hr)

---

### Task T12: Dynamic prompt builder [alpha]

**File**: `vector_os_nano/llm/prompts.py`

**Depends on**: T9 (schema has enum), T11 (skills have enum values in params)

**Change**: Modify `build_planning_prompt()` (L165-175) to dynamically inject constraints instead of hardcoding PLACE LOCATIONS.

Replace the hardcoded `PLACE LOCATIONS` block (L82-91 of `PLANNING_SYSTEM_PROMPT`) with a `{constraints_block}` placeholder, then generate it dynamically.

New `PLANNING_SYSTEM_PROMPT` -- replace L82-91:
```
PLACE LOCATIONS (map user language to these values):
- "前面/前方/front" → "front"
...
```
With:
```
{constraints_block}
```

New `build_planning_prompt()`:
```python
def build_planning_prompt(
    skill_schemas: list[dict[str, Any]],
    world_state: dict[str, Any],
) -> str:
    skills_json = json.dumps(skill_schemas, indent=2, ensure_ascii=False)
    world_state_json = json.dumps(world_state, indent=2, ensure_ascii=False)

    # Build dynamic constraints block
    constraints_parts: list[str] = []

    # 1. Extract enum constraints from skill schemas
    for schema in skill_schemas:
        for param_name, param_def in schema.get("parameters", {}).items():
            if isinstance(param_def, dict) and "enum" in param_def:
                constraints_parts.append(
                    f"VALID VALUES for {schema['name']}.{param_name}: {', '.join(str(v) for v in param_def['enum'])}"
                )

    # 2. Extract available objects from world state
    objects = world_state.get("objects", [])
    if objects:
        labels = [o.get("label", "unknown") for o in objects]
        constraints_parts.append(f"AVAILABLE OBJECTS: {', '.join(labels)}")
    else:
        constraints_parts.append("AVAILABLE OBJECTS: none detected")

    # 3. Extract gripper state
    robot = world_state.get("robot", {})
    held = robot.get("held_object")
    if held:
        constraints_parts.append(f"GRIPPER: holding {held}")
    else:
        gripper_state = robot.get("gripper_state", "unknown")
        constraints_parts.append(f"GRIPPER: {gripper_state} (not holding anything)")

    constraints_block = "\n".join(constraints_parts)

    return PLANNING_SYSTEM_PROMPT.format(
        skills_json=skills_json,
        world_state_json=world_state_json,
        constraints_block=constraints_block,
    )
```

The hardcoded Chinese/English location mapping (L82-91) is REMOVED. The LLM will see `VALID VALUES for place.location: front, front_left, ...` instead, which is dynamically generated from the skill schema. The MULTI-OBJECT EXAMPLE and other fixed rules stay.

**Test file**: `tests/unit/test_dynamic_prompts.py` (NEW)
- `test_planning_prompt_includes_enum_constraints`
- `test_planning_prompt_includes_available_objects`
- `test_planning_prompt_includes_gripper_state`
- `test_planning_prompt_empty_world`

**Estimated effort**: Medium (< 1 hr)

---

### Task T13: MCP schema pass-through for enum [gamma]

**File**: `vector_os_nano/mcp/tools.py`

**Verify** that `skill_schema_to_mcp_tool()` already passes through `enum` values. Looking at L128-129:
```python
if "enum" in param_def:
    prop["enum"] = param_def["enum"]
```

This already works. No code change needed -- just verify with a test.

Also verify `failure_modes` is accessible from the MCP tool schema. The `skill_schema_to_mcp_tool` doesn't currently pass `failure_modes` into the tool definition. We should add it as a top-level field on the tool schema (outside `inputSchema`, since it's not a parameter).

```python
tool = {
    "name": schema["name"],
    "description": schema.get("description", ""),
    "inputSchema": input_schema,
}
if "failure_modes" in schema:
    tool["failure_modes"] = schema["failure_modes"]
return tool
```

**Test file**: `tests/unit/test_skill_schema_enhanced.py`
- `test_mcp_tool_schema_has_enum` (already listed)

**Estimated effort**: Small (< 20 min)

---

### P1 Execution Waves

```
Wave 4 (parallel):
  Alpha: T9 (Skill protocol + to_schemas)
  Beta:  T11 (Skill param annotations)
  Gamma: T13 (MCP schema verification)

Wave 5 (after Wave 4):
  Alpha: T12 (Dynamic prompts)
  Beta:  Run P1 test suite
  Gamma: (available for P2 start)
```

**P1 total tasks**: 4 (T9, T11, T12, T13)

---

## 4. Phase 3 (P2): PlanValidator

### Dependencies

```
+-------------------+     +-------------------+
| T14: Validation   |     | T15: Validator    |
| types             |     | core logic        |
| (plan_validator)  |     | (plan_validator)  |
+--------+----------+     +--------+----------+
         |                          |
         +----------+---------------+
                    |
           +--------v----------+
           | T16: Auto-repair  |
           | logic             |
           | (plan_validator)  |
           +--------+----------+
                    |
           +--------v----------+
           | T17: Agent        |
           | integration       |
           | (agent.py)        |
           +--------+----------+
                    |
           +--------v----------+
           | T18: Integration  |
           | tests             |
           +-------------------+
```

### Task T14: Validation types [alpha]

**File**: `vector_os_nano/core/plan_validator.py` (NEW)

Define the data types:

```python
"""Plan validation and auto-repair for LLM-generated task plans.

Pure symbolic validation -- no LLM calls. Runs between plan generation
and execution in the agent pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationError:
    """A single validation error or warning."""
    step_id: str
    field: str          # "skill_name", "parameters.location", etc.
    code: str           # "unknown_skill", "invalid_enum", "missing_required", etc.
    message: str
    suggestion: str     # What to change


@dataclass(frozen=True)
class Repair:
    """Record of an auto-repair applied to a plan."""
    step_id: str
    field: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    """Result of plan validation."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
```

**Test file**: `tests/unit/test_plan_validator.py` (NEW)
- `test_validation_result_default_valid`
- `test_validation_error_frozen`

**Estimated effort**: Small (< 20 min)

---

### Task T15: PlanValidator core validation [beta]

**File**: `vector_os_nano/core/plan_validator.py`

Add `PlanValidator` class with `validate()` method:

```python
class PlanValidator:
    """Validate LLM-generated task plans before execution."""

    def __init__(self, skill_registry: Any, world_model: Any) -> None:
        self._registry = skill_registry
        self._world_model = world_model
        # Build skill name set and alias map for fuzzy matching
        self._skill_names: set[str] = set(skill_registry.list_skills())
        self._alias_map: dict[str, str] = {}
        for name in self._skill_names:
            skill = skill_registry.get(name)
            if skill:
                for alias in getattr(skill, '__skill_aliases__', []):
                    self._alias_map[alias.lower()] = name
                self._alias_map[name.lower()] = name

    def validate(self, plan: TaskPlan) -> ValidationResult:
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        step_ids = {s.step_id for s in plan.steps}

        for step in plan.steps:
            # 1. Skill existence
            if step.skill_name not in self._skill_names:
                errors.append(ValidationError(
                    step_id=step.step_id,
                    field="skill_name",
                    code="unknown_skill",
                    message=f"Skill {step.skill_name!r} not found in registry",
                    suggestion=self._suggest_skill_name(step.skill_name),
                ))

            else:
                skill = self._registry.get(step.skill_name)
                if skill:
                    # 2. Required parameters
                    for pname, pdef in skill.parameters.items():
                        if isinstance(pdef, dict):
                            has_default = "default" in pdef
                            explicitly_optional = pdef.get("required") is False
                            if not has_default and not explicitly_optional:
                                if pname not in step.parameters:
                                    warnings.append(ValidationError(
                                        step_id=step.step_id,
                                        field=f"parameters.{pname}",
                                        code="missing_required",
                                        message=f"Required parameter {pname!r} missing",
                                        suggestion=f"Add {pname} to parameters",
                                    ))

                    # 3. Enum conformance
                    for pname, pvalue in step.parameters.items():
                        pdef = skill.parameters.get(pname, {})
                        if isinstance(pdef, dict) and "enum" in pdef:
                            if pvalue not in pdef["enum"]:
                                errors.append(ValidationError(
                                    step_id=step.step_id,
                                    field=f"parameters.{pname}",
                                    code="invalid_enum",
                                    message=f"Value {pvalue!r} not in enum {pdef['enum']}",
                                    suggestion=f"Use one of: {pdef['enum']}",
                                ))

                    # 4. Type conformance
                    for pname, pvalue in step.parameters.items():
                        pdef = skill.parameters.get(pname, {})
                        if isinstance(pdef, dict) and "type" in pdef:
                            expected = pdef["type"]
                            if not self._check_type(pvalue, expected):
                                errors.append(ValidationError(
                                    step_id=step.step_id,
                                    field=f"parameters.{pname}",
                                    code="type_mismatch",
                                    message=f"Expected type {expected}, got {type(pvalue).__name__}",
                                    suggestion=f"Convert value to {expected}",
                                ))

            # 5. Dependency graph
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(ValidationError(
                        step_id=step.step_id,
                        field="depends_on",
                        code="missing_dependency",
                        message=f"Dependency {dep!r} not found in plan",
                        suggestion="Remove or fix the dependency reference",
                    ))

        # 6. Cycle detection (if no missing deps)
        if not any(e.code == "missing_dependency" for e in errors):
            if self._has_cycle(plan.steps):
                errors.append(ValidationError(
                    step_id="plan",
                    field="depends_on",
                    code="circular_dependency",
                    message="Circular dependency detected in plan",
                    suggestion="Remove circular dependency",
                ))

        # 7. Precondition satisfiability
        for step in plan.steps:
            for pred in step.preconditions:
                if not self._world_model.check_predicate(pred):
                    warnings.append(ValidationError(
                        step_id=step.step_id,
                        field="preconditions",
                        code="precondition_unsatisfiable",
                        message=f"Precondition {pred!r} not satisfied by current world state",
                        suggestion="Ensure prior steps establish this condition",
                    ))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
```

Helper methods:
```python
def _suggest_skill_name(self, name: str) -> str:
    """Suggest closest skill name using substring matching."""
    name_lower = name.lower()
    # Check alias map first
    if name_lower in self._alias_map:
        return f"Did you mean {self._alias_map[name_lower]!r}?"
    # Substring match
    for known in self._skill_names:
        if name_lower in known.lower() or known.lower() in name_lower:
            return f"Did you mean {known!r}?"
    return "No close match found"

@staticmethod
def _check_type(value: Any, expected: str) -> bool:
    """Check if value matches expected type string."""
    type_map = {
        "string": str, "str": str,
        "float": (int, float), "number": (int, float),
        "int": int, "integer": int,
        "bool": bool, "boolean": bool,
    }
    expected_types = type_map.get(expected)
    if expected_types is None:
        return True  # Unknown type, pass
    return isinstance(value, expected_types)

@staticmethod
def _has_cycle(steps: list) -> bool:
    """Detect cycle in dependency graph using DFS."""
    adj: dict[str, list[str]] = {s.step_id: list(s.depends_on) for s in steps}
    visited: set[str] = set()
    in_stack: set[str] = set()

    def dfs(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for dep in adj.get(node, []):
            if dfs(dep):
                return True
        in_stack.remove(node)
        return False

    return any(dfs(s.step_id) for s in steps if s.step_id not in visited)
```

**Test file**: `tests/unit/test_plan_validator.py`
- `test_valid_plan_passes`
- `test_unknown_skill_detected`
- `test_invalid_enum_detected`
- `test_missing_required_param_detected`
- `test_circular_dependency_detected`
- `test_type_mismatch_detected`
- `test_precondition_unsatisfiable`

**Estimated effort**: Large (1.5-2 hrs)

---

### Task T16: PlanValidator auto-repair [gamma]

**File**: `vector_os_nano/core/plan_validator.py`

Add `validate_and_repair()` method:

```python
def validate_and_repair(self, plan: TaskPlan) -> tuple[TaskPlan, list[Repair]]:
    """Validate and auto-repair common LLM errors.

    Returns a NEW TaskPlan (original is immutable) and list of repairs applied.
    """
    repairs: list[Repair] = []
    new_steps: list[TaskStep] = []

    for step in plan.steps:
        skill_name = step.skill_name
        parameters = dict(step.parameters)

        # Repair 1: Skill name fuzzy match
        if skill_name not in self._skill_names:
            matched = self._fuzzy_match_skill(skill_name)
            if matched:
                repairs.append(Repair(
                    step_id=step.step_id,
                    field="skill_name",
                    old_value=skill_name,
                    new_value=matched,
                    reason=f"Unknown skill {skill_name!r} matched to {matched!r}",
                ))
                skill_name = matched

        # Repair 2: Enum normalization
        skill = self._registry.get(skill_name)
        if skill:
            for pname, pvalue in list(parameters.items()):
                pdef = skill.parameters.get(pname, {})
                if isinstance(pdef, dict) and "enum" in pdef:
                    if pvalue not in pdef["enum"]:
                        normalized = self._normalize_enum(pvalue, pdef["enum"])
                        if normalized:
                            repairs.append(Repair(
                                step_id=step.step_id,
                                field=f"parameters.{pname}",
                                old_value=pvalue,
                                new_value=normalized,
                                reason=f"Normalized {pvalue!r} to {normalized!r}",
                            ))
                            parameters[pname] = normalized

            # Repair 3: Fill missing defaults
            for pname, pdef in skill.parameters.items():
                if isinstance(pdef, dict) and "default" in pdef:
                    if pname not in parameters:
                        parameters[pname] = pdef["default"]
                        # Don't record as repair -- defaults are expected

        new_steps.append(TaskStep(
            step_id=step.step_id,
            skill_name=skill_name,
            parameters=parameters,
            depends_on=list(step.depends_on),
            preconditions=list(step.preconditions),
            postconditions=list(step.postconditions),
        ))

    new_plan = TaskPlan(
        goal=plan.goal,
        steps=new_steps,
        requires_clarification=plan.requires_clarification,
        clarification_question=plan.clarification_question,
        message=plan.message,
    )
    return new_plan, repairs

def _fuzzy_match_skill(self, name: str) -> str | None:
    """Match unknown skill name to registered skill via aliases + substring."""
    name_lower = name.lower().strip()
    # Direct alias match
    if name_lower in self._alias_map:
        return self._alias_map[name_lower]
    # Substring match (e.g. "pickup" contains "pick")
    for known in self._skill_names:
        if known.lower() in name_lower or name_lower in known.lower():
            return known
    return None

@staticmethod
def _normalize_enum(value: Any, enum_values: list) -> Any | None:
    """Try to normalize a value to match an enum entry."""
    if not isinstance(value, str):
        return None
    v = value.lower().strip()
    # Exact case-insensitive match
    for ev in enum_values:
        if isinstance(ev, str) and ev.lower() == v:
            return ev
    # Common normalizations: strip "side", underscores, spaces
    v_clean = v.replace(" ", "_").replace("-", "_")
    for ev in enum_values:
        if isinstance(ev, str) and ev.lower() == v_clean:
            return ev
    # Try removing trailing words like "side"
    for suffix in ("side", "area", "part", "position"):
        if v.endswith(f" {suffix}") or v.endswith(f"_{suffix}"):
            v_stripped = v.replace(f" {suffix}", "").replace(f"_{suffix}", "").strip()
            for ev in enum_values:
                if isinstance(ev, str) and ev.lower() == v_stripped:
                    return ev
    return None
```

**Test file**: `tests/unit/test_plan_validator.py`
- `test_unknown_skill_auto_repaired`
- `test_invalid_enum_auto_repaired`
- `test_repair_returns_modified_plan`

**Estimated effort**: Medium (< 1 hr)

---

### Task T17: Agent pipeline integration [alpha]

**File**: `vector_os_nano/core/agent.py`

**Depends on**: T14-T16 (PlanValidator exists and works)

**Change**: In `_handle_task()`, after `plan = self._llm.plan(...)` and before execution, insert validation + repair. Approximately at L467 (after plan is parsed, before on_message push):

```python
# ---- Validate & repair plan ----
from vector_os_nano.core.plan_validator import PlanValidator
validator = PlanValidator(self._skill_registry, self._world_model)
plan, repairs = validator.validate_and_repair(plan)
if repairs:
    repair_log = "; ".join(f"{r.field}: {r.old_value!r}->{r.new_value!r}" for r in repairs)
    logger.info("[Agent] Plan auto-repaired: %s", repair_log)

validation = validator.validate(plan)
if not validation.valid:
    error_log = "; ".join(f"{e.step_id}.{e.field}: {e.message}" for e in validation.errors)
    logger.warning("[Agent] Plan validation failed: %s", error_log)
    # Feed validation errors back to LLM as failure context for re-planning
    self._memory.add_assistant_message(
        f"Plan validation failed: {error_log}. Please fix and re-plan.",
        entry_type="task",
    )
    continue  # Go to next planning attempt
```

This goes inside the `for attempt in range(max_retries):` loop, after `plan = self._llm.plan(...)` returns and after the `requires_clarification` / empty steps checks, but before `on_message` and execution.

**Test file**: `tests/unit/test_plan_validator.py` (integration-style unit test)
- Verify that `_handle_task` with a mock LLM returning a plan with `skill_name="pickup"` gets auto-repaired to `"pick"`.

Actually, testing agent.py integration is better as an integration test. Unit tests for PlanValidator are sufficient for T15/T16.

**Test file**: `tests/integration/test_observable_pipeline.py` (NEW)
- `test_mcp_pick_failure_returns_structured_json`
- `test_multi_pick_world_model_consistent`
- `test_detect_merge_preserves_ids`
- `test_executor_trace_carries_diagnostics`

**Estimated effort**: Medium (< 1 hr)

---

### Task T18: Integration tests + final pass [beta]

**Files**: `tests/integration/test_observable_pipeline.py`

Write integration tests that exercise the full pipeline end-to-end using mock hardware.

```python
class TestObservablePipeline:
    """Integration tests for v0.3.0 Observable Symbolic Layer."""

    def test_mcp_pick_failure_returns_structured_json(self):
        """Full MCP pick call with unreachable object returns parseable JSON."""
        # Setup: Agent with mock arm (IK returns None for far positions)
        # Call: handle_tool_call(agent, "pick", {"object_label": "mug"})
        # Assert: response is valid JSON with "diagnosis" in steps

    def test_multi_pick_world_model_consistent(self):
        """3-object scene: pick first, world model retains other 2."""
        # Setup: WorldModel with banana_0, mug_0, bottle_0
        # Action: apply_skill_effects("pick", {"mode": "drop", "object_label": "banana"}, success_result)
        # Assert: mug_0 and bottle_0 still in world model

    def test_detect_merge_preserves_ids(self):
        """Two detect calls reuse first call's object IDs."""
        # Setup: Agent with mock perception returning [banana, mug]
        # Action: detect twice
        # Assert: second detect produces same object_ids

    def test_executor_trace_carries_diagnostics(self):
        """Full plan execution trace has result_data on every step."""
        # Setup: plan with scan -> detect -> pick
        # Action: executor.execute()
        # Assert: all trace entries have result_data with "diagnosis"
```

**Estimated effort**: Large (1.5-2 hrs)

---

### P2 Execution Waves

```
Wave 6 (parallel):
  Alpha: T14 (Validation types)
  Beta:  T15 (Validator core logic)
  Gamma: T16 (Auto-repair logic)

Wave 7 (serial, depends on Wave 6):
  Alpha: T17 (Agent pipeline integration)
  Beta:  T18 (Integration tests)
  Gamma: (available for bug fixes)

Wave 8 (final):
  All: Run full test suite, fix any regressions
  Lead: Version bump, tag release
```

**P2 total tasks**: 5 (T14-T18)

---

## 5. Task Summary

| ID | Phase | Description | File(s) | Agent | Depends | Est. |
|----|-------|-------------|---------|-------|---------|------|
| T1 | P0 | StepTrace.result_data field | types.py | alpha | -- | S |
| T2 | P0 | Executor copies result_data | executor.py | alpha | T1 | S |
| T3 | P0 | WorldModel pick fix | world_model.py | beta | -- | M |
| T4 | P0 | PickSkill fix + diagnostics | pick.py | beta | T3 | L |
| T5 | P0 | PlaceSkill diagnostics | place.py | gamma | -- | S |
| T6 | P0 | DetectSkill merge + diagnostics | detect.py | gamma | -- | M |
| T7 | P0 | Simple skills diagnostics | scan/home/gripper/wave.py | alpha | -- | S |
| T8 | P0 | MCP structured JSON response | tools.py, agent.py | gamma | T1,T2 | M |
| T9 | P1 | Skill protocol + to_schemas | skill.py | alpha | -- | S |
| T11 | P1 | Skill param annotations | all skills | beta | T9 | M |
| T12 | P1 | Dynamic prompt builder | prompts.py | alpha | T9,T11 | M |
| T13 | P1 | MCP schema pass-through | tools.py | gamma | T9 | S |
| T14 | P2 | Validation types | plan_validator.py | alpha | -- | S |
| T15 | P2 | Validator core logic | plan_validator.py | beta | T14 | L |
| T16 | P2 | Auto-repair logic | plan_validator.py | gamma | T14 | M |
| T17 | P2 | Agent integration | agent.py | alpha | T15,T16 | M |
| T18 | P2 | Integration tests | tests/integration/ | beta | T17 | L |

**Total: 17 tasks** (6 Small, 6 Medium, 5 Large)

Size legend: S = <45min, M = <1hr, L = 1.5-2hr

---

## 6. Execution Wave Schedule

```
WAVE 1 ──────────────────────────────────
  Alpha: T1 + T2  (StepTrace + Executor)
  Beta:  T3       (WorldModel fix)
  Gamma: T5       (PlaceSkill diag)

WAVE 2 ──────────────────────────────────
  Alpha: T7       (Simple skills diag)
  Beta:  T4       (PickSkill fix + diag)
  Gamma: T6       (DetectSkill merge)

WAVE 3 ──────────────────────────────────
  Alpha: [free or help]
  Beta:  [free or help]
  Gamma: T8       (MCP structured JSON)

--- P0 GATE: all existing tests pass, new P0 tests pass ---

WAVE 4 ──────────────────────────────────
  Alpha: T9       (Skill protocol)
  Beta:  T11      (Skill annotations)
  Gamma: T13      (MCP schema verify)

WAVE 5 ──────────────────────────────────
  Alpha: T12      (Dynamic prompts)
  Beta:  [test suite]
  Gamma: [free]

--- P1 GATE: P1 tests pass ---

WAVE 6 ──────────────────────────────────
  Alpha: T14      (Validation types)
  Beta:  T15      (Validator core)
  Gamma: T16      (Auto-repair)

WAVE 7 ──────────────────────────────────
  Alpha: T17      (Agent integration)
  Beta:  T18      (Integration tests)
  Gamma: [bug fixes]

--- P2 GATE: full suite green, 80%+ coverage ---
--- VERSION BUMP: v0.3.0, tag release ---
```

---

## 7. Test Strategy

### New test files (6 files)

| File | Phase | Tests |
|------|-------|-------|
| `tests/unit/test_step_trace_diagnostics.py` | P0 | 6 |
| `tests/unit/test_skill_diagnostics.py` | P0 | 11 |
| `tests/unit/test_mcp_structured_response.py` | P0 | 6 |
| `tests/unit/test_world_model_consistency.py` | P0 | 9 |
| `tests/unit/test_skill_schema_enhanced.py` | P1 | 6 |
| `tests/unit/test_dynamic_prompts.py` | P1 | 4 |
| `tests/unit/test_plan_validator.py` | P2 | 10 |
| `tests/integration/test_observable_pipeline.py` | P2 | 4 |

**Total new tests**: ~56

### TDD enforcement

Every task follows RED-GREEN-REFACTOR:
1. Agent writes test file first (RED -- tests fail because feature doesn't exist)
2. Agent implements the feature (GREEN -- tests pass)
3. Agent refactors for clarity (REFACTOR -- tests still pass)

### Regression checks

After each wave, run:
```bash
python -m pytest tests/ -x -q
```

All 34 existing test files must continue to pass with zero modifications.

---

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| `failure_modes` on Skill Protocol breaks `isinstance()` | Existing code may check `isinstance(x, Skill)` | Grep for usage; add `failure_modes: list[str] = []` default to all skills in same PR |
| MCP consumers expect text, get JSON | Breaking change for existing integrations | `_format_execution_result` returns JSON for ExecutionResult but still returns plain text for string inputs. MCP consumers that parsed the text format will need to update. Document in release notes. |
| `_normalize_enum` too aggressive | Auto-repairs valid but unusual values | Only normalize when exact match fails AND a close match exists. Log all repairs. |
| Detect merge creates stale references | If two objects have same label at different positions, merge picks wrong one | Use `get_objects_by_label` which returns closest match. For v0.3.0 this is acceptable; F2 (confidence-weighted resolution) is future work. |
| TaskPlan.from_dict missing `message` field | L289-295: `from_dict` doesn't parse `message` | Not blocking for v0.3.0 (plan is built internally, not deserialized from external sources) |

---

## 9. Backward Compatibility Checklist

- [x] `StepTrace()` without `result_data` works (default empty dict)
- [x] `SkillResult()` unchanged -- only contract (what goes in result_data) changes
- [x] `ExecutionResult` unchanged -- no new fields
- [x] `_format_execution_result(str_input)` returns string unchanged
- [x] `Skill` protocol: `failure_modes` added but `getattr` used in all production code
- [x] `build_planning_prompt()` signature unchanged
- [x] `skill_schema_to_mcp_tool()` passes through new fields but ignores missing ones
- [x] All existing tests pass without modification
