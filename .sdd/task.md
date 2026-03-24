# v0.3.0 Task List

## Execution Status
- Total tasks: 17
- Completed: 0
- In progress: 0
- Pending: 17

---

## Tasks

### Task T1: Add result_data to StepTrace
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: alpha
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/types.py`
- **Test file**: `tests/unit/test_step_trace_diagnostics.py`
- **TDD Deliverables**:
  - RED: Write `test_executor_result_data_default_empty`, `test_step_trace_serialization_with_result_data` — both fail
  - GREEN: Add `result_data: dict[str, Any] = field(default_factory=dict)` to `StepTrace`; update `to_dict()` and `from_dict()` to include `result_data`
  - REFACTOR: Confirm backward compat — existing `StepTrace` construction without `result_data` still works
- **Acceptance Criteria**:
  - `StepTrace(step_id="s1", skill_name="x", status="success").result_data == {}`
  - Round-trip through `to_dict()` / `from_dict()` preserves `result_data`
  - All existing tests still pass (no regression)

---

### Task T2: Executor copies result_data into StepTrace
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: alpha
- **Depends**: T1
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/executor.py`
- **Test file**: `tests/unit/test_step_trace_diagnostics.py`
- **TDD Deliverables**:
  - RED: Write `test_executor_copies_result_data_on_success`, `test_executor_copies_result_data_on_failure`, `test_executor_skill_not_found_has_diagnosis`, `test_executor_precondition_failed_has_diagnosis` — all fail
  - GREEN: Update all 6 `StepTrace` construction sites in `executor.py`:`execute()` to pass `result_data`
  - REFACTOR: Extract a helper if the pattern is repetitive
- **Acceptance Criteria**:
  - Success trace entry: `result_data == skill_result.result_data`
  - Failure trace entries: `result_data["diagnosis"]` is a non-empty string
  - `skill_not_found` trace: `result_data == {"diagnosis": "skill_not_found"}`
  - `precondition_failed` trace: `result_data` contains `"diagnosis"` and `"predicate"` keys

---

### Task T3: WorldModel.apply_skill_effects() pick fix
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: beta
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/world_model.py`
- **Test file**: `tests/unit/test_world_model_consistency.py`
- **TDD Deliverables**:
  - RED: Write `test_pick_drop_removes_only_picked_object`, `test_pick_hold_marks_object_grasped`, `test_pick_drop_clears_held_object`, `test_apply_skill_effects_pick_hold`, `test_apply_skill_effects_pick_drop`, `test_multi_pick_preserves_remaining` — all fail
  - GREEN: Replace `_objects.clear()` with surgical remove/update logic in `apply_skill_effects()` pick branch
  - REFACTOR: Ensure `mode="hold"` sets object state to `"grasped"` and `mode="drop"` (default) removes only the target object
- **Acceptance Criteria**:
  - 3-object scene: pick(banana, mode=drop) leaves mug_0 and bottle_0 in world model
  - pick(banana, mode=hold): banana_0.state == "grasped", robot.held_object == "banana_0"
  - pick(banana, mode=drop): robot.held_object is None, gripper_state == "open"

---

### Task T4: PickSkill world model fix + diagnostics
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: beta
- **Depends**: T3
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/skills/pick.py`
- **Test file**: `tests/unit/test_skill_diagnostics.py`
- **TDD Deliverables**:
  - RED: Write `test_pick_failure_ik_returns_diagnosis`, `test_pick_failure_workspace_returns_diagnosis`, `test_pick_failure_no_arm_returns_diagnosis`, `test_pick_failure_no_detections_returns_diagnosis`, `test_pick_success_returns_diagnosis_ok` — all fail
  - GREEN: Replace `_objects.clear()` in `_single_pick_attempt()` with targeted remove; add `result_data={"diagnosis": ...}` to every return path; track `_last_perception_diag` for perception sub-failures; propagate `last_diagnosis` through retry loop
  - REFACTOR: Verify `_sample_from_perception()` sets `self._last_perception_diag` before returning None
- **Acceptance Criteria**:
  - Every SkillResult from pick contains `result_data["diagnosis"]`
  - IK failure: `diagnosis == "ik_unreachable"`, contains `target_base_cm`
  - Workspace failure: `diagnosis == "out_of_workspace"`, contains `distance_cm`
  - No arm: `diagnosis == "no_arm"`
  - No detections: `diagnosis == "no_detections"`
  - Success: `diagnosis == "ok"`

---

### Task T5: PlaceSkill diagnostics
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: gamma
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/skills/place.py`
- **Test file**: `tests/unit/test_skill_diagnostics.py`
- **TDD Deliverables**:
  - RED: Write `test_place_failure_ik_returns_diagnosis`, `test_place_success_returns_diagnosis_ok` — both fail
  - GREEN: Add `result_data={"diagnosis": ...}` to every return path in PlaceSkill
  - REFACTOR: Confirm `diagnosis: "ok"` is added to existing success `result_data` (merged, not replaced)
- **Acceptance Criteria**:
  - No arm: `diagnosis == "no_arm"`
  - IK failures: `diagnosis == "ik_unreachable"`, contains `target_cm`
  - Move failures: `diagnosis == "move_failed"`, contains `phase`
  - Success: existing `result_data["placed_at"]` preserved and `diagnosis == "ok"` added

---

### Task T6: DetectSkill merge + diagnostics
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: gamma
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/skills/detect.py`
- **Test file**: `tests/unit/test_world_model_consistency.py`, `tests/unit/test_skill_diagnostics.py`
- **TDD Deliverables**:
  - RED: Write `test_detect_merges_existing_objects`, `test_detect_creates_new_for_unknown`, `test_detect_updates_position_on_merge`, `test_detect_failure_no_perception_returns_diagnosis`, `test_detect_success_returns_diagnosis_ok` — all fail
  - GREEN: Replace index-based ID generation with label-merge logic (check world model before generating new ID); add `diagnosis` to all return paths; track `merged_count` for success result
  - REFACTOR: Ensure collision-free ID generation when label is new
- **Acceptance Criteria**:
  - Two detect calls on same object reuse the same `object_id`
  - New object gets a unique ID not already in world model
  - No perception: `diagnosis == "no_perception"`
  - No detections: `diagnosis == "no_detections"`, contains `query`
  - Success: `diagnosis == "ok"`, contains `merged_count`

---

### Task T7: Simple skills diagnostics (scan, home, gripper, wave)
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: alpha
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/skills/scan.py`, `vector_os_nano/skills/home.py`, `vector_os_nano/skills/gripper.py`, `vector_os_nano/skills/wave.py`
- **Test file**: `tests/unit/test_skill_diagnostics.py`
- **TDD Deliverables**:
  - RED: Write `test_scan_failure_returns_diagnosis`, `test_home_failure_returns_diagnosis` — both fail
  - GREEN: Add `result_data={"diagnosis": ...}` to failure and success return paths in all 4 skill files
  - REFACTOR: Verify pattern is consistent: no_arm failure = `{"diagnosis": "no_arm"}`, move failure = `{"diagnosis": "move_failed"}`, success = `{"diagnosis": "ok"}`
- **Acceptance Criteria**:
  - scan: no_arm → `{"diagnosis": "no_arm"}`; move fail → `{"diagnosis": "move_failed"}`; success → `{"diagnosis": "ok"}`
  - home: same pattern as scan
  - gripper open/close: no_arm → `{"diagnosis": "no_arm"}`; success → `{"diagnosis": "ok"}`
  - wave: no_arm → `{"diagnosis": "no_arm"}`; raise fail → `{"diagnosis": "move_failed"}`; success → `{"diagnosis": "ok"}`

---

### Task T8: MCP structured JSON response
- **Status**: [ ] pending
- **Phase**: P0
- **Agent**: gamma
- **Depends**: T1, T2
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/mcp/tools.py`, `vector_os_nano/core/agent.py`
- **Test file**: `tests/unit/test_mcp_structured_response.py`
- **TDD Deliverables**:
  - RED: Write `test_format_execution_result_returns_json`, `test_format_result_json_has_steps_with_result_data`, `test_format_result_json_has_world_state`, `test_format_result_string_passthrough`, `test_format_result_json_has_failure_reason`, `test_format_result_json_success_case` — all fail
  - GREEN: Rewrite `_format_execution_result()` to return JSON string for `ExecutionResult` inputs; add optional `world_state` parameter; update `handle_tool_call` to pass `agent.world.to_dict()` as `world_state`; ensure all failure paths in `agent.py` include `world_model_diff`
  - REFACTOR: Confirm string passthrough for non-ExecutionResult inputs (backward compat)
- **Acceptance Criteria**:
  - Non-ExecutionResult string input: returned unchanged
  - ExecutionResult input: valid JSON with `success`, `status`, `steps`, `steps_completed`, `steps_total`, `total_duration_sec`
  - Each step in `steps` has `step_id`, `skill_name`, `status`, `duration_sec`, `result_data`
  - Failure case: `failure_reason` key present
  - `world_state` key present when world snapshot provided

---

### Task T9: Skill protocol + to_schemas() failure_modes
- **Status**: [ ] pending
- **Phase**: P1
- **Agent**: alpha
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/skill.py`
- **Test file**: `tests/unit/test_skill_schema_enhanced.py`
- **TDD Deliverables**:
  - RED: Write `test_to_schemas_includes_failure_modes`, `test_skill_protocol_has_failure_modes` — both fail
  - GREEN: Add `failure_modes: list[str]` to `Skill` Protocol; update `to_schemas()` to include `failure_modes` via `getattr` with default `[]`
  - REFACTOR: Confirm `getattr(s, 'failure_modes', [])` is safe — no `isinstance(x, Skill)` checks broken
- **Acceptance Criteria**:
  - `Skill` Protocol has `failure_modes: list[str]` field
  - `to_schemas()` output includes `"failure_modes"` key for skills that define it
  - Skills without `failure_modes` attr do not produce `"failure_modes"` key in schema (empty list omitted)

---

### Task T11: Skill parameter annotations (enum, source, failure_modes)
- **Status**: [ ] pending
- **Phase**: P1
- **Agent**: beta
- **Depends**: T9
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/skills/pick.py`, `vector_os_nano/skills/place.py`, `vector_os_nano/skills/detect.py`, `vector_os_nano/skills/scan.py`, `vector_os_nano/skills/home.py`, `vector_os_nano/skills/gripper.py`, `vector_os_nano/skills/wave.py`
- **Test file**: `tests/unit/test_skill_schema_enhanced.py`
- **TDD Deliverables**:
  - RED: Write `test_pick_schema_has_mode_enum`, `test_place_schema_has_location_enum`, `test_schema_source_annotation`, `test_mcp_tool_schema_has_enum`, `test_all_skills_have_failure_modes_attr` — all fail
  - GREEN: Add `enum` and `source` annotations to relevant parameters in each skill; add `failure_modes: list[str]` class attribute to every skill
  - REFACTOR: Verify place.py uses `list(_LOCATION_MAP.keys())` for location enum
- **Acceptance Criteria**:
  - `pick.parameters["mode"]["enum"] == ["drop", "hold"]`
  - `place.parameters["location"]["enum"]` is the list of named locations
  - All 7 skill classes have `failure_modes` attribute (non-empty for hardware skills)
  - `pick.parameters["object_id"]["source"] == "world_model.objects.object_id"`

---

### Task T12: Dynamic prompt builder
- **Status**: [ ] pending
- **Phase**: P1
- **Agent**: alpha
- **Depends**: T9, T11
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/llm/prompts.py`
- **Test file**: `tests/unit/test_dynamic_prompts.py`
- **TDD Deliverables**:
  - RED: Write `test_planning_prompt_includes_enum_constraints`, `test_planning_prompt_includes_available_objects`, `test_planning_prompt_includes_gripper_state`, `test_planning_prompt_empty_world` — all fail
  - GREEN: Replace hardcoded `PLACE LOCATIONS` block in `PLANNING_SYSTEM_PROMPT` with `{constraints_block}` placeholder; implement `build_planning_prompt()` with dynamic constraints generation from skill schemas and world state
  - REFACTOR: Confirm Chinese/English location hardcoding is fully removed; LLM sees `VALID VALUES for place.location: front, front_left, ...`
- **Acceptance Criteria**:
  - Prompt contains `VALID VALUES for place.location:` with the actual enum values
  - Prompt contains `AVAILABLE OBJECTS:` with labels from world state
  - Prompt contains `GRIPPER:` with current gripper state
  - Empty world: `AVAILABLE OBJECTS: none detected`
  - Holding object: `GRIPPER: holding <object_id>`

---

### Task T13: MCP schema pass-through for enum and failure_modes
- **Status**: [ ] pending
- **Phase**: P1
- **Agent**: gamma
- **Depends**: T9
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/mcp/tools.py`
- **Test file**: `tests/unit/test_skill_schema_enhanced.py`
- **TDD Deliverables**:
  - RED: Write `test_mcp_tool_schema_has_enum` (verify enum pass-through); write test for `failure_modes` on MCP tool — fail
  - GREEN: Verify `skill_schema_to_mcp_tool()` already passes `enum` (L128-129); add `failure_modes` pass-through as top-level field on tool dict
  - REFACTOR: Confirm `failure_modes` is outside `inputSchema` (it is metadata, not a parameter)
- **Acceptance Criteria**:
  - `skill_schema_to_mcp_tool(pick_schema)["inputSchema"]["properties"]["mode"]["enum"] == ["drop", "hold"]`
  - `skill_schema_to_mcp_tool(pick_schema)["failure_modes"]` contains the pick failure modes list

---

### Task T14: PlanValidator data types
- **Status**: [ ] pending
- **Phase**: P2
- **Agent**: alpha
- **Depends**: none
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/plan_validator.py` (NEW FILE)
- **Test file**: `tests/unit/test_plan_validator.py`
- **TDD Deliverables**:
  - RED: Write `test_validation_result_default_valid`, `test_validation_error_frozen` — both fail (file does not exist)
  - GREEN: Create `plan_validator.py` with frozen dataclasses `ValidationError`, `Repair`, `ValidationResult`
  - REFACTOR: Confirm all three types are `frozen=True`; `ValidationResult.errors` and `warnings` default to empty lists
- **Acceptance Criteria**:
  - `ValidationResult(valid=True).errors == []`
  - `ValidationResult(valid=True).warnings == []`
  - `ValidationError` instances are immutable (frozen dataclass)
  - `Repair` instances are immutable (frozen dataclass)

---

### Task T15: PlanValidator core validation logic
- **Status**: [ ] pending
- **Phase**: P2
- **Agent**: beta
- **Depends**: T14
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/plan_validator.py`
- **Test file**: `tests/unit/test_plan_validator.py`
- **TDD Deliverables**:
  - RED: Write `test_valid_plan_passes`, `test_unknown_skill_detected`, `test_invalid_enum_detected`, `test_missing_required_param_detected`, `test_circular_dependency_detected`, `test_type_mismatch_detected`, `test_precondition_unsatisfiable` — all fail
  - GREEN: Add `PlanValidator` class with `validate()` method covering 7 checks: skill existence, required params, enum conformance, type conformance, dependency graph, cycle detection, precondition satisfiability
  - REFACTOR: Extract `_suggest_skill_name()`, `_check_type()`, `_has_cycle()` as helpers; verify alias map is built in `__init__`
- **Acceptance Criteria**:
  - Valid plan: `ValidationResult.valid == True`, no errors
  - Unknown skill: `ValidationError.code == "unknown_skill"`, suggestion contains closest match
  - Invalid enum: `ValidationError.code == "invalid_enum"`, lists valid values
  - Circular dependency: `ValidationError.code == "circular_dependency"`
  - Type mismatch: `ValidationError.code == "type_mismatch"`
  - Unsatisfiable precondition: appears as warning, not error

---

### Task T16: PlanValidator auto-repair logic
- **Status**: [ ] pending
- **Phase**: P2
- **Agent**: gamma
- **Depends**: T14
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/plan_validator.py`
- **Test file**: `tests/unit/test_plan_validator.py`
- **TDD Deliverables**:
  - RED: Write `test_unknown_skill_auto_repaired`, `test_invalid_enum_auto_repaired`, `test_repair_returns_modified_plan` — all fail
  - GREEN: Add `validate_and_repair()` method with 3 repair strategies: fuzzy skill name match, enum normalization (case-insensitive + suffix stripping), fill missing defaults
  - REFACTOR: Add `_fuzzy_match_skill()` and `_normalize_enum()` helpers; confirm original plan is not mutated (new `TaskPlan` returned)
- **Acceptance Criteria**:
  - `"pickup"` skill name repaired to `"pick"` with `Repair` record
  - `"Front"` location value normalized to `"front"` with `Repair` record
  - `"front side"` normalized to `"front"`
  - Repairs list contains one entry per change applied
  - Original `TaskPlan` object unchanged (immutability preserved)

---

### Task T17: Agent pipeline integration
- **Status**: [ ] pending
- **Phase**: P2
- **Agent**: alpha
- **Depends**: T15, T16
- **Package**: vector_os_nano
- **Files**: `vector_os_nano/core/agent.py`
- **Test file**: `tests/integration/test_observable_pipeline.py`
- **TDD Deliverables**:
  - RED: Write `test_mcp_pick_failure_returns_structured_json`, `test_multi_pick_world_model_consistent`, `test_detect_merge_preserves_ids`, `test_executor_trace_carries_diagnostics` — all fail
  - GREEN: In `_handle_task()`, after `plan = self._llm.plan(...)` and before execution, insert `PlanValidator.validate_and_repair()` then `validate()`; on validation failure, add error to memory and `continue` to next retry
  - REFACTOR: Ensure import of `PlanValidator` is local (avoid circular imports); log repairs at INFO and failures at WARNING
- **Acceptance Criteria**:
  - Plan with `skill_name="pickup"` auto-repaired to `"pick"` before execution
  - Plan with unresolvable validation errors triggers re-plan (not execution)
  - Repair log message visible at INFO level
  - Validation failure message fed back to LLM memory for re-planning

---

### Task T18: Integration tests
- **Status**: [ ] pending
- **Phase**: P2
- **Agent**: beta
- **Depends**: T17
- **Package**: vector_os_nano
- **Files**: `tests/integration/test_observable_pipeline.py` (NEW FILE)
- **Test file**: `tests/integration/test_observable_pipeline.py`
- **TDD Deliverables**:
  - RED: All 4 integration tests fail (no implementation yet when writing tests)
  - GREEN: Write `TestObservablePipeline` with 4 tests using mock hardware — all pass after T17 is done
  - REFACTOR: Ensure tests use mock arm (IK returns None for far positions) and mock perception; no real hardware required
- **Acceptance Criteria**:
  - `test_mcp_pick_failure_returns_structured_json`: response is valid JSON, `steps[0]["result_data"]["diagnosis"]` is a string
  - `test_multi_pick_world_model_consistent`: after picking banana_0, mug_0 and bottle_0 still in world model
  - `test_detect_merge_preserves_ids`: two detect calls produce identical `object_id` values for same object
  - `test_executor_trace_carries_diagnostics`: every trace entry has non-empty `result_data`

---

## Dependency Graph

```
T1 (types.py)
  └── T2 (executor.py)
        └── T8 (tools.py + agent.py)

T3 (world_model.py)
  └── T4 (pick.py)

T5 (place.py)          -- independent
T6 (detect.py)         -- independent
T7 (scan/home/gripper/wave) -- independent

T9 (skill.py)
  ├── T11 (all skills)
  │     └── T12 (prompts.py)
  └── T13 (tools.py)

T14 (plan_validator.py types)
  ├── T15 (validator core)
  └── T16 (auto-repair)
        └── T17 (agent.py)
              └── T18 (integration tests)
```

---

## Execution Waves

### Wave 1 (P0 start, parallel)
| Agent | Task | Description |
|-------|------|-------------|
| Alpha | T1 then T2 | StepTrace field, then Executor update (serial within Alpha) |
| Beta  | T3 | WorldModel pick fix |
| Gamma | T5 | PlaceSkill diagnostics |

### Wave 2 (P0, after Wave 1, parallel)
| Agent | Task | Description |
|-------|------|-------------|
| Alpha | T7 | Simple skills diagnostics |
| Beta  | T4 | PickSkill fix + diagnostics |
| Gamma | T6 | DetectSkill merge + diagnostics |

### Wave 3 (P0 close, after Wave 2)
| Agent | Task | Description |
|-------|------|-------------|
| Gamma | T8 | MCP structured JSON response |
| Alpha | -- | Run full test suite, fix integration issues |
| Beta  | -- | Available for P1 start |

### Wave 4 (P1 start, parallel)
| Agent | Task | Description |
|-------|------|-------------|
| Alpha | T9  | Skill protocol + to_schemas |
| Beta  | T11 | Skill parameter annotations |
| Gamma | T13 | MCP schema pass-through verification |

### Wave 5 (P1 close, after Wave 4)
| Agent | Task | Description |
|-------|------|-------------|
| Alpha | T12 | Dynamic prompt builder |
| Beta  | -- | Run P1 test suite |
| Gamma | -- | Available for P2 start |

### Wave 6 (P2 start, parallel)
| Agent | Task | Description |
|-------|------|-------------|
| Alpha | T14 | Validation data types |
| Beta  | T15 | Validator core logic |
| Gamma | T16 | Auto-repair logic |

### Wave 7 (P2, after Wave 6)
| Agent | Task | Description |
|-------|------|-------------|
| Alpha | T17 | Agent pipeline integration |
| Beta  | T18 | Integration tests |
| Gamma | -- | Available for bug fixes |

### Wave 8 (final)
| Agent | Task | Description |
|-------|------|-------------|
| All   | -- | Run full test suite, fix regressions |
| Lead  | -- | Version bump, tag release |
