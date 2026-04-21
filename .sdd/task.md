# v2.4 Perception Overhaul — Task Breakdown

**Status**: DRAFT — blocked on CEO approval of spec.md § Open Questions
**Agents**: Alpha / Beta / Gamma (Sonnet, serial) + QA (code-review + security-review) + Scribe
**Prereq**: spec.md approved; env probe (T0) complete

---

## Execution Summary

| Metric | Value |
|---|---|
| Total code tasks | 10 (T0–T10) |
| Waves | 6 |
| New files | ~12 (4 impl modules + 1 protocol + ~6 test files + 10 MJCF fragments) |
| Modified files | 6 |
| Deleted files | 2 (`vlm_qwen.py` + its test) |
| Unit tests added | ≥ 32 |
| Integration tests added | ≥ 3 |
| E2E dry-run | 1 updated |
| Coverage floor | 90 % on new perception modules |

### Serial-subagent safety (carried forward from v2.3)

- Subagents MUST NOT run `pytest tests/` or `pytest tests/integration/`.
- Narrow scope: `pytest tests/unit/perception/test_<file>.py` only.
- Forbidden-import list in every subagent prompt:
  `pipeline, track_anything, mujoco, realsense, tracker, ultralytics`
  (ultralytics is heavyweight — only enable in Wave 5 GPU smoke).
- Dispatcher runs narrow-scope regressions at wave gates.
- Memory `feedback_no_parallel_agents.md` still authoritative.

---

## Wave 0 — Environment Probe

### T0 — Probe dependency + GSO access

**Agent**: Dispatcher (no subagent)
**Depends**: CEO approval of spec
**Deliverable**: probe report in `agents/devlog/v24-env-probe.md`

Steps:

1. `pip install -U 'ultralytics>=8.3.237'` in project venv. Log version.
2. `python -c "from ultralytics import YOLOE, SAM; from ultralytics.models.sam import SAM3SemanticPredictor"` — confirm imports.
3. Check `~/.cache/ultralytics/` for existing weights.
4. Attempt `YOLOE('yoloe-11s-seg.pt')` — auto-download check.
5. Attempt `SAM('sam2.pt')` — auto-download check.
6. Probe `~/.cache/ultralytics/sam3.pt` — expected missing → document HF access step for Yusen.
7. Clone `git clone --depth 1 https://github.com/kevinzakka/mujoco_scanned_objects` to `/tmp/gso` — verify structure, count models.
8. Pick 10 candidate `pickable_*` object IDs and document in probe report.
9. Add `ultralytics>=8.3.237` to `pyproject.toml [perception]` extras.

**Gate**: probe report posted. If SAM 3 access pending, proceed with SAM 2.1 fallback primary.

---

## Wave 1 — Core Perception Modules (3 parallel tasks, SERIAL dispatch)

### T1 — YoloeDetector + OpenVocabDetector protocol

**Agent**: Alpha
**Wave**: 1
**Depends**: T0
**Package**: `vector_os_nano/perception/detectors/`

**Scope**:
Implement the 2D open-vocabulary detector. Wrap `ultralytics.YOLOE`,
cache class set to avoid re-compilation, map common CN/EN keywords
to YOLOE class strings.

**Files**:
- NEW `vector_os_nano/perception/detectors/__init__.py`
- NEW `vector_os_nano/perception/detectors/base.py` (OpenVocabDetector Protocol)
- NEW `vector_os_nano/perception/detectors/yoloe_detector.py` (~130 LoC)
- NEW `vector_os_nano/perception/detectors/query_map.py` (~60 LoC)
- NEW `tests/unit/perception/detectors/test_yoloe_detector.py`
- NEW `tests/unit/perception/detectors/test_query_map.py`

**TDD (RED → GREEN → REFACTOR)**:

RED — unit tests (must fail with `ImportError` first):

1. `test_detect_empty_image_returns_empty` — all-zero image → `[]`.
2. `test_detect_with_bbox_returns_detection` — mocked YOLOE returning one box → `Detection` with correct label/bbox/confidence.
3. `test_set_classes_cached_across_calls` — same query twice → `set_classes` called once.
4. `test_confidence_threshold_filters_low` — mock returns two boxes conf=0.1/0.9, threshold=0.25 → only 0.9 returned.
5. `test_device_fallback_to_cpu` — no CUDA → device="cpu", no crash.
6. `test_colour_query_kept_for_downstream` — `"blue bottle"` → YOLOE sees class `"bottle"`, Detection.label preserves full query so colour filter can use it.

Query-map tests (in `test_query_map.py`):

1. `test_bottle_keywords` — `"bottle" / "瓶子" / "瓶" / "blue bottle"` → `("bottle",)`.
2. `test_all_objects_returns_wildcard` — `"all objects" / "所有物体" / "everything"` → `("*",)`.
3. `test_unknown_query_falls_back_to_wildcard` — `"widget"` → `("*",)`.

GREEN — minimal implementation per spec §7.2 + plan §3.1.

REFACTOR — extract `_results_to_detections` helper, add type hints.

**Verify**: `pytest tests/unit/perception/detectors/ -v` (target: 9 green)

### T2 — Sam3Segmenter + SAM 2.1 fallback

**Agent**: Beta
**Wave**: 1
**Depends**: T0
**Package**: `vector_os_nano/perception/segmenters/`

**Scope**:
Wrap SAM 3 `SAM3SemanticPredictor`; on missing weights, fallback to
SAM 2.1 via `SAM("sam2.pt")`. Uniform `segment(image, bboxes) -> list[mask]`
contract.

**Files**:
- NEW `vector_os_nano/perception/segmenters/__init__.py`
- NEW `vector_os_nano/perception/segmenters/sam3_segmenter.py` (~160 LoC)
- NEW `tests/unit/perception/segmenters/test_sam3_segmenter.py`

**TDD**:

RED:

1. `test_sam3_primary_path` — mock `SAM3SemanticPredictor` present → `backend_name == "sam3"`.
2. `test_sam2_fallback_when_weights_missing` — patch `_weights_available` → False → `backend_name == "sam2.1"`, warning logged.
3. `test_segment_empty_prompts_returns_empty` — `[]` → `[]`, no predictor call.
4. `test_segment_single_bbox_returns_mask_with_correct_shape` — 640×480 image + one bbox → mask shape `(480, 640)` bool.
5. `test_segment_multi_bbox_returns_matching_list_length` — 3 bboxes → 3 masks.
6. `test_mask_upscaled_if_predictor_downsamples` — mock returns 320×240 mask → output 640×480.

GREEN: per plan §3.2.

REFACTOR: extract `_weights_available`, `_mask_to_bool`, `_upscale` helpers.

**Verify**: `pytest tests/unit/perception/segmenters/ -v` (target: 6 green)

### T3 — pointcloud_projection + sanity_gates

**Agent**: Gamma
**Wave**: 1
**Depends**: T0
**Package**: `vector_os_nano/perception/`

**Scope**:
Mask-filtered depth projection with statistical outlier removal, plus
sanity-gate module.

**Files**:
- NEW `vector_os_nano/perception/pointcloud_projection.py` (~120 LoC)
- NEW `vector_os_nano/perception/sanity_gates.py` (~80 LoC)
- NEW `tests/unit/perception/test_pointcloud_projection.py`
- NEW `tests/unit/perception/test_sanity_gates.py`

**TDD**:

RED (pointcloud_projection):

1. `test_projection_cube_in_front_of_camera` — synthetic mask over a 10×10 cube depth=1.0 → centroid `(0, 0, 1)` with `fx=fy=100, cx=cy=50`.
2. `test_empty_mask_returns_empty_array` — mask all False → `np.empty((0,3))`.
3. `test_all_invalid_depth_returns_empty` — valid mask, depth all `NaN` → empty.
4. `test_depth_range_clips_samples_outside_bounds` — depth 0.05 (too near) and 8.0 (too far) → excluded.
5. `test_statistical_filter_removes_outliers` — 100 points around 1.0 + 5 points at 5.0 → outliers removed.
6. `test_centroid_methods` — median/mean give expected values on skewed distribution.

RED (sanity_gates):

1. `test_gate_depth_out_of_range` — depth 6.0 → rejected `depth_out_of_range`.
2. `test_gate_height_out_of_range_world` — world z 2.0 → rejected.
3. `test_gate_mask_area_too_small` — 50 px → rejected.
4. `test_gate_mask_area_too_large` — 60 000 px → rejected.
5. `test_gate_sparse_pointcloud` — 10 points → rejected.
6. `test_all_gates_pass` — nominal values → `(True, "")`.

GREEN: per plan §3.3/3.4.

REFACTOR: scipy KDTree wrapped in try/except with numpy fallback
(O(N²) but runs).

**Verify**: `pytest tests/unit/perception/test_pointcloud_projection.py tests/unit/perception/test_sanity_gates.py -v` (target: 12 green)

---

## Wave 2 — Integration (3 parallel, SERIAL dispatch)

### T4 — Go2Perception v2 rewrite

**Agent**: Alpha
**Wave**: 2
**Depends**: T1, T2, T3
**Package**: `vector_os_nano/perception/`

**Scope**:
Rewrite `Go2Perception` to use detector + segmenter + pointcloud +
sanity_gates. Structured per-detection log line.

**Files**:
- MOD `vector_os_nano/perception/go2_perception.py` (~280 LoC; delete bbox-centre depth code)
- MOD `tests/unit/perception/test_go2_perception.py` (rewrite)

**TDD**:

RED:

1. `test_detect_delegates_to_detector` — mock detector → returns same list.
2. `test_track_runs_segment_and_projects` — mock seg returns mask → pose present.
3. `test_sanity_gate_rejection_sets_pose_none` — pointcloud sparse → `pose is None`.
4. `test_structured_log_emitted_per_detection` — caplog shows `[go2_perception] label=...` INFO.
5. `test_get_point_cloud_delegates_to_mask_projection` — returns array with correct shape.
6. `test_caption_and_visual_query_delegate_to_vlm` — `Go2VLMPerception` wrapper unchanged.

GREEN: per plan §3.5.

**Verify**: `pytest tests/unit/perception/test_go2_perception.py -v` (target: 6 green)

### T5 — xmat LEFT/RIGHT fix

**Agent**: Beta
**Wave**: 2
**Depends**: — (parallel with T4)
**Package**: `vector_os_nano/hardware/sim/`

**Scope**:
Fix `get_camera_pose` to REP-103 convention. Update regression test
expected values. Verify `depth_projection.camera_to_world` still
consistent (no change needed — the xmat columns changed, transform
formula is invariant).

**Files**:
- MOD `vector_os_nano/hardware/sim/go2_ros2_proxy.py` (2 lines)
- MOD `tests/integration/test_go2_camera_pose.py` (update expected)

**TDD**:

RED:

1. Existing `test_right_column_is_minus_y_at_heading_zero` (or similar) — update to expect `right == (0, -1, 0)` at heading=0.
2. `test_up_column_is_plus_z_at_heading_zero` — expect `up == (0, 0, 1)`.
3. `test_xmat_orthonormal` — det==1, columns unit-length.

GREEN: change `right = (sin_h, -cos_h, 0)` and `up = np.cross(right, fwd)`.

**Verify**: `pytest tests/integration/test_go2_camera_pose.py -v` and
`pytest tests/integration/test_piper_grasp_dry_run.py -v` (the 30/30 pick
regression file — confirm still green with centred bottles).

### T6 — Qwen grounding removal

**Agent**: Gamma
**Wave**: 2
**Depends**: T4 (Go2Perception must be detector-based first)
**Package**: `vector_os_nano/perception/`

**Scope**:
Delete `vlm_qwen.py` + `test_vlm_qwen.py`. Grep for any residual
imports and remove.

**Files**:
- DEL `vector_os_nano/perception/vlm_qwen.py`
- DEL `tests/unit/perception/test_vlm_qwen.py`
- MOD (possibly) `vector_os_nano/vcli/tools/sim_tool.py`, any
  `perception/__init__.py` re-exports.

**Procedure**:

1. `rg "QwenVLMDetector|vlm_qwen"` — enumerate all references.
2. Remove each reference; prefer replacing with `None` or
   detector-based path.
3. `git rm` the two files.
4. Ruff check: `ruff check vector_os_nano/perception/ vector_os_nano/vcli/`.

**Verify**: `rg "QwenVLMDetector"` returns zero matches. Ruff clean.

---

## Wave 3 — Scene + Wiring (2 parallel, SERIAL dispatch)

### T7 — GSO asset curation + scene XML

**Agent**: Alpha
**Wave**: 3
**Depends**: T0 (probe lists candidates)
**Package**: `vector_os_nano/hardware/sim/`

**Scope**:
Select 10 GSO objects, copy into `mjcf/pickable_assets/`, configure
Git-LFS, include in `go2_room.xml`.

**Files**:
- NEW `vector_os_nano/hardware/sim/mjcf/pickable_assets/<id>/model.xml` × 10
- NEW corresponding `.obj` + textures (Git-LFS tracked)
- NEW `vector_os_nano/hardware/sim/mjcf/pickable_assets/include.xml` (aggregator)
- MOD `vector_os_nano/hardware/sim/go2_room.xml`
- NEW `scripts/fetch_gso_assets.sh` (fallback download path)
- MOD `.gitattributes` (LFS track `*.obj *.png` under `pickable_assets/`)

**Procedure**:

1. From T0 candidate list (≥10), verify each is <5 cm diameter, has
   valid MJCF with V-HACD submeshes.
2. Copy 10 dirs to `mjcf/pickable_assets/`.
3. `git lfs track "vector_os_nano/hardware/sim/mjcf/pickable_assets/**/*.obj" "vector_os_nano/hardware/sim/mjcf/pickable_assets/**/*.png"`.
4. Write `include.xml` aggregating mesh+material assets.
5. In `go2_room.xml`, delete lines 585–601 (3 capsule pickable_* bodies),
   insert 10 new `<body name="pickable_<id>">` around `pos="11.0 3.0 0.250"`
   arranged on the pick table surface. Use `freejoint` so they are
   pickable. Keep `friction="2 0.05 0.001" priority="2"` for grip.
6. Raise pick_table size to accommodate (was 20×25×10 → try 25×50×10).

**Integration test**:
- `tests/integration/test_scene_load.py::test_go2_room_loads_with_gso` —
  `mujoco.MjModel.from_xml_path(go2_room.xml)` succeeds, `nbody >=
  gso_count + baseline_nbody`.

**Verify**: `python -c "import mujoco; m = mujoco.MjModel.from_xml_path('vector_os_nano/hardware/sim/go2_room.xml'); print(m.nbody, m.ngeom)"`.

### T8 — sim_tool wire-up + DetectSkill summary + MobilePick distance warning

**Agent**: Beta
**Wave**: 3
**Depends**: T4
**Package**: `vector_os_nano/vcli/tools/`, `vector_os_nano/skills/`

**Scope**:
Glue the new Go2Perception v2 into `sim_tool._start_go2`; improve
`DetectSkill` output summary; add distance warning in `MobilePickSkill`.

**Files**:
- MOD `vector_os_nano/vcli/tools/sim_tool.py`
- MOD `vector_os_nano/skills/detect.py`
- MOD `vector_os_nano/skills/mobile_pick.py`
- NEW `tests/integration/test_sim_tool_v24_wireup.py`

**TDD**:

RED:

1. `test_start_go2_wires_v24_perception` — mock weight presence → `agent._perception` is `Go2Perception` with `detector` attr of type `YoloeDetector`.
2. `test_detectskill_summary_field_present` — mock perception with 2 detections → `result_data["summary"]` contains both labels + cm coords.
3. `test_mobile_pick_logs_distance_warning_over_3m` — mock target at 4 m → caplog has `far approach`.

GREEN: per plan §3.7/3.8/3.9.

**Verify**: narrow pytest.

---

## Wave 4 — Diagnostics + E2E (2 parallel, SERIAL dispatch)

### T9 — Live diagnostic script

**Agent**: Alpha
**Wave**: 4
**Depends**: T4, T7
**Deliverable**: `scripts/debug_perception_live.py`

Spec in plan §7 "GPU smoke (manual)". Write HTML report with
RGB/mask/depth overlays.

**Verify**: script runs to completion with mocked detector/segmenter
(CI-safe dry mode).

### T10 — verify_perception_pick.py dry-run update + QUICKSTART doc

**Agent**: Gamma
**Wave**: 4
**Depends**: T4, T7, T8
**Deliverable**:
- MOD `scripts/verify_perception_pick.py` — mock detector+segmenter,
  assert world coords ≈ GSO position.
- MOD `docs/v2.4_live_repl_checklist.md` — 5-step guide for Yusen.
- MOD `progress.md` — v2.4 change log section.
- MOD `agents/devlog/status.md` — reflect v2.4 progress.

**Verify**: `scripts/verify_perception_pick.py --dry-run` exit 0 in <2 s.

---

## Wave 5 — QA

Parallel code-review + security-review (subagents).

### Focus areas

- `yoloe_detector.py`: class-caching correctness, GPU fallback, query translation edge cases.
- `sam3_segmenter.py`: fallback determinism, mask upscale quality.
- `pointcloud_projection.py`: numeric stability, integer-division bugs at bbox boundaries.
- `sanity_gates.py`: boundary inclusivity (≤ vs <), error messages.
- `go2_perception.py`: thread-safety (still single-thread guarantee?), log noise level.
- `go2_room.xml`: MJCF validation, collision efficiency.
- `.gitattributes`: LFS tracking is correct (avoid committing huge files accidentally).
- New deps: `ultralytics` supply chain, license (AGPL-3 — confirm
  OK for VectorRobotics redistribution model).

### Gate

All CRITICAL / HIGH findings fixed or explicitly deferred with CEO
sign-off. Commit fix batch as `fix(v2.4): QA round — findings addressed`.

---

## Wave 6 — CEO live-REPL smoke + release

Yusen runs `go2sim with_arm=1` → `抓起蓝色瓶子` × 5. Expected
criteria:

1. At least 4/5 attempts return valid world coords within 0.1 m of the
   targeted GSO bottle pose (read from MJCF).
2. SAM 3 path active (or fallback SAM 2.1 with logged WARN).
3. CLI shows object summary line with label + cm coords.
4. No phantom far-distance navigations (sanity gates working).

On success: merge to main (or protected branch), tag `v2.4.0-rc1`.

---

## Task dispatch order (dispatcher script)

```
W0  T0   (dispatcher)
W1  T1   Alpha  →  T2   Beta  →  T3   Gamma       # serial
W1  gate: pytest tests/unit/perception/ -v  (≥ 27 new green)
W2  T4   Alpha  →  T5   Beta  →  T6   Gamma       # serial
W2  gate: pytest tests/unit/perception/ tests/integration/test_go2_camera_pose.py -v
W3  T7   Alpha  →  T8   Beta                       # serial
W3  gate: scene load + narrow integration pytest
W4  T9   Alpha  →  T10  Gamma                      # serial
W4  gate: verify_perception_pick.py --dry-run exit 0
W5  QA   code-reviewer  +  security-reviewer      # parallel subagents (read-only)
W6  CEO  live-REPL smoke  +  release decision
```

Estimated wall-clock: **2–3 days** at serial-subagent cadence.

---

## Subagent prompt template (for dispatcher)

Every task prompt MUST include:

```
[context]
Task {ID} of v2.4 Perception Overhaul SDD cycle.
Spec: .sdd/spec.md
Plan: .sdd/plan.md §{section}
Depends on tasks: {list} (already landed at commits {SHAs})

[safety rules]
- DO NOT run `pytest tests/` or `pytest tests/integration/` — only
  the narrow file listed in "Verify" below.
- DO NOT import from: pipeline, track_anything, mujoco, realsense,
  tracker, ultralytics (unless explicitly in your task's scope).
- Stick to the files listed in "Files" — don't refactor adjacent code.
- Follow TDD: write RED tests first, confirm they fail, then GREEN,
  then REFACTOR.

[deliverable]
{files list, test names, verify command}

[completion criteria]
- All tests listed pass in the narrow-scope verify command.
- ruff check clean on modified files.
- Commit under `[alpha|beta|gamma] {type}(v2.4): <description>`.
- Report commit SHA and test pass count back to dispatcher.
```
