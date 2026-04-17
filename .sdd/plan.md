# SDD Plan — v2.0.1 V-Graph Cross-Room Edge Fix (Phase A)

**Type**: Non-architectural (bug fix + revert + test cleanup)
**Spec**: `.sdd/spec.md` — approved 2026-04-16
**Base**: `feat/v2.0-vectorengine-unification` (uncommitted tree)
**CEO gate**: Async review (no new interfaces, no new nodes)

---

## 1. Design Summary

No new components. This is a **targeted revert + config fix + test cleanup**.

```
┌────────────────────┐       ┌─────────────────────────┐
│ go2_vnav_bridge    │──────▶│ /registered_scan        │
│  (SIMPLIFIED)      │       │ (live lidar, sole pub)  │
│  REMOVED:          │       └─────────────────────────┘
│  - _terrain_map_pub│                  │
│  - _terrain_map_   │                  ▼
│    ext_pub         │       ┌──────────────────────┐  ┌──────────────────┐
│  - _sync_terrain_  │       │ terrainAnalysis      │─▶│ /terrain_map     │ (single pub)
│    to_far()        │       │ (ros2 run + --ros-   │  └──────────────────┘
│  - _publish_       │       │  args,                │          │
│    accumulated_    │       │  maxRelZ=1.5 …)      │          ▼
│    terrain()       │       └──────────────────────┘  ┌──────────────────┐
│  - terrain_map     │       ┌──────────────────────┐  │   FAR planner    │
│    pub in replay   │       │ terrainAnalysisExt   │─▶│  (unchanged)     │
│  KEPT:             │       │ (ros2 run + --ros-   │  │  builds V-Graph  │
│  - TerrainAccumul. │       │  args)                │  │  from correct    │
│  - save/load       │       └──────────────────────┘  │  obstacle cloud  │
│  - _replay_terrain │                                 └──────────────────┘
│    → only to       │
│    /registered_scan│
└────────────────────┘
```

Key invariants after fix:
1. **Single publisher** per topic — bridge only publishes `/registered_scan`, not terrain_map/ext.
2. **`terrainAnalysis` gets correct params** — driven by `--ros-args -p maxRelZ:=1.5 …`, same as `launch_explore.sh` already uses.
3. **TerrainAccumulator kept** for persistence (save/load, startup replay to /registered_scan only). Accumulator itself is fine; only its misuse as a second `/terrain_map` publisher was wrong.

## 2. Files Changed

### A. Bridge (Python) — `scripts/go2_vnav_bridge.py`

Cuts:
- `self._terrain_map_pub = self.create_publisher(... "/terrain_map", ...)` (line 231-233)
- `self._terrain_map_ext_pub = self.create_publisher(... "/terrain_map_ext", ...)` (line 234-236)
- `self.create_timer(5.0, self._sync_terrain_to_far)` (line 287)
- Method `_sync_terrain_to_far(self)` (lines 489-516)
- Method `_publish_accumulated_terrain(self)` alias (lines 517-519)
- Two lines in `_replay_terrain()`: `self._terrain_map_pub.publish(msg)` + `self._terrain_map_ext_pub.publish(msg)` (lines 479-480) — keep `self._pc_pub.publish(msg)`
- The shutdown hook call `self._publish_accumulated_terrain()` (line 1470)

Keeps:
- `_pc_pub` publishing to `/registered_scan` (unchanged).
- `TerrainAccumulator` class (unchanged).
- `save_terrain()` / `load_terrain()` for persistence.
- `_replay_terrain()` publishes to `/registered_scan` only (terrain replay on startup still works — TARE and terrain_analysis pick it up naturally).

Estimated diff: −55 lines.

### B. Launch scripts (shell)

**`scripts/launch_nav_only.sh`** — replace lines 66-69:
```bash
# BEFORE
ros2 run terrain_analysis terrainAnalysis &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt &
PIDS+=($!)

# AFTER (matches launch_explore.sh pattern verbatim)
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p clearDyObs:=true \
  -p minDyObsDis:=0.14 \
  -p minOutOfFovPointNum:=20 \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 \
  -p limitGroundLift:=true \
  -p maxGroundLift:=0.05 \
  -p minDyObsVFOV:=-30.0 \
  -p maxDyObsVFOV:=35.0 &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 &
PIDS+=($!)
```

**`scripts/launch_nav_explore.sh`** — same fix at lines 58-61.
**`scripts/test_integration.sh`** — same fix at lines 61-63.

No changes to `launch_explore.sh`, `launch_vnav.sh` (already correct).

### C. Tests (updates + new)

Tests to DELETE (enforce reverted-out bug):
- `tests/harness/test_level40_far_terrain_feed.py` (177 lines, entire file) — asserts `_publish_accumulated_terrain` + terrain_map publishing, both removed.

Tests to UPDATE:
- `tests/harness/test_level36_terrain_replay.py` — remove/invert `TestTerrainReplayPublisherDeclarations` class. Assertions flip: bridge must NOT have `_terrain_map_pub` / `_terrain_map_ext_pub`. Reduce to testing only `_pc_pub` on `/registered_scan`. Expected: drops from ~15 tests to ~6.

Tests to ADD:
- `tests/harness/test_level41_vgraph_bridge_cleanup.py` (new) — T1 regression guards:
  ```python
  def test_bridge_has_no_terrain_map_publisher(): ...
  def test_bridge_has_no_sync_terrain_to_far(): ...
  def test_bridge_has_no_publish_accumulated_terrain(): ...
  def test_replay_terrain_publishes_only_registered_scan(): ...
  ```
- `tests/harness/test_level42_launch_terrain_params.py` (new) — T2:
  ```python
  @pytest.mark.parametrize("script", [
      "launch_nav_only.sh",
      "launch_nav_explore.sh",
      "test_integration.sh",
  ])
  def test_script_passes_terrain_params(script): ...
  ```
- `tests/harness/test_level43_vgraph_integration.py` (new, opt-in) — T3:
  ```python
  @pytest.mark.ros2  # skipped unless ROS2 + nav_stack installed
  @pytest.mark.slow  # ~60s
  def test_vgraph_forms_cross_room_edge_through_living_hall_door(): ...
  ```

## 3. Data Flow After Fix

```
MuJoCo lidar
  ↓ Go2VNavBridge._publish_pointcloud (20 Hz)
  ↓
/registered_scan  [intensity = z - ground_z, ceiling-filtered at 1.0m]
  │
  ├────▶ terrainAnalysis (maxRelZ=1.5)
  │         ↓
  │      /terrain_map [intensity = z - local_ground]
  │         ↓
  │      FAR (as /scan_cloud) → ScanCallBack
  │
  ├────▶ terrainAnalysisExt (maxRelZ=1.5)
  │         ↓
  │      /terrain_map_ext [intensity = z - local_ground]
  │         ↓
  │      FAR (as /terrain_cloud) → TerrainCallBack
  │         ↓
  │      CropBoxCloud 7.5m → ExtractFreeAndObsCloud (kFreeZ=0.15)
  │         ↓
  │      surround_obs_cloud_ → contour_detector → corners → V-Graph
  │
  └────▶ TARE (localFrontier, waypoint generation)
  └────▶ FAR (as /terrain_local_cloud) → TerrainLocalCallBack
```

After fix: FAR's `surround_obs_cloud_` actually contains wall and doorframe points → `contour_detector` places corners at door edges → V-Graph gets nodes → edge voting can succeed.

## 4. Wave Structure (Task Grouping)

Work is parallelizable; grouping for Phase 3 (sdd-tasks) task decomposition:

### Wave 1 — Parallel, independent file touches
| # | Task | Agent | Files | ~LOC |
|---|------|-------|-------|------|
| W1-A | Bridge revert | Alpha | `scripts/go2_vnav_bridge.py` | −55 |
| W1-B | Launch script fix ×3 | Beta | `scripts/launch_nav_only.sh`, `scripts/launch_nav_explore.sh`, `scripts/test_integration.sh` | +30 |
| W1-C | Test cleanup | Gamma | Delete `test_level40_*.py`, update `test_level36_*.py` | −200 |

### Wave 2 — Depends on Wave 1
| # | Task | Agent | Files |
|---|------|-------|-------|
| W2-A | T1 regression tests | Alpha | `test_level41_vgraph_bridge_cleanup.py` |
| W2-B | T2 launch script tests | Beta | `test_level42_launch_terrain_params.py` |
| W2-C | T3 integration harness | Gamma | `test_level43_vgraph_integration.py` |

### Wave 3 — Verification (sequential)
| # | Task | Owner |
|---|------|-------|
| W3-1 | Full `pytest tests/` (exclude stale prompt tests) | Dispatcher |
| W3-2 | Commit (a) bridge revert — atomic | Dispatcher |
| W3-3 | Commit (b) launch script fix — atomic | Dispatcher |
| W3-4 | **AC6 manual**: Yusen runs `vector-cli → "去卧室"`, observes FAR V-Graph route | CEO |

If W3-4 passes: Phase A complete, Phase B not needed, ready for v2.0 merge.
If W3-4 fails: escalate to Phase B (new SDD round for FAR internals).

## 5. Test Strategy

### Unit-style (static analysis, no ROS2)
Pattern already established in `tests/harness/test_level30_*.py`, `test_level36_*.py`: read source text, assert presence/absence of symbols. Use `nav_debug_helpers.read_bridge_source()`.

These tests are fast (<1s), runnable in any CI, reliable.

### Integration (live ROS2)
Pattern from existing `tests/e2e/test_explore_navigate_e2e.py`: subprocess-launch the nav stack, wait for readiness, drive via `/joy` or direct /speed, subscribe to output topics.

T3 harness will:
1. Launch `scripts/launch_nav_only.sh` in subprocess.
2. Launch bridge in same process group.
3. Wait for `/robot_vgraph` topic to exist (timeout 15s).
4. Send /joy forward commands for 20s to drive Go2 through living_room→hallway door.
5. Subscribe to `/robot_vgraph`, collect graph snapshots.
6. Assert: ≥1 edge connects two nodes whose positions fall in different room bounding boxes (from `scene_room.xml` known layout).

Marked `@pytest.mark.ros2 @pytest.mark.slow`, opt-in via `pytest -m ros2`.

### Regression baseline
`pytest tests/ --ignore=tests/unit/vcli/test_prompt.py -q` must show 0 failures after all waves.

Expected delta vs current state:
- −177 tests from deleted `test_level40_*.py`
- −9 tests invalidated in `test_level36_*.py` (from `TestTerrainReplayPublisherDeclarations`)
- +4 from new `test_level41_*.py`
- +3 from new `test_level42_*.py` (3 parametrized)
- +0 from new `test_level43_*.py` (skipped without ROS2)

Net: ~3267 → ~3088 total, all passing.

## 6. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Phase A insufficient → R3 (FAR internal) is the real cause | Med | High | Phase B contingency already scoped in spec §9. Off-ramp clear. |
| 2 | Removing `_publish_accumulated_terrain` breaks a shutdown code path I missed | Low | Med | Grep pass before removal; shutdown hook `save_terrain()` is separate and stays intact. |
| 3 | `TerrainAccumulator` still exists but nothing consumes `to_pointcloud()` — dead code? | Low | Low | `_replay_terrain()` still uses it (publishes replay to /registered_scan on startup). Not dead. |
| 4 | Integration test T3 flakes on slow machines (subprocess ROS2 bringup) | Med | Low | Long timeouts (15s bringup, 20s drive); opt-in via `@pytest.mark.ros2`; no CI-blocking. |
| 5 | Launch script edit breaks line-count-sensitive external users | Very low | Low | No one depends on shell script line numbers. |
| 6 | Regression: terrainAnalysis with `maxRelZ=1.5` lets ceiling points pollute obs cloud | Low | Med | Bridge ceiling filter at 1.0m on `/registered_scan` upstream already caps input. |
| 7 | `launch_nav_explore.sh` is only referenced in `explore.py:_launch_nav_explore()` — does it actually get called? | Med | Low | Check call site; if dead, just delete instead of fix. |

## 7. Pre-execution Verification Checks

Before starting Wave 1:
```bash
# Confirm the uncommitted changes match spec scope (shouldn't be surprises)
git status
git diff scripts/go2_vnav_bridge.py | head -30

# Confirm launch_explore.sh is the right reference
diff <(grep "ros2 run terrain_analysis" scripts/launch_explore.sh) \
     <(grep "ros2 run terrain_analysis" scripts/launch_nav_only.sh)

# Baseline test run (capture pre-fix state)
pytest tests/harness/ -q --tb=no 2>&1 | tail -5
```

## 8. CEO-Visible Summary

- **Architectural?** No. Non-architectural — revert of recent bridge experiment + config parameter sync + test cleanup.
- **Nodes affected?** None added/removed. `terrainAnalysis` / `terrainAnalysisExt` get correct params (they were already in launch graph).
- **Topics affected?** `/terrain_map` + `/terrain_map_ext` become single-publisher (from 2 → 1).
- **Interfaces changed?** No.
- **External dependencies?** None new.
- **Estimated effort?** ~2 hours: 3 parallel Wave-1 tasks (~30min each), 3 parallel Wave-2 tasks (~30min each), + commit + AC6 manual test.

## 9. Handoff to Phase 3 (sdd-tasks)

Phase 3 will break the waves above into TDD-structured tasks with explicit Red/Green/Refactor cycles per task:

- W1-A: [Red] write regression tests asserting publishers are absent → fail (they still exist) → [Green] delete publishers + methods → tests pass
- W1-B: [Red] write parametrized script-content tests → fail → [Green] patch scripts → tests pass
- W1-C: straight test deletions — no TDD cycle, just remove
- W2-A/B: tests written in W1 are already these — merge into W1 tasks
- W2-C: integration harness scaffold → [Manual] run live once → [Commit] mark passing

---

**Classification**: Non-architectural. CEO can review async; agent team proceeds.
