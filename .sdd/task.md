# SDD Tasks — v2.0.1 V-Graph Cross-Room Edge Fix

**Spec**: `.sdd/spec.md`
**Plan**: `.sdd/plan.md`
**Total tasks**: 9
**Waves**: 3

---

## Dependency Graph

```
                          ┌───────────┐
                          │ T1 RED    │  ─┐
                          │ bridge    │   │
                          │ cleanup   │   │
                          │ tests     │   │ Wave 1
                          └─────┬─────┘   │ (3 parallel, independent test writes)
                                │         │
     ┌───────────┐              │         │
     │ T3 RED    │             │         │
     │ launch    │             │         │
     │ param     │─────────────┼────── ──┤
     │ tests     │              │
     └─────┬─────┘              │
           │        ┌───────────┐
           │        │ T5        │
           │        │ integration
           │        │ harness    ─ ─ ─ ─ ┘
           │        │ (scaffold,
           │        │  opt-in)   │
           │        └─── ── ─ ──┘
           │              │
  ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ │─ ─ ─ ─ ─ ─ ─ ─ Wave 1 gate: T1+T3 RED confirmed
           │              │
           ▼              ▼
  ┌────────────┐    ┌────────────┐
  │ T2 GREEN   │    │ T4 GREEN   │
  │ bridge     │    │ launch     │         Wave 2 (2 parallel)
  │ revert +   │    │ scripts ×3 │
  │ test clean │    │ patch      │
  └─────┬──────┘    └─────┬──────┘
        │                 │
  ─ ─ ─ │─ ─ ─ ─ ─ ─ ─ ─  │─ ─ ─ ─ ─ ─ ─ ─ Wave 2 gate: T1+T3 tests GREEN
        └────────┬────────┘
                 ▼
          ┌──────────────┐
          │ T6 full      │
          │ regression   │
          │ pytest       │
          └──────┬───────┘
                 │
                 ├──────────┬──────────┐   Wave 3 (sequential + parallel commits)
                 ▼          ▼          │
          ┌───────────┐ ┌───────────┐  │
          │ T7 commit │ │ T8 commit │  │
          │ (a)       │ │ (b)       │  │
          │ bridge    │ │ launch    │  │
          │ revert    │ │ fix       │  │
          └─────┬─────┘ └─────┬─────┘  │
                │             │        │
                └──────┬──────┘        │
                       ▼               │
                ┌──────────────┐       │
                │ T9 AC6 CEO   │ ◀─────┘
                │ manual test  │
                └──────────────┘
```

---

## Task List

### Task 1: Bridge cleanup regression tests (RED)
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: none
- **Package**: vector_os_nano (tests)
- **Wave**: 1
- **Input**:
  - `scripts/go2_vnav_bridge.py` (to understand current structure)
  - `tests/harness/nav_debug_helpers.py` (for `read_bridge_source()`)
  - `tests/harness/test_level36_terrain_replay.py` (pattern reference)
- **Output**: `tests/harness/test_level41_vgraph_bridge_cleanup.py` (new file)
- **TDD Deliverables**:
  - RED: Write 6 tests asserting the bridge has NO terrain_map publisher code:
    1. `test_bridge_has_no_terrain_map_publisher_declaration` — `"_terrain_map_pub"` NOT in bridge __init__
    2. `test_bridge_has_no_terrain_map_ext_publisher_declaration` — `"_terrain_map_ext_pub"` NOT in bridge __init__
    3. `test_bridge_has_no_sync_terrain_to_far_method` — `"def _sync_terrain_to_far"` NOT in bridge source
    4. `test_bridge_has_no_publish_accumulated_terrain_method` — `"def _publish_accumulated_terrain"` NOT in bridge
    5. `test_replay_terrain_publishes_only_registered_scan` — `_replay_terrain` body references `_pc_pub` but NOT `_terrain_map_pub` / `_terrain_map_ext_pub`
    6. `test_no_terrain_map_topic_in_publisher_declarations` — `"/terrain_map"` NOT in bridge __init__ (but may appear in comments/docstrings)
  - These tests will FAIL (RED) because bridge still has those — confirms TDD gate.
- **Acceptance Criteria**: File created, 6 tests present, all 6 FAIL on current bridge
- **Verify**:
  ```bash
  pytest tests/harness/test_level41_vgraph_bridge_cleanup.py -v --tb=short
  # Expected: 6 failed (RED confirmed)
  ```

### Task 2: Bridge revert + test_level36 cleanup + test_level40 deletion (GREEN)
- **Status**: [ ] pending
- **Agent**: alpha
- **Depends**: Task 1 (RED confirmed)
- **Package**: vector_os_nano (scripts + tests)
- **Wave**: 2
- **Input**:
  - `.sdd/plan.md` §2.A (bridge cut list)
  - `.sdd/plan.md` §2.C (test file handling)
  - `scripts/go2_vnav_bridge.py` current state
  - `tests/harness/test_level36_terrain_replay.py` (to surgical-edit)
  - `tests/harness/test_level40_far_terrain_feed.py` (to delete)
- **Output**:
  - `scripts/go2_vnav_bridge.py` — modified
  - `tests/harness/test_level36_terrain_replay.py` — modified (remove WRONG-direction assertions)
  - `tests/harness/test_level40_far_terrain_feed.py` — DELETED
- **TDD Deliverables**:
  - GREEN: Delete from `scripts/go2_vnav_bridge.py`:
    - Publisher lines (~lines 231–236): `self._terrain_map_pub = ...` and `self._terrain_map_ext_pub = ...`
    - Timer line (~line 287): `self.create_timer(5.0, self._sync_terrain_to_far)`
    - Method `_sync_terrain_to_far()` (~lines 489–516, ~28 lines)
    - Method `_publish_accumulated_terrain()` alias (~lines 517–519, 3 lines)
    - In `_replay_terrain()`: lines 479–480 (`self._terrain_map_pub.publish(msg)` and `self._terrain_map_ext_pub.publish(msg)`) — keep `self._pc_pub.publish(msg)` at line 478
    - Shutdown hook call at ~line 1470: `self._publish_accumulated_terrain()` — remove entire line
  - Surgical update `tests/harness/test_level36_terrain_replay.py`:
    - Remove entire `TestTerrainReplayPublisherDeclarations` class (tests 63–98ish)
    - In any other class: flip assertions that check for `_terrain_map_pub` / `_terrain_map_ext_pub` presence to ABSENCE, OR just delete those test methods
    - Keep tests asserting `_pc_pub` + `/registered_scan` invariants
    - Keep tests for TerrainAccumulator roundtrip (those belong to L30 anyway)
  - Delete `tests/harness/test_level40_far_terrain_feed.py` entirely.
  - REFACTOR: Run `pytest tests/harness/test_level41_*.py test_level36_*.py test_level30_*.py -v` → expect all GREEN.
- **Acceptance Criteria**:
  - All 6 T1 tests pass
  - `test_level36_terrain_replay.py` passes with reduced test count (~6 tests remain vs ~15 before)
  - `test_level40_far_terrain_feed.py` does not exist
  - `test_level30_terrain_persist.py` still passes (accumulator tests unaffected)
- **Verify**:
  ```bash
  pytest tests/harness/test_level41_vgraph_bridge_cleanup.py -v
  pytest tests/harness/test_level36_terrain_replay.py tests/harness/test_level30_terrain_persist.py -v
  ls tests/harness/test_level40_far_terrain_feed.py 2>&1 | grep -q "No such" && echo "OK deleted"
  ```

### Task 3: Launch script parameter tests (RED)
- **Status**: [ ] pending
- **Agent**: beta
- **Depends**: none
- **Package**: vector_os_nano (tests)
- **Wave**: 1
- **Input**:
  - `scripts/launch_nav_only.sh` (to see current broken state)
  - `scripts/launch_explore.sh` (reference for correct pattern — lines 97–108)
- **Output**: `tests/harness/test_level42_launch_terrain_params.py` (new file)
- **TDD Deliverables**:
  - RED: Write parametrized tests. Structure:
    ```python
    import re
    import pytest
    from pathlib import Path

    _REPO = Path(__file__).resolve().parent.parent.parent
    
    _REQUIRED_PARAMS = {
        "maxRelZ": 1.5,
        "clearDyObs": "true",
        "obstacleHeightThre": 0.15,
        "maxGroundLift": 0.05,
    }

    @pytest.mark.parametrize("script", [
        "launch_nav_only.sh",
        "launch_nav_explore.sh",
        "test_integration.sh",
    ])
    class TestTerrainAnalysisParams:
        def test_script_passes_maxRelZ(self, script): ...
        def test_script_passes_clearDyObs(self, script): ...
        def test_script_passes_obstacleHeightThre(self, script): ...
        def test_script_uses_ros_args(self, script): ...
        def test_ext_also_has_params(self, script): ...  # terrainAnalysisExt line
    ```
  - These tests FAIL (RED) against the 3 broken scripts.
- **Acceptance Criteria**: 15 parametrized tests (5 × 3 scripts), all FAIL
- **Verify**:
  ```bash
  pytest tests/harness/test_level42_launch_terrain_params.py -v --tb=short
  # Expected: 15 failed
  ```

### Task 4: Launch script patch ×3 (GREEN)
- **Status**: [ ] pending
- **Agent**: beta
- **Depends**: Task 3 (RED confirmed)
- **Package**: vector_os_nano (scripts)
- **Wave**: 2
- **Input**:
  - `.sdd/plan.md` §2.B (exact replacement block)
  - `scripts/launch_explore.sh` lines 97–108 (reference)
- **Output**:
  - `scripts/launch_nav_only.sh` — patched (lines ~66–69 replaced)
  - `scripts/launch_nav_explore.sh` — patched (lines ~58–61 replaced)
  - `scripts/test_integration.sh` — patched (lines ~61–63 replaced)
- **TDD Deliverables**:
  - GREEN: Patch all 3 scripts to match `launch_explore.sh` pattern. Use identical param values:
    ```bash
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
  - Preserve any surrounding comments / echo statements.
  - REFACTOR: Run T3 tests, confirm GREEN.
- **Acceptance Criteria**: All 15 T3 tests pass
- **Verify**:
  ```bash
  pytest tests/harness/test_level42_launch_terrain_params.py -v
  # Expected: 15 passed
  bash -n scripts/launch_nav_only.sh && bash -n scripts/launch_nav_explore.sh && bash -n scripts/test_integration.sh
  # Expected: all 3 shell-syntax OK
  ```

### Task 5: V-Graph integration harness (scaffold, opt-in)
- **Status**: [ ] pending
- **Agent**: gamma
- **Depends**: none
- **Package**: vector_os_nano (tests)
- **Wave**: 1
- **Input**:
  - `tests/harness/test_vgraph_debug.py` (existing pattern for MuJoCo raycasting)
  - `tests/e2e/test_explore_navigate_e2e.py` (existing pattern for subprocess-launching nav stack)
  - `vector_os_nano/hardware/sim/scene_room.xml` (room bounding boxes for edge-spans-rooms assertion)
  - `scripts/launch_nav_only.sh` (subprocess target)
- **Output**: `tests/harness/test_level43_vgraph_integration.py` (new file)
- **TDD Deliverables**:
  - Scaffold a SINGLE test:
    ```python
    @pytest.mark.ros2
    @pytest.mark.slow
    def test_vgraph_forms_cross_room_edge_through_living_hall_door():
        """AC4: after driving Go2 through living_hall door, /robot_vgraph has ≥1 cross-room edge."""
        # 1. Launch scripts/launch_nav_only.sh as subprocess group
        # 2. Launch go2_vnav_bridge as subprocess (spawn Go2 at living_room center)
        # 3. Wait for /robot_vgraph publisher (timeout 15s)
        # 4. Publish /joy commands to drive forward through door for 20s
        # 5. Collect /robot_vgraph snapshots via subscriber
        # 6. Parse graph: find edge whose endpoints fall in different known rooms
        # 7. Assert ≥1 cross-room edge exists
        # 8. Teardown: kill nav stack process group
    ```
  - Include room bounding boxes inline as constants (derived from `scene_room.xml`):
    ```python
    LIVING_ROOM_BBOX = (0, 6, 0, 5)    # x_min, x_max, y_min, y_max
    HALLWAY_BBOX = (6, 14, 0, 10)
    # ... (6 rooms)
    ```
  - Helper: `_points_in_different_rooms(p1, p2) -> bool`.
  - Mark `@pytest.mark.ros2 @pytest.mark.slow` — skipped by default, run via `pytest -m ros2`.
- **Acceptance Criteria**:
  - File created
  - `pytest tests/harness/test_level43_vgraph_integration.py --collect-only` shows 1 test
  - `pytest tests/harness/test_level43_vgraph_integration.py -q` (default marks) reports 0 tests run (skipped)
  - `pytest -m ros2 --collect-only tests/harness/test_level43_vgraph_integration.py` shows 1 test
- **Verify**:
  ```bash
  pytest tests/harness/test_level43_vgraph_integration.py --collect-only 2>&1 | grep "1 test"
  pytest tests/harness/test_level43_vgraph_integration.py -q 2>&1 | grep "deselected\|skipped"
  ```
- **Note**: Actual PASS verification is deferred to AC6 (manual CEO run). This task only scaffolds.

### Task 6: Full regression suite
- **Status**: [ ] pending
- **Agent**: dispatcher (me)
- **Depends**: Task 2, Task 4, Task 5
- **Package**: vector_os_nano (all)
- **Wave**: 3
- **Input**: working tree with Wave 2 complete
- **Output**: regression report
- **Deliverables**:
  - Run full suite excluding known-stale prompt tests
  - Count pass/fail/skip
  - If any failure: escalate to Architect (root cause, not patch-over)
- **Acceptance Criteria**:
  - 0 failures (apart from 2 stale `test_prompt.py` tests, which are excluded)
  - Total tests ≈ 3088 (down from 3267 due to removed 180 + added 7)
- **Verify**:
  ```bash
  pytest tests/ --ignore=tests/unit/vcli/test_prompt.py -q --tb=no 2>&1 | tail -5
  # Expected: "XXXX passed" with 0 failed
  ```

### Task 7: Commit (a) — bridge revert
- **Status**: [ ] pending
- **Agent**: dispatcher (me)
- **Depends**: Task 6 (green)
- **Package**: vector_os_nano (git)
- **Wave**: 3
- **Input**: working tree with Task 2 changes
- **Output**: git commit
- **Deliverables**:
  - Stage ONLY bridge-related files:
    - `scripts/go2_vnav_bridge.py`
    - `tests/harness/test_level36_terrain_replay.py`
    - `tests/harness/test_level40_far_terrain_feed.py` (deletion)
    - `tests/harness/test_level41_vgraph_bridge_cleanup.py`
  - Commit message:
    ```
    revert: remove bridge duplicate publishers on /terrain_map and /terrain_map_ext

    The bridge was also publishing to /terrain_map and /terrain_map_ext in
    addition to terrainAnalysis/terrainAnalysisExt, causing duplicate-
    publisher data pollution for FAR. Its intensity semantics differed
    (absolute z vs height-above-ground), and FAR crops global terrain to
    7.5m anyway.

    - Remove _sync_terrain_to_far() periodic timer
    - Remove _publish_accumulated_terrain() alias
    - Remove _terrain_map_pub and _terrain_map_ext_pub publishers
    - _replay_terrain() now publishes only /registered_scan
    - Delete test_level40_far_terrain_feed.py (enforced removed invariant)
    - Trim test_level36_terrain_replay.py to keep only /registered_scan
    - Add test_level41_vgraph_bridge_cleanup.py as regression guard

    Net: -260 lines.
    ```
- **Acceptance Criteria**: Commit created, `git log -1 --stat` shows expected files
- **Verify**:
  ```bash
  git log -1 --stat | head -15
  ```

### Task 8: Commit (b) — launch script fix
- **Status**: [ ] pending
- **Agent**: dispatcher (me)
- **Depends**: Task 6 (green)
- **Package**: vector_os_nano (git)
- **Wave**: 3 (can run parallel with T7 — different files)
- **Input**: working tree with Task 4 changes
- **Output**: git commit
- **Deliverables**:
  - Stage ONLY launch/test files:
    - `scripts/launch_nav_only.sh`
    - `scripts/launch_nav_explore.sh`
    - `scripts/test_integration.sh`
    - `tests/harness/test_level42_launch_terrain_params.py`
    - `tests/harness/test_level43_vgraph_integration.py`
  - Commit message:
    ```
    fix: pass Go2 terrain_analysis params in launch_nav_* scripts

    launch_nav_only.sh and launch_nav_explore.sh used `ros2 run
    terrain_analysis terrainAnalysis` without parameters, inheriting
    the C++ default maxRelZ=0.2. For Go2 (vehicleZ~0.27m), this
    rejects any lidar point above z=0.47 at close range — wall and
    doorframe points get dropped, so contour_detector never sees
    door edges and FAR never builds V-Graph corners at doorways.

    Aligned with the same pattern launch_explore.sh and launch_vnav.sh
    already use: --ros-args -p maxRelZ:=1.5 plus other Go2-tuned flags.

    - Patch launch_nav_only.sh, launch_nav_explore.sh, test_integration.sh
    - Add test_level42 param regression guard
    - Scaffold test_level43 V-Graph integration harness (opt-in)
    ```
- **Acceptance Criteria**: Commit created, `git log -1 --stat` shows expected files
- **Verify**:
  ```bash
  git log -1 --stat | head -15
  ```

### Task 9: AC6 — CEO manual verification
- **Status**: [ ] pending
- **Agent**: CEO (Yusen)
- **Depends**: Task 7, Task 8
- **Package**: vector_os_nano
- **Wave**: 3
- **Input**: running laptop with clean working tree post-commits
- **Output**: CEO verdict (PASS / FAIL)
- **Deliverables**:
  1. Dispatcher hands control to Yusen with clear instructions:
     ```
     git status         # clean
     vector-cli          # launch REPL
     # In REPL: "启动仿真"    # spawns MuJoCo + nav stack via launch_nav_only.sh
     # In REPL: "去卧室"     # test cross-room nav
     # Open RViz (separate terminal): rviz2 -d config/vnav.rviz
     # Watch /robot_vgraph marker array — should show nodes + edges
     ```
  2. CEO observes:
     - V-Graph nodes appear at wall corners and doorways
     - At least one edge crosses through a doorway (connects rooms)
     - Go2 navigates via FAR route (not door-chain fallback)
  3. CEO reports PASS or FAIL to dispatcher
- **Acceptance Criteria (spec AC6)**:
  - Yusen visually confirms V-Graph edge forms
  - Go2 reaches target room
- **Verify**: CEO verbal/text confirmation
- **If FAIL**: Dispatcher opens Phase B SDD round. Commits T7+T8 remain (they're good hygiene regardless); Phase B investigates `IsInDirectConstraint` / vote accumulation via FAR debug logs.

---

## Execution Waves

| Wave | Tasks | Agents | Gate |
|------|-------|--------|------|
| 1 | T1, T3, T5 | Alpha, Beta, Gamma | T1 + T3 tests are RED on current tree; T5 test skeleton compiles |
| 2 | T2, T4 | Alpha, Beta (Gamma idle) | T1 + T3 tests flip to GREEN |
| 3 | T6 → T7 + T8 (parallel) → T9 | Dispatcher + CEO | Full suite green, 2 commits created, CEO manual PASS |

**Wave 1 parallelism**: T1/T3/T5 touch completely disjoint files. Safe to run 3 agents concurrently.

**Wave 2 parallelism**: T2 touches bridge + tests; T4 touches launch scripts. Disjoint. 2 agents.

**Wave 3**: T6 must complete before commits; T7/T8 touch disjoint file sets and can commit in parallel; T9 is manual and sequential after commits.

---

## QA Gate Policy

Per plan classification (non-architectural), no CEO approval gate on task.md itself. QA review:
- ✅ Tasks atomic (each has one deliverable)
- ✅ TDD deliverables explicit (RED tests in T1/T3 before GREEN in T2/T4)
- ✅ Dependencies correct (Task 2 depends on Task 1; Task 4 on Task 3; etc.)
- ✅ Wave grouping optimal (max parallelism respected)
- ✅ Atomic commits planned (2 logical commits matching CEO D4)
- ✅ CEO only needed at AC6 (final manual test)

Ready to execute.
