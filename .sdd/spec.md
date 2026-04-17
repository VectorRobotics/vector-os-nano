# SDD Spec — V-Graph Cross-Room Edge Fix

**Version**: v2.0.1-vgraph-fix
**Date**: 2026-04-16
**Architect**: Opus (Dispatcher)
**Type**: Bug fix + Diagnostic (cross-package)
**Base**: feat/v2.0-vectorengine-unification (uncommitted)

---

## 1. Intent

Diagnose and fix why **FAR V-Graph never builds edges through doorways**, blocking cross-room navigation. Navigate skill currently falls back to door-chain (slow, unreliable); FAR V-Graph is supposed to be the primary planner.

## 2. Context (from deep analysis already performed)

### Empirically known (from tests/harness/test_vgraph_debug.py)
- At ceiling_filter=1.0m, MuJoCo raycaster proves 5/6 doors have **0 obstacles on visibility ray** (terrain is not the problem).
- V-Graph in `/robot_vgraph` observed to remain empty or without cross-room edges.

### Found by code analysis this session
Three candidate root causes, ranked:

**R1 (HIGH probability)** — `explore.py` calls `scripts/launch_nav_only.sh` which starts `ros2 run terrain_analysis terrainAnalysis` **with no parameters**, inheriting the C++ defaults:
- `maxRelZ=0.2` (vs 1.5 in `launch_explore.sh`/`launch_vnav.sh` which were manually tuned)
- `clearDyObs=false`
- `obstacleHeightThre=0.2`

Effect: With `vehicleZ=0.27` and `disRatioZ=0.2`, terrain_analysis rejects any lidar point with `z > 0.27+0.2+0.2*dis`. At close range (< 2m), that's anything above ~0.5m. **Doorframes (~2m tall), walls (~2m), and even most wall midsections get discarded**, so `contour_detector` can't see the door frame shape → no corner nodes placed at doorways → no V-Graph to form edges from.

**R2 (CONFIRMED BUG)** — Bridge's uncommitted `_sync_terrain_to_far()` publishes to `/terrain_map` and `/terrain_map_ext`, same topics `terrainAnalysis`/`terrainAnalysisExt` already publish. `_replay_terrain()` does the same on startup. Both use `TerrainAccumulator.to_pointcloud()` which sets `intensity = z` (absolute), but FAR expects `intensity = z - ground_z`. Double-publishing + wrong semantics → FAR sees mixed/incorrect terrain data. Also wasteful: FAR crops `terrain_cloud` to 7.5m anyway (`terrain_range=7.5`).

**R3 (UNVERIFIED)** — FAR internal edge constraints may block cross-door edges even with correct data:
- `IsInDirectConstraint` rejects edges not within a corner's `surf_dirs` "reduced dir" cone.
- `connect_votes_size=5` at 5Hz requires 1s of consistent voting during traversal.
Only explored once R1+R2 are eliminated.

## 3. Scope

### In scope
- **R1 fix**: Patch `scripts/launch_nav_only.sh` + `scripts/launch_nav_explore.sh` + `scripts/test_integration.sh` to pass the same `--ros-args` that `launch_explore.sh` and `launch_vnav.sh` already use. No new launch file.
- **R2 fix**: Remove `_sync_terrain_to_far()`, `_publish_accumulated_terrain()`, and terrain_map/ext publishing from `_replay_terrain()`. Drop the `_terrain_map_pub` / `_terrain_map_ext_pub` publishers and the 5s timer.
- **Diagnostic tooling**: Keep `monitor_vgraph.py` and `tests/harness/test_vgraph_debug.py`. Add a new integration harness `test_vgraph_cross_room.py` that launches the stack, drives Go2 through doors, and counts cross-room edges.
- **Debug flag**: Enable `is_debug_output: true` in `tare_go2_indoor.yaml`… wait, wrong file — `far_planner` debug flag lives in **`vector_navigation_stack/src/route_planner/far_planner/config/indoor.yaml`**. Spec says to **temporarily flip `is_debug_output: true`** for diagnostic runs only. Not committed permanently.

### Out of scope
- **FAR source code changes** — we don't own that upstream.
- **Nav stack architectural changes** — no new nodes, no new topics.
- **v2.0 merge to master** — separate task. V-Graph fix gates that merge.
- **Unrelated uncommitted work preservation** — the VGG flow fixes (engine.py/intent_router.py), nav cascade fix (go_to_waypoint), wall escape refactor, and TARE margins are **kept as-is** (not reverted). Only terrain-sync code is removed.
- **R3 escalation** — if R1+R2 fixes are insufficient, we enter a Phase B debug round (FAR debug logs, cv::imshow contours). Phase B is a separate SDD round.

## 4. Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC1 | `scripts/launch_nav_only.sh` starts `terrainAnalysis` with `maxRelZ≥1.0`, `clearDyObs=true`, `obstacleHeightThre≤0.15` | `grep "maxRelZ:=" scripts/launch_nav_only.sh` returns a value ≥ 1.0 |
| AC2 | Bridge does NOT publish to `/terrain_map` or `/terrain_map_ext` | `grep -c "_terrain_map_pub\|_terrain_map_ext_pub" scripts/go2_vnav_bridge.py` == 0 |
| AC3 | Existing harness `test_vgraph_debug.py` still passes (terrain is clear at 1.0m) | `pytest tests/harness/test_vgraph_debug.py -k terrain -v` → PASS |
| AC4 | New harness `tests/harness/test_vgraph_integration.py` asserts: after launching nav stack + bridge + driving through a door, `/robot_vgraph` reports ≥1 cross-room edge within 30s | `pytest tests/harness/test_vgraph_integration.py -m ros2 -v` → PASS |
| AC5 | No regression in the pre-existing 3267-test suite (excluding 2 stale `test_prompt.py` tests) | `pytest tests/ --ignore=tests/unit/vcli/test_prompt.py -q` → 0 failures |
| AC6 | Manual CEO check: Yusen runs `vector-cli → "去卧室"`, observes Go2 traversing ≥1 door via FAR V-Graph path (not door-chain) | CEO confirms visually |

AC6 is the final gate. AC1–AC5 are automated proxies.

## 5. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| R1+R2 fixes insufficient, need R3 deep-dive | Med | Blocks AC6 | Phase-gated spec; R3 escalates to new SDD round with FAR debug logs |
| Removing terrain_map publisher breaks something I missed | Low | Med | Grep all callers pre-removal; confirm only `_replay_terrain` and `_sync_terrain_to_far` use it |
| New `test_vgraph_integration.py` is flaky (ROS2 bringup) | Med | Low | Mark `@pytest.mark.ros2`, skip by default, run manually |
| Changing launch_nav_only.sh breaks explore.sh via shared pattern | Low | Med | Each launch script is standalone; no shared code |
| Maxed `maxRelZ=1.5` lets ceiling points in | Low | Low | `_CEILING_FILTER_HEIGHT=1.0` at bridge level already filters |

## 6. Non-Goals

- No FAR planner source modification.
- No new ROS2 messages/services/actions.
- No refactor of launch scripts into a shared library (defer).
- No attempt to fix the 2 stale `test_prompt.py` tests (pre-existing, unrelated).
- No v2.0 PR creation (follows after AC6).

## 7. Interface Definitions

No new interfaces. Affected topics unchanged:

| Topic | Publisher (correct) | Subscriber | Semantics |
|-------|-------------------|------------|-----------|
| `/registered_scan` | `go2_vnav_bridge` | `terrainAnalysis`, `terrainAnalysisExt`, `TARE`, FAR (as `/terrain_local_cloud`) | raw lidar PointCloud2, intensity=z-ground |
| `/terrain_map` | `terrainAnalysis` only | FAR (as `/scan_cloud`) | local terrain, intensity=z-ground |
| `/terrain_map_ext` | `terrainAnalysisExt` only | FAR (as `/terrain_cloud`) | wider terrain, intensity=z-ground |
| `/robot_vgraph` | `far_planner` | diagnostic tools, `graph_msger` | FAR visibility graph |

**Key change from current state**: bridge removes its publishers on `/terrain_map` and `/terrain_map_ext`. Single-publisher invariant restored.

## 8. Test Contracts

### T1 — Bridge cleanup unit tests
```python
def test_bridge_does_not_publish_terrain_map(bridge_source):
    """AC2: bridge has no publishers on /terrain_map or /terrain_map_ext."""
    assert "_terrain_map_pub" not in bridge_source
    assert "_terrain_map_ext_pub" not in bridge_source

def test_bridge_has_no_sync_terrain_function(bridge_source):
    """R2 regression: the _sync_terrain_to_far pattern cannot reappear."""
    assert "def _sync_terrain_to_far" not in bridge_source
    assert "def _publish_accumulated_terrain" not in bridge_source
```

### T2 — Launch script parameter test
```python
def test_launch_nav_only_passes_terrain_params():
    """AC1: launch_nav_only.sh starts terrainAnalysis with Go2-safe maxRelZ."""
    content = Path("scripts/launch_nav_only.sh").read_text()
    assert "maxRelZ:=" in content
    match = re.search(r"maxRelZ:=([0-9.]+)", content)
    assert match and float(match.group(1)) >= 1.0
    assert "clearDyObs:=true" in content
```
(Same for `launch_nav_explore.sh`, `test_integration.sh`.)

### T3 — V-Graph integration harness (ROS2 live)
```python
@pytest.mark.ros2
@pytest.mark.slow
def test_vgraph_forms_cross_room_edge():
    """AC4: after driving Go2 through a door, /robot_vgraph gets ≥1 cross-room edge within 30s."""
    # launches launch_nav_only.sh + bridge via subprocess.Popen
    # drives via /joy to move Go2 through living_hall door
    # subscribes to /robot_vgraph, asserts edges[] non-empty with at least one edge
    # whose endpoints span rooms A and B based on known room-center bounding boxes
    ...
```

### T4 — Harness regression
Existing `tests/harness/test_vgraph_debug.py::test_door_visibility_at_1m` must still pass (proves terrain-data layer is still clean).

### T5 — Suite regression
`pytest tests/ --ignore=tests/unit/vcli/test_prompt.py -q` — 0 failures.

## 9. Phased Execution

**Phase A — Safe fixes (R1 + R2)**
1. Revert the three terrain-map-polluting code paths in bridge
2. Add `--ros-args` to the three bad launch scripts
3. Add unit tests T1 + T2
4. Add integration harness T3 (skippable)
5. Run T4 + T5 — ensure no regression

**Phase A gate**: Yusen runs vector-cli live, observes FAR V-Graph forming. → AC6.

**Phase B — R3 escalation (only if Phase A fails AC6)**
- Enable `is_debug_output: true`, `is_opencv_visual: true` in `indoor.yaml`
- Capture FAR debug output during live door traversal
- New SDD round analyzing `IsInDirectConstraint` / vote accumulation evidence

This spec covers Phase A only. Phase B is a new SDD if needed.

## 10. Decisions Needing CEO Approval

### D1. Scope: Phase A only, or include Phase B commitment?
- **Option A** (recommended): Phase A only. Re-spec if R3 needs work. Avoids over-commitment.
- Option B: Spec covers A+B upfront. Longer timeline, more unknowns.

### D2. Launch script strategy
- **Option A** (recommended): Match `launch_explore.sh` pattern — `ros2 run` + `--ros-args -p ...` for `terrainAnalysis` and `terrainAnalysisExt` in the three broken scripts (`launch_nav_only.sh`, `launch_nav_explore.sh`, `test_integration.sh`). Minimal diff, consistent with what already works.
- Option B: Switch all scripts to `ros2 launch terrain_analysis.launch.py`. Cleaner but breaks the established `--ros-args` pattern and re-scopes work.

### D3. Parameter values for terrain_analysis
Match `launch_explore.sh`:
```
-p clearDyObs:=true
-p minDyObsDis:=0.14
-p minOutOfFovPointNum:=20
-p obstacleHeightThre:=0.15
-p maxRelZ:=1.5
-p limitGroundLift:=true
-p maxGroundLift:=0.05
-p minDyObsVFOV:=-30.0
-p maxDyObsVFOV:=35.0
```
These are **already proven** in `launch_explore.sh` (which Yusen said does sometimes work). No invention.

### D4. Commit strategy for this fix
- **Option A** (recommended): Two commits — (a) `revert: bridge terrain_map duplicate publishers`, (b) `fix: pass terrain_analysis params in launch scripts`. Atomic, bisectable.
- Option B: Single commit. Less clean history.

## 11. Test-First Plan Hint for Phase 2

Phase 2 (plan.md) will break this into tasks:
- Red: write T1 + T2 unit tests (fail)
- Green: apply R2 bridge revert (T1 passes) + R1 launch script patch (T2 passes)
- Red: write T3 skeleton (fails without ROS2)
- Green: flesh out T3, verify live bringup
- Verify: run T4 + T5 + AC6 (manual)

---
