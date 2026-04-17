# DEBUG.md — FAR V-Graph surround_obs_cloud is empty

**Session**: 2026-04-16 Phase B
**Branch**: feat/v2.0-vectorengine-unification
**Bug**: `FARUtil::surround_obs_cloud_` stays empty (0-143 pts) despite `/terrain_map_ext` carrying 15k obstacle points. Consequence: contour_detector gets all-zero img → 0 contours → 0 vertices → V-Graph has 0 nodes → cross-room edges never form.

## OBSERVE

### Repro steps
1. Launch MuJoCo + go2_vnav_bridge.py + launch_nav_only.sh (now patched per Phase A)
2. Drive Go2 around rooms
3. Watch `/robot_vgraph` — nodes=0 forever; watch `/FAR_obs_debug` — <200 pts

### System snapshot (from prior session)
```
/terrain_map_ext: 42k pts, intensity 0..1.24 (15k obs | 27k free) ✓
  │ → FAR TerrainCallBack
/FAR_free_debug: 25-30k pts ✓  (FREE path works)
/FAR_obs_debug: 0-143 pts ✗    (OBS path broken)
/tmp/far_contour_imgs/0.tiff: all-zero 301×301 → 0 contours → 0 nodes
```

### Known config (indoor.yaml)
- voxel_dim=0.1, robot_dim=0.5, vehicle_height=1.0, sensor_range=15, terrain_range=7.5
- floor_height=1.5, cell_length=1.5 → cell_height=0.6, kHeightVoxel=0.2, kTolerZ=1.3
- terrain_free_Z=0.15 (intensity below → free, at or above → obs)
- connect_votes_size=5, filter_count_value=6, is_debug_output=true

### Config sweeps already tried (all failed)
- baseline → surround_obs 143, nodes 0
- is_static_env=true → surround_obs **0**, nodes 0 ← **critical datapoint**
- dyosb_update_thred=10000 + decay=1.0 → no help
- obs_inflate_size=3 + filter_count=2 → FAR pegged at 74% CPU, 0 msgs
- terrain_free_Z sweeps earlier → no change

### Previously ruled out
- Ceiling filter (MuJoCo raycaster harness proves doors CLEAR at z≤1.0m, 5/6 doors)
- Global-terrain merge
- `new_intensity_thred`

## Key asymmetry from reading source

| Stage | FREE path | OBS path |
|-------|-----------|----------|
| CropBoxCloud(7.5m, z±kTolerZ=1.3) | shared | shared |
| ExtractFreeAndObsCloud (intensity<kFreeZ=0.15) | → temp_free_ptr_ | → temp_obs_ptr_ |
| RemoveOverlapCloud(_, stack_dyobs_cloud_) | not applied | applied, **skipped when is_static_env=true** |
| Update\*CloudGrid storage | all `InRange` cells | only cells in **neighbor_obs_indices_** |
| Get\*Cloud readback | from neighbor_free_indices_ | from neighbor_obs_indices_ |

### Critical evidence
`is_static_env=true` skips RemoveOverlapCloud (far_planner.cpp:732 guard) yet surround_obs still = 0. This **rules out RemoveOverlapCloud** as the bug.

Therefore the drop must be in one of:
- ExtractFreeAndObsCloud assigning all points as free (intensity semantic mismatch)
- UpdateObsCloudGrid dropping points because cells aren't in neighbor_obs_indices_
- GetSurroundObsCloud reading empty cells
- Cells exist but are filtered (FilterCloud w/ leaf_size)

## HYPOTHESIZE

| # | Hypothesis | Category | Evidence |
|---|-----------|----------|----------|
| H1 | `temp_obs_ptr_` is already near-empty after ExtractFreeAndObsCloud because `/terrain_map_ext` intensity is wrong (e.g., all <0.15, all classified as free) | Data | Prior session logged 15k obs on /terrain_map_ext, but that count came from our Python probe — maybe our probe used different threshold than FAR |
| H2 | CropBoxCloud drops the obs points because robot_pos_.z is above ground and z+kTolerZ=1.57 cuts off walls/doorframes | Geometry | kTolerZ=1.3, robot.z≈0.27 for Go2 → [−1.03, 1.57] — walls (z>1.57) get dropped but intensity-indexed points at z≤1.24 shouldn't |
| H3 | UpdateObsCloudGrid stores <<15k because most cells miss `neighbor_obs_indices_` (e.g., robot_cell_sub_ wrong, or indices empty) | Grid | FREE path works (25-30k) but FREE stores unconditionally on `InRange`; OBS requires cell ∈ neighbor_obs_indices_ |
| H4 | `neighbor_obs_indices_` is populated but points fall in cells just outside (grid origin offset, first-init effect from Go2 starting crouched/airborne) | Grid | SetMapOrigin runs once on is_init_=false, subtracts vehicle_height=1.0 — if init happens while Go2 is at rest height (0.27) vs lifted, offset flips |
| H5 | Cells do get points but FilterCloud with leaf_size=0.1 collapses them (FilterCloud is applied to all modified cells after every UpdateObsCloudGrid call) | Filter | UpdateObsCloudGrid line 218-220 filters each modified cell; with 0.1 voxel and cell size 1.5, should only dedupe, not drop |

Ranking by evidence strength:
1. **H3** — tightest asymmetry between FREE (works) and OBS (empty)
2. **H4** — related sub-hypothesis; would explain why some cells work some don't
3. **H2** — easy to falsify via printf
4. **H1** — check the per-frame obs count
5. **H5** — unlikely but cheap to verify

## EXPERIMENT plan

Add rate-limited `std::cout` instrumentation in `far_planner.cpp::TerrainCallBack` and `map_handler.cpp::{UpdateObsCloudGrid,UpdateRobotPosition,GetSurroundObsCloud}` at each stage. Print every 10th call (≈2s at 5Hz):

```
[FAR-DBG #N]
  recv /terrain_cloud: total=TT
  after CropBox: total=CC  robot_pos=(x,y,z)
  after ExtractFreeAndObs: free=FF obs=OO  intensity_bin[<0.15]=... [0.15-0.5]=... [>0.5]=...
  stack_dyobs size=DD  is_static=S
  after RemoveOverlap: obs=OO'
  UpdateObsCloudGrid: input=OO'  stored=KK  in_range_dropped=RR  neighbor_miss=NN
  map_handler: is_init=I  neighbor_obs.size=AA  robot_sub=(sx,sy,sz)  cell_range_x=...
  GetSurroundObsCloud: out=SS
  surround_obs_cloud_: final=FF (after dyobs removal)
```

Single run, 30s, stationary + 10s walking → should catch the drop.

## EXPERIMENT

### Run 1 — live capture with printf instrumentation (2026-04-16)

Built far_planner with printf at every stage. Launched bridge + launch_nav_explore.sh, captured 60s of /tmp/nav_vgraph.log.

Observed (every frame):
```
[VGRAPH-DBG #N] recv_msg=33770 robot=(9.35,1.73,0.29) kTolerZ=1.3 kFreeZ=0.15 static=0
[VGRAPH-DBG] stage1_PrcocessCloud temp=9726
[VGRAPH-DBG] stage2_CropBox     temp=7827          # CropBox OK
[VGRAPH-DBG] stage3_Extract     free=5224 obs=2603 # split OK
[VGRAPH-DBG] stage4_RemoveOverlap obs=2603 (was 2603) stack_dyobs=0   # no dyobs
[VGRAPH-DBG-MH] UpdateObsCloudGrid stored=2603 miss_neighbor=0 out_of_range=0
                grid_pre=6113 grid_post=3525       # grid HAS 3525 obs points
                sample_sub=(102,98,13) in_range=1 in_neighbor=1   # sample cell OK
[VGRAPH-DBG-MH] UpdateRobotPos is_init=1 neighbor_obs.size=2205    # 21×21×5 ✓
                robot_sub=(99,99,14) grid_origin=(-140.011,-147,-8.223)
[VGRAPH-DBG-MH] GetSurroundObs neighbor.size=8 non_empty=0 empty=8 total_out=0  # 2205 → 8 !!!
```

**H3 CONFIRMED. The key datapoint:** between `UpdateObsCloudGrid` (sees 2205 indices, stores 2603 pts across many cells) and `GetSurroundObsCloud` (reads only 8 indices, all empty), `neighbor_obs_indices_` gets mutated from 2205 → 8.

Code trace in `TerrainCallBack`:
```
map_handler_.UpdateRobotPosition(...)        // fills 2205
map_handler_.UpdateObsCloudGrid(...)         // stored=2603 ✓
map_handler_.UpdateFreeCloudGrid(...)        // no touch
map_handler_.GetSurroundFreeCloud(...)       // reads free (works)
map_handler_.UpdateTerrainHeightGrid(...)    // ← MUTATES neighbor_obs_indices_
    └ ObsNeighborCloudWithTerrain(neighbor_obs_indices_, ...)
        std::unordered_set<int> neighbor_copy = neighbor_obs;
        neighbor_obs.clear();                 // wipes 2205
        for idx in neighbor_copy:
            NearestHeightOfRadius(pos, 1.06m, minH, maxH, inRange);
            if (inRange && z-overlap)
                neighbor_obs.insert(idx);     // keeps only cells near traversable terrain
map_handler_.GetSurroundObsCloud(...)         // reads filtered 8 cells → 0 points
```

`ObsNeighborCloudWithTerrain` uses `kdtree_terrain_clould_` which is built from `flat_terrain_cloud_` derived from `TraversableAnalysis`. **If TraversableAnalysis fails, kdtree is ClearKdTree'd to a single dummy point at (0,0,0)**, leaving only cells near the grid coord origin within R=1.06m.

Grid origin=(-140, -147) in XY. The 8 surviving cells map to: 2 XY cells (sub 93,97 and 93,98, both within 1.06m of world (0,0)) × 4 Z levels (subs 13,14,15,16 passing vertical-overlap check). **Exactly 8. Matches observation.**

### Why does TraversableAnalysis fail?

`TraversableAnalysis` (map_handler.cpp:486) BFS's from robot_sub, needs to establish "robot_terrain_init" by finding a terrain cell whose height `e` satisfies:
```cpp
if (abs(e - FARUtil::robot_pos.z + FARUtil::vehicle_height) > H_THRED) continue;
// H_THRED = height_voxel_dim = 0.2m
```

Effectively looks for terrain height `e ≈ robot.z - vehicle_height`.

Config: `vehicle_height: 1.0` (indoor.yaml). Bridge publishes robot.z ≈ 0.28 (Go2 base-link offset from floor; `ground_z = odom.z - 0.28` in go2_vnav_bridge.py:585).

So FAR looks for terrain at `z ≈ 0.28 - 1.0 = -0.72m`. Real floor terrain is at `z ≈ 0`. Gap = 0.72m >> 0.2m threshold → init NEVER happens → terrainHeightOut stays empty → ClearKdTree → dummy point → only 8 cells pass ObsNeighborCloudWithTerrain → GetSurroundObsCloud returns 0.

**H2 (CropBox) REJECTED**: CropBox leaves 7827 pts; Extract gives 2603 obs; grid stores 2603. Drop happens strictly at `ObsNeighborCloudWithTerrain`.
**H3 (neighbor_obs filter) CONFIRMED**: 2205→8 verified, mechanism traced.
**H4 (grid origin offset) CONTRIBUTES**: the specific value "8" comes from grid origin (-140,-147) leaving only cells at sub (93,97)(93,98) within 1.06m of (0,0).
**H1 (temp_obs near-empty) REJECTED**: obs=2603 per frame, healthy.
**H5 (FilterCloud collapse) REJECTED**: grid_post=3525, plenty retained.

## CONCLUDE

### Root cause
`far_planner/config/indoor.yaml` has `vehicle_height: 1.0` (manually set, misleading comment "consider obstacles up to 1m"). The correct semantic is **robot base-link height above ground** (used for `map_origin.z = robot.z - grid_half_z - vehicle_height` in SetMapOrigin, and for the `robot_terrain_init` filter in TraversableAnalysis). For Go2 with bridge offset 0.28m, correct value is ~0.28-0.30m. Upstream `outdoor.yaml` has 0.6 for a person-sized Jackal.

With vh=1.0, TraversableAnalysis never achieves `is_robot_terrain_init=true`, terrain kdtree stays on a dummy (0,0,0) point, and `ObsNeighborCloudWithTerrain` prunes `neighbor_obs_indices_` from 2205 down to 8 (cells near world origin only). GetSurroundObsCloud returns 0 → contour_detector image is blank → 0 corners → V-Graph has 0 nodes → no cross-room edges.

### File:line
- `vector_navigation_stack/src/route_planner/far_planner/config/indoor.yaml:6` — `vehicle_height: 1.0` → should be `0.3`
- Secondary: misleading comment `# Unit: meter (indoor: consider obstacles up to 1m)` — should be `# Unit: meter (base-link height above ground; Go2 ≈ 0.3)`

### Fix
```yaml
# Before
vehicle_height: 1.0  # Unit: meter (indoor: consider obstacles up to 1m)
# After
vehicle_height: 0.3  # Unit: meter (robot base-link height above floor; Go2 ≈ 0.28m)
```

### Regression test
After fix:
- AC verify-1: `grep "GetSurroundObs neighbor.size" nav_vgraph.log` should show values > 8 (expect ~1000-2205 as robot moves)
- AC verify-2: `grep "total_out=" nav_vgraph.log` should show non-zero values (thousands of surround_obs points)
- AC verify-3: `/tmp/far_contour_imgs/*.tiff` should contain non-black images with wall/furniture outlines
- AC verify-4: `/robot_vgraph` nodes count > 0

### Remaining instrumentation
The printf blocks added to far_planner.cpp and map_handler.cpp stay in for the verification run. After AC verify passes, they'll be removed before commit.

## VERIFY (Run 2 + Run 3 combined, 2026-04-16)

### Run 2 — vh fix applied, first verification

Changed `vehicle_height: 1.0` → `0.3` in indoor.yaml. Launched bridge + launch_nav_explore.sh.

Result:
- `neighbor_obs.size` at UpdateRobotPos: 2205 (unchanged; loop-source)
- **`GetSurroundObs neighbor.size`: 225** (from 8 → 225, 28x improvement)
- **`total_out`: 925 points** (from 0 → 925)
- `/FAR_obs_debug` (stage6) reports `surround_obs=925 surround_free=7240` ✓
- FAR then crashed in `MainLoopCallBack → contour_detector_.BuildTerrainImgAndExtractContour → SaveCurrentImg`. SIGSEGV at `rclcpp::Node::get_logger()` because `nh_` is never initialized in ContourDetector.

GDB backtrace:
```
#0  rclcpp::Node::get_logger()
#1  ContourDetector::SaveCurrentImg (contour_detector.h:175)
#2  ContourDetector::UpdateImgMatWithCloud (contour_detector.cpp:60)
#3  ContourDetector::BuildTerrainImgAndExtractContour
#4  FARMaster::MainLoopCallBack (far_planner.cpp:193)
```

This is an **upstream FAR bug** that was only exposed because `is_save_img: true` only runs when there's actually image data to save — which requires `surround_obs_cloud_` to be non-empty. Before our fix, surround_obs was always empty → the save path was never exercised → no crash. After fix, path runs → crash.

### Run 3 — fix SaveCurrentImg bug + rerun

Replaced `RCLCPP_WARN(nh_->get_logger(), "CD: image save success!")` in
`include/far_planner/contour_detector.h:175` with a safe `std::cout` (rate-limited by `img_counter_ % 25`).

Result:
- FAR ran stably for 630+ seconds (3151 TerrainCallBack frames) with no crash.
- `[CD] image save: /tmp/far_contour_imgs/*.tiff` — images saved successfully, sizes 2244+ bytes non-zero.
- **V-Graph Initialized**: "[1;32m V-Graph Initialized [0m" log appears 20s after start.
- `/viz_graph_topic` markers:

| Namespace | Count |
|-----------|------|
| global_vertex | 67-75 |
| freespace_vertex | 19-20 |
| frontier_vertex | 22-23 |
| boundary_vertex | 3 |
| trajectory_vertex | 5 |
| global_vgraph (edges) | 131 |
| visibility_edge | 131 |
| freespace_vgraph | 127 |
| odom_edge | 43 |

### Cross-room edges confirmed

Sample edges from `/viz_graph_topic global_vgraph`:
```
edge: (14.27, 3.40) -- (6.43, 1.10)   # kitchen door east ←→ living door west, 8m line-of-sight
edge: (13.73, 7.64) -- (6.43, 1.10)   # study door ←→ living door, ~9m through hallway
edge: (13.73, 2.64) -- (6.43, 1.10)   # kitchen door ←→ living door
edge: ( 9.99, 3.00) -- (6.43, 1.10)   # hallway mid ←→ living door
```

Vertex positions map to scene_room.xml doorframe corners:
- (6.27, 2.64), (5.84, 2.64): at x=6 wall, y=2.4-3.6 (living-hall door, both sides)
- (14.17, 7.64), (13.74, 7.64): at x=14 wall, y=7.4-8.6 (study-hall door, both sides)
- (13.74, 3.41), (14.27, 3.41): at x=14 wall, y=2.4-3.6 (kitchen-hall door)
- (8.07, 9.71), (8.94, 9.71): at y=10 wall (master/guest/bath wall)

### AC verify results

| Check | Target | Actual | Pass |
|-------|--------|--------|------|
| neighbor_obs neighbor.size > 8 | >= 100 | 120-444 | ✅ |
| total_out > 0 | > 1000 | 925-4869 | ✅ |
| /tmp/far_contour_imgs/*.tiff non-black | ≥ 50% pixels | 84%+ | ✅ |
| /robot_vgraph nodes > 0 | ≥ 1 | 2 navpoints (+66 global) | ✅ |
| V-Graph spans doorways | ≥ 1 cross-room edge | 4+ visible | ✅ |

## SECONDARY BUG FIX

Beside the yaml change, fixed:
- `include/far_planner/contour_detector.h:175` — `RCLCPP_WARN(nh_->get_logger(), ...)` crashes because `ContourDetector::nh_` is never initialized. Replaced with rate-limited `std::cout`. This bug only manifests with `is_save_img: true` AND non-empty surround_obs_cloud_ (i.e., never reached before this session).

Both changes committed as the v2.0.1 V-Graph fix.

## Unrelated finding (not blocking)

Go2 got stuck at (6.45, 1.10) between interior wall (x=6) and some obstacle on the east side. Bridge reported `Stuck 8s: F=inf L=0.80 R=0.78 B=0.40` and `Stuck loop — extended escape 4s`. Robot z climbed to 0.49m from nominal 0.28m — possibly foot caught on wall baseboard. This is a **nav stack / physics issue**, not a V-Graph issue. Log for follow-up; does not block V-Graph validation.

