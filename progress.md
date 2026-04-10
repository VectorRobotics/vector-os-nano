# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-09
**Version:** v1.5.0-dev
**Branch:** robo-cli (29 commits ahead of master)

## VGG: Verified Goal Graph — Complete Framework

Cognitive layer — ALL actionable commands flow through VGG. LLM decomposes complex tasks into verifiable sub-goal trees. Simple commands get 1-step GoalTrees without LLM call.

```
User input
  ↓
should_use_vgg?
  ├─ Action → VGG
  │    ├─ Simple (skill match) → 1-step GoalTree (fast, no LLM)
  │    └─ Complex (multi-step) → LLM decomposition → GoalTree
  │    ↓
  │  VGG Harness: 3-layer feedback loop
  │    Layer 1: step retry (alt strategies)
  │    Layer 2: continue past failure
  │    Layer 3: re-plan with failure context
  │    ↓
  │  GoalExecutor → verify → trace → stats
  │
  └─ Conversation → tool_use path (LLM direct)
```

### Cognitive Layer (vcli/cognitive/)

| Component | Purpose |
|-----------|---------|
| GoalDecomposer | LLM → GoalTree; template + skill fast path |
| GoalVerifier | Safe sandbox for verify expressions |
| StrategySelector | Rule + stats-driven strategy selection |
| GoalExecutor | Execute + verify + fallback + stats recording |
| VGGHarness | 3-layer feedback loop (retry → continue → re-plan) |
| CodeExecutor | RestrictedPython sandbox (velocity clamped) |
| StrategyStats | Persistent success rate tracking |
| ExperienceCompiler | Traces → parameterized templates |
| TemplateLibrary | Store + match + instantiate templates |
| ObjectMemory | Time-aware object tracking with exponential confidence decay |
| predict | Rule-based state prediction from room topology |
| VisualVerifier | VLM-based visual verification fallback |

### Primitives API (vcli/primitives/)

30 functions across 4 categories:
- **locomotion** (8): get_position, get_heading, walk_forward, turn, stop, stand, sit, set_velocity
- **navigation** (5): nearest_room, publish_goal, wait_until_near, get_door_chain, navigate_to_room
- **perception** (6): capture_image, describe_scene, detect_objects, identify_room, measure_distance, scan_360
- **world** (11): query_rooms, query_doors, query_objects, get_visited_rooms, path_between, world_stats, last_seen, certainty, find_object, objects_in_room, room_coverage

### CLI Integration

- Async execution — CLI never blocks during navigation/explore
- GoalTree plan shown before execution
- Step-by-step [idx/total] progress feedback
- VGG only active after sim start (requires functioning robot)

Design spec: `docs/vgg-design-spec.md`

## Sensor Configuration

- **Lidar**: Livox MID-360, -20 deg downward tilt (match real Go2)
- **Terrain Analysis**: VFoV -30/+35 deg (matched to MID-360)
- **VLM**: OpenRouter (google/gemma-4-31b-it)
- **Ceiling filter**: points > 1.8m filtered from /registered_scan (fixes V-Graph)

## Navigation Pipeline

```
Explore: TARE → room detection → SceneGraph doors
Navigate: FAR V-Graph → door-chain fallback (nav stack waypoints)
Path follower: TRACK/TURN modes, cylinder body safety
Stuck recovery: boxed-in detection → 3-4s sustained reverse
```

## Vector CLI

```bash
vector                    # Interactive REPL (VGG cognitive layer)
vector go2 stand          # One-shot Go2 commands
vector sim start          # Simulation lifecycle
vector ros nodes          # ROS2 diagnostics
vector chat               # LLM agent mode
```

## 备注：开发测试流程

所有测试和验证只通过 vector-cli 启动，直接对话交互。不单独脚本调用 MuJoCo/ROS2。

## Test Coverage: 630+ VGG tests, 1150+ total

| Suite | Tests | Status |
|-------|-------|--------|
| Locomotion L0-L4 | 26 | pass |
| Agent+Go2 | 5 | pass |
| VLM+Scene L0-L9 | 200+ | pass |
| Nav L17-L33 | 247 | pass |
| Sim-to-Real L34-L38 | 120+ | pass |
| Nav fixes L39-L40 | 27 | pass |
| VGG Phase 1 L41-L46 | 187 | pass |
| VGG Phase 2 L47-L50 | 87 | pass |
| VGG CLI L51 | 25 | pass |
| Door-chain L52 | 18 | pass |
| Ceiling filter L53 | 21 | pass |
| VGG Integration L54 | 29 | pass |
| CLI Scenarios L55 | 52 | pass |
| VGG Harness L56 | 24 | pass |
| ObjectMemory L57 | 39 | pass |
| predict L58 | 35 | pass |
| VisualVerifier L59 | 28 | pass |
| Namespace Integration L60 | 21 | pass |
| Auto-Observe L61 | 36 | pass |
| Other | 80+ | pass |

## Phase 3: Active World Model

```
ObjectMemory: SceneGraph → TrackedObject (指数衰减: conf * exp(-0.001 * elapsed))
  ↓
GoalVerifier namespace: last_seen(), certainty(), find_object(), objects_in_room(), room_coverage(), predict_navigation()
  ↓
VisualVerifier: verify 失败 → VLM 拍照二次确认 (感知步骤才触发)
  ↓
Auto-Observe: 探索时每个新 viewpoint → VLM 自动识别物体 → SceneGraph + ObjectMemory
```

## Foxglove 可视化 — feat/web-viz

Foxglove Studio + foxglove_bridge 替代 RViz (ADR-004 revised)。
Three.js 自建方案已废弃 — 开发成本高、效果调试困难。

```
ROS2 Topics → foxglove_bridge (ws://8765) → Foxglove Studio
  Dashboard: foxglove/vector-os-dashboard.json
  6 面板: 3D 透视 + 3D 俯视 + 摄像头 + SceneGraph JSON + Raw Messages + 速度曲线
  14 topic: registered_scan(turbo) + free_paths + path + markers + camera + scene_graph JSON...
  启动: vector CLI → "打开可视化" 或 ./foxglove/launch_foxglove.sh
  CLI tool: open_foxglove (start/stop/status)
  JSON topic: /vector_os/scene_graph (0.5Hz, rooms+doors+objects+stats)
```

已知限制: V-Graph 线段在 Foxglove 中只能显示为散点或很细的 marker（PointCloud2 无法渲染为 LineSegments，MarkerArray 线宽由 FAR 源码决定）。MuJoCo sim 的点云密度和视觉效果不如真实 LiDAR。

## TODO

- Isaac Lab 集成 — 替代/补充 MuJoCo 仿真（更好的物理+视觉）
- Foxglove 自定义面板 (Wave 2): SceneGraph/ObjectMemory/VGG React 扩展
- V-Graph 线段渲染: 需要 Foxglove custom panel 或回到 Three.js 方案
- MuJoCo sim 环境太简单 — 需要增加家具、物品、更复杂的房间布局
- TARE kAutoStart 需要在 indoor_small.yaml 里手动设为 true（会被 linter 改回 false）

## FAR V-Graph 调参记录 (2026-04-09)

根本原因: `new_intensity_thred: 2.0` — bridge 的 intensity = height above ground，ceiling filter 1.8m 导致所有点 < 2.0，FAR 不认为有"新"障碍物 → V-Graph 不建。改为 0.5 后数据层有改善。

调参尝试:
- `is_static_env: true` → 让 FAR 行为更差（obs 归零），不要用
- 降低 `connect_votes_size` 10→3, `node_finalize_thred` 6→3 → 效果不明显
- 增大 `dynamic_obs_dacay_time` 2→30, `new_points_decay_time` 1→15 → 效果更差

最终配置: 只改 `new_intensity_thred: 0.5`，其余保持原始值。

实际状态: FAR 数据层在工作 — `/free_paths` 18k-60k 点持续发布，`/global_path` 5-58 poses。V-Graph edges 存在但 RViz 看不到（PointCloud2 用 2px Points 渲染，不是线段）。

结论: 不是 FAR 不工作，是 RViz 不适合显示 V-Graph。→ 自建 Web 可视化。

## Known Limitations

- VGG complex decomposition quality depends on LLM model
- Async skills (explore, patrol) report "launched" not "completed" in VGG
- Real-world room detection needs SLAM + spatial understanding
