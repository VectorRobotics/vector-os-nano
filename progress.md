# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-14
**Version:** v1.8.0
**Branch:** master (merged from gazebo-better-render)

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

## 仿真后端 (2026-04-11)

### MuJoCo — 主仿真后端 (默认)

MuJoCo 3.6.0。四足步态物理验证、VGG 认知层开发。vector-cli 默认后端。

```bash
vector-cli → "启动仿真" → MuJoCo Go2 + 室内环境
```

**传感器配置 (匹配真实 Go2 硬件)**

| 传感器 | 挂载 | 参数 |
|--------|------|------|
| Livox MID-360 | trunk + (0.15, 0, 0.14)m, 前倾 20 deg | raycast lidar, 360 HFoV |
| RealSense D435 | trunk + (0.25, 0, 0.10)m, 下倾 5 deg | 640x480, fovy=42, RGB+Depth |
| IMU | trunk center | quat + gyro + acc |

**渲染增强 (done)**

OBJ mesh 家具 (Kenney Furniture Kit, CC0):
- 27 个 OBJ 模型, 按 MTL 材质拆分为 68 个子 mesh
- 每个子 mesh 独立 rgba 颜色 (沙发红, 木纹棕, 金属银, 绿植绿...)
- inertia="shell" 全部家具 mesh (防止薄片 mesh 报错)
- Y-up → Z-up: euler="1.5708 0 0"
- 碰撞体: 隐形 box (group="3"), 与视觉 mesh 分离

场景优化:
- Shadow: 2048 (10+ 灯 × shadowsize^2 = GPU 炸弹, 不能太大)
- 相机默认: 640x480, shadow + reflection 渲染标志
- 填充光: 4 个 bounce lights (castshadow=false) + sun 增强
- 墙面纹理: gradient plaster, 木地板 texrepeat 10x10
- 踢脚线 + 门框: visual-only geoms
- GUI 追踪相机: distance=5.5, elevation=-20, 跟随机器人 XY

MuJoCo 家具导入 pipeline:
```
OBJ+MTL → split by usemtl → sub-OBJ per material
  → <mesh inertia="shell"> + <geom type="mesh" euler="1.5708 0 0" rgba="...">
  → invisible collision box (group="3")
```

**下一步: OBJ mesh 家具 + PNG 纹理**

MuJoCo 3.6.0 支持:
- OBJ mesh 导入 (带 UV + 法线)
- PNG 纹理贴图 (albedo)
- PBR 材质 (metallic/roughness, 3.2+)

资源:
- [furniture_sim](https://github.com/vikashplus/furniture_sim) — 14 件带纹理家具 (Apache 2.0)
- Gazebo Fuel / 3DGEMS / Sketchfab — OBJ 家具模型 (转换为 MuJoCo XML)

### Gazebo Harmonic — 暂停

方向暂停，代码保留在: `gazebo/`, `scripts/launch_gazebo.sh`, `GazeboGo2Proxy`

### Isaac Sim — 归档

归档在 `web-viz-Isaac-sim` 分支。cmd_vel 链路未通。

## FAR V-Graph 配置 (2026-04-14, fixed)

`config/far_go2_indoor.yaml` 关键参数:
- `new_intensity_thred: 0.5` — bridge intensity = height above ground, 1.8m ceiling filter后所有点<1.8, 阈值必须<1.0
- `dynamic_obs_dacay_time: 999999` — 静态室内，障碍物数据不需要衰减
- `connect_votes_size: 5` — 加速V-Graph边建立（原始10太慢）

FAR数据层正常: `/free_paths` 18k-60k点, `/global_path` 5-58 poses。

## Known Limitations

- VGG complex decomposition quality depends on LLM model
- Async skills (explore, patrol) report "launched" not "completed" in VGG
- Real-world room detection needs SLAM + spatial understanding
