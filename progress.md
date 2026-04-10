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

## TODO

- MuJoCo sim 环境太简单 — 需要增加家具、物品、更复杂的房间布局来测试 Phase 3 物体追踪
- FAR V-Graph 不映射 — 可能是 vector-cli 启动方式的线程冲突问题（见下方分析）

## Known Limitations

- VGG complex decomposition quality depends on LLM model
- Async skills (explore, patrol) report "launched" not "completed" in VGG
- Real-world room detection needs SLAM + spatial understanding

## V-Graph 线程冲突分析

FAR V-Graph 一直不建图，之前假设是天花板点云污染（已加 ceiling filter）。
但更可能的原因是 **MuJoCo 线程不安全 + vector-cli 的多线程访问冲突**。

当前线程模型（vector-cli --sim-go2 启动）：

```
Thread 1: MuJoCo physics loop (1kHz)
  └── mj_step1/mj_step2, _update_odometry, _update_lidar
      直接读写 mj.data (qpos, qvel, sensordata)

Thread 2: ROS2 bridge spin (rclpy.spin in daemon thread)
  └── _publish_odom (200Hz): 读 go2.get_position/get_heading/get_odometry
  └── _publish_pointcloud (5Hz): 读 go2.get_3d_pointcloud → 读 _last_pointcloud
  └── _publish_camera (5Hz): 读 go2.get_camera_frame → mj.Renderer.render()
  └── _follow_path (20Hz): 写 go2.set_velocity

Thread 3: CLI main thread (用户交互)
  └── VGG execute → skill.execute → go2.set_velocity / get_position
```

问题：
- Thread 1 的 physics loop 在 mj_step 期间修改 mj.data
- Thread 2 的 bridge 同时读 mj.data（get_position、get_3d_pointcloud）
- **mj.Renderer.render()** 和 **mj_step** 不能并发 — MuJoCo 文档明确说不安全
- get_3d_pointcloud 返回的 _last_pointcloud 在 physics thread 里更新，bridge thread 里读取，无锁保护
- _cmd_lock 只保护速度命令，不保护传感器读取

后果：
- /registered_scan 可能包含不完整或错误的点云数据
- FAR 收到乱数据 → 无法建立 visibility edges → V-Graph 为空
- 间歇性问题（取决于线程调度时机）

潜在修复方向：
1. 给 MuJoCoGo2 加 data_lock (RLock)，所有传感器读取和 physics step 互斥
2. 或: physics loop 每步结束时做 snapshot（copy qpos/pointcloud），bridge 只读 snapshot
3. 或: bridge 和 physics 合并到同一线程（bridge timer 在 physics loop 内调用）
