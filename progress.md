# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-16
**Version:** v2.0-dev (branch: feat/v2.0-vectorengine-unification)
**Base:** v1.8.0

## v2.0.1 V-Graph 跨房间修复 — 完成 (2026-04-16)

### 问题
FAR V-Graph 一直是空的（0 nodes），探索时机器人只能用慢 fallback 的 door-chain。

### Phase A（SDD）
- Bridge 删除重复的 `/terrain_map` 发布（-55 行），恢复 single-publisher
- 3 个 launch 脚本加上 Go2 terrain_analysis 参数（maxRelZ=1.5 等）
- 22 个新 test，1 个删除，54 个 stale 测试清理

### Phase B（根因定位）
核心 bug 在 FAR（vector_navigation_stack fork）：
- `indoor.yaml` 里 `vehicle_height: 1.0` 被误解为"障碍物 1m 以下截止"
- 实际是 **robot base_link 离地面高度**（Go2 ≈ 0.28m）
- `TraversableAnalysis` 找不到 `robot.z - vh ≈ -0.72m` 的地面 → kdtree 清空 → `ObsNeighborCloudWithTerrain` 把 2205 cells 砍到 8 cells → `GetSurroundObsCloud` 返回 0 点

**修复（vector_navigation_stack）**:
- `far_planner/config/indoor.yaml`: `vehicle_height: 1.0 → 0.3`
- `far_planner/include/far_planner/contour_detector.h`: `SaveCurrentImg` 里段错误的 `RCLCPP_WARN(nh_->...)` 换成 `std::cout`（nh_ 从未初始化，是 upstream latent bug）

### 验证
| 指标 | 修前 | 修后 |
|---|---|---|
| /FAR_obs_debug | 0 pts | 4800+ pts |
| global_vertex | 0 | 75 |
| visibility_edge | 0 | 131 |
| 跨房间边 | 无 | 有（如 kitchen↔living 门经过 hallway） |

### 文档
- `.sdd/DEBUG.md` — 完整 OBSERVE/HYPOTHESIZE/EXPERIMENT/CONCLUDE 记录
- `~/.claude/skills/learned/far-vgraph-pipeline.md` — 可复用模式

---

## v2.0 架构统一 — Wave 1-3 完成

### 改了什么

v1.8.0 有两条独立的执行路径：

```
旧架构:
  vector-cli → VectorEngine (新引擎，VGG 认知层)
  vector-os-mcp → Agent.execute() (旧引擎，llm/ 模块)
  两条路径共享 skills/hardware，但执行逻辑完全独立
```

v2.0 统一为一条路径：

```
新架构:
  vector-cli  ─┐
               ├→ VectorEngine → VGG / tool_use → skill.execute()
  vector-os-mcp┘
  一条路径，一套逻辑，一次修 bug 全局生效
```

### 删了什么

| 删除项 | 行数 | 原因 |
|--------|------|------|
| `robo/` (Click CLI) | 1,200 | 旧命令行，已被 vector-cli 替代 |
| `cli/` (SimpleCLI, dashboard) | 2,200 | 旧界面，已被 vector-cli 替代 |
| `web/` (FastAPI) | 1,100 | 旧 web 界面 |
| `run.py` (启动器) | 950 | 旧启动器，MCP 有自己的工厂函数 |
| `llm/` (LLM 模块) | 700 | 旧 LLM 层，vcli/backends/ 替代 |
| `core/agent_loop.py` | 243 | 旧迭代循环，VGG GoalExecutor 替代 |
| `core/mobile_agent_loop.py` | 474 | 旧移动循环，VGG Harness 替代 |
| `core/tool_agent.py` | 355 | 旧工具代理，VectorEngine 替代 |
| `core/memory.py` | 263 | 旧会话记忆，vcli Session 替代 |
| `core/plan_validator.py` | 375 | 旧计划验证，VGG 内部验证替代 |
| 旧测试 (27 文件) | 6,500 | 测试已删除模块 |
| **总计** | **~17,700** | |

### 保留了什么

| 保留项 | 角色 |
|--------|------|
| `vcli/` | 主引擎：VectorEngine + VGG 认知层 + 39 个工具 |
| `mcp/` | MCP 桥接：现在用 VectorEngine（与 CLI 同一引擎）|
| `core/agent.py` (瘦身) | 硬件容器：arm/gripper/base/perception 引用 |
| `core/types.py` | 全局数据类型 |
| `core/skill.py` | 技能协议 + 注册表 |
| `core/world_model.py` | 世界状态 |
| `core/scene_graph.py` | 空间层级 |
| `skills/` | 22 个机器人技能 |
| `hardware/` | 硬件抽象（MuJoCo / ROS2 / Isaac Sim）|
| `perception/` | 感知管线 |

### 入口点

只剩两个：
```bash
vector-cli        # 交互式 REPL（人类用）
vector-os-mcp     # MCP 服务器（Claude Code 用）
```

### 测试

- 3,219 个测试收集成功
- MCP 测试 52/52 通过
- 0 个新回归

## Wave 2: Abort 信号 — 完成

- 全局 abort 模块 (vcli/cognitive/abort.py)
- P0 stop 绕过: "stop/停/halt/freeze/别动/停止" 硬编码匹配, <100ms, 不走 LLM
- VGGHarness + GoalExecutor + navigate + explore 全部检查 abort
- async skill wait: explore 完成才执行下一步
- 14 个测试全通过

## Wave 3: 全栈升级 — 完成

### 导航可靠性 (A1-A5)
- Nav stack 健康监控 (5s poll, 崩溃自动重启)
- 单一 TARE 启动 (explore 不再重复启动 TARE)
- 卡死检测 30s 上限 (写 /tmp/vector_nav_stalled)
- navigate_to 全局超时 + abort 检查
- door-chain 超时动态分配 (remaining / n_waypoints, min 5s)

### 反馈可观测性 (B1-B4)
- 导航进度回调 (每 2s: "距目标 3.2m, 已走 6s")
- 探索进度每 5s 报告 (rooms_found, position, elapsed)
- Camera/depth 时间戳 + >1s 过期警告

### 引擎质量 (C1-C6)
- config/nav.yaml: 8 个可调参数 (ceiling, arrival, timeout, stall...)
- GoalDecomposer prompt caching (cache_control ephemeral + instance cache)
- World context 5s TTL 缓存 (emergency stop 失效)
- 日志轮转 (5MB 截断)
- VGG init 每组件独立 try/except + 命名错误日志

## VGG: Verified Goal Graph

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

## Navigation Pipeline

```
Explore: TARE → room detection → SceneGraph doors
Navigate: FAR V-Graph → door-chain fallback (nav stack waypoints)
Path follower: TRACK/TURN modes, cylinder body safety
Stuck recovery: boxed-in detection → 3-4s sustained reverse
```

## Sensor Configuration

- **Lidar**: Livox MID-360, -20 deg downward tilt (match real Go2)
- **Terrain Analysis**: VFoV -30/+35 deg (matched to MID-360)
- **VLM**: OpenRouter (google/gemma-4-31b-it)
- **Ceiling filter**: points > 1.8m filtered from /registered_scan (fixes V-Graph)

## Simulation

### MuJoCo — 主仿真后端 (默认)

MuJoCo 3.6.0。vector-cli 默认后端。

```bash
vector-cli → "启动仿真" → MuJoCo Go2 + 室内环境
```

### Gazebo Harmonic — 暂停

代码保留在: `gazebo/`

### Isaac Sim — 归档

归档在 `web-viz-Isaac-sim` 分支。

## V-Graph Debug — 2026-04-15

### Harness 创建

`tests/harness/test_vgraph_debug.py` — headless MuJoCo 激光雷达分析门口可见性。

### 结论：地形数据没问题

5/6 门在 ceiling_filter=1.0m 下完全畅通（0个障碍物在可见性射线上）。
只有 kitchen_study 被家具阻挡。

之前以为需要全局地形合并到 /registered_scan — **不需要**。

### 根因缩小到 FAR 内部

- **H_A**: FAR 需要 5 次连续投票确认边（`connect_votes_size=5`）。穿门时间不够积累。
- **H_B**: `IsInDirectConstraint` 检查节点法线方向，门附近的节点可能法线朝墙内。
- 下一步：开启 FAR debug 日志 (`is_debug_output: true`)，观察穿门时哪个条件失败。

### 测试修复

- 修了 5 个 test_level12_tare_chain.py 的测试（`_start_tare` mock 错误）
- 3267 tests, 0 collection errors

## V-Graph Phase B — 2026-04-16 (BLOCKED)

### Phase A DONE（未 commit）

SDD 走完 spec→plan→task→execute。修 R1 + R2：
- bridge 删 55 行（移除 `/terrain_map` 双发布冲突）
- 3 launch 脚本加 `--ros-args -p maxRelZ:=1.5 …`（同 `launch_explore.sh`）
- 54 个 pre-existing stale 测试清理（Alpha 10 files + Beta 5 files + Gamma 1 file + dispatcher 2 files）
- production fixes: `vcli/engine.py` vgg_execute clear_abort、`scene_graph_viz.py` _build_object_markers、`skills/go2/look.py` objects 管道
- **harness: 1349 passed / 0 real failures**（41 bulk = MuJoCo pollution）

### Phase B BLOCKED — V-Graph 不建图

**核心谜团**：`/terrain_map_ext` 有 15k obstacle，`/FAR_free_debug` 25k，但 `/FAR_obs_debug` 一直 0-143 点。free 路径 OK，**obs 路径断**。

contour_detector 图像（`/tmp/far_contour_imgs/0.tiff`）全黑 → findContours 0 个 → 0 nodes。

**试过失败的 FAR yaml 调参**：`is_static_env=true`、`dyosb_update_thred=10000`、`decay=1.0`、`obs_inflate_size=3`、`filter_count_value=2`。全部无进展甚至更糟。

### 下 session 方向

停止瞎调参。需要 C++ printf debug：
1. `map_handler.cpp::UpdateObsCloudGrid` 加 log 确认 grid 是否真填充
2. `temp_obs_ptr_` 在 `ExtractFreeAndObsCloud` 后的 size
3. `neighbor_obs_indices_` 的 size
4. `map_handler::is_init_` 状态

config 现状：`is_debug_output: true`, `c_detector/is_save_img: true` 开着，其他 baseline。

## Known Limitations

- VGG complex decomposition quality depends on LLM model
- Real-world room detection needs SLAM + spatial understanding
- C4 session 智能压缩尚未实现（低优先级，可后续迭代）
- **V-Graph 未建图** — Phase B 待解
