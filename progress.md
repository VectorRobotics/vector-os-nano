# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-19
**Version:** v2.1-dev (branch: feat/v2.0-vectorengine-unification)
**Base:** v1.8.0

## v2.1 Phase B+C refactor — ROS2 bridge for arm, full stack + arm together (2026-04-19)

**Problem Yusen reported**: `go2sim` with `with_arm=1` started bare MuJoCo
with no rviz / no nav stack (my earlier `_start_go2_local` took an
in-process shortcut). Could not navigate + manipulate in one session.
Plus objects were 50 cm from dog — too close, dog didn't need to move.

### Architecture change
Removed `_start_go2_local`. `with_arm=True` now goes through the same
subprocess path as `with_arm=False` — `launch_explore.sh` brings up
bridge + nav stack + rviz. The bridge (`scripts/go2_vnav_bridge.py`)
detects the arm (MuJoCo `nq ≥ 27`) and auto-enables three ROS2 topics:

- `/piper/joint_state`  bridge→ (JointState, 20 Hz, 6 arm + gripper)
- `/piper/joint_cmd`    →bridge (Float64MultiArray, 6 targets)
- `/piper/gripper_cmd`  →bridge (Float64, 0..1 normalized)

Main process uses new `PiperROS2Proxy` + `PiperGripperROS2Proxy`
(`vector_os_nano/hardware/sim/piper_ros2_proxy.py`) that implement
ArmProtocol / GripperProtocol via these topics. IK / FK runs locally
in the main process on an **isolated** MjModel loaded from the same
MJCF — concurrent MuJoCo API on one model segfaults. Dog base pose
pulled from `Go2ROS2Proxy.get_position()` + `get_heading()` (yaw-only
quaternion, flat-floor assumption).

### Scene changes
- `pick_table` moved from (10.4, 3.0) to (11.0, 3.0) — **1.1 m from
  dog spawn** so the dog has to walk ~60 cm forward before picking.
- Table enlarged to 40×50 cm so objects can spread laterally.
- Three objects same h=8 cm, radii 2.8-3.3 cm (thin bottles <2.5 cm
  radius can't be held reliably by Piper's 35 mm jaws with position-
  only control). Arranged in a row at y=2.85 / 3.00 / 3.15.
- Friction bumped to 2.0 tangential to help grip without force
  sensor.

### sim_tool wiring
`_start_go2` after Go2ROS2Proxy connects:
- If `with_arm=True`, construct `PiperROS2Proxy` + `PiperGripperROS2Proxy`,
  register `PickTopDownSkill`, populate `world_model` via
  `_populate_pickables_from_mjcf` (loads scene XML locally).
- `SimStopTool._shutdown_agent` also disconnects arm/gripper cleanly.

### Pitfalls documented
- MuJoCo API concurrency: any call on a shared MjModel while another
  thread runs mj_step can segfault. IK **must** use an isolated
  model.
- Dog teleport to y=3.0 aligns with a bottle at y=3.0 → physics
  contact impulse knocks the bottle off the table. Offset dog y by
  5 cm in verify_pick_top_down.py to avoid.
- Short bottles (h < pre_grasp_height + half_h) clip on pre-grasp
  descent. All 3 verify objects standardized to h=8 cm.
- `grasp_heuristic=False` with `lift<1cm` = gripper missed. Check
  jaw separation at close for debugging.

### Test status
- 30 unit tests still pass (17 MuJoCoPiper direct + 13 skill mocks)
- E2E verify — in-flight retest after refactor (see next session note)

### Pending
- **Yusen verify**: does `go2sim with_arm=1` now bring up rviz + nav
  stack + arm? Manual check needed.
- **Mobile manipulation**: skill now REQUIRES dog to be close enough
  (55 cm from target). User must walk dog manually or via navigate.
- Place/drop skill (reverse flow) still not written.

---



## v2.1 Phase B+C —— Piper top-down 抓取 pipeline (2026-04-17)

### 交付
- `MuJoCoPiper` (`hardware/sim/mujoco_piper.py`) — ArmProtocol 实现。用**独立的** MjModel+MjData 跑 FK/IK（避开 physics 1kHz 线程的并发 segfault）。IK 是 6-DoF Jacobian DLS + 多 seed（canned "top-down ready" 初值），可靠收敛。
- `MuJoCoPiperGripper` (`hardware/sim/mujoco_piper_gripper.py`) — GripperProtocol。位置控制 `piper_gripper` actuator，open/close/is_holding/get_position。
- `PickTopDownSkill` (`skills/pick_top_down.py`) — 从 world_model 读目标位姿 → IK pre_grasp → IK grasp → open → move → descent → close → lift → hold. 不回 home（避免 wrist 转 90° 把物体翻出）。
- 场景物体：`pick_table` + 3 个 `pickable_*` cylinder（蓝瓶/绿瓶/红罐）加到 `go2_room.xml`，距狗 spawn 50cm。
- `ee_site` (piper.xml link6) — grasp 参考坐标系，FK/IK 都瞄准这个点。
- `sim_tool._start_go2_local` — `with_arm=True` 走**进程内** sim（没有 nav stack，但 arm 能共享 MuJoCo 状态）。world_model 在 connect 时扫描 `pickable_*` body 自动注册。
- `MuJoCoGo2._scene_xml_path` — 暴露加载的 scene 路径给下游（MuJoCoPiper 的独立 IK model 需要）。

### 关键设计决策（避开了 5 个连续的并发 bug）
1. **只读 live data 会崩**：IK 期间读 `data.qpos` 并发 physics `mj_step` 间歇性 segfault → 换成 brief `_pause_physics` during snapshot read
2. **共享 model 读取也会崩**：即使 `mj_forward(m, scratch)` 的 m 是 const，MuJoCo 底层的并发仍不安全 → IK 用**独立 load 的** MjModel + MjData
3. **多次 MjData allocation 积累**：每次 IK 分配新 scratch 数据 = 崩 → connect 时一次性 load 独立模型，所有 IK 复用
4. **`mj_forward` in `_populate_pickables` 崩**：scan pickable 时多余的 mj_forward 和 physics 抢 → 删掉，physics 自己更新 xpos
5. **Home pose 松手**：grasp 后回 URDF-zero 让 wrist 从向下转到朝前，物体翻出 → 不回 home，停在 pre-grasp

### 测试
| 测试 | 规模 | 结果 |
|---|---|---|
| `test_mujoco_piper.py` unit | 17 tests (FK/IK/move/gripper/fixtures) | 17/17 passed |
| `test_pick_top_down.py` mock | 13 tests (失败模式/调用顺序/参数 routing) | 13/13 passed |
| `verify_pick_top_down.py --repeat 10` E2E headless | 3 objects × 10 = 30 picks in fresh subprocesses | 30/30 passed |
| 物体 lift | blue 1.6cm / green 2.1cm / red 2.1cm | held=True 每次 |

### 使用方式
```bash
# REPL:
vector-cli
> go2sim            # 选 with_arm=1 (进入 manipulation mode, 无 nav stack)
> 抓起蓝色瓶子     # or "pick up the blue bottle", "grab red can" 等

# 纯验证:
.venv-nano/bin/python scripts/verify_pick_top_down.py --repeat 5

# pytest:
.venv-nano/bin/python -m pytest tests/hardware/sim/test_mujoco_piper.py tests/skills/test_pick_top_down.py
```

### 显式假设（demo-quality，完整列表见 `docs/pick_top_down_spec.md`）
- 只做 top-down（夹爪永远朝世界 -Z）
- 物体位姿事先已知（`world_model` 从 MJCF `pickable_*` body 读）
- 狗站立不动，无 base-arm 协调
- 无碰撞检测（狗身、墙、其他物体都不查）
- 物体半径 ≤3cm、质量 ≤0.2kg、粗体可 top-down 抓取
- 抓后不回 home，停在 pre-grasp 位姿保持夹持

### 遗留 / 未完成
- **Real hardware 未接**：需要 Piper ROS2 driver
- **nav-stack + arm 同会话不能用**：with_arm=True 走进程内路径，没 FAR/TARE。要两者共存需加 `PiperROS2Proxy`（下个 phase）
- Place / drop skill 未写（pick 现在只到 hold，不能放下）
- 无感知路径（object.pose 只从 world_model 读，不做 RealSense/SAM3D 检测）

---

## v2.1 Phase A —— Piper 手臂挂载 + 双模式仿真 (2026-04-17)

### 目标
把 AgileX Piper 6-DoF 机械臂装到 Go2 背上（仿真），为后续 manipulation 准备基础。

### 资产
- 资产来源：MuJoCo Menagerie 官方 `agilex_piper` (MJCF + meshes + LICENSE)
- 拷贝到 `vector_os_nano/hardware/sim/mjcf/piper/`
- 合成工具：`mjcf/go2_piper/build_go2_piper.py` 用 MuJoCo 3.6 `MjSpec.attach()`
- 挂载位置：`pos=(-0.02, 0, 0.06)` — Go2 trunk 顶面居中，雷达后方 15cm
- Piper body 全部加前缀 `piper_` 避免和 Go2 命名冲突
- Default class 自动 scope 到 `piper_main / piper_visual / piper_collision`
- 生成的合成 model：`go2_piper.xml`，29 bodies / 21 joints / 19 actuators

### 双模式（用户启动时选择）
- **no-arm** (默认): `scene_room.xml`, nq=19, convex_mpc gait （更流畅）
- **with-arm**: `scene_room_piper.xml`, nq=27, sinusoidal gait（convex_mpc PinGo2Model 是 12-DoF 固定，不兼容 nq=27）
- `sim_tool` input_schema 加 `with_arm` 参数，description 要求 LLM 启动前询问用户
- `VECTOR_SIM_WITH_ARM` 环境变量从 vector-cli 传到 subprocess
- `_build_room_scene_xml(with_arm=None)` 自动读 env var

### Debug 过程 (`.sdd/DEBUG.md`)
按 Hypothesis Loop 四轮：
1. **H1 假设：bridge `_cmd_vel_cb` 不刷新 `_teleop_until`** → 看 git diff 发现本来就有，否证
2. **H2 假设：bridge `_follow_path` 20Hz override set_velocity** → 加 thread-ID gate，独立测试通过，但 Yusen 真机仍不走
3. **H3 盲点：Yusen 用 `go2sim` (sim_tool, Go2ROS2Proxy path) 不是 `--sim-go2` (MuJoCoGo2 path)** → 我改的 MuJoCoGo2 thread gate 在正确位置，但核心 bug 不在此
4. **H4 真因：convex_mpc 的 PinGo2Model 是 12-DoF 专用，MuJoCo nq=27 导致 `pin.forwardKinematics` 每 tick 抛 `ValueError: expected 19, got 27`，physics 线程崩溃** — 狗永远不动

### Root Cause + Fix
`_init_mpc_stack` 加维度守卫：
```python
if self._pin.model.nq != self._mj.model.nq:
    raise RuntimeError(...)  # caller fallback to sinusoidal
```
物理线程不再崩 → sinusoidal gait 接管 → 狗能走 (with-arm 模式)

### SimStopTool 新增
之前说"关闭仿真"被 VGG fast-path 误匹配到 `gripper_close_skill`。加了 `stop_simulation` tool，tool_use 优先级高于 skill fast-path：
- `SimStartTool._shutdown_agent()` 复用的 teardown 工具
- `_start_go2` 存 `vnav_proc` 到 `base._sim_subprocess`
- `SimStopTool.execute` 调 killpg + disconnect + unregister go2 skill tools

### 测试状态
- Python 独立测试：no-arm (MPC) / with-arm (sinusoidal) 两模式都通过
- Yusen 机器：`go2sim` → 选模式 → 启动 OK；`走两米` 在 with-arm 模式下能走（步态粗糙是 sinusoidal 本身的限制）
- `关闭仿真` 待 Yusen 下 session 验证

### 遗留
- with-arm 模式下 Piper 的 home pose (joint2=1.57, joint3=-1.3485) 从某些角度看像"展开"，实际是 Piper URDF 原厂 home keyframe。可以后续自定义 `_PIPER_STOW_QPOS` 让 arm 贴 trunk 更紧凑
- with-arm 模式真要上 MPC gait 需要在 Pinocchio 里 rebuild 含 Piper 的 URDF + retrain 模型 (scope 大，Phase D+)
- Phase B: `MuJoCoPiper` 类实现 `ArmProtocol`（joint/FK/IK/gripper），让 arm 能被 skill 调用
- Phase C: 扩展 PickSkill 支持 6-DoF Piper，E2E "去厨房拿杯子"

---

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
