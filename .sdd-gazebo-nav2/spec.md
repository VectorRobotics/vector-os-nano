# Spec: Go2 Gazebo Simulation with Nav2 Navigation

**Version**: 0.1.0-draft
**Status**: PENDING CEO/CTO APPROVAL
**Date**: 2026-03-26
**Author**: Architect (Opus)

---

## 1. Problem Statement

当前 Go2 导航仿真存在两个阻塞问题:

1. **MuJoCo 桥接卡死**: go2_bridge.py 发布的 fake joystick 无法可靠触发 CMU pathFollower 的 autonomy mode（C++ 硬编码 joystick axes[2] 判断），导致 cmd_vel 为零
2. **Unity 仿真黑盒**: 预编译可执行文件，无法修改机器人模型或传感器配置

这两个问题阻止了 Go2 自主导航的开发和测试。同时，当前传感器仿真（手写 mj_ray）与真实硬件传感器栈（Livox MID360 + RealSense D435）差距较大。

## 2. Proposed Solution

构建基于 **Gazebo + Nav2** 的全新仿真导航环境，完全替代 MuJoCo+CMU nav stack 用于导航开发:

```
Vector OS Nano Agent (brain)
  │ NavStackClient (updated for Nav2 action interface)
  ▼
Nav2 Stack (SLAM Toolbox + Smac Planner + DWB Controller)
  │ /cmd_vel
  ▼
CHAMP Controller (gait → 12 joint torques)
  │ ros2_control
  ▼
Gazebo (Go2 + MID360 + D435 + Indoor House Scene)
  │ sensor plugins → /livox/pointcloud, /camera/*, /imu/data
  ▲
Nav2 (subscribes sensor data for costmap + localization)
```

**不替代 MuJoCo 用于步态研发** — MuJoCo 物理精度更高，继续保留用于 MPC 步态调优。本 spec 仅覆盖导航仿真。

## 3. Goals

| ID | Goal | Success Criteria |
|----|------|-----------------|
| G1 | Go2 在 Gazebo 室内场景中自主导航到指定位置 | Agent 发 navigate_to(x,y) → 机器人到达目标 ±0.5m |
| G2 | 传感器仿真匹配真实硬件 | MID360 产生 PointCloud2, D435 产生 depth+RGB+pointcloud |
| G3 | SLAM 建图可用 | SLAM Toolbox 在仿真中建出可用的 2D occupancy grid |
| G4 | Vector OS Nano Agent 无缝接入 | NavStackClient API 不变，仅内部实现切换 Nav2 |
| G5 | 保留 MuJoCo 步态研发能力 | 现有 --sim-go2 模式不受影响 |

## 4. Non-Goals

- 真实 Go2 硬件驱动（unitree_sdk2py 集成）
- 粗糙地形 / 楼梯导航
- 多机器人仿真
- FAR Planner 集成到 Nav2（可作为后续扩展）
- 手臂（SO-101）在 Gazebo 中的仿真

## 5. Architecture

### 5.1 Package Structure

新建 ROS2 workspace: `~/Desktop/vector_go2_sim/`

```
vector_go2_sim/
├── src/
│   ├── vector_go2_description/     # Go2 URDF + MID360 + D435 传感器
│   │   ├── urdf/
│   │   │   ├── go2.urdf.xacro          # 基础 Go2 模型 (from unitree-go2-ros2)
│   │   │   ├── sensors/
│   │   │   │   ├── mid360.urdf.xacro   # Livox MID360 lidar
│   │   │   │   └── d435.urdf.xacro     # RealSense D435 depth camera
│   │   │   └── go2_sensor.urdf.xacro   # Go2 + all sensors (顶层)
│   │   ├── meshes/                      # Go2 STL/DAE meshes
│   │   └── config/
│   │       ├── ros_control.yaml         # joint controller config
│   │       └── gait.yaml               # CHAMP gait parameters
│   │
│   ├── vector_go2_gazebo/          # Gazebo worlds + launch
│   │   ├── worlds/
│   │   │   ├── indoor_house.world       # 20x14m 7-room house (对标 MuJoCo scene)
│   │   │   └── empty_flat.world         # 空地测试
│   │   ├── models/                      # Gazebo SDF 模型 (家具、墙壁)
│   │   └── launch/
│   │       ├── gazebo.launch.py         # Gazebo + Go2 spawn
│   │       └── full_stack.launch.py     # Gazebo + Nav2 + CHAMP 一键启动
│   │
│   ├── vector_go2_navigation/      # Nav2 配置
│   │   ├── config/
│   │   │   ├── nav2_params.yaml         # Nav2 全栈参数
│   │   │   ├── slam_params.yaml         # SLAM Toolbox 参数
│   │   │   └── costmap_params.yaml      # Costmap 层配置
│   │   ├── behavior_trees/
│   │   │   └── navigate_bt.xml          # Nav2 行为树
│   │   └── launch/
│   │       ├── navigation.launch.py     # Nav2 stack
│   │       └── slam.launch.py          # SLAM mapping mode
│   │
│   └── champ/ (git submodule)      # CHAMP 步态控制器 (from unitree-go2-ros2)
│       ├── champ_base/
│       ├── champ_bringup/
│       └── champ_gazebo/
```

### 5.2 Sensor Configuration

#### Livox MID360

| Parameter | Value |
|-----------|-------|
| Horizontal FOV | 360° |
| Vertical FOV | -7° to +52° |
| Range | 0.1 - 40m |
| Point rate | ~200,000 pts/s |
| Update rate | 10 Hz |
| Gazebo plugin | `livox_laser_simulation` (非重复扫描模式) |
| ROS2 topic | `/livox/pointcloud` (PointCloud2) |
| Mount position | Go2 背部中央，z=+0.10m |

**Fallback**: 若 `livox_laser_simulation` 与 Gazebo 11/Humble 不兼容，用 Velodyne VLP16 插件近似（16层, 360°, 130m range）。差异: 重复扫描模式 vs MID360 非重复模式，对 SLAM 影响可接受。

#### RealSense D435

| Parameter | Value |
|-----------|-------|
| Depth resolution | 640×480 @ 30fps |
| RGB resolution | 640×480 @ 30fps |
| Depth range | 0.2 - 10m |
| Depth FOV | 86° × 57° |
| Gazebo plugin | `libgazebo_ros_openni_kinect.so` 或 `realsense_gazebo_plugin` |
| ROS2 topics | `/camera/depth/image_raw`, `/camera/color/image_raw`, `/camera/depth/points` |
| Mount position | Go2 正前方头部，x=+0.25m, z=+0.05m |

### 5.3 Nav2 Configuration

#### Global Planner: SmacPlanner2D
- Grid-based A* variant，适合室内 2D 环境
- Resolution: 0.05m (匹配 SLAM Toolbox output)
- Cost: lethal=254, inscribed=253, inflation 0.55m

#### Local Controller: DWB (Dynamic Window Approach)
- 接收全局路径，输出 `/cmd_vel`
- Max velocity: vx=0.5 m/s, vy=0.25 m/s, vyaw=1.0 rad/s
- Simulation granularity: 0.025s
- Critics: PathAlign, GoalAlign, PathDist, GoalDist, ObstacleFootprint

#### Localization
- **Mapping mode**: SLAM Toolbox (online_async) — 建图 + 定位同时
- **Navigation mode**: AMCL + pre-built map — 纯定位
- Scan matcher: Ceres solver
- Scan topic: `/livox/pointcloud` → 2D projection (via pointcloud_to_laserscan)

#### Costmap Layers
- **Static Layer**: 从 SLAM 建的 OccupancyGrid
- **Obstacle Layer**: MID360 pointcloud + D435 depth pointcloud
- **Inflation Layer**: radius=0.4m (Go2 bodywidth=0.5m / 2 + margin)
- **Voxel Layer** (local costmap): 3D 障碍物检测，高度 0.1-0.6m

#### Recovery Behaviors (Behavior Tree)
- Spin (原地旋转)
- BackUp (后退 0.3m)
- Wait (等待 5s)
- 3 次重试后 abort

### 5.4 NavStackClient 适配

当前 NavStackClient 使用 CMU nav stack 的简单 topic 接口:
```python
# 当前: publish /way_point (PointStamped), subscribe /goal_reached (Bool)
nav.navigate_to(x, y, timeout=60.0)
```

Nav2 使用 action server:
```
# Nav2: NavigateToPose action (/navigate_to_pose)
# Goal: PoseStamped
# Feedback: current_pose, distance_remaining, navigation_time
# Result: empty (success) or error code
```

**适配方案**: 扩展 NavStackClient，双模式支持:

```python
class NavStackClient:
    def __init__(self, mode: str = "auto"):
        # mode: "auto" | "nav2" | "cmu"
        # auto: 尝试 Nav2 action server，fallback 到 CMU topic

    def navigate_to(self, x, y, timeout=60.0) -> bool:
        if self._mode == "nav2":
            return self._nav2_navigate(x, y, timeout)
        else:
            return self._cmu_navigate(x, y, timeout)

    def _nav2_navigate(self, x, y, timeout) -> bool:
        # 1. Create NavigateToPose goal
        # 2. Send via action client
        # 3. Wait for result with timeout
        # 4. Return success/failure

    def get_state_estimation(self) -> Odometry:
        # Nav2 mode: subscribe /odom or /amcl_pose
        # CMU mode: subscribe /state_estimation (unchanged)
```

**API 保持不变**: `navigate_to(x, y)` 签名不变，技能层和 Agent 层零改动。

### 5.5 Indoor House World

对标当前 MuJoCo 室内场景 (20×14m, 7 rooms + hallway):

```
+--------------------------------------------------+
|                                                    |
|  Master       Guest                                |
|  Bedroom      Bedroom     Bathroom                 |
|  (4×5)        (4×4)       (3×3)                   |
|                                                    |
+--------+  +--------+  +-----------+               |
|        |  |        |  |           |               |
|   Hallway (central corridor, 2m wide)              |
|        |  |        |  |           |               |
+--------+  +--------+  +-----------+               |
|                                                    |
|  Living       Dining       Kitchen                 |
|  Room         Room         (4×4)                   |
|  (6×5)        (4×5)                               |
|                                                    |
+--------------------------------------------------+
```

- 墙壁高度: 2.5m
- 门洞宽度: 0.9m
- 家具: 桌、椅、沙发、床（简单碰撞几何体）
- 材质: 基础颜色贴图（不追求渲染质量）
- 格式: SDF world file + Gazebo model refs

### 5.6 系统交互图

```
┌──────────────────────────────────────────────────────────┐
│                  Vector OS Nano Agent                     │
│                                                           │
│   NavigateSkill → NavStackClient.navigate_to(x,y)        │
│                      │                                    │
│                      │ (mode=nav2)                        │
│                      ▼                                    │
│               Nav2 Action Client                          │
│               /navigate_to_pose                           │
└──────────────────┬───────────────────────────────────────┘
                   │ ROS2 Action
┌──────────────────▼───────────────────────────────────────┐
│                    Nav2 Stack                              │
│                                                           │
│  SLAM Toolbox ← /livox/pointcloud (PointCloud2)         │
│       │                                                   │
│       ▼                                                   │
│  /map (OccupancyGrid)                                    │
│       │                                                   │
│  Smac Planner 2D (global path)                           │
│       │                                                   │
│  DWB Controller (local, obstacle avoidance)              │
│       │                                                   │
│       ▼                                                   │
│  /cmd_vel (Twist)                                        │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│              CHAMP Gait Controller                        │
│                                                           │
│  /cmd_vel → IK → 12 joint torques → ros2_control        │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│                 Gazebo Simulation                          │
│                                                           │
│  Go2 URDF (12 joints + 4 legs)                           │
│  Livox MID360 → /livox/pointcloud (PointCloud2, 10Hz)   │
│  RealSense D435 → /camera/depth/*, /camera/color/*      │
│  IMU → /imu/data (100Hz)                                 │
│  Ground Truth Odom → /odom/ground_truth                  │
│  Indoor House World (20×14m, 7 rooms)                    │
└──────────────────────────────────────────────────────────┘
```

## 6. Scope & Deliverables

### Phase 1: Gazebo Go2 基础 (P0)
- [ ] D1: Go2 URDF + CHAMP 在 Gazebo 中站立行走（验证参考仓库可编译运行）
- [ ] D2: 替换传感器为 MID360 + D435（URDF xacro + Gazebo plugins）
- [ ] D3: 键盘遥控 Go2 在空世界中移动，传感器数据可在 rviz2 中查看

### Phase 2: 室内场景 + Nav2 (P1)
- [ ] D4: 室内 house world（SDF，7 rooms，对标 MuJoCo scene）
- [ ] D5: Nav2 配置（SLAM Toolbox + Smac + DWB + costmap）
- [ ] D6: Go2 在室内场景中 SLAM 建图（手动遥控一圈，保存 map.pgm）
- [ ] D7: Go2 用 pre-built map + Nav2 自主导航到 rviz2 指定目标

### Phase 3: Agent 接入 (P2)
- [ ] D8: NavStackClient 扩展 Nav2 action 模式
- [ ] D9: NavigateSkill 通过 NavStackClient 在 Gazebo 中导航成功
- [ ] D10: ToolAgent 语音/文字指令 → "去厨房" → Go2 自主导航到厨房
- [ ] D11: 集成测试 + 启动脚本（一键 launch）

## 7. Dependencies

### ROS2 Packages (apt)
```
ros-humble-navigation2
ros-humble-nav2-bringup
ros-humble-slam-toolbox
ros-humble-robot-localization
ros-humble-gazebo-ros-pkgs
ros-humble-gazebo-ros2-control
ros-humble-ros2-controllers
ros-humble-pointcloud-to-laserscan
ros-humble-xacro
ros-humble-joint-state-publisher
ros-humble-robot-state-publisher
```

### External Packages (source build)
```
CHAMP: git submodule from unitree-go2-ros2
livox_laser_simulation: Gazebo plugin for Livox lidars (if compatible)
realsense_gazebo_plugin: D435 simulation (optional, generic depth camera as fallback)
```

### Existing (unchanged)
```
Vector OS Nano SDK: ~/Desktop/vector_os_nano/
MuJoCo Go2 sim: 保留，不修改
```

## 8. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CHAMP 步态在 Gazebo Humble 上编译失败 | 高 — 阻塞全部工作 | 参考仓库声明支持 Humble; 备选: gazebo_ros2_control + simple velocity plugin |
| livox_laser_simulation 与 Gazebo 11 不兼容 | 中 — 传感器不匹配 | Fallback 到 Velodyne VLP16 插件近似 |
| Nav2 DWB 控制器对四足不友好 | 中 — 导航抖动/不稳定 | 调参: 降低加速度限制, 增大 path tolerance; 备选: RegulatedPurePursuit |
| 室内 world 构建耗时 | 低 — 不阻塞导航验证 | Phase 1 用 empty world 验证; 可从 Gazebo 模型库下载家具 |
| NavStackClient Nav2 适配改动影响现有功能 | 中 — CMU nav stack 模式回退 | 双模式设计 (auto/nav2/cmu), 现有测试不受影响 |

## 9. Testing Strategy

| Level | Test | Tool |
|-------|------|------|
| Unit | NavStackClient Nav2 mode (mock action server) | pytest |
| Unit | Nav2 参数加载验证 | launch_testing |
| Integration | Gazebo spawn + CHAMP + 遥控移动 | launch_testing + ros2 topic echo |
| Integration | SLAM 建图质量 (map coverage > 80%) | 手动 + map_server save |
| System | Agent → navigate_to("kitchen") → 到达 | end-to-end script |
| System | 连续导航 5 个房间，全部到达 | automated test script |

## 10. Timeline Estimate

| Phase | Deliverables | Parallel Agents |
|-------|-------------|-----------------|
| Phase 1 | D1-D3 | Alpha + Beta |
| Phase 2 | D4-D7 | Alpha + Beta + Gamma |
| Phase 3 | D8-D11 | Alpha + Beta |

## 11. Open Questions

1. **Gazebo Classic (11) vs Gz Sim (Harmonic)?**
   - 参考仓库用 Classic (Humble 默认)。Harmonic 更新但 Humble 需要额外配置。
   - **建议**: Classic，与参考仓库保持一致，降低风险。

2. **保留 CMU nav stack 还是完全切换 Nav2?**
   - 本 spec 采用完全 Nav2。CMU nav stack 保留在原位置不修改。
   - NavStackClient 双模式设计允许随时切换回 CMU stack。

3. **360 度摄像头 vs D435?**
   - D435 先做（Gazebo 插件成熟）。360 摄像头作为后续扩展。

4. **CHAMP 步态够用吗?**
   - 平地室内导航完全够用。Max 0.3 m/s 偏慢但可调。
   - 若需要更好步态，后续可替换为 RL policy 或移植现有 MPC。
