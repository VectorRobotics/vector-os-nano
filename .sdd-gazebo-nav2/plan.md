# Plan: Go2 Gazebo Simulation with Nav2 Navigation

**Spec**: spec.md (approved 2026-03-26)
**Status**: PENDING CEO/CTO APPROVAL
**Date**: 2026-03-26

---

## Module Overview

| # | Module | Purpose | New/Modify |
|---|--------|---------|------------|
| M1 | Workspace + CHAMP | ROS2 workspace scaffolding, CHAMP submodule, build验证 | New workspace |
| M2 | Go2 Description | Go2 URDF + MID360 lidar + D435 camera sensor xacro | New package |
| M3 | Gazebo World | 室内 house scene (20×14m, 7 rooms) | New package |
| M4 | Gazebo Launch | 启动文件: spawn Go2, load controllers, sensor verification | New package |
| M5 | Nav2 Config | Nav2 全栈参数: SLAM, planner, controller, costmap | New package |
| M6 | NavStackClient | Nav2 action 模式扩展 (双模式: nav2/cmu) | Modify existing |
| M7 | Integration | 全栈 launch, Agent 接入, 端到端验证 | New + Modify |

## Technical Decisions

### D1: CHAMP 作为 Git Submodule
从参考仓库 fork CHAMP 代码（champ/ + robots/），作为 submodule 引入。
理由：CHAMP 是成熟步态控制器，接收 /cmd_vel 输出 12 关节力矩，已验证可用于 Gazebo + Go2。

### D2: 传感器 Gazebo 插件选型

| 传感器 | 插件 | 备选 |
|--------|------|------|
| MID360 lidar | `livox_laser_simulation_ros2` (LCAS fork, 支持 Humble) | Velodyne VLP16 (`libgazebo_ros_velodyne_laser.so`) |
| D435 depth | `libgazebo_ros_camera.so` (depth=true, 通用) | `realsense_gazebo_plugin` (pal-robotics fork) |
| IMU | `libgazebo_ros_imu_sensor.so` (参考仓库已有) | — |

MID360 插件如果编译失败，立即 fallback 到 Velodyne — 对 SLAM/Nav2 无实质影响（都是 PointCloud2）。

D435 用通用深度相机插件即可，不需要 vendor-specific plugin。配置 FOV 和分辨率匹配 D435 spec。

### D3: pointcloud_to_laserscan 桥接
SLAM Toolbox 需要 2D `/scan` (LaserScan)。MID360 输出 3D PointCloud2。
用 `ros-humble-pointcloud-to-laserscan` 节点做实时转换。

### D4: Nav2 参数基线
基于参考仓库 `navigation.yaml`，做以下修改：

| 参数 | 参考仓库值 | 修改为 | 原因 |
|------|-----------|--------|------|
| Planner | NavfnPlanner | SmacPlanner2D | 更好的路径质量 |
| max_vel_x | 0.4 | 0.5 | Go2 实际能力 0.875 m/s, 保守取 0.5 |
| max_vel_theta | 0.75 | 1.0 | Go2 实际能力更高 |
| robot_radius | 0.22 | 0.30 | Go2 body width 0.5m / 2 + margin |
| inflation_radius | 0.55 | 0.45 | 室内走廊 0.9m 门洞，需要更紧凑 |
| scan_topic (costmap) | /scan | /scan (from pointcloud_to_laserscan) | 统一 2D 输入 |
| depth_topic (voxel) | /camera/depth/color/points | /camera/depth/points | 匹配我们的 D435 topic |
| slam max_laser_range | 5.0 | 12.0 | MID360 range 40m, 室内用 12m |

### D5: NavStackClient 双模式设计

```python
class NavStackClient:
    def __init__(self, node=None, mode="auto", timeout=60.0):
        # mode: "auto" | "nav2" | "cmu"
        # auto: 检测 /navigate_to_pose action server → nav2, 否则 fallback cmu

    def navigate_to(self, x, y, timeout=None) -> bool:
        # 签名不变，内部根据 mode 路由
```

Nav2 模式使用 `rclpy.action.ActionClient` 调用 `NavigateToPose`。
不引入 `nav2_simple_commander` — 避免增加依赖，原生 action client 够用。

### D6: TF Tree

```
map
 └── odom                    (robot_localization EKF)
      └── base_footprint     (state_estimation)
           └── base_link     (URDF, fixed)
                ├── trunk    (URDF, fixed)
                │   ├── imu_link (fixed)
                │   ├── mid360_link (fixed, z=+0.10)
                │   └── d435_link (fixed, x=+0.25, z=+0.05)
                └── {lf,rf,lh,rh}_hip → upper_leg → lower_leg → foot
```

SLAM Toolbox 发布 map→odom transform。
EKF (robot_localization) 发布 odom→base_footprint。
robot_state_publisher 发布 base_footprint 以下所有 static transforms。

---

## Task Breakdown

### Wave 1: Foundation (并行, 无依赖)

#### T1: Workspace 创建 + CHAMP 构建
**Agent**: Alpha
**Input**: 参考仓库 CHAMP 代码
**Output**: `~/Desktop/vector_go2_sim/` workspace, `colcon build` 通过

Steps:
1. 创建 workspace 目录结构:
   ```
   ~/Desktop/vector_go2_sim/
   ├── src/
   │   ├── champ/           (git submodule or copy from ref)
   │   ├── go2_config/      (from ref, modified)
   │   └── go2_description/ (from ref, will be modified in T2)
   ```
2. 从 `/tmp/unitree-go2-ros2-ref/` 复制 champ/ + robots/ 代码
3. 安装 apt 依赖:
   ```bash
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
     ros-humble-ros2-controllers ros-humble-robot-localization \
     ros-humble-xacro ros-humble-joint-state-publisher-gui \
     ros-humble-navigation2 ros-humble-nav2-bringup \
     ros-humble-slam-toolbox ros-humble-pointcloud-to-laserscan
   ```
4. `colcon build` 验证 CHAMP 编译通过
5. 简单测试: `ros2 launch go2_config gazebo.launch.py` — Go2 出现在 Gazebo

**验收**: Gazebo 窗口中看到 Go2 站立，`ros2 topic list` 显示 /joint_states, /odom, /imu/data

#### T2: Go2 URDF + MID360 + D435 Sensor Xacro
**Agent**: Beta
**Input**: 参考仓库 Go2 URDF, MID360/D435 spec
**Output**: `go2_description/urdf/go2_sensor.urdf.xacro`

Steps:
1. 基于参考仓库 `robot_VLP.xacro`，创建 `go2_sensor.urdf.xacro` (顶层)
2. 新建 `sensors/mid360.urdf.xacro`:
   - Link: `mid360_link`, 固定在 trunk 上方 z=+0.10m
   - Gazebo sensor plugin:
     - 首选: `livox_laser_simulation_ros2` (clone, colcon build)
     - Fallback: `libgazebo_ros_velodyne_laser.so` 配置为 360° 16层
   - 输出: `/mid360/pointcloud` (PointCloud2)
   - 参数: range 0.1-40m, 10Hz, gaussian noise 0.005m
3. 新建 `sensors/d435.urdf.xacro`:
   - Link: `d435_link`, 固定在 trunk 前方 x=+0.25m, z=+0.05m
   - Gazebo sensor plugin: `libgazebo_ros_camera.so`
   - Depth camera: 640×480 @ 15fps, FOV 86°×57°, range 0.2-10m
   - 输出: `/camera/depth/image_raw`, `/camera/color/image_raw`, `/camera/depth/points`
4. 保留 IMU (从参考仓库 gazebo.xacro 不变)
5. 删除 Velodyne 和 Hokuyo xacro (不需要)

**验收**: `xacro go2_sensor.urdf.xacro` 无错误输出完整 URDF; TF tree 包含 mid360_link + d435_link

---

### Wave 2: 环境 + 验证 (并行, 依赖 Wave 1)

#### T3: 室内 House World
**Agent**: Gamma
**Input**: MuJoCo scene 布局 (20×14m, 7 rooms)
**Output**: `vector_go2_gazebo/worlds/indoor_house.world`

Steps:
1. 新建 ROS2 package `vector_go2_gazebo` (ament_cmake, launch + worlds)
2. 创建 SDF world file: `indoor_house.world`
   - 地面: 24×18m plane (留 margin)
   - 外墙: 4 面 box collisions, 高度 2.5m, 厚度 0.15m
   - 内墙: 分隔 7 房间 + hallway (box models)
   - 门洞: 0.9m 宽, 2.0m 高 (墙壁留空)
   - 房间布局:
     ```
     North: master_bedroom(4×5) | guest_bedroom(4×4) | bathroom(3×3)
     Center: hallway (2m wide corridor)
     South: living_room(6×5) | dining_room(4×5) | kitchen(4×4)
     ```
   - 基础家具 (每房间 1-2 件, box/cylinder 碰撞体):
     - living_room: sofa (2×0.8×0.8m), coffee_table (1×0.5×0.4m)
     - kitchen: counter (2×0.6×0.9m), table (1×1×0.75m)
     - master_bedroom: bed (2×1.5×0.5m), nightstand (0.5×0.4×0.5m)
     - dining_room: table (1.5×0.9×0.75m), 4 chairs
     - 其他房间: 1 件家具
   - 光照: sun directional light + ceiling ambient
   - 物理: ODE, 1ms timestep
3. 创建 `empty_flat.world` (纯平地, 用于快速测试)

**验收**: `gazebo indoor_house.world` 打开显示完整房屋; 墙壁碰撞体正确 (球体不穿墙)

#### T4: Gazebo Launch 文件
**Agent**: Alpha (T1 完成后)
**Input**: T1 workspace + T2 URDF
**Output**: `vector_go2_gazebo/launch/gazebo.launch.py`

Steps:
1. 创建 launch file: `gazebo.launch.py`
   - 参数: `world` (default: indoor_house.world), `rviz` (default: true)
   - 启动:
     a. `gzserver` + `gzclient` (via `gazebo_ros/launch/gazebo.launch.py`)
     b. `robot_state_publisher` (from go2_sensor.urdf.xacro)
     c. `spawn_entity.py` (spawn Go2 at hallway center, z=0.275)
     d. `controller_manager` + load `joint_states_controller` + `joint_group_effort_controller`
     e. `champ_bringup/bringup.launch.py` (quadruped_controller + state_estimation + EKF)
   - 包含 `pointcloud_to_laserscan` 节点:
     ```python
     Node(
         package='pointcloud_to_laserscan',
         executable='pointcloud_to_laserscan_node',
         parameters=[{
             'target_frame': 'base_link',
             'min_height': 0.05,
             'max_height': 0.5,
             'range_min': 0.1,
             'range_max': 12.0,
         }],
         remappings=[
             ('cloud_in', '/mid360/pointcloud'),
             ('scan', '/scan'),
         ],
     )
     ```
2. 创建 rviz config: `config/default.rviz`
   - Displays: TF, Robot Model, PointCloud2 (/mid360/pointcloud), LaserScan (/scan), Image (/camera/color/image_raw), Map (/map)

**验收**: `ros2 launch vector_go2_gazebo gazebo.launch.py` — Go2 在 house 中站立, rviz 显示 lidar + camera 数据

#### T5: 遥控行走验证
**Agent**: Beta (T2 完成后)
**Input**: T4 launch
**Output**: 验证截图 / topic echo 记录

Steps:
1. 启动 gazebo.launch.py
2. 另一终端启动 `champ_teleop` 键盘控制:
   ```bash
   ros2 launch champ_teleop teleop.launch.py
   ```
3. 验证:
   - 键盘控制 Go2 前进、转弯、侧移
   - `/odom` topic 有合理的 pose 变化
   - `/mid360/pointcloud` 在 rviz 中显示 3D 点云
   - `/camera/depth/points` 在 rviz 中显示深度点云
   - `/camera/color/image_raw` 在 rviz 中显示 RGB 图像
   - `/scan` (from pointcloud_to_laserscan) 在 rviz 中显示 2D 扫描
   - `/imu/data` topic 有数据
4. 记录 topic hz: `ros2 topic hz /mid360/pointcloud /scan /odom /imu/data`

**验收**: 所有传感器有数据, Go2 可遥控在室内移动, 不穿墙

---

### Wave 3: Nav2 配置 (依赖 Wave 2)

#### T6: Nav2 参数文件
**Agent**: Alpha
**Input**: 参考仓库 navigation.yaml, 本 plan 的 D4 修改表
**Output**: `vector_go2_navigation/config/nav2_params.yaml`

Steps:
1. 新建 ROS2 package `vector_go2_navigation` (ament_cmake, config + launch)
2. 从参考仓库 `navigation.yaml` 复制，修改:
   - Controller: DWB, max_vel_x=0.5, max_vel_theta=1.0, robot_radius=0.30
   - Planner: SmacPlanner2D, tolerance=0.3
   - Local costmap: `/scan` (2D from pointcloud_to_laserscan) + `/camera/depth/points` (3D voxel)
   - Global costmap: same sensor sources + static layer
   - Inflation: radius=0.45, cost_scaling=3.0
   - Recovery: spin + backup + wait
3. 创建 `slam_params.yaml`:
   - 基于参考仓库 `slam.yaml`
   - scan_topic: `/scan`
   - max_laser_range: 12.0
   - resolution: 0.05
4. 创建 `costmap_params.yaml` (如果需要独立配置，否则内联在 nav2_params.yaml)

**验收**: YAML 语法正确, 参数值合理

#### T7: SLAM Launch + 建图验证
**Agent**: Beta
**Input**: T4 Gazebo launch, T6 SLAM params
**Output**: `vector_go2_navigation/launch/slam.launch.py`, 建好的 map.pgm + map.yaml

Steps:
1. 创建 `slam.launch.py`:
   - Include Gazebo launch (with world=indoor_house)
   - Launch `slam_toolbox/online_async_launch.py` with slam_params
   - Launch Nav2 controller (DWB) for manual teleop navigation
   - Launch rviz with SLAM display config
2. 运行 SLAM 建图:
   - 启动 slam.launch.py
   - 用 champ_teleop 手动遥控 Go2 遍历所有房间
   - 在 rviz 中观察 map 逐步构建
3. 保存地图:
   ```bash
   ros2 run nav2_map_server map_saver_cli -f ~/Desktop/vector_go2_sim/maps/indoor_house
   ```
4. 验证地图质量: 墙壁清晰, 房间形状正确, 门洞可见

**验收**: `indoor_house.pgm` + `indoor_house.yaml` 地图文件, 覆盖 >80% 房间面积

---

### Wave 4: 自主导航 (依赖 Wave 3)

#### T8: Nav2 Navigation Launch + 验证
**Agent**: Alpha
**Input**: T6 Nav2 params, T7 保存的地图
**Output**: `vector_go2_navigation/launch/navigation.launch.py`

Steps:
1. 创建 `navigation.launch.py`:
   - Include Gazebo launch
   - Launch `nav2_bringup/bringup_launch.py` with:
     - map: `maps/indoor_house.yaml`
     - params_file: `config/nav2_params.yaml`
     - use_sim_time: true
   - Launch rviz with navigation display config
2. 验证自主导航:
   - 启动 navigation.launch.py
   - 在 rviz 中用 "2D Goal Pose" 发送目标到不同房间
   - Go2 自动规划路径 + 避障 + 到达目标
   - 测试场景:
     a. hallway → kitchen (直线短距离)
     b. living_room → master_bedroom (需要穿过 hallway + 转弯)
     c. kitchen → bathroom (需要穿过多个门洞)
     d. 有家具障碍物的路径
3. 调参 (如果导航不稳定):
   - DWB critics 权重
   - 加速度限制
   - inflation radius
   - 如果 DWB 不好，切换到 RegulatedPurePursuit

**验收**: Go2 能自主导航到至少 5 个不同房间目标, 不碰撞墙壁/家具

#### T9: Full Stack Launch Script
**Agent**: Gamma
**Input**: T8 navigation launch
**Output**: `vector_go2_gazebo/launch/full_stack.launch.py`

Steps:
1. 创建一键启动 launch:
   - Gazebo + Go2 + sensors
   - Nav2 full stack (AMCL + planner + controller + costmap)
   - rviz
   - pointcloud_to_laserscan
2. 参数化:
   - `mode`: "slam" (建图) vs "nav" (导航, 默认)
   - `world`: world file path
   - `map`: map file path (nav mode)
   - `rviz`: true/false
3. 创建便捷 shell 脚本 `scripts/launch_go2_gazebo.sh`:
   ```bash
   #!/bin/bash
   source /opt/ros/humble/setup.bash
   source ~/Desktop/vector_go2_sim/install/setup.bash
   ros2 launch vector_go2_gazebo full_stack.launch.py mode:=nav
   ```

**验收**: 一键启动，Go2 在室内场景自主导航

---

### Wave 5: Agent 接入 (与 Wave 4 并行部分)

#### T10: NavStackClient Nav2 模式
**Agent**: Beta
**Input**: 现有 nav_client.py, Nav2 action interface
**Output**: 修改 `vector_os_nano/core/nav_client.py`

Steps:
1. 扩展 `__init__` 增加 `mode` 参数:
   ```python
   def __init__(self, node=None, mode="auto", timeout=60.0):
       self._mode = mode  # "auto" | "nav2" | "cmu"
   ```
2. `_setup_ros2()` 根据 mode 配置:
   - `"cmu"`: 现有逻辑不变 (/way_point, /goal_reached, /state_estimation)
   - `"nav2"`: 创建 NavigateToPose action client + /odom subscriber
   - `"auto"`: 尝试等待 Nav2 action server 2s, 有则 nav2, 无则 cmu
3. `navigate_to()` 路由:
   ```python
   def navigate_to(self, x, y, timeout=None):
       if self._active_mode == "nav2":
           return self._nav2_navigate(x, y, timeout)
       return self._cmu_navigate(x, y, timeout)  # 现有逻辑
   ```
4. `_nav2_navigate()` 实现:
   - 构造 `NavigateToPose.Goal` (PoseStamped, frame_id="map")
   - `send_goal_async()` → wait for acceptance → wait for result
   - 支持 timeout (cancel goal if exceeded)
   - 解析 result status → return True/False
5. `get_state_estimation()`:
   - Nav2 模式: subscribe `/odom` (nav_msgs/Odometry)
   - CMU 模式: subscribe `/state_estimation` (不变)
6. `cancel()`:
   - Nav2 模式: cancel goal via action client
   - CMU 模式: publish /cancel_goal (不变)
7. 保持所有 imports lazy (rclpy, nav2_msgs)
8. 现有单元测试不应 break (mock CMU mode)
9. 新增 Nav2 mode 单元测试 (mock action server)

**验收**: `pytest tests/unit/test_nav_client.py` 全部通过 (含新测试)

#### T11: NavigateSkill 集成验证
**Agent**: Alpha (T8 完成后)
**Input**: T10 NavStackClient, T8 Navigation launch
**Output**: 端到端验证脚本

Steps:
1. 创建 `scripts/test_nav2_brain.py`:
   ```python
   # 启动 ROS2 node, NavStackClient(mode="nav2"), 发送导航目标
   # 测试: navigate_to(kitchen_x, kitchen_y) → 等待到达
   ```
2. 创建 `scripts/run_nav2_brain.sh`:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/Desktop/vector_go2_sim/install/setup.bash
   export PYTHONPATH="$NANO_ROOT:$PYTHONPATH"
   python3 test_nav2_brain.py
   ```
3. 验证流程:
   - Terminal 1: `full_stack.launch.py`
   - Terminal 2: `run_nav2_brain.sh`
   - 观察: Agent 发目标 → Nav2 规划 → Go2 导航 → goal reached
4. 测试 NavigateSkill (在 Vector OS Nano 内):
   - `--sim-go2` 模式下，配置 NavStackClient mode="nav2"
   - ToolAgent: "去厨房" → NavigateSkill → NavStackClient → Nav2 → Go2 moves

**验收**: Agent 文字指令 → Go2 在 Gazebo 中导航到目标房间

---

### Wave 6: 收尾 (依赖所有)

#### T12: 测试套件
**Agent**: Beta
**Output**: 新增测试文件

Steps:
1. `tests/unit/test_nav_client_nav2.py`:
   - Mock NavigateToPose action server
   - Test navigate_to() with nav2 mode
   - Test auto mode detection
   - Test timeout + cancellation
   - Test get_state_estimation() from /odom
2. `tests/integration/test_gazebo_nav.py` (launch_testing):
   - 验证 Gazebo spawn 成功
   - 验证传感器 topic 有数据
   - 验证 Nav2 action server available
3. 运行全量测试: `python -m pytest tests/ -x -q`

**验收**: 所有测试通过, 无 regression

#### T13: 文档 + Progress
**Agent**: Scribe
**Output**: 更新 progress.md, 启动脚本文档

Steps:
1. 更新 `~/Desktop/vector_os_nano/progress.md`
2. 在 `vector_go2_sim/` 创建 `QUICKSTART.md`:
   - 依赖安装
   - 一键启动命令
   - SLAM 建图步骤
   - 导航验证步骤

---

## Execution Plan

```
Wave 1 (并行):  T1 [Alpha]  +  T2 [Beta]
                    │              │
Wave 2 (并行):  T4 [Alpha]  +  T5 [Beta]  +  T3 [Gamma]
                    │              │              │
Wave 3 (并行):  T6 [Alpha]  +  T7 [Beta]
                    │              │
Wave 4 (并行):  T8 [Alpha]  +  T9 [Gamma]  +  T10 [Beta]
                    │                              │
Wave 5:         T11 [Alpha] (needs T8 + T10)
                    │
Wave 6 (并行):  T12 [Beta]  +  T13 [Scribe]
```

## File Change Summary

### New Files (vector_go2_sim workspace)
```
~/Desktop/vector_go2_sim/
├── src/
│   ├── champ/                          # submodule / copy
│   ├── go2_description/
│   │   ├── urdf/go2_sensor.urdf.xacro
│   │   ├── urdf/sensors/mid360.urdf.xacro
│   │   ├── urdf/sensors/d435.urdf.xacro
│   │   ├── config/ros_control.yaml
│   │   ├── config/gait.yaml
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   ├── vector_go2_gazebo/
│   │   ├── worlds/indoor_house.world
│   │   ├── worlds/empty_flat.world
│   │   ├── launch/gazebo.launch.py
│   │   ├── launch/full_stack.launch.py
│   │   ├── config/default.rviz
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── vector_go2_navigation/
│       ├── config/nav2_params.yaml
│       ├── config/slam_params.yaml
│       ├── launch/navigation.launch.py
│       ├── launch/slam.launch.py
│       ├── maps/                        # generated by SLAM
│       ├── CMakeLists.txt
│       └── package.xml
├── scripts/
│   ├── launch_go2_gazebo.sh
│   └── test_nav2_brain.py
└── QUICKSTART.md
```

### Modified Files (vector_os_nano)
```
vector_os_nano/core/nav_client.py        # +Nav2 action mode (~80 lines added)
scripts/run_nav2_brain.sh                 # new launcher script
tests/unit/test_nav_client_nav2.py        # new test file
tests/unit/test_nav_client.py             # ensure no regression
progress.md                              # updated status
```

## Risks & Mitigations

| Risk | Trigger | Mitigation |
|------|---------|------------|
| CHAMP 不编译 | colcon build 失败 | 检查 Humble 兼容性; 备选: 从参考仓库 fork 修复 |
| livox_laser_simulation 不工作 | Gazebo plugin crash | 立即切 Velodyne VLP16 fallback (T2 备选) |
| Gazebo 太慢 | RTF < 0.5 | 降低传感器频率 (lidar 5Hz, camera 10fps); 减少世界复杂度 |
| Nav2 DWB 导航抖动 | Go2 原地震荡 | 调 DWB critics; 切 RegulatedPurePursuit controller |
| SLAM 建图质量差 | 墙壁模糊、漂移 | 增大 max_laser_range; 用 ground truth odom 替代 EKF |
| Door 太窄通不过 | inflation + robot_radius | 减小 inflation_radius 到 0.35; 调 costmap 参数 |
