# Spec: Go2 Locomotion Abstraction Layer

**Version**: 0.1.0
**Status**: APPROVED (CEO directive: "一起做吧")
**Date**: 2026-03-26
**Parent**: spec.md (Gazebo + Nav2)

---

## 1. Problem

CHAMP 步态控制器在 Gazebo 中不稳定 — Go2 站立时持续漂移/弹跳。
这阻塞了 Nav2 导航验证的全部后续工作。

根因: CHAMP 是简化的 Raibert 步态，PID 力矩控制在 Gazebo ODE 物理引擎中
无法稳定保持站姿。这不是调参能解决的问题 — 需要更好的控制器或绕过方案。

## 2. Solution: Locomotion Mode Abstraction

通过 launch 参数切换步态控制方式，Nav2 层完全不变:

```
ros2 launch vector_go2_gazebo gazebo.launch.py locomotion:=planar   # Phase 1
ros2 launch vector_go2_gazebo gazebo.launch.py locomotion:=mpc      # Phase 2
ros2 launch vector_go2_gazebo gazebo.launch.py locomotion:=champ    # Legacy
```

三种模式共享同一套 Nav2 配置、传感器、URDF 基础模型。只有控制器不同。

```
Nav2 (/cmd_vel → /odom, /scan, /tf)
  │
  ▼
┌─────────────────────────────────────────┐
│         Locomotion Mode Switch          │
│                                         │
│  planar:  gazebo_ros_planar_move        │
│           直接移动，无腿部物理            │
│           5 分钟搞定，立刻测 Nav2        │
│                                         │
│  mpc:     quadruped_ros2_control        │
│           OCS2 MPC 或 RL policy         │
│           真实步态，需要集成             │
│                                         │
│  champ:   CHAMP (current, unstable)     │
│           保留但不推荐                   │
└─────────────────────────────────────────┘
  │
  ▼
Gazebo (Go2 + sensors + world)
```

## 3. Phase 1: planar_move (立刻做)

### 原理
Gazebo 内置插件 `libgazebo_ros_planar_move.so`:
- 订阅 `/cmd_vel` (Twist) → 直接移动 base_link
- 发布 `/odom` (Odometry) + TF odom→base_link
- 不需要腿部关节、ros2_control、PID — 全部跳过
- TurtleBot3 导航仿真就是这个原理

### 改动
1. 新建 `go2_description/xacro/go2_planar.xacro` — 包含 base Go2 模型 + sensors + planar_move 插件（不包含 ros2_control/leg transmission）
2. 修改 `gazebo.launch.py` — 根据 `locomotion` 参数选择 URDF 和启动内容:
   - `planar`: 用 go2_planar.xacro，不启动 CHAMP/ros2_control
   - `champ`: 用 go2_sensor.xacro，启动完整 CHAMP stack (现有行为)
3. 传感器 (MID360, D435, IMU) 在所有模式下都工作

### planar_move 插件配置
```xml
<gazebo>
  <plugin name="go2_planar_move" filename="libgazebo_ros_planar_move.so">
    <update_rate>50</update_rate>
    <publish_rate>50</publish_rate>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```

### 效果
Go2 在地面滑动（腿不动），但:
- Nav2 能正常规划和控制
- SLAM 能正常建图
- 传感器数据正常
- 导航全链路可验证

## 4. Phase 2: quadruped_ros2_control (并行做)

### 来源
`legubiao/quadruped_ros2_control` (GitHub, 121 commits, actively maintained)
- 支持 Go2 in Gazebo + MuJoCo
- 控制器: OCS2 (Model Predictive Control), RL (PPO), Unitree Guide
- ros2_control hardware interface
- ROS2 Jazzy 主支，有 Humble 分支

### 集成方案
1. Clone 仓库，checkout Humble 分支
2. 将 Go2 MPC controller 集成到 vector_go2_sim workspace
3. 创建 `locomotion:=mpc` launch 模式
4. 验证 Go2 站立稳定、接受 /cmd_vel、正确反馈 /odom

### 风险
- Humble 分支可能不完整 (主开发在 Jazzy)
- OCS2 依赖复杂 (Pinocchio, Eigen, casadi)
- 编译时间长
- 如果 Humble 不行，可用 Docker 隔离

## 5. Deliverables

| Phase | 产出 | 验收标准 |
|-------|------|---------|
| P1 | planar mode | Go2 在 AWS house 中接受 /cmd_vel 移动，不漂移 |
| P1 | Nav2 SLAM | planar 模式遥控建图，保存 map |
| P1 | Nav2 导航 | rviz 发目标 → Go2 自主到达 |
| P2 | mpc mode | Go2 用 MPC 步态站稳，接受 /cmd_vel 行走 |
| P2 | Nav2 + MPC | 同 P1 验收标准，但用真实步态 |

## 6. Non-Goals

- 真实 Go2 硬件驱动 (unitree_sdk2py)
- RL policy 训练
- 粗糙地形步态
