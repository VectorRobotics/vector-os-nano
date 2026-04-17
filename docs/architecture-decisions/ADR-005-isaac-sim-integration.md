# ADR-005: Isaac Sim 集成 — Docker 化高保真仿真后端

**Date:** 2026-04-10
**Status:** Accepted
**Author:** Yusen (CEO) + Architect

## 背景

MuJoCo 仿真的视觉质量不满足 demo 和 manipulation 测试需求。物体只有简单几何体 + 纯色材质，环境只有基本墙壁和障碍物，无法做视觉 sim-to-real 验证。

需要一个高保真仿真后端：光追渲染、复杂室内环境、GPU 物理、RTX 传感器仿真。

## 决定

使用 NVIDIA Isaac Sim 5.1 (Docker 容器) 作为主仿真后端，通过 DDS 桥接到 host 的 ROS2 Jazzy。MuJoCo 保留用于快速迭代和 CI 测试。

### 为什么选 Docker

- Isaac Sim 5.1 官方不支持 Ubuntu 24.04 — 容器内是 Ubuntu 22.04
- 不污染 host 的 Python / pip / ROS2 环境
- 可重复构建，团队共享同一镜像
- 干净卸载: `docker compose down`

### 为什么选 Isaac Sim 5.1 (不是 v3.0-beta)

- 5.1 是稳定版，API 成熟
- v3.0-beta 是 Isaac Sim 6.0 Early Developer Release，不稳定
- RTX 5080 driver 580.x 兼容 Isaac Sim 5.1 容器
- Isaac Lab 2.3.x 可在 5.1 上运行 Go2 locomotion 训练

### DDS 跨版本通信

Docker 内 ROS2 Humble + CycloneDDS ↔ Host ROS2 Jazzy + CycloneDDS。
`--network host` 消除网络隔离，DDS wire protocol 版本兼容。

## 架构

```
Host (Ubuntu 24.04 + ROS2 Jazzy)
  vector-cli → VGG → IsaacSimProxy (BaseProtocol)
       ↕ DDS (CycloneDDS, domain 0)
Docker (Isaac Sim 5.1 + ROS2 Humble)
  isaac_sim_bridge.py → PhysX + RTX sensors → ROS2 topics
```

### Topic 兼容性

Isaac Sim bridge 发布与 MuJoCo bridge 完全相同的 topic:
- /state_estimation, /registered_scan, /camera/image, /camera/depth
- /tf, /joint_states, /joy, /speed
- 订阅: /cmd_vel, /cmd_vel_nav

Nav stack (FAR/TARE/pathFollower) 不感知后端差异。

### Protocol 复用

`IsaacSimProxy` 继承 `Go2ROS2Proxy`，覆盖:
- `name` → "isaac_go2"
- `_NODE_NAME` → "isaac_sim_proxy"
- `supports_lidar` → True
- `connect()` → 加 Docker 健康检查

其余方法 (set_velocity, walk, navigate_to, get_position 等) 全部复用。

## 传感器配置

| 传感器 | 挂载偏移 | 参数 |
|--------|----------|------|
| Livox MID-360 | (0.3, 0, 0.2)m, -20 deg pitch | -7~+52 deg VFoV, 360 deg, 30 rings, 0.1-12m |
| RealSense D435 | (0.3, 0, 0.05)m, -5 deg pitch | 640x480, FoV 42 deg, RGB + depth aligned |

自定义 Livox MID-360 RTX lidar 配置: `bridge/lidar_configs/Livox_MID360.json`

## 场景

| 场景 | 用途 |
|------|------|
| flat | 基础测试 |
| room | 4x5m 单房间 |
| apartment | 3 房间 + 门 |
| navigation | 60m2 五房间公寓 (走廊/客厅/厨房/卧室/浴室) — 完整导航 + 操作测试 |
| hospital | Isaac Sim 内置医院 — 最复杂导航测试 |

## 文件结构

```
docker/isaac-sim/
  Dockerfile                    — Isaac Sim 4.2 + ROS2 Humble + Isaac Lab + CycloneDDS
  docker-compose.yaml           — GPU, host network, shader cache, health check
  cyclonedds.xml                — SharedMemory disabled, domain 0
  docker-entrypoint.sh          — Source ROS2 → exec Isaac Python
  bridge/
    isaac_sim_bridge.py         — 主 bridge (所有 topic 发布/订阅)
    go2_scene.py                — 6 个场景构建器
    go2_sensors.py              — Livox MID-360 + D435 传感器配置
    lidar_configs/Livox_MID360.json  — 自定义 RTX lidar

vector_os_nano/hardware/sim/
  isaac_sim_proxy.py            — IsaacSimProxy (BaseProtocol)
  isaac_sim_arm_proxy.py        — IsaacSimArmProxy (ArmProtocol)

scripts/
  launch_isaac.sh               — 一键启动 (Docker 检查 + compose up + 健康等待)
  stop_isaac.sh                 — docker compose down
```

## 测试

263 新测试 (200 passed, 51 skipped, 0 failed):
- Protocol 合规, Docker 检查, 状态查询, 运动, 导航
- Docker 配置验证 (Dockerfile, compose, CycloneDDS)
- Topic 兼容性 (名称/类型/QoS/帧率/传感器参数)
- CLI backend 路由
- 全链路 E2E: Docker → DDS → Proxy → Primitives → VGG

## 风险

| 风险 | 缓解 |
|------|------|
| Isaac Sim GPU 显存 > 8GB | RTX 5080 有 16GB |
| 首次启动 shader 编译 5-10 min | Docker volume 持久化 shader cache |
| DDS Humble↔Jazzy 不兼容 | CycloneDDS 两侧一致 + host network |
| Isaac Sim 容器 ~20GB | 一次拉取，Docker volume 缓存 |

## 后续

- Wire RTX lidar annotator reads (替换 placeholder)
- Wire RenderProduct camera (替换零帧)
- Isaac Lab RL locomotion policy (替换 _apply_cmd_vel)
- Go2 步态在 Isaac Sim 中配置
