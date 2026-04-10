# ADR-004: ROS2 Web Visualization 替代 RViz

**Date:** 2026-04-09
**Status:** Proposed
**Author:** Yusen (CEO) + Architect

## 背景

Vector OS Nano 目前依赖 RViz2 做导航和探索的实时可视化。在 MuJoCo 仿真 + FAR/TARE nav stack 的调试过程中，暴露了多个 RViz 的局限：

1. **V-Graph 不可见** — FAR 的 `/free_paths` (PointCloud2) 用 Points 样式渲染，2px 大小在黑色背景上几乎不可见。数据实际在流（50k+ 点），但看不到。
2. **自定义消息类型不支持** — `/decoded_vgraph` (visibility_graph_msg/msg/Graph) 在 RViz 中无法解析，报 "type invalid"。
3. **无法显示 VGG 数据** — GoalTree 执行进度、ObjectMemory 置信度、SceneGraph 房间拓扑等 Vector OS 特有数据在 RViz 中无法呈现。
4. **配置脆弱** — RViz display 的 topic/QoS/样式配置容易出错，调试耗时。
5. **CEO 不友好** — RViz 是给 ROS 开发者设计的，不适合非开发者做决策和验收。

## 决定

用 rosbridge_websocket + roslibjs + Three.js 自建 Web 可视化前端，作为 Vector OS Nano 的调试和演示工具。保留 RViz 作为备选，但日常调试和演示用 Web 前端。

## 架构

```
ROS2 Topics (MuJoCo bridge)
  ↓
rosbridge_websocket (ws://localhost:9090)
  ↓
Browser (localhost:8080)
  ├── roslibjs — ROS2 topic 订阅/发布
  ├── Three.js — 3D 渲染
  │     ├── PointCloud (BufferGeometry + Points)
  │     ├── V-Graph edges (LineSegments)
  │     ├── Robot model (简化几何体 + 朝向箭头)
  │     ├── Path (Line)
  │     ├── Waypoint markers (Sphere)
  │     └── Room labels (CSS2DRenderer)
  └── UI overlay
        ├── SceneGraph 房间列表 + 连通性
        ├── ObjectMemory 物体列表 + 置信度
        ├── VGG GoalTree 执行进度
        └── 导航状态 (FAR/TARE/bridge)
```

## 订阅的 Topics

| Topic | Type | 渲染方式 | 频率 |
|-------|------|---------|------|
| /registered_scan | PointCloud2 | Points (高度着色) | 5 Hz |
| /free_paths | PointCloud2 | LineSegments (绿色) | 5 Hz |
| /state_estimation | Odometry | 机器人位置 + 朝向 | 降采样 10 Hz |
| /path | Path | Line (青色) | 按需 |
| /way_point | PointStamped | Sphere marker (红色) | 1 Hz |
| /global_path | Path | Line (黄色) | 按需 |
| /exploration_path | Path | Line (橙色) | 按需 |
| /camera/image | Image | 2D 画中画 | 5 Hz |

## 不做什么

- 不做完整的 RViz 替代品（不支持任意 display 插件）
- 不做 TF 树可视化（用坐标直接渲染）
- 不做 URDF 模型加载（用简化几何体）
- 不做远程部署（只 localhost 开发用）

## 技术选择

| 组件 | 选择 | 原因 |
|------|------|------|
| WebSocket bridge | rosbridge_suite | ROS2 官方支持，apt 安装即用 |
| JS ROS client | roslibjs | 成熟，配合 rosbridge |
| 3D 渲染 | Three.js | 最成熟的 Web 3D 库，点云渲染性能好 |
| UI | 原生 HTML/CSS | 不需要 React/Vue，保持简单 |
| 静态服务 | Python http.server | 不需要 Node.js |
| PointCloud2 解析 | 手写 DataView 解析 | roslibjs 不内置 PointCloud2 解析 |

## 文件结构

```
vector_os_nano/web/viz/
  index.html          — 主页面
  js/
    app.js            — 入口，初始化 Three.js + roslibjs
    ros_bridge.js     — ROS topic 订阅管理
    scene.js          — Three.js 场景设置（相机、灯光、地面）
    pointcloud.js     — PointCloud2 解析 + 渲染
    vgraph.js         — free_paths → LineSegments
    robot.js          — 机器人位置 + 朝向渲染
    paths.js          — Path 消息 → Line 渲染
    ui.js             — 状态面板 (SceneGraph, ObjectMemory, VGG)
  css/
    style.css         — 布局样式
  launch_viz.sh       — 一键启动 rosbridge + HTTP server
```

## 启动方式

```bash
# 在 vector-cli 启动后（ROS2 topics 已经在发）
cd ~/Desktop/vector_os_nano
./web/viz/launch_viz.sh
# 浏览器打开 http://localhost:8080
```

或集成到 vector-cli 里作为一个命令：
```
vector> /viz
  Opening http://localhost:8080 ...
```

## 开发计划

### Wave 1: 最小可用 (MVP)
- rosbridge 连接 + Three.js 场景
- /registered_scan 点云渲染（高度着色）
- /state_estimation 机器人位置
- /path 当前路径线
- 基本相机控制（轨道、缩放）

### Wave 2: V-Graph + 导航
- /free_paths V-Graph edges (LineSegments)
- /way_point 目标点 marker
- /global_path 全局路径
- /exploration_path 探索路径
- Room labels (从 SceneGraph HTTP API 或 hardcode 布局)

### Wave 3: Vector OS 集成
- /camera/image 画中画
- SceneGraph 房间/门/物体面板
- ObjectMemory 置信度可视化
- VGG GoalTree 执行进度
- 集成到 vector-cli (/viz 命令)

## 风险

| 风险 | 缓解 |
|------|------|
| PointCloud2 解析性能 | 降采样：只渲染每 N 个点，或用 Web Worker |
| rosbridge 带宽 | 大 topic (pointcloud) 降频到 2Hz |
| Three.js 点云内存 | 复用 BufferGeometry，不每帧 new |
| 浏览器兼容性 | 只支持 Chrome/Firefox，不做兼容 |
