# ADR-004: ROS2 Web Visualization 替代 RViz

**Date:** 2026-04-09
**Status:** Accepted (revised)
**Author:** Yusen (CEO) + Architect
**Revision:** 2026-04-09 — 方案从 Three.js 自建改为 Foxglove Studio

## 背景

Vector OS Nano 目前依赖 RViz2 做导航和探索的实时可视化。在 MuJoCo 仿真 + FAR/TARE nav stack 的调试过程中，暴露了多个 RViz 的局限：

1. **V-Graph 不可见** — FAR 的 `/free_paths` (PointCloud2) 用 Points 样式渲染，2px 大小在黑色背景上几乎不可见。数据实际在流（50k+ 点），但看不到。
2. **自定义消息类型不支持** — `/decoded_vgraph` (visibility_graph_msg/msg/Graph) 在 RViz 中无法解析，报 "type invalid"。
3. **无法显示 VGG 数据** — GoalTree 执行进度、ObjectMemory 置信度、SceneGraph 房间拓扑等 Vector OS 特有数据在 RViz 中无法呈现。
4. **配置脆弱** — RViz display 的 topic/QoS/样式配置容易出错，调试耗时。
5. **CEO 不友好** — RViz 是给 ROS 开发者设计的，不适合非开发者做决策和验收。

## 决定

~~用 rosbridge_websocket + roslibjs + Three.js 自建 Web 可视化前端。~~

**修订 (2026-04-09):** 改用 Foxglove Studio + foxglove_bridge。原因：
- Three.js 自建方案开发成本高 (~1600 行代码)，视觉效果调试困难 (bloom 光污染)
- Foxglove 内置 PointCloud2 渲染 (turbo colormap)、路径显示、摄像头画中画、标记可视化
- DimOS (Unitree Go2 项目) 已验证 Foxglove 方案可行
- 自定义面板可通过 React 扩展实现，覆盖 SceneGraph/ObjectMemory/VGG 需求

## 架构

```
ROS2 Topics (MuJoCo bridge)
  ↓
foxglove_bridge (ws://localhost:8765, Foxglove WebSocket 协议)
  ↓
Foxglove Studio (app.foxglove.dev 或桌面版)
  ├── 3D Panel — PointCloud2, Path, Marker, Odometry (内置)
  ├── Image Panel — /camera/image (内置)
  ├── Plot Panel — 速度曲线 (内置)
  ├── Raw Messages — 调试用 (内置)
  └── Custom Extension Panel — SceneGraph/VGG (Wave 2, React)
```

## Topic 配置

Dashboard 中配置的 topic 及 Foxglove 渲染参数：

| Topic | Type | Foxglove 渲染 | 说明 |
|-------|------|-------------|------|
| /registered_scan | PointCloud2 | turbo colormap by z, pointSize=2.5, decay=3s | 累积点云地图 |
| /free_paths | PointCloud2 | turbo colormap by intensity, pointSize=1.5 | FAR V-Graph 可通行路径 |
| /path | Path | 白色线, lineWidth=0.05 | 当前导航路径 |
| /global_path | Path | teal 线, lineWidth=0.04 | FAR 全局路径 |
| /exploration_path | Path | 橙色线, lineWidth=0.03 | TARE 探索路径 |
| /way_point | PointStamped | 红色 marker | 当前目标点 |
| /goal_point | PointStamped | 橙色 arrow | FAR 全局目标 |
| /scene_graph_markers | MarkerArray | 原样显示 | 房间标签、门、轨迹 |
| /camera/image | Image | Image Panel | RGB 画中画 |
| /state_estimation | Odometry | Raw Messages + Plot | 位姿 + 速度曲线 |
| /viz_graph_topic | MarkerArray | freespace_vgraph + polygon_edge 可见 | FAR V-Graph 可视化 |

## 技术选择

| 组件 | 选择 | 原因 |
|------|------|------|
| WebSocket bridge | foxglove_bridge | C++ 高性能，原生 Foxglove WebSocket 协议，自动发现所有 topic |
| 可视化前端 | Foxglove Studio | 内置 PointCloud2/Path/Marker/Image 渲染，支持自定义扩展 |
| 自定义面板 | Foxglove Extension (React) | 用于 SceneGraph/ObjectMemory/VGG 数据 (Wave 2) |
| 安装方式 | apt (ros-jazzy-foxglove-bridge) | 无需 npm/Node.js/三方库 |

## 文件结构

```
vector_os_nano/foxglove/
  vector-os-dashboard.json   — Dashboard 布局配置 (导入 Foxglove)
  launch_foxglove.sh         — 一键启动 foxglove_bridge
```

## 启动方式

```bash
cd ~/Desktop/vector_os_nano
./foxglove/launch_foxglove.sh
# 浏览器打开 app.foxglove.dev → Open connection → ws://localhost:8765
# 导入 foxglove/vector-os-dashboard.json
```

## 开发计划

### Wave 1: Dashboard 配置 (当前)
- foxglove_bridge + foxglove_msgs 安装
- Dashboard JSON (3D 透视 + 俯视 + 摄像头 + 速度曲线)
- 所有 nav stack topic 配置
- 启动脚本

### Wave 2: 自定义面板
- Foxglove Extension: SceneGraph 房间/门/物体面板
- Foxglove Extension: ObjectMemory 置信度面板
- Foxglove Extension: VGG GoalTree 执行进度
- vector-cli /viz 命令集成

## 废弃的方案

Three.js 自建方案 (rosbridge + roslibjs + Three.js) 开发成本高、bloom 效果调试困难。
DimOS 团队也使用 Foxglove (+ Rerun) 而非自建渲染引擎。详见 git 历史。

## 风险

| 风险 | 缓解 |
|------|------|
| Foxglove web 版需要网络 | 可用桌面版或 Docker 自托管 |
| 自定义面板需要 npm 生态 | Wave 2 才涉及，Dashboard JSON 不需要 |
| foxglove_bridge CPU 占用 | C++ 实现，比 Python rosbridge 高效得多 |
