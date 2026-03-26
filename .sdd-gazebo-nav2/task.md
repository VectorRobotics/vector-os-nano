# Tasks: Go2 Gazebo + Nav2 Navigation

**Plan**: plan.md (approved 2026-03-26)
**Status**: IN PROGRESS

---

## Wave 1: Foundation

- [x] T1: Workspace 创建 + CHAMP 构建 [Alpha] -- colcon build clean, all packages installed
- [x] T2: Go2 URDF + MID360 + D435 Sensor Xacro [Alpha] -- mid360.xacro, d435.xacro, go2_sensor.xacro validated

## Wave 2: 环境 + 验证

- [x] T3: 室内 House World [Alpha] -- 589 lines, 34 models, 7 rooms + hallway + furniture
- [x] T4: Gazebo Launch 文件 [Alpha] -- gazebo.launch.py + full_stack.launch.py
- [ ] T5: 遥控行走验证 [pending] -- needs manual Gazebo launch + teleop test

## Wave 3: Nav2 配置

- [x] T6: Nav2 参数文件 [Alpha] -- nav2_params.yaml (297 lines) + slam_params.yaml (65 lines)
- [ ] T7: SLAM Launch + 建图验证 [pending] -- needs Gazebo running + teleop mapping

## Wave 4: 自主导航 + Agent 接入

- [ ] T8: Nav2 Navigation Launch + 验证 [pending] -- needs map from T7
- [x] T9: Full Stack Launch Script [Alpha] -- full_stack.launch.py with mode=slam/nav
- [x] T10: NavStackClient Nav2 模式 [Beta] -- 406 lines, dual-mode (nav2/cmu/auto), 53 tests passing

## Wave 5: 端到端验证

- [x] T11: NavigateSkill 集成验证 [Dispatcher] -- test_nav2_brain.py + 38 skill tests passing

## Wave 6: 收尾

- [x] T12: 测试套件 [Beta + Dispatcher] -- 53 (nav_client_nav2) + 38 (navigate_skill_nav2) + 16 (nav_client existing) = 107 new/verified tests
- [ ] T13: 文档 + Progress [pending]

## Bonus: Bug Fixes

- [x] NavigateSkill._navigate_with_nav_stack() ignored nav.navigate_to() return value -- always returned success=True
- [x] go2/__init__.py imported deleted explore module -- caused 3 test failures
