# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-27
**Current version:** v0.5.0-dev

---

## Active: Go2 MuJoCo Navigation Simulation

### Architecture (sim-to-real ready)
```
unitree_mujoco (MuJoCo physics, indoor house scene)
  ↕ unitree_sdk2 DDS (domain=1, lo)  ← same interface as real Go2
hardware_unitree_sdk2 (ros2_control)
  ↕
unitree_guide_controller (Unitree official gait FSM)
  ↕ /control_input
cmd_vel_bridge.py (/cmd_vel → /control_input)
  ↕ /cmd_vel
Nav2 (SLAM Toolbox + MPPI controller + SmacPlanner2D)
  ↕
Vector OS Nano Agent (NavStackClient, NavigateSkill)
```

### Status
- [x] Ubuntu 24.04 + ROS2 Jazzy (upgraded from 22.04 + Humble)
- [x] MuJoCo Go2 simulation — stable trotting gait in indoor house
- [x] unitree_sdk2 interface — identical for sim and real hardware
- [x] Sensor bridges: /odom (50Hz), /scan (10Hz lidar via mj_ray), odom→base TF
- [x] cmd_vel_bridge: /cmd_vel + /cmd_vel_nav → /control_input (velocity ramping)
- [x] Nav2 config: Jazzy default params, MPPI controller, base frame="base"
- [x] SLAM Toolbox: builds map from /scan
- [ ] Nav2 autonomous navigation — costmap bounds issue (robot at map edge)
- [ ] Agent integration: NavStackClient → Nav2 → Go2

### Known Issues
1. Nav2 costmap: robot starts at (10,3) which is at SLAM map edge → "out of bounds". Need to teleop first to grow map.
2. cmd_vel_bridge 50Hz publishing overrides keyboard_input → kill bridge before manual control.
3. Gz Sim (DART) trotting unstable — MuJoCo confirmed as the correct sim backend.

### Workspaces
| Workspace | Location | Purpose |
|-----------|----------|---------|
| vector_os_nano | ~/Desktop/vector_os_nano/ | Python SDK (.venv needs rebuild for 3.12) |
| vector_go2_sim | ~/Desktop/vector_go2_sim/ | ROS2 Nav2 + controllers |
| unitree_mujoco | ~/Desktop/unitree_mujoco/ | MuJoCo Go2 simulator |
| unitree_sdk2 | /opt/unitree_robotics/ | C++ SDK (system install) |

---

## v0.5.0-dev — NavStackClient + HAL

### NavStackClient Nav2 Mode — DONE
- Dual-mode: auto/nav2/cmu — API unchanged
- Nav2 NavigateToPose action client + feedback + cancel
- 53 unit tests + 16 existing (zero regression)

### Bug Fixes — DONE
- NavigateSkill ignored navigate_to() return value
- go2/__init__.py imported deleted explore module

### Hardware Abstraction Layer — DONE
- BaseProtocol, Odometry/LaserScan types, SkillContext dict registries

---

## Previous Milestones
- v0.4.0: Go2 MuJoCo Milestone 1 (convex MPC, 6 skills, 48 tests)
- v0.2.0: MCP + Memory + Router
- v0.1.0: Foundation

---

## Test Summary
| Suite | Count | Status |
|-------|-------|--------|
| NavStackClient Nav2 | 53 | PASS |
| NavigateSkill Nav2 | 38 | PASS |
| Nav2 config validation | 27 | PASS |
| Full unit suite | 1262 | PASS |

## Launch Commands
```bash
# MuJoCo + Nav2 full stack
cd ~/Desktop/vector_go2_sim && ./launch_nav.sh --rviz

# Keyboard control (separate terminal)
ros2 run keyboard_input keyboard_input  # press 2→4→WASD

# SO-101 arm sim
cd ~/Desktop/vector_os_nano && python run.py --sim
```
