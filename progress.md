# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-26
**Current version:** v0.5.0-dev
**Focus:** Gazebo + Nav2 Navigation Simulation

---

## v0.5.0-dev — IN PROGRESS

### Gazebo Go2 Simulation (NEW — 2026-03-26)
- [x] Workspace: ~/Desktop/vector_go2_sim/ (14 packages, colcon build clean)
- [x] Go2 URDF: Unitree official model from quadruped_ros2_control (FR/FL/RR/RL naming)
- [x] Sensors: MID360 lidar (Velodyne VLP16 plugin) + D435 depth camera + IMU
- [x] World: AWS RoboMaker Small House (realistic furnished residential environment)
- [x] Planar move mode: gravity-free floating, /cmd_vel → /odom, all sensors working
- [x] Nav2 config: SmacPlanner2D + DWB controller + SLAM Toolbox (params written, untested)
- [x] Launch: `go2launch` alias, world presets (aws_house/indoor_house/empty), locomotion modes
- [ ] Nav2 SLAM mapping (needs Gazebo + teleop drive-through)
- [ ] Nav2 autonomous navigation (needs pre-built map)
- [ ] Agent integration: NavStackClient → Nav2 → Go2 in Gazebo

### Locomotion Controller Status
- CHAMP: unstable in Gazebo (drift/bounce), abandoned
- unitree_guide_controller: built OK, but leg_pd_controller (ChainableController) crashes gzserver on Humble — chained controllers need Jazzy
- Planar move: working, used for Nav2 development
- **Plan: upgrade Ubuntu 22.04 → 24.04 + ROS2 Jazzy for proper controller support**

### NavStackClient Nav2 Mode (NEW — 2026-03-26)
- [x] Dual-mode: auto/nav2/cmu — API signature unchanged
- [x] Nav2: NavigateToPose action client, feedback, cancel, timeout
- [x] CMU: /way_point + /goal_reached (preserved, zero regression)
- [x] 53 unit tests (test_nav_client_nav2.py)
- [x] All 16 existing tests pass (test_nav_client.py)

### Bug Fixes (2026-03-26)
- [x] NavigateSkill._navigate_with_nav_stack() ignored navigate_to() return value — always returned success=True
- [x] go2/__init__.py imported deleted skills.explore module — 3 tests failing

### Hardware Abstraction Layer — DONE
- [x] BaseProtocol, Odometry/LaserScan types, SkillContext dict registries
- [x] MuJoCoGo2: 1kHz physics thread, dual velocity modes, 3D lidar
- [x] NavStackClient, NavigateSkill (hardware-agnostic)

### Navigation Stack Integration
- [x] Unity sim: brain → nav stack → robot loop proven
- [x] MuJoCo bridge: publishes /state_estimation + /registered_scan
- [ ] pathFollower autonomy mode not resolved (CMU nav stack specific)

---

## v0.4.0 — Go2 MuJoCo Milestone 1 — DONE

- Convex MPC locomotion, 6 Go2 skills, indoor house scene
- `python run.py --sim-go2`, 48 tests

## v0.2.0 — MCP + Memory + Router — DONE
## v0.1.0 — Foundation — DONE

---

## Test Summary (2026-03-26)

| Suite | Count | Status |
|-------|-------|--------|
| NavStackClient Nav2 | 53 | PASS |
| NavigateSkill Nav2 | 38 | PASS |
| NavStackClient CMU | 16 | PASS |
| NavigateSkill existing | 20 | PASS |
| Gazebo URDF planar | 27 | PASS |
| Full unit suite | 1262 | PASS |

---

## Launch Commands

```bash
# SO-101 arm
python run.py --sim

# Go2 MuJoCo (locomotion R&D)
python run.py --sim-go2

# Go2 Gazebo (navigation, default: AWS house + planar mode)
go2launch

# Go2 Gazebo with specific world/mode
GO2_WORLD=empty go2launch
GO2_WORLD=indoor_house go2launch
GO2_LOCOMOTION=champ go2launch

# Nav2 brain test
./scripts/run_nav2_brain.sh
```
