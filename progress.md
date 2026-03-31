# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-31
**Version:** v0.5.0-dev
**Branch:** feat/vector-os-nano-python-sdk

---

## Current: Go2 MuJoCo + Vector Navigation Stack

### Architecture
```
User
  ├── RViz teleop panel ──→ /joy ──→ bridge (direct velocity)
  ├── "走到厨房" ──→ Agent ──→ NavigateSkill
  └── /goal_point ──→ FAR planner ──→ /way_point ──→ localPlanner
                                                        ↓
                                                      /path
                                                        ↓
MuJoCoGo2 (convex MPC, 1kHz)  ←── bridge ←── /navigation_cmd_vel
  ├── publishes: /state_estimation (200Hz), /registered_scan (5Hz, 10k+ pts)
  ├── publishes: /camera/image (5Hz, 320x240), /speed (2Hz)
  ├── TF: map→sensor, map→vehicle
  └── terrain_analysis produces /terrain_map (~49k pts)
```

### Harness Results
| Suite | Result | Details |
|-------|--------|---------|
| Locomotion (pytest) | **26/26** | L0 physics → L4 navigation |
| Agent+Go2 | **5/5** | walk, turn, stand, sit, skills |
| Nav2 integration | **11/11** | AMCL + MPPI, goal arrival |
| SLAM mapping | **3/3** | map grows during movement |
| VNav integration | **~15/18** | teleop, camera, terrain pass |

### What Works
- Go2 walks with unitree convex MPC (auto-detected, sinusoidal fallback)
- Livox MID360 simulation: 30° tilt, -7° to +52° FOV, 10k+ points/scan
- Point cloud intensity = height above ground (terrain_analysis compatible)
- Vector Nav Stack: localPlanner, pathFollower, terrain_analysis, FAR planner
- Teleop from RViz panel (direct /joy → velocity)
- Camera RGB from MuJoCo renderer
- Nav2 + SLAM alternatives also available
- Agent SDK: natural language → Go2 skills

### Known Issues
1. FAR planner publishes /way_point but not /global_path (graph_decoder issue)
2. Camera depth rendering needs MuJoCo API fix
3. Long-distance autonomous navigation needs FAR graph accumulation

### Scripts
| Script | Purpose |
|--------|---------|
| `./scripts/launch_vnav.sh` | Full Vector Nav Stack + RViz |
| `./scripts/launch_nav2.sh --rviz` | Nav2 + AMCL alternative |
| `./scripts/launch_slam.sh` | SLAM real-time mapping |
| `.venv-nano/bin/python3 scripts/go2_demo.py` | Visual locomotion demo |
| `.venv-nano/bin/python3 run.py --sim-go2` | Agent mode (NL control) |
| `.venv-nano/bin/python3 -m pytest tests/harness/ -v` | Locomotion harness |
| `./scripts/test_integration.sh` | Full integration harness |

---

## Session Log (2026-03-30/31)

1. **Go2 locomotion**: Replaced broken convex_mpc dependency with sinusoidal gait (Backend A), then installed convex_mpc on Python 3.12 as Backend B. Dual-backend auto-detection.

2. **Nav2**: Bridge publishes /odom + /scan + TF. Fixed QoS (RELIABLE for /scan). Fixed cmd_vel topic (Nav2 Jazzy uses /cmd_vel_nav). 11/11 pass.

3. **SLAM**: slam_toolbox online mapping verified. Map grows during movement.

4. **Visualization**: 3D point cloud (AxisColor by Z), terrain map, paths in RViz.

5. **Vector Nav Stack**: Rebuilt for Jazzy. Bridge adapted: /state_estimation, /registered_scan, /joy, /speed. localPlanner + pathFollower + terrain_analysis running. FAR planner integrated.

6. **MID360 lidar**: Correct asymmetric FOV (-7° to +52°), 30° downward pitch tilt in body frame, sensor at (0.2, 0, 0.1). Intensity = height above ground.

7. **Teleop**: Direct /joy → velocity bypass (pathFollower needs /path to move). Priority mechanism prevents path follower override.

8. **Camera**: MuJoCo renderer → /camera/image at 5Hz.

9. **Agent integration**: `run.py --sim-go2` works. Walk/turn/sit skills verified.
