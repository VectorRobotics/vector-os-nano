# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-31
**Current version:** v0.5.0-dev

---

## Active: Go2 MuJoCo + Vector Navigation Stack

### Locomotion Harness: 26/26 PASS
- Dual-backend MPC (convex_mpc) + sinusoidal fallback
- Forward/backward/turn/velocity/navigation all verified

### Vector Nav Stack Integration
```
MuJoCoGo2 (MPC, 1kHz physics)
  ↕ go2_vnav_bridge.py
  ├── /state_estimation (200Hz, map→sensor)
  ├── /registered_scan (5Hz, 10776 pts, MID360 FOV, 30° tilt, intensity=height)
  ├── /camera/image (5Hz, 320x240 RGB)
  ├── /speed (2Hz, 0.5 m/s)
  ├── subscribe: /joy (teleop), /navigation_cmd_vel, /cmd_vel_nav
  └── TF: map→sensor, map→vehicle
        ↕
  Vector Nav Stack (Jazzy)
  ├── localPlanner + pathFollower (unitree_go2 config)
  ├── terrain_analysis + ext (~49k pts)
  ├── FAR planner (running, graph building needs movement)
  ├── odomTransformer (sensor→base_link)
  └── sensorScanGeneration
```

### Harness Status
| Suite | Pass | Notes |
|-------|------|-------|
| Locomotion | 26/26 | L0-L4 all pass |
| Agent+Go2 | 5/5 | walk/turn/sit via SDK |
| Integration | ~15/18 | teleop+camera pass, FAR path WIP |

### Known Issues
1. FAR planner needs robot movement to build visibility graph (no static planning)
2. Camera depth rendering intermittent
3. Integration test timing-sensitive (startup order)

### Scripts
- `./scripts/launch_vnav.sh` — full stack with RViz
- `./scripts/go2_demo.py` — visual locomotion demo
- `./scripts/test_integration.sh` — full integration harness
- `.venv-nano/bin/python3 -m pytest tests/harness/` — locomotion harness

---

## Previous Milestones
- Go2 dual-backend locomotion (sinusoidal + MPC)
- Nav2 integration (11/11 pass)
- SLAM real-time mapping (slam_toolbox)
- Agent + Go2 skills (Phase 1)
- Point cloud visualization
