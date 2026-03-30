# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-30
**Current version:** v0.5.0-dev

---

## Active: Go2 Dual-Backend Locomotion — ALL LEVELS PASS

### Locomotion Harness Results (2026-03-30)
```
21 passed, 1 skipped (lateral), 0 failed — 79.7s
```
| Level | Test | Result |
|-------|------|--------|
| L0 | Model loading, physics, actuators | 5/5 PASS |
| L1 | PD standing, stability, tracking | 5/5 PASS |
| L2 | Forward, backward, turn, upright | 5/5 PASS |
| L3 | Velocity tracking, yaw, stop | 4/4 PASS (1 skip) |
| L4 | Waypoint navigation, heading | 2/2 PASS |

### Dual-Backend Architecture
```
MuJoCoGo2(backend="auto")
  ├── Backend A: Sinusoidal trot (mujoco + numpy only, zero deps)
  └── Backend B: Convex MPC (convex_mpc + casadi + pinocchio)

connect() auto-detects convex_mpc → uses MPC if available, falls back to sinusoidal
```

### Backend B: convex_mpc on Python 3.12
- casadi 3.7.2 + pinocchio 3.9.0 + scipy installed in .venv
- convex_mpc installed from ~/Desktop/go2-convex-mpc (relaxed Python constraint)
- Full MPC stack: CentroidalMPC + LegController + Gait + ComTraj
- QP solver failures caught gracefully (falls back to PD hold)

### Nav2 Integration — READY
- Bridge: `scripts/go2_nav_bridge.py` — publishes /odom, /scan, TF; subscribes /cmd_vel
- Launch: `./scripts/launch_nav2.sh --rviz`
- Map: house.pgm (20x14m), Go2 starts at (10, 3)

### Next Steps
1. Test Nav2 end-to-end (send goal, Go2 walks)
2. Add Level 5 harness test
3. Run full 1262-test suite

---

## v0.5.0-dev — NavStackClient + HAL

### NavStackClient Nav2 Mode — DONE
- Dual-mode: auto/nav2/cmu
- 53 unit tests + 16 existing

### Hardware Abstraction Layer — DONE
- BaseProtocol, Odometry/LaserScan types, SkillContext dict registries

---

## Previous Milestones
- v0.4.0: Go2 MuJoCo Milestone 1 (convex MPC — now replaced with sinusoidal gait)
- v0.2.0: MCP + Memory + Router
- v0.1.0: Foundation

---

## Workspaces
| Workspace | Location | Purpose |
|-----------|----------|---------|
| vector_os_nano | ~/Desktop/vector_os_nano/ | Python SDK |
| vector_go2_sim | ~/Desktop/vector_go2_sim/ | ROS2 Nav2 + controllers |
| unitree_mujoco | ~/Desktop/unitree_mujoco/ | MuJoCo Go2 simulator (legacy) |

## Test Summary
| Suite | Count | Status |
|-------|-------|--------|
| Locomotion harness | 7/7 | PASS |
| NavStackClient Nav2 | 53 | PASS (last run 2026-03-27) |
| NavigateSkill Nav2 | 38 | PASS (last run 2026-03-27) |
| Full OS Nano unit suite | 1262 | PASS (last run 2026-03-27, needs rerun) |
