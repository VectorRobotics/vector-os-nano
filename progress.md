# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-08
**Version:** v1.3.0
**Branch:** robo-cli (8 commits, +5400 lines, 431 new tests)

## Vector CLI (NEW)

Unified `vector` command — all robot interaction from the terminal.

```bash
vector                    # Interactive REPL
vector go2 stand          # One-shot Go2 command (12 commands)
vector arm home           # Arm commands (8 commands)
vector gripper open       # Gripper commands
vector perception detect  # VLM commands
vector sim start          # Simulation lifecycle
vector ros nodes          # ROS2 diagnostics (via rosm)
vector status             # Hardware status (22 skills registered)
vector skills             # List all skills with aliases
vector chat               # LLM agent mode (vcli engine)
```

Package: `vector_os_nano/robo/` (13 files, 1419 lines)
Entry point: `vector` (pyproject.toml)
Framework: Click (same as rosm)

## Vibe Code for Robotics (NEW)

AI dev environment — write code + control robot in one session.

New capabilities added to VectorEngine:
- **RobotContextProvider** — position, room, SceneGraph, nav state injected into LLM prompt
- **7 new tools**: scene_graph_query, ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status, skill_reload
- **CategorizedToolRegistry** — tools grouped by category (code/robot/diag/system), enable/disable at runtime
- **REPL → VectorEngine** — `vector` REPL now uses full agent loop with all tools
- **Hot reload** — edit skill code, reload without restarting sim

Package: `vcli/tools/` (4 new files) + `vcli/robot_context.py`
Tests: 415 vcli unit tests (all green)

## Architecture

```
vector-cli (agent process)          launch_explore.sh (subprocess)
  LLM + SceneGraph + VLM              MuJoCoGo2 (convex MPC, 1kHz)
  Go2ROS2Proxy ◄── ROS2 ──►           Go2VNavBridge (200Hz odom)
  12 Go2 skills                        localPlanner + FAR + TARE
                                       terrainAnalysis + RViz
```

## Navigation Pipeline

```
Sim startup:
  config/room_layout.yaml → SceneGraph (8 rooms, 7 doors, instant)

Explore:
  TARE autonomous → position-based room detection (nearest_room)
  Room transitions → SceneGraph learns doors (add_door)

Navigate:
  Phase 1 (5s): /goal_point → wait for FAR /way_point
  Phase 2: FAR V-Graph routing (/goal_point only)
  Phase 3: door-chain fallback (SceneGraph BFS → /way_point → localPlanner)
```

## Path Follower (two-mode quadruped controller)

```
TRACK mode (heading error < 60°):
  vx = speed × cos(err), vy = -speed × sin(err)
  Space-aware: open → 0.8 m/s, tight → proportional slowdown
  Gentle yaw correction (gain=4.0, smoothed 0.04/tick)

TURN mode (heading error > 60°, hysteresis at 30°):
  vx = 0.05 (gait creep), vy = 0 (no strafe)
  Snappy rotation (gain=6.0, smoothed 0.08/tick)

All axes smoothed: vx 0.04/tick, vy 0.02/tick, vyaw mode-dependent
Deceleration 2x faster than acceleration
Cylinder body safety: 3 zones (comfort/push/danger) based on gap
```

## SceneGraph
- Rooms + doors from config (sim) or exploration (real)
- APIs: add_door(), get_door(), get_door_chain() BFS, nearest_room()
- Persistence: YAML save/load
- load_layout(): seed from config file on startup

## Terrain Persistence
- TerrainAccumulator: 2D voxel grid → ~/.vector_os_nano/terrain_map.npz
- Auto-save every 30s during explore
- Replay to /registered_scan + /terrain_map + /terrain_map_ext

## Local VLM (Ollama)
- Ollama + gemma4:e4b installed (RTX 5080 16GB)
- VLM backend switchable via VECTOR_VLM_URL env var
- Auto-look disabled for sim (room detection is config-based)
- Available for real-world scene description when needed

## Harness Tests: 750+ total
| Suite | Tests | Status |
|-------|-------|--------|
| Locomotion L0-L4 | 26 | pass |
| Agent+Go2 | 5 | pass |
| VLM+Scene L0-L9 | 200+ | pass |
| Nav L17-L33 | 247 | pass |
| Sim-to-Real L34-L38 | 120+ | pass |
| Other | 150+ | pass |

## Sim-to-Real Design

| Component | Sim (current) | Real (future) |
|-----------|--------------|---------------|
| Localization | MuJoCo ground-truth | SLAM (arise_slam_mid360) |
| Room detection | config/room_layout.yaml | SLAM + semantic understanding |
| Room identification | Position-based (instant) | VLM or topological segmentation |
| VLM | Ollama gemma4:e4b (disabled for sim) | Active for scene description |
| Interface | /state_estimation (same) | /state_estimation (same) |

## CLI Commands
- `/reset` — one-click recovery from tip-over (stand up at current position)
- `/clear_memory` — clear SceneGraph (auto-reloads layout on next explore)
- `stop` — immediately cancels navigation (0.5s response)

## Known Limitations
- FAR V-Graph coverage depends on TARE exploration thoroughness
- TARE sometimes misses rooms → door-chain fallback handles it (10s stall detection)
- Real-world room detection needs SLAM + spatial understanding
- Ollama + Open WebUI installed (Docker needs reboot for iptables)
