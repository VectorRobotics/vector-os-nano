# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-08
**Version:** v1.4.0-dev
**Branch:** robo-cli (13 commits ahead of master)

## VGG: Verified Goal Graph — Phase 1+2 Complete

Cognitive layer — LLM decomposes complex tasks into verifiable sub-goal trees. Full implementation with code execution sandbox, persistent strategy learning, and template compilation.

```
User: "去厨房看看有没有杯子"
  ↓
GoalDecomposer (LLM → GoalTree JSON, template matching first)
  ├─ reach_kitchen    verify: nearest_room() == 'kitchen'    strategy: navigate_skill
  ├─ observe_table    verify: 'table' in describe_scene()    strategy: look_skill
  └─ detect_cup       verify: len(detect_objects('cup')) > 0  strategy: vlm_detect
  ↓
GoalExecutor: execute step → verify → fallback if failed
  ↓
StrategyStats: record success rate, feed into StrategySelector
  ↓
ExperienceCompiler: convert successful traces → reusable templates
```

### Cognitive Layer (vcli/cognitive/)

| Component | Purpose |
|-----------|---------|
| GoalDecomposer | LLM → GoalTree, template matching first |
| GoalVerifier | Safe sandbox for verify expressions |
| StrategySelector | Rule + stats-driven strategy selection |
| GoalExecutor | Execute + verify + fallback + stats recording |
| CodeExecutor | RestrictedPython sandbox (velocity clamped, AST validated, 30s timeout) |
| StrategyStats | Persistent success rate tracking (~/.vector_os_nano/strategy_stats.json) |
| ExperienceCompiler | Traces → parameterized templates |
| TemplateLibrary | Store + match + instantiate templates |

### Primitives API (vcli/primitives/)

25 functions across 4 categories:
- **locomotion** (8): get_position, get_heading, walk_forward, turn, stop, stand, sit, set_velocity
- **navigation** (5): nearest_room, publish_goal, wait_until_near, get_door_chain, navigate_to_room
- **perception** (6): capture_image, describe_scene, detect_objects, identify_room, measure_distance, scan_360
- **world** (6): query_rooms, query_doors, query_objects, get_visited_rooms, path_between, world_stats

### CLI Integration

- IntentRouter.is_complex(): keyword + multi-verb detection
- GoalTree plan displayed BEFORE execution
- Step-by-step feedback with [idx/total] prefix
- vgg_decompose() / vgg_execute() split for plan-then-execute flow

Design spec: `docs/vgg-design-spec.md`

## Sensor Configuration

- **Lidar**: Livox MID-360, -20 deg downward tilt (match real Go2 hardware)
- **Terrain Analysis**: VFoV -30/+35 deg (matched to MID-360 at 20deg tilt, world frame)
- **VLM**: OpenRouter (google/gemma-4-31b-it) — replaces local Ollama

## Navigation Pipeline

```
Explore: TARE autonomous → position-based room detection → SceneGraph doors
Navigate: FAR V-Graph → door-chain fallback (nav stack waypoints with obstacle avoidance)
Path follower: TRACK/TURN modes, cylinder body safety, space-aware speed
```

### V-Graph Ceiling Fix (In Progress)

Root cause identified: ceiling points (intensity > 1.8m) pollute FAR's obstacle cloud.
- Fix: filter ceiling points in _publish_pointcloud() before publishing /registered_scan
- FAR now only sees ground + walls → visibility checks work correctly

### Door-chain Navigation (In Progress)

Fallback mechanism: when FAR graph is incomplete, fall back to SceneGraph BFS waypoint chain.
- Changed from dead-reckoning (straight line through walls) to nav stack waypoints
- Uses base.navigate_to() with localPlanner obstacle avoidance

## Vector CLI

Unified `vector` command — all robot interaction from terminal.

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
vector chat               # LLM agent mode (vgg cognitive layer)
```

## Vibe Code for Robotics

AI dev environment — write code + control robot in one session.
- CategorizedToolRegistry: 39 tools (code/robot/diag/system)
- IntentRouter: keyword-based tool routing (~52% token savings)
- DynamicSystemPrompt: robot state refreshed every LLM turn
- Hot reload: edit skill code, reload without restarting sim

## Architecture

```
vector-cli (agent process)          launch_explore.sh (subprocess)
  VGG cognitive layer                 MuJoCoGo2 (convex MPC, 1kHz)
  LLM + SceneGraph + VLM              Go2VNavBridge (200Hz odom)
  Go2ROS2Proxy ◄── ROS2 ──►           localPlanner + FAR + TARE
  22 Go2 skills                        terrainAnalysis + RViz
```

## Test Coverage: 1000+ Tests

| Suite | Tests | Status |
|-------|-------|--------|
| Locomotion L0-L4 | 26 | pass |
| Agent+Go2 | 5 | pass |
| VLM+Scene L0-L9 | 200+ | pass |
| Nav L17-L33 | 247 | pass |
| Sim-to-Real L34-L38 | 120+ | pass |
| Nav fixes L39-L40 | 27 | pass |
| VGG Phase 1 L41-L46 | 187 | pass |
| VGG Phase 2 L47-L50 | 87 | pass |
| VGG CLI Feedback | 25 | pass |
| Other | 80+ | pass |

## Known Limitations

- FAR V-Graph coverage depends on TARE exploration thoroughness
- TARE sometimes misses rooms → door-chain fallback handles it
- Real-world room detection needs SLAM + spatial understanding
- VGG GoalDecomposer quality depends on LLM (needs live testing)

## Next Milestones

- Complete V-Graph ceiling filter, deploy on real Go2
- Test door-chain obstacle avoidance
- Run VGG live on real navigation tasks
- Train ExperienceCompiler templates from user sessions
- Phase 3: RL executor integration
