# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-08
**Version:** v1.4.0-dev
**Branch:** robo-cli (13 commits ahead of master)

## VGG: Verified Goal Graph (NEW)

Cognitive layer — LLM decomposes complex tasks into verifiable sub-goal trees.

```
User: "去厨房看看有没有杯子"
  ↓
GoalDecomposer (LLM → GoalTree JSON)
  ├─ reach_kitchen    verify: nearest_room() == 'kitchen'    strategy: navigate_skill
  ├─ observe_table    verify: 'table' in describe_scene()    strategy: look_skill
  └─ detect_cup       verify: len(detect_objects('cup')) > 0  strategy: vlm_detect
  ↓
GoalExecutor: execute each step → verify → fallback if failed
  ↓
ExecutionTrace: transparent record of what happened and why
```

### Cognitive Layer (vcli/cognitive/)

| Component | File | Purpose |
|-----------|------|---------|
| GoalDecomposer | goal_decomposer.py | LLM → GoalTree, template matching first |
| GoalVerifier | goal_verifier.py | Safe sandbox for verify expressions |
| StrategySelector | strategy_selector.py | Rule + stats-driven strategy selection |
| GoalExecutor | goal_executor.py | Execute + verify + fallback + stats recording |
| CodeExecutor | code_executor.py | Code-as-Policy sandbox (velocity clamped) |
| StrategyStats | strategy_stats.py | Persistent success rate tracking |
| ExperienceCompiler | experience_compiler.py | Traces → parameterized templates |
| TemplateLibrary | template_library.py | Store + match + instantiate templates |

### Primitives API (vcli/primitives/)

25 functions across 4 categories wrapping existing interfaces:
- **locomotion**: get_position, get_heading, walk_forward, turn, stop, stand, sit, set_velocity
- **navigation**: nearest_room, publish_goal, wait_until_near, get_door_chain, navigate_to_room
- **perception**: capture_image, describe_scene, detect_objects, identify_room, measure_distance, scan_360
- **world**: query_rooms, query_doors, query_objects, get_visited_rooms, path_between, world_stats

### Integration

- IntentRouter.is_complex(): keyword-based complexity detection
- Simple tasks → existing skill dispatch (zero overhead)
- Complex tasks → VGG goal decomposition pipeline
- VectorEngine.init_vgg() / try_vgg(): optional VGG path

Design spec: `docs/vgg-design-spec.md`

## Vector CLI

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

## Navigation Pipeline

```
Explore: TARE autonomous → position-based room detection → SceneGraph doors
Navigate: FAR V-Graph → door-chain fallback (SceneGraph BFS)
Path follower: TRACK/TURN modes, cylinder body safety, space-aware speed
```

## Sensor Config

- Livox MID-360: -7 to +52 deg FOV, 20 deg downward tilt
- terrainAnalysis: VFoV -30/+35 deg (matched to MID-360)
- Bridge terrain feed: bypasses terrainAnalysis 3.5m range filter for FAR

## Harness Tests: 1000+ total

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
| Other | 150+ | pass |

## Known Limitations

- FAR V-Graph coverage depends on TARE exploration thoroughness
- TARE sometimes misses rooms → door-chain fallback handles it
- Real-world room detection needs SLAM + spatial understanding
- VGG GoalDecomposer quality depends on LLM (needs live testing)
