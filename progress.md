# Vector OS Nano SDK — Progress

**Last updated:** 2026-04-09
**Version:** v1.5.0-dev
**Branch:** robo-cli (29 commits ahead of master)

## VGG: Verified Goal Graph — Complete Framework

Cognitive layer — ALL actionable commands flow through VGG. LLM decomposes complex tasks into verifiable sub-goal trees. Simple commands get 1-step GoalTrees without LLM call.

```
User input
  ↓
should_use_vgg?
  ├─ Action → VGG
  │    ├─ Simple (skill match) → 1-step GoalTree (fast, no LLM)
  │    └─ Complex (multi-step) → LLM decomposition → GoalTree
  │    ↓
  │  VGG Harness: 3-layer feedback loop
  │    Layer 1: step retry (alt strategies)
  │    Layer 2: continue past failure
  │    Layer 3: re-plan with failure context
  │    ↓
  │  GoalExecutor → verify → trace → stats
  │
  └─ Conversation → tool_use path (LLM direct)
```

### Cognitive Layer (vcli/cognitive/)

| Component | Purpose |
|-----------|---------|
| GoalDecomposer | LLM → GoalTree; template + skill fast path |
| GoalVerifier | Safe sandbox for verify expressions |
| StrategySelector | Rule + stats-driven strategy selection |
| GoalExecutor | Execute + verify + fallback + stats recording |
| VGGHarness | 3-layer feedback loop (retry → continue → re-plan) |
| CodeExecutor | RestrictedPython sandbox (velocity clamped) |
| StrategyStats | Persistent success rate tracking |
| ExperienceCompiler | Traces → parameterized templates |
| TemplateLibrary | Store + match + instantiate templates |
| ObjectMemory | Time-aware object tracking with exponential confidence decay |
| predict | Rule-based state prediction from room topology |
| VisualVerifier | VLM-based visual verification fallback |

### Primitives API (vcli/primitives/)

30 functions across 4 categories:
- **locomotion** (8): get_position, get_heading, walk_forward, turn, stop, stand, sit, set_velocity
- **navigation** (5): nearest_room, publish_goal, wait_until_near, get_door_chain, navigate_to_room
- **perception** (6): capture_image, describe_scene, detect_objects, identify_room, measure_distance, scan_360
- **world** (11): query_rooms, query_doors, query_objects, get_visited_rooms, path_between, world_stats, last_seen, certainty, find_object, objects_in_room, room_coverage

### CLI Integration

- Async execution — CLI never blocks during navigation/explore
- GoalTree plan shown before execution
- Step-by-step [idx/total] progress feedback
- VGG only active after sim start (requires functioning robot)

Design spec: `docs/vgg-design-spec.md`

## Sensor Configuration

- **Lidar**: Livox MID-360, -20 deg downward tilt (match real Go2)
- **Terrain Analysis**: VFoV -30/+35 deg (matched to MID-360)
- **VLM**: OpenRouter (google/gemma-4-31b-it)
- **Ceiling filter**: points > 1.8m filtered from /registered_scan (fixes V-Graph)

## Navigation Pipeline

```
Explore: TARE → room detection → SceneGraph doors
Navigate: FAR V-Graph → door-chain fallback (nav stack waypoints)
Path follower: TRACK/TURN modes, cylinder body safety
Stuck recovery: boxed-in detection → 3-4s sustained reverse
```

## Vector CLI

```bash
vector                    # Interactive REPL (VGG cognitive layer)
vector go2 stand          # One-shot Go2 commands
vector sim start          # Simulation lifecycle
vector ros nodes          # ROS2 diagnostics
vector chat               # LLM agent mode
```

## 开发测试流程

所有实机验证通过 vector-cli 对话完成，不直接脚本调用 MuJoCo/ROS2：

```bash
cd ~/Desktop/vector_os_nano
python3 -m vector_os_nano.vcli.cli --sim-go2
```

启动后在 REPL 中用自然语言交互：
- `探索` — 触发 TARE 自主探索
- `去厨房` — VGG 分解 + nav stack 导航
- `看看有没有杯子` — VLM 感知
- `去厨房看看有没有杯子` — VGG 多步分解 + 执行 + 验证

单元测试（无 MuJoCo）：
```bash
python3 -m pytest tests/harness/test_level57_object_memory.py tests/harness/test_level58_predict.py tests/harness/test_level59_visual_verifier.py tests/harness/test_level60_namespace_integration.py tests/harness/test_level60_visual_verify_integration.py tests/harness/test_level61_auto_observe.py -v
```

MuJoCo 集成测试：
```bash
python3 -m pytest tests/harness/test_level62_phase3_mujoco.py -v
```

## Test Coverage: 630+ VGG tests, 1150+ total

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
| VGG CLI L51 | 25 | pass |
| Door-chain L52 | 18 | pass |
| Ceiling filter L53 | 21 | pass |
| VGG Integration L54 | 29 | pass |
| CLI Scenarios L55 | 52 | pass |
| VGG Harness L56 | 24 | pass |
| ObjectMemory L57 | 39 | pass |
| predict L58 | 35 | pass |
| VisualVerifier L59 | 28 | pass |
| Namespace Integration L60 | 21 | pass |
| Auto-Observe L61 | 36 | pass |
| Other | 80+ | pass |

## Phase 3: Active World Model

```
ObjectMemory: SceneGraph → TrackedObject (指数衰减: conf * exp(-0.001 * elapsed))
  ↓
GoalVerifier namespace: last_seen(), certainty(), find_object(), objects_in_room(), room_coverage(), predict_navigation()
  ↓
VisualVerifier: verify 失败 → VLM 拍照二次确认 (感知步骤才触发)
  ↓
Auto-Observe: 探索时每个新 viewpoint → VLM 自动识别物体 → SceneGraph + ObjectMemory
```

## Known Limitations

- VGG complex decomposition quality depends on LLM model
- Async skills (explore, patrol) report "launched" not "completed" in VGG
- FAR V-Graph ceiling fix needs live validation
- Real-world room detection needs SLAM + spatial understanding
- Phase 3 functions need live validation with real LLM + sim
