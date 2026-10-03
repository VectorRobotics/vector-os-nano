# MuJoCo Sim Developer Guide

## Development Philosophy

Speed commands are the ONLY interface — don't test gait details.

- Sim: sinusoidal gait (mujoco only) or MPC (needs convex_mpc)
- Real: Unitree SDK / ROS2 cmd_vel
- Test with `set_velocity(vx, vy, vyaw)` and `walk(vx, vy, vyaw, duration)`

Never assert on joint angles, gait frequency, or internal controller state. Only
assert on observable outputs: position, heading, lidar scan, camera frame.

## Gate Strategy

| Condition | Effect |
|-----------|--------|
| `mujoco` not installed | Skip all sim tests (`pytest.importorskip`) |
| `convex_mpc` not installed | Skip MPC-specific tests; sinusoidal tests still run |
| LLM API key missing | VGG decompose uses MockLLMBackend; simple commands never call LLM |

Apply the gate at the top of each test file:

```python
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
```

For MPC-specific tests, add a second guard inside the test:

```python
pytest.importorskip("convex_mpc", reason="convex_mpc not installed")
```

## Test Layers

| Layer | Location | What it tests |
|-------|----------|---------------|
| Unit | `tests/unit/test_mujoco_sim.py` | Scene XML generation, connection lifecycle |
| Unit | `tests/unit/test_mujoco_go2.py` | Stand/sit/walk interface, state queries |
| Unit | `tests/unit/test_mujoco_perception.py` | Camera frame, lidar scan, depth |
| Harness | `tests/harness/test_level62_phase3_mujoco.py` | Phase 3 world model + real robot |
| E2E | `tests/harness/test_mujoco_vgg_e2e.py` | Full pipeline: user -> VGG -> physics -> verify |

### When to add to each layer

- **Unit**: testing a single method on MuJoCoGo2 or a utility function
- **Harness level 62+**: testing component interactions that require a live simulation
- **E2E**: testing user-facing commands through the complete VGG stack

## Running Tests

```bash
# Unit only (fast, no physics wait)
pytest tests/unit/test_mujoco_sim.py tests/unit/test_mujoco_go2.py tests/unit/test_mujoco_perception.py -v

# VGG E2E (real physics, ~30s for full suite)
pytest tests/harness/test_mujoco_vgg_e2e.py -v

# All MuJoCo tests (unit + harness)
pytest tests/ -k mujoco -v

# Single test class
pytest tests/harness/test_mujoco_vgg_e2e.py::TestVGGDecomposeWithRealRobot -v
```

## Fixture Scope Rules

MuJoCo is expensive to start (~1-2s per instance). Use `scope="module"` to
share one instance across a test module and reset posture between tests.

```python
@pytest.fixture(scope="module")
def go2_module():
    robot = MuJoCoGo2(gui=False, room=False)
    robot.connect()
    robot.stand(duration=2.0)
    yield robot
    robot.disconnect()

@pytest.fixture
def vgg_engine(go2_module):
    go2_module.stand(duration=1.0)   # reset posture before each test
    engine, agent = _make_engine(go2_module)
    yield engine, go2_module
```

Do NOT create a new MuJoCoGo2 per test in physics test files. One instance per
module is the correct pattern.

## Sim vs Real Interface

| Interface | Sim (MuJoCoGo2) | Real (Go2ROS2Proxy) |
|-----------|-----------------|---------------------|
| `set_velocity` | writes `_cmd_vel` -> the runtime's gait controller | ROS2 `/cmd_vel` topic |
| `walk` | `set_velocity` + `wait_sim` (SIM seconds) | same |
| `stand` | PD control to `_STAND_JOINTS` | Unitree SDK high-level |
| `sit` | PD control to `_SIT_JOINTS` | Unitree SDK high-level |
| `get_position` | `qpos[0:3]` | ROS2 odometry |
| `get_heading` | quaternion from `qpos[3:7]` | ROS2 odometry yaw |
| `get_lidar_scan` | one `mj_multiRay` over a step-boundary copy (~31 rings) | Livox MID360 pointcloud |
| `get_camera_frame` / `get_rgbd_frame` | `SimRuntime.render` (d435_rgb, one instant) | RealSense D435 |

## Physics ownership (SimRuntime)

`MuJoCoGo2.runtime` (`vector_os_nano/sim/runtime.py`) is the ONE owner of MjData:
one stepper runs the gait controller every 5 ms tick (1 kHz physics), paced to an
absolute wall schedule. Never write `base._mj.data` from another thread and never
`mj_forward` it: mutate with `base.runtime.call(lambda m, d: ...)` (applied at a step
boundary), read `base.runtime.snapshot()` (qpos/qvel of one instant) or
`base.get_rgbd_frame(w, h)` (RGB, depth, seg and camera pose of one `sim_time`, `K`
from the frame shape). Timed motions count SIM seconds (`wait_sim`), so they cover
the same ground at any real-time ratio. `VECTOR_SIM_LOCKSTEP=1` starts no stepper:
physics advances only inside `wait_sim`/`step`, so a scripted run is deterministic
(two runs end in identical ground truth). Generated scenes go to
`~/.cache/vector_os_nano/scenes/` (content-addressed), never the package dir.

## MuJoCo Go2 Backends

| Backend | Controller | Dependencies | When to use |
|---------|-----------|--------------|-------------|
| `sinusoidal` | Open-loop trot gait + PD torque | `mujoco`, `numpy` | Loud fallback when the MPC stack cannot start |
| `convex_mpc` | Centroidal MPC + leg controller | vendored `third_party/convex_mpc` + `casadi`, `pin` (`[go2]` extra) | Default (`auto`); the readiness line and the verdict say which one ran (`gait=`) |

Select backend:

```python
robot = MuJoCoGo2(gui=False, room=False)          # auto (MPC; sinusoidal + warning if it cannot start)
robot = MuJoCoGo2(gui=False, backend="sinusoidal") # force sinusoidal
robot = MuJoCoGo2(gui=False, backend="mpc")        # force MPC (raises if unavailable)
```

## VGG Integration Pattern

The VGG pipeline requires a fake agent with `_base`, `_spatial_memory`, and
`_skill_registry`. SceneGraph is optional (set to None for pure physics tests).

```python
from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills.go2 import get_go2_skills
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.intent_router import IntentRouter

class _FakeAgent:
    def __init__(self, base, skill_registry):
        self._base = base
        self._spatial_memory = None
        self._vlm = None
        self._skill_registry = skill_registry

registry = SkillRegistry()
for s in get_go2_skills():
    registry.register(s)

agent = _FakeAgent(go2, registry)
engine = VectorEngine(backend=MockLLMBackend(), intent_router=IntentRouter())
engine.init_vgg(agent=agent, skill_registry=registry)
```

For simple commands (`站起来`, `stop`, `walk`), `try_vgg` takes the fast path
and never calls the LLM backend.

## ARM Sim World (SO-101 / `vector-cli --sim`)

### Overview

The SO-101 arm runs in a separate sim world from the Go2. When an arm agent is
connected, its 10 skills are automatically wrapped as tools in the `robot` category
and become available to the agent loop.

### Arm Skill Registry

The 10 built-in arm skills (registered at startup by `cli._init_agent` when `--sim`
is passed):

| Skill | Category | Notes |
|-------|----------|-------|
| `home` | motor | move arm to home pose |
| `wave` | motor | wave gesture |
| `scan` | motor | sweep arm through scan arc |
| `detect` | perception | detect objects in scene |
| `describe` | perception | caption / visual query of current view |
| `pick` | motor | scan → detect → grasp (auto-steps via `__skill_auto_steps__`) |
| `place` | motor | place held object |
| `handover` | motor | hand object to user |
| `gripper_open` | motor | open gripper |
| `gripper_close` | motor | close gripper |

The `pick` skill declares `__skill_auto_steps__`, so `SkillWrapperTool` executes
a scan → detect → pick chain automatically rather than a bare grasp.

### Headless vs Window on macOS

MuJoCo's passive viewer requires the Cocoa main thread, which only `mjpython`
provides. Since Stage 0 the window is the DEFAULT and `--headless` is the opt-out:
`vector-cli --sim` (or NL "start the arm sim") detects when a window is wanted and
re-execs the whole REPL under mjpython automatically (`locate_mjpython()` derives it
from `sys.executable`'s venv bin — see `vcli/tools/sim_tool.py`).

| Mode | Command | Viewer | Use-case |
|------|---------|--------|----------|
| Headless | `vector-cli --sim --headless` | none | tests, remote, CI |
| Window | `vector-cli --sim` (or `scripts/vector-sim`) | MuJoCo viewer window | interactive dev |

`scripts/vector-sim` is a thin backward-compat alias: it resolves the repo venv
interpreter (`.venv/bin/python`, legacy `.venv-nano` fallback) and execs
`python -m vector_os_nano.vcli.cli --sim "$@"` — the CLI itself then handles the
mjpython re-exec.

### Test Suite

All arm sim tests live in `tests/vcli/test_level71_robot_control.py` (15 tests). Tests
split into two groups:

**Pure-logic (no mujoco dep):** `test_robot_context_arm_only_reports_connected`,
`test_start_simulation_in_sim_category`, `test_start_simulation_visible_in_dev_world`,
`test_sim_start_reachable_via_router_in_dev_world`, `test_model_flag_no_sentinel`,
`test_wave_scan_classified_motor` — run without any hardware.

**Sim-backed (skip if mujoco absent):** use the `arm` fixture (`MuJoCoArm(gui=False)`)
and cover gripper/perception wiring, VGG gate, perception word-boundary,
`SkillWrapperTool` auto-steps, `SimStartTool`/`SimStopTool` lifecycle, and
`DynamicSystemPrompt` refresh correctness.

```bash
# Run only the arm control tests
pytest tests/vcli/test_level71_robot_control.py -v

# Skip sim-backed tests when mujoco is absent
pytest tests/vcli/test_level71_robot_control.py -v -k "not (arm or sim)"
```

The `arm` fixture uses `scope` per-function (each test gets a fresh `MuJoCoArm`)
because arm state resets are cheap compared to Go2 physics. All tests run headless
(`gui=False`); no viewer is spawned in CI.

## Physics Assertions

Use conservative bounds that hold for both backends:

| State | Assert |
|-------|--------|
| Standing | `z in [0.20, 0.45]` |
| Sitting | `z < 0.35` |
| Not fallen | `z > 0.15` |
| Walking (not crashed) | `z > 0.15` |

Do not assert on exact position after walking — sinusoidal and MPC backends
produce different displacements. Assert that the robot did not fall.
