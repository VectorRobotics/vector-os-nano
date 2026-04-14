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
| `set_velocity` | writes `_cmd_vel` -> physics thread | ROS2 `/cmd_vel` topic |
| `walk` | `set_velocity` + sleep | same |
| `stand` | PD control to `_STAND_JOINTS` | Unitree SDK high-level |
| `sit` | PD control to `_SIT_JOINTS` | Unitree SDK high-level |
| `get_position` | `qpos[0:3]` | ROS2 odometry |
| `get_heading` | quaternion from `qpos[3:7]` | ROS2 odometry yaw |
| `get_lidar_scan` | `mj_ray` 360-degree scan (30 rings) | Livox MID360 pointcloud |
| `get_camera_frame` | `mj.Renderer` (d435_rgb) | RealSense D435 |

## MuJoCo Go2 Backends

| Backend | Controller | Dependencies | When to use |
|---------|-----------|--------------|-------------|
| `sinusoidal` | Open-loop trot gait + PD torque | `mujoco`, `numpy` | Default, always available |
| `convex_mpc` | Centroidal MPC + leg controller | `convex_mpc`, `casadi`, `pinocchio` | Optional, physics-accurate |

Select backend:

```python
robot = MuJoCoGo2(gui=False, room=False)          # auto (sinusoidal if no MPC)
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
