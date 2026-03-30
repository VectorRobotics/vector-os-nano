# Plan: Go2 MuJoCo Locomotion Harness

**Version:** 0.1.0
**Date:** 2026-03-30
**Spec:** `.sdd-locomotion/spec.md`

---

## 1. Architecture Overview

```
mujoco_go2.py
  └── _Go2Model               # replaces MuJoCo_GO2_Model from convex_mpc
        ├── model / data       # loaded from local mjcf/go2/go2.xml
        ├── base_bid           # body id of base_link
        └── set_joint_torque() # writes data.ctrl[0:12]

  └── _SinusoidalGait         # replaces convex_mpc locomotion stack
        ├── compute(t, vx, vy, vyaw) -> np.ndarray  # 12 joint targets
        └── params from harness.yaml / constructor

tests/harness/
  ├── conftest.py              # shared fixtures (Go2 instance, room scene)
  ├── level0_physics.py        # L0 test
  ├── level1_standing.py       # L1 test
  ├── level2_walking.py        # L2 test
  ├── level3_velocity.py       # L3 test
  └── level4_navigation.py     # L4 test (ROS2/Nav2, separate run)

scripts/
  └── harness_run.py           # CLI: run L0–L3 (or L0–L4), auto-tune, report

.sdd-locomotion/
  └── harness.yaml             # all tunable parameters
```

---

## 2. Phase 1: Decouple from convex_mpc

**Goal:** `connect()` works on Python 3.12 with no `convex_mpc` installed.

### 2.1 _Go2Model Wrapper

New inner class in `mujoco_go2.py` (replaces `MuJoCo_GO2_Model`):

```python
class _Go2Model:
    """Minimal MuJoCo model wrapper — no convex_mpc required."""
    MJCF_PATH = Path(__file__).parent / "mjcf" / "go2" / "go2.xml"

    def __init__(self, scene_xml: Path | None = None) -> None:
        mj = _get_mujoco()
        xml_path = scene_xml or self.MJCF_PATH
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.base_bid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "base_link"
        )

    def set_joint_torque(self, tau: np.ndarray) -> None:
        self.data.ctrl[0:12] = tau
```

Room scene: a new `_build_room_scene_xml_local()` function reads the template from
`sim/go2_room.xml` and substitutes the local MJCF path instead of using
`convex_mpc.__file__` path resolution.

### 2.2 _SinusoidalGait

New class in `mujoco_go2.py` (or extracted to `sim/sinusoidal_gait.py` if >100 lines):

```python
@dataclass
class GaitParams:
    frequency_hz: float = 2.0
    thigh_amplitude: float = 0.30
    calf_amplitude: float = 0.30
    calf_phase_offset: float = math.pi / 2.0
    hip_amplitude: float = 0.15
    duty_cycle: float = 0.6

class _SinusoidalGait:
    LEG_PHASES = {"FL": 0.0, "FR": math.pi, "RL": math.pi, "RR": 0.0}
    STAND = {"thigh": 0.9, "calf": -1.8}
    K_YAW = 0.05

    def __init__(self, params: GaitParams) -> None: ...

    def compute(
        self, t: float, vx: float, vy: float, vyaw: float
    ) -> np.ndarray:
        """Return 12-element joint target array."""
        ...
```

`compute()` implements the equations from spec Section 5.2–5.3.

### 2.3 connect() Refactor

Remove all `convex_mpc` imports from `connect()`. New `connect()`:

1. Instantiate `_Go2Model` (with room scene if `self._room`)
2. Set `model.opt.timestep = 0.001`
3. Call `mj.mj_forward()`
4. Instantiate `_SinusoidalGait(GaitParams())` — stored as `self._gait`
5. Open viewer if `gui=True`
6. Start physics thread

Remove fields: `_pin`, `_traj`, `_mpc`, `_leg_ctrl`.
Add field: `_gait: _SinusoidalGait`.

### 2.4 _physics_loop Refactor

Replace MPC branch with sinusoidal gait:

```python
if is_moving:
    if sim_step % _CTRL_DECIM == 0:
        q_des = self._gait.compute(time_now, vx, vy, vyaw)
        q_cur = data.qpos[7:19]
        dq_cur = data.qvel[6:18]
        tau = _KP * (q_des - q_cur) - _KD * dq_cur
        tau = np.clip(tau, -_TAU_LIMITS, _TAU_LIMITS)
        tau_hold = tau
```

Idle branch (PD hold standing) unchanged.

---

## 3. Phase 2: Harness Infrastructure

### 3.1 conftest.py

Pytest fixtures:

```python
@pytest.fixture
def go2_no_gui() -> Generator[MuJoCoGo2, None, None]:
    robot = MuJoCoGo2(gui=False, room=False)
    robot.connect()
    yield robot
    robot.disconnect()

@pytest.fixture
def go2_room_no_gui() -> Generator[MuJoCoGo2, None, None]:
    robot = MuJoCoGo2(gui=False, room=True)
    robot.connect()
    yield robot
    robot.disconnect()
```

Load `harness.yaml` via a `config` fixture (reads from `.sdd-locomotion/harness.yaml`).

### 3.2 level0_physics.py

```python
def test_l0_model_loads(go2_no_gui): ...
def test_l0_torque_produces_displacement(go2_no_gui, config): ...
def test_l0_no_nan_in_state(go2_no_gui): ...
```

### 3.3 level1_standing.py

```python
def test_l1_height_after_stand(go2_no_gui, config): ...
def test_l1_angular_velocity_stable(go2_no_gui, config): ...
```

### 3.4 level2_walking.py

```python
def test_l2_forward_displacement(go2_no_gui, config): ...
def test_l2_no_fall_during_walk(go2_no_gui, config): ...
```

### 3.5 level3_velocity.py

```python
def test_l3_mean_velocity_above_threshold(go2_no_gui, config): ...
def test_l3_no_fall_during_velocity_command(go2_no_gui, config): ...
```

### 3.6 level4_navigation.py

ROS2 launch_testing test. Requires Nav2 stack running.
Marked `@pytest.mark.nav2` — skipped unless `--run-nav2` flag passed to pytest.

### 3.7 harness_run.py

CLI runner (`python scripts/harness_run.py [--levels 0-3] [--tune] [--config path]`):

1. Loads `harness.yaml`
2. Runs L0–L3 (or L0–L4) via `subprocess` calling pytest on each level file
3. If level fails and `--tune` is set, runs parameter grid search
4. Writes JSON result report to `agents/devlog/harness_report_<date>.json`
5. Prints summary table to stdout

---

## 4. Phase 3: Parameter Tuning Loop

**Only triggered by `harness_run.py --tune` after L2 or L3 failure.**

Algorithm:
```
for param_set in itertools.product(freq_range, thigh_range, calf_range, kp_range):
    patch harness.yaml with param_set
    run pytest tests/harness/level{N}.py
    record (param_set, metric, pass/fail)
    if pass: break
    if attempts >= max_iterations_per_level: try next backend
```

Best-found parameters are written back to `harness.yaml` on success.

---

## 5. Phase 4: Nav2 Integration (Level 4)

**Blocked on:** T10 (Go2 ROS2 Bridge nodes) from tasks.md Wave 4.

When T10 is complete:
1. `go2_nav2.launch.py` launches bridge nodes + Nav2 stack
2. L4 test sends NavigateToPose goal via action client
3. Verifies arrival within tolerance

Nav2 parameter files: `config/nav2_go2_params.yaml` (tuned for Go2 footprint + speed).

---

## 6. File Manifest

### Modified

| File | Change |
|------|--------|
| `vector_os_nano/hardware/sim/mujoco_go2.py` | Remove convex_mpc; add `_Go2Model`, `_SinusoidalGait`, `GaitParams`; refactor `connect()` and `_physics_loop()` |

### New

| File | Purpose |
|------|---------|
| `vector_os_nano/hardware/sim/mjcf/go2/` | Local MJCF files (already copied — T1 DONE) |
| `tests/harness/conftest.py` | Shared pytest fixtures |
| `tests/harness/level0_physics.py` | L0 tests |
| `tests/harness/level1_standing.py` | L1 tests |
| `tests/harness/level2_walking.py` | L2 tests |
| `tests/harness/level3_velocity.py` | L3 tests |
| `tests/harness/level4_navigation.py` | L4 tests (launch_testing, Nav2) |
| `scripts/harness_run.py` | One-command harness runner with auto-tune |
| `.sdd-locomotion/harness.yaml` | All tunable parameters |

### Unchanged (verify no regression)

| File | Why checked |
|------|-------------|
| `tests/unit/test_mujoco_go2_streaming.py` | Physics thread tests |
| `tests/unit/test_go2_sensors.py` | Odometry + lidar tests |
| `tests/unit/test_run_go2.py` | Skill + agent integration |

---

## 7. Dependencies

Runtime: `mujoco`, `numpy` (already installed).
Test: `pytest`, `pyyaml` (for loading `harness.yaml`).
No new pip dependencies required.

---

## 8. Risk Register

| Risk | Mitigation |
|------|-----------|
| Sinusoidal gait insufficient for Nav2 (not reactive) | L3 sets 50% tolerance — generous. If tracking fails, Backend C (position actuators) as fallback |
| MJCF actuator type mismatch (motor vs torque) | L0 verifies torque produces displacement; fix MJCF if needed |
| Physics thread timing drift at 1 kHz | Accept ±5% timing jitter — gait is time-based not step-count-based |
| Room scene XML path resolution | `_build_room_scene_xml_local()` uses absolute paths, verified in L1 |
