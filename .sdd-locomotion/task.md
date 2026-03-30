# Task Board: Go2 Locomotion Harness

**Created:** 2026-03-30
**Spec:** `.sdd-locomotion/spec.md`
**Plan:** `.sdd-locomotion/plan.md`
**Owner:** Alpha

---

## Task List

### T1: Copy MJCF files — DONE

- `vector_os_nano/hardware/sim/mjcf/go2/go2.xml` and `assets/` present
- No action required

---

### T2: Refactor mujoco_go2.py [alpha]

**Priority:** P0 — blocks everything
**File:** `vector_os_nano/hardware/sim/mujoco_go2.py`

Subtasks:
- [ ] T2a: Add `GaitParams` frozen dataclass
- [ ] T2b: Implement `_Go2Model` inner class (loads from local MJCF, no convex_mpc)
- [ ] T2c: Implement `_build_room_scene_xml_local()` (path resolution from local MJCF dir)
- [ ] T2d: Implement `_SinusoidalGait.compute(t, vx, vy, vyaw) -> np.ndarray`
- [ ] T2e: Refactor `connect()` — remove all convex_mpc imports, use `_Go2Model` + `_SinusoidalGait`
- [ ] T2f: Refactor `_physics_loop()` — replace MPC branch with sinusoidal gait PD
- [ ] T2g: Remove unused fields: `_pin`, `_traj`, `_mpc`, `_leg_ctrl`

Acceptance:
- `MuJoCoGo2(gui=False).connect()` succeeds on Python 3.12
- No `import convex_mpc` anywhere in the file
- All existing tests in `test_mujoco_go2_streaming.py` and `test_go2_sensors.py` pass

---

### T3: Write harness benchmarks (L0–L3) [alpha]

**Priority:** P1 — depends on T2
**Files:**
- `tests/harness/__init__.py`
- `tests/harness/conftest.py`
- `tests/harness/level0_physics.py`
- `tests/harness/level1_standing.py`
- `tests/harness/level2_walking.py`
- `tests/harness/level3_velocity.py`

Subtasks:
- [ ] T3a: `conftest.py` — `go2_no_gui`, `go2_room_no_gui`, `config` fixtures
- [ ] T3b: `level0_physics.py` — model load, torque displacement, no-NaN (3 tests)
- [ ] T3c: `level1_standing.py` — height check, angular velocity check (2 tests)
- [ ] T3d: `level2_walking.py` — displacement check, no-fall check (2 tests)
- [ ] T3e: `level3_velocity.py` — mean vx check, no-fall check (2 tests)

Acceptance:
- All L0 tests pass immediately after T2
- L1–L3 are runnable (may fail before tuning — that is expected)

---

### T4: Harness runner script [alpha]

**Priority:** P1 — depends on T3
**File:** `scripts/harness_run.py`

Subtasks:
- [ ] T4a: CLI argument parsing (`--levels`, `--tune`, `--config`, `--report`)
- [ ] T4b: Sequential level runner (subprocess pytest, capture pass/fail + metric)
- [ ] T4c: Parameter grid search loop (reads `harness.yaml`, patches, reruns)
- [ ] T4d: JSON report writer (`agents/devlog/harness_report_<date>.json`)
- [ ] T4e: Stdout summary table

Acceptance:
- `python scripts/harness_run.py --levels 0-1` runs without error
- Report JSON written after run

---

### T5: Run L0–L1, verify pass [alpha]

**Priority:** P2 — depends on T2, T3
**Action:** Execution task (no new code unless fix needed)

Steps:
1. `python scripts/harness_run.py --levels 0-1`
2. Confirm L0 pass: model loads, torque works, no NaN
3. Confirm L1 pass: robot stands at z > 0.25 m for 1 s

Pass criteria from spec Section 6.2.
If L0 fails: fix `_Go2Model` MJCF loading (T2b).
If L1 fails: tune `_KP`/`_KD` in `harness.yaml`.

---

### T6: Run L2, tune gait until pass [alpha]

**Priority:** P3 — depends on T5
**Action:** Execution + tuning task

Steps:
1. `python scripts/harness_run.py --levels 2 --tune`
2. Runner performs grid search over `frequency_hz`, `thigh_amplitude`, `calf_amplitude`, `pd.kp`
3. Best params written to `harness.yaml`
4. Confirm L2 pass: displacement > 0.30 m in 5 s, no fall

If L2 still fails after 10 iterations: switch to Backend C (position actuators), document result.

---

### T7: Run L3, tune velocity mapping until pass [alpha]

**Priority:** P4 — depends on T6
**Action:** Execution + tuning task

Steps:
1. `python scripts/harness_run.py --levels 3 --tune`
2. Adjust velocity-to-amplitude mapping (`K_yaw`, amplitude scale)
3. Confirm L3 pass: mean vx ≥ 0.15 m/s after 3 s settle, no fall

---

### T8: Nav2 L4 integration [alpha] — BLOCKED

**Priority:** P5 — blocked on T10 (Go2 ROS2 Bridge) from tasks.md
**File:** `tests/harness/level4_navigation.py`

Steps:
1. Write launch_testing L4 test (go2_nav2.launch.py + NavigateToPose goal)
2. `python scripts/harness_run.py --levels 4`
3. Confirm Nav2 reaches goal within 0.5 m in 60 s

Blocker: T10 (Go2 ROS2 Bridge nodes) must be complete first.

---

### T9: Full end-to-end verification [alpha]

**Priority:** P6 — depends on T7 (T8 optional)
**Action:** Regression + sign-off

Steps:
1. `python scripts/harness_run.py --levels 0-3`
2. Verify all existing tests still pass: `pytest tests/unit/ -x`
3. Update `agents/devlog/status.md`
4. Tag Lead for L4 sign-off if Nav2 integration is ready

---

## Dependency Graph

```
T1 (DONE)
  |
  v
T2 (refactor mujoco_go2.py)
  |
  v
T3 (harness benchmarks L0-L3) ──> T4 (runner script)
  |                                       |
  v                                       v
T5 (run L0-L1, verify)          T6 (run L2, tune)
                                         |
                                         v
                                T7 (run L3, tune)
                                         |
                     T8 (L4 Nav2) ───────v
                     [BLOCKED]           |
                                         v
                                T9 (full E2E verify)
```

---

## Current Status

| Task | Status   | Notes                              |
|------|----------|------------------------------------|
| T1   | DONE     | MJCF files at mjcf/go2/            |
| T2   | TODO     | First task to start                |
| T3   | TODO     | After T2                           |
| T4   | TODO     | After T3                           |
| T5   | TODO     | Execution                          |
| T6   | TODO     | Execution + tuning                 |
| T7   | TODO     | Execution + tuning                 |
| T8   | BLOCKED  | Waiting on T10 (tasks.md)          |
| T9   | TODO     | Final sign-off                     |
