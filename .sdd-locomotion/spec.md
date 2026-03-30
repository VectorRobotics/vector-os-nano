# Spec: Go2 MuJoCo Locomotion Harness

**Version:** 0.1.0
**Date:** 2026-03-30
**Status:** Draft

---

## 1. Problem Statement

`mujoco_go2.py` has a hard dependency on `convex_mpc` (Python bindings for CasADi + Pinocchio)
which requires Python 3.10 and is incompatible with the current Python 3.12 environment.

Current state:
- `connect()` — fails at `from convex_mpc.mujoco_model import MuJoCo_GO2_Model`
- PD idle (stand/sit/lie_down) — blocked because `connect()` never completes
- MPC locomotion (`walk()`, `set_velocity()`) — completely broken
- Local MJCF files already copied to `vector_os_nano/hardware/sim/mjcf/go2/`

Goal: make Go2 walk in MuJoCo with zero dependency on `convex_mpc`, then verify with a
bottom-up automated harness.

---

## 2. Solution Overview

Replace the `convex_mpc` dependency with:

1. A local `_Go2Model` wrapper (loads MJCF directly, no `convex_mpc` required)
2. A pure sinusoidal trotting gait (numpy only, no CasADi/Pinocchio)
3. An automated bottom-up verification harness (5 levels, L0–L4)

No existing public API changes (`connect`, `disconnect`, `stand`, `walk`, `set_velocity`,
`get_position`, `get_heading`, `get_odometry`, `get_lidar_scan` all preserved).

---

## 3. Joint Layout

MuJoCo `ctrl` and `qpos[7:19]` ordering — 12 joints total:

| Index | Leg | Joint  | Torque Limit |
|-------|-----|--------|-------------|
| 0     | FL  | hip    | 21.3 Nm     |
| 1     | FL  | thigh  | 21.3 Nm     |
| 2     | FL  | calf   | 40.9 Nm     |
| 3     | FR  | hip    | 21.3 Nm     |
| 4     | FR  | thigh  | 21.3 Nm     |
| 5     | FR  | calf   | 40.9 Nm     |
| 6     | RL  | hip    | 21.3 Nm     |
| 7     | RL  | thigh  | 21.3 Nm     |
| 8     | RL  | calf   | 40.9 Nm     |
| 9     | RR  | hip    | 21.3 Nm     |
| 10    | RR  | thigh  | 21.3 Nm     |
| 11    | RR  | calf   | 40.9 Nm     |

Standing pose (per-joint targets): `[0.0, 0.9, -1.8] * 4`

---

## 4. Control Backends

Three backends are defined. The harness selects the active backend via `harness.yaml`.

### Backend A — Sinusoidal Trotting Gait (primary)

Pure Python/NumPy. No external dependencies beyond `mujoco`.

### Backend B — convex_mpc (future)

Restored when `convex_mpc` is ported to Python 3.12. Currently disabled/skipped.

### Backend C — MJCF Position Actuators

Uses MuJoCo `position` actuators in the MJCF directly (kp set in XML).
Useful as a physics-validity baseline; not suitable for dynamic locomotion.

---

## 5. Backend A: Sinusoidal Trotting Gait

### 5.1 Trot Diagonal Pairs

Trot gait: FL+RR swing together, FR+RL swing together (diagonal pairs).

```
Pair A (phase=0):         FL, RR
Pair B (phase=π):         FR, RL
```

### 5.2 Joint Angle Equations

For each leg, given global time `t` (seconds) and leg phase offset `φ_leg`:

```
θ(t) = t * 2π * f_gait   # global gait phase (radians)
φ_swing = θ + φ_leg       # leg-specific phase

hip(t)   =  A_hip  * sin(φ_swing)
thigh(t) =  θ_stand_thigh + A_thigh * sin(φ_swing)
calf(t)  =  θ_stand_calf  + A_calf  * sin(φ_swing + φ_calf_offset)
```

Where:
- `f_gait` = gait frequency (Hz), default 2.0
- `A_hip` = hip swing amplitude (rad), default 0.15
- `A_thigh` = thigh lift amplitude (rad), default 0.30
- `A_calf` = calf amplitude (rad), default 0.30
- `φ_calf_offset` = π/2 (calf leads thigh by 90°)
- `θ_stand_thigh` = 0.9 rad (standing thigh offset)
- `θ_stand_calf` = −1.8 rad (standing calf offset)

Leg phase offsets (trot):
```
φ_FL = 0        φ_FR = π
φ_RL = π        φ_RR = 0
```

### 5.3 Velocity Steering

Forward velocity `vx` modulates hip amplitude:
```
A_hip_effective = A_hip * clamp(|vx| / VX_MAX, 0, 1)
```

Yaw rate `vyaw` adds differential hip offset:
```
hip_left  += K_yaw * vyaw    (FL, RL)
hip_right -= K_yaw * vyaw    (FR, RR)
```

Where `K_yaw` = 0.05 (rad·s/rad).

### 5.4 PD Torque Law

```
τ = Kp * (q_des - q_cur) - Kd * dq_cur
τ = clip(τ, -τ_limit, +τ_limit)
```

Gains: `Kp = 120.0`, `Kd = 3.5` (same as existing idle PD).

### 5.5 Control Rate

Gait targets computed at `CTRL_HZ = 200 Hz` (every 5 physics steps at 1 kHz).

---

## 6. Verification Harness

### 6.1 Architecture

Bottom-up automated harness. Each level is independent; failures at level N block level N+1.

```
L0: Physics Validity
L1: Standing Stability
L2: Open-Loop Walking
L3: Velocity Tracking
L4: Nav2 Navigation
```

### 6.2 Level Definitions and Pass/Fail Criteria

#### Level 0 — Physics Validity

Verifies MuJoCo model loads and actuators produce torque.

Test procedure:
1. Load MJCF from local `mjcf/go2/go2.xml` (no room, flat ground)
2. Apply a known torque command (10 Nm) to all joints for 100 steps
3. Measure joint displacement

Pass criteria:
- Model loads without exception
- Joint displacement > 0.01 rad for at least 1 joint after 100 steps
- No NaN in `qpos`, `qvel`

Fail: model load error, zero displacement, NaN in state.

#### Level 1 — Standing Stability

Verifies PD idle controller keeps robot upright.

Test procedure:
1. `connect()` (no GUI)
2. `stand(duration=2.0)`
3. Hold for `hold_duration_s` = 1.0 s
4. Sample `z`, `angular_vel` at 10 Hz

Pass criteria:
- `z > 0.25 m` throughout hold period
- `|angular_velocity| < 0.5 rad/s` (no tipping)
- Completes within `timeout_s` = 10.0 s

Fail: robot falls (z ≤ 0.15 m), tips (angular_vel > 0.5), or times out.

#### Level 2 — Open-Loop Walking

Verifies sinusoidal gait produces forward displacement.

Test procedure:
1. `connect()` → `stand()`
2. `set_velocity(vx=0.3, vy=0, vyaw=0)`
3. Run for `walk_duration_s` = 5.0 s
4. Measure horizontal displacement from start

Pass criteria:
- Horizontal displacement > 0.3 m after 5 s
- `z > 0.15 m` throughout (no fall)
- No NaN in position

Fail: displacement < 0.3 m, robot falls, or NaN in state.

#### Level 3 — Velocity Tracking

Verifies commanded velocity is approximately achieved.

Test procedure:
1. `connect()` → `stand()` → `set_velocity(vx=0.3)`
2. Wait `settle_time_s` = 3.0 s
3. Measure actual `vx` over `duration_s` = 5.0 s (mean)

Pass criteria:
- Mean actual `vx` ≥ `velocity_tolerance * cmd_vx` = 0.5 × 0.3 = 0.15 m/s
- `z > 0.15 m` throughout

Fail: mean vx < 0.15 m/s, robot falls.

#### Level 4 — Nav2 Navigation

Verifies end-to-end Nav2 path following with Go2.

Test procedure:
1. Launch ROS2 bridge nodes + Nav2 stack (go2_nav2.launch.py)
2. Send NavigateToPose goal: `(12.0, 3.0)` from start `(10.0, 3.0)` in room scene
3. Wait up to `timeout_s` = 60.0 s

Pass criteria:
- Robot reaches within `arrival_tolerance_m` = 0.5 m of goal
- Nav2 action returns SUCCESS
- Robot does not fall during navigation

Fail: timeout, Nav2 ABORTED/CANCELED, robot falls.

### 6.3 Summary Table

| Level | Name              | Key Threshold              | Timeout |
|-------|-------------------|---------------------------|---------|
| L0    | Physics Validity  | displacement > 0.01 rad   | N/A     |
| L1    | Standing          | z > 0.25 m for 1 s        | 10 s    |
| L2    | Walking           | displacement > 0.30 m     | 5 s     |
| L3    | Velocity Tracking | mean_vx ≥ 0.15 m/s        | 8 s     |
| L4    | Nav2 Navigation   | dist_to_goal ≤ 0.5 m      | 60 s    |

---

## 7. Automated Optimization Loop

When a level fails, the runner performs a grid search over gait parameters before
escalating to manual intervention.

Optimization scope: L2 and L3 only (L0/L1 are pass/fail on correctness; L4 is integration).

Tunable parameters (from `harness.yaml`):
- `gait.frequency_hz`: [1.5, 2.0, 2.5, 3.0]
- `gait.thigh_amplitude`: [0.2, 0.3, 0.4]
- `gait.calf_amplitude`: [0.2, 0.3, 0.4]
- `pd.kp`: [80, 100, 120, 150]

After `max_iterations_per_level` = 10 parameter sets without pass, the runner:
1. Logs all attempted configurations and outcomes
2. Reports the best-achieved metric
3. If `backend_switch_after` = 5 consecutive failures, switches to Backend C as fallback

---

## 8. Out of Scope

- Real hardware deployment (Go2 SDK, CycloneDDS to physical robot)
- Stair climbing, dynamic jumping, external perturbations
- Training neural network policies
