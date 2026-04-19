# Top-Down Grasp Pipeline (Piper on Go2) — Spec & Assumptions

Phase C of the v2.1 manipulation stack. Given an object whose world pose is
known a priori, the Piper arm performs a fixed top-down grasp. Demo-quality
only — see the Assumptions section before using this for anything real.

## Status

- Piper 6-DoF arm: works in MuJoCo headless and GUI.
- Objects: three pickables pre-placed on a low table (`pick_table` in
  `go2_room.xml`).
- Tested end-to-end: 10/10 picks on each object across fresh Python
  subprocesses (`scripts/verify_pick_top_down.py --repeat 10`).
- NOT tested: real hardware. NOT wired: ROS2 bridge for arm (so nav stack
  + arm in one session doesn't work yet — see "Future work").

## Pipeline

```
User: "抓起蓝色瓶子" / "pick up the blue bottle"
  ↓
VGG / tool_use routes to PickTopDownSkill
  ↓
Resolve target object (world_model lookup by id / label / explicit xyz)
  ↓
IK top-down for pre-grasp pose  (object.z + 4 cm)
IK top-down for grasp pose       (object.z + 1 cm)
  ↓
Open gripper → pre-grasp move (4 s) → descent (2 s)
  ↓
Close gripper → wait 0.8 s for grip to settle
  ↓
Lift back to pre-grasp (2 s)      ← hold here; do NOT return home
  ↓
Report SkillResult(success, grasped_heuristic, object_id)
```

`ik_top_down` targets world coordinates. The IK rotates the gripper so its
finger axis (local +Z) aligns with world −Z (straight down), then solves
6-DoF Jacobian damped least-squares. A canned set of "top-down ready" seeds
is tried in order because solving from the URDF-zero config (arm fully
extended forward) rarely converges in reasonable iterations.

## Assumptions (all of them — this pipeline is demo-quality)

1. **Top-down only.** Gripper z-axis → world −Z. No angled or side
   approach. Objects whose best grasp is from the side (mugs by the
   handle, books on edge, etc.) will not be picked reliably.
2. **Known object pose.** The skill looks up `object_id` / `object_label`
   in `world_model`, which is populated at sim startup from MJCF bodies
   whose name begins with `pickable_`. No perception, no re-detection.
3. **Static scene.** Object positions are read once on connect. If a
   skill moves an object, `world_model` is not refreshed — a second pick
   on the same object will target the stale (original) position.
4. **Dog standing still.** The arm does not coordinate with base motion.
   Attempting a pick while the dog is walking / being pushed will smear
   the IK target and likely miss. `walk` / `navigate` skills should
   complete before `pick_top_down` is invoked.
5. **Reach envelope is below the Piper base.** Piper base sits at about
   `z ≈ 0.34 m` when Go2 stands. Targets whose pre-grasp Z would exceed
   the base height force the arm into fully-folded configurations
   outside the IK solver's attraction basin; pre-grasp height defaults
   to 4 cm above object centre so it remains reachable. Objects taller
   than ~8 cm above the table may still fail.
6. **Objects are graspable from above.** Radius ≤ ~3 cm (Piper jaw
   max opening 70 mm), mass ≤ ~0.2 kg (jaws are position-controlled with
   kp=40 and have no force sensor — heavier objects slip out).
7. **No collision checking.** IK does not avoid the dog's own trunk,
   the table, or other objects. Objects should be placed in the clear.
8. **After lift, the skill holds the object at the pre-grasp pose.**
   It does NOT return to the URDF-zero "home" because rotating the
   wrist 90° with a held object causes it to tip out of the jaws. A
   follow-up `place` / `drop` skill is expected to dispose of the
   object.
9. **Grasp confirmation is position-only.** `gripper.is_holding()`
   returns True when commanded-closed AND jaw position is held open by
   contact. No force sensor → no real force feedback. Heuristic can
   false-negative if the object is very thin.
10. **MuJoCo sim only.** Not run on real hardware. Real Piper + Go2
    integration requires a ROS2 topic bridge (see Future work).

## API

### Skill invocation

```python
from vector_os_nano.skills.pick_top_down import PickTopDownSkill
from vector_os_nano.core.skill import SkillContext

skill = PickTopDownSkill()
ctx = SkillContext(arm=piper, gripper=gripper, base=go2,
                   world_model=wm, config=cfg)

# By object id (canonical):
r = skill.execute({"object_id": "pickable_bottle_blue"}, ctx)

# By label (world_model does case-insensitive + substring match):
r = skill.execute({"object_label": "blue bottle"}, ctx)

# Or supply an explicit world-frame target (debug / programmatic):
r = skill.execute({"target_xyz": [10.4, 3.0, 0.25]}, ctx)
```

### Result

`SkillResult.result_data` on success:

| key | example |
|---|---|
| `diagnosis` | `"ok"` or `"possibly_missed"` (lift succeeded but `is_holding` is False) |
| `object_id` | `"pickable_bottle_blue"` |
| `grasp_world` | `[10.35, 2.90, 0.28]` — world xyz of the grasp point |
| `grasped_heuristic` | `True` / `False` — position-based grip confirmation |

### Failure modes (`diagnosis`)

| code | meaning |
|---|---|
| `no_arm` | `context.arm` is None |
| `no_gripper` | `context.gripper` is None |
| `no_world_model` | `context.world_model` is None |
| `arm_unsupported` | arm does not implement `ik_top_down` (SO-101 goes here) |
| `object_not_found` | resolved neither `object_id` nor `object_label` |
| `ik_unreachable` | IK failed (target outside reach or orientation-incompatible). Includes `phase: "pre_grasp" \| "grasp"` |
| `move_failed` | `arm.move_joints(...)` returned False (rare) |

### Tunable config (`config.skills.pick_top_down`)

| key | default | what it does |
|---|---|---|
| `pre_grasp_height` | 0.04 m | vertical offset of pre-grasp hover above object centre |
| `grasp_z_above` | 0.01 m | vertical offset of grasp target above object centre |

## Scene

Three pickable cylinders sit on a `pick_table` at (10.4, 3.0, 0) — 50 cm
in front of the Go2 spawn point (9.9, 3.0). Top of the table is at
z = 0.20 m. Defined in `vector_os_nano/hardware/sim/go2_room.xml`:

- `pickable_bottle_blue` — cylinder r=2.5 cm, h=16 cm, blue
- `pickable_bottle_green` — cylinder r=3.0 cm, h=10 cm, green
- `pickable_can_red` — cylinder r=3.3 cm, h=9 cm, red

Names MUST begin with `pickable_` for auto-registration into
`world_model` on sim connect.

## Hardware layer

- `MuJoCoPiper` (`vector_os_nano/hardware/sim/mujoco_piper.py`) —
  ArmProtocol. Ctrl writes go to live `MjData`; FK / IK run on an
  isolated `MjModel` + `MjData` loaded separately from the same MJCF.
  The isolation prevents races with the Go2 1 kHz physics thread.
  The IK base pose is sync'd from live at the start of each
  `ik_top_down` call, under a brief (<1 ms) physics pause.
- `MuJoCoPiperGripper` (`mujoco_piper_gripper.py`) —
  GripperProtocol. Single `piper_gripper` position actuator writes
  ctrl[18] of the live data. Fully closed = 0 m, fully open = 0.035 m
  per finger (70 mm total jaw separation).

## Integration

`sim_tool.SimStartTool` routes `with_arm=True` to the **local** (in-
process) start path (`_start_go2_local`) instead of the subprocess nav-
stack path. Local mode runs MuJoCoGo2 + MuJoCoPiper + MuJoCoPiperGripper
in the same Python process so the arm can share the MJCF's `MjData`.
Trade-off: the nav stack (TARE, FAR, door-chain) is NOT launched in
this mode, so explore / door-chain skills will not work simultaneously
with arm manipulation. Run separate sessions for now.

## Verification

### Unit tests

```bash
.venv-nano/bin/python -m pytest tests/hardware/sim/test_mujoco_piper.py
.venv-nano/bin/python -m pytest tests/skills/test_pick_top_down.py
```

Headless (no GUI), single MuJoCo instance, disconnects on teardown.

### End-to-end verification

```bash
# Single sweep (3 objects × 1 pick each), headless:
.venv-nano/bin/python scripts/verify_pick_top_down.py

# Stress: 3 objects × 10 picks each:
.venv-nano/bin/python scripts/verify_pick_top_down.py --repeat 10

# Only one object:
.venv-nano/bin/python scripts/verify_pick_top_down.py --object pickable_can_red
```

Each run launches a fresh subprocess so MjModel allocation isolation is
guaranteed. The script exits 0 only when every pick reports `lift ≥ 1 cm`
and `held=True`.

### Manual (REPL / GUI)

```
vector-cli
> go2sim  (pick with_arm=1 when prompted)
> 抓起蓝色瓶子
```

## Known issues

- IK `ik_top_down` can fail at the reach edge (>55 cm from Piper base
  or object above Piper base height). The skill returns `ik_unreachable`
  with `phase: "pre_grasp"`.
- MuJoCo sometimes segfaults if multiple sims are instantiated in the
  same Python process. Always use subprocess for repeated tests
  (`verify_pick_top_down.py` does this). Tests in this repo use
  module-scope fixtures to keep a single sim per test file.

## Future work (not in this phase)

- ROS2 bridge for arm topics so nav-stack + arm can run together.
- Perception-driven grasp: replace `world_model` lookup with live
  depth+detection pipeline (RealSense / SAM3D).
- Orientation-aware grasp: angled or side approach, useful for mugs
  and objects with a preferred grasp axis.
- Place skill: reverse flow (hold → pre-place → place → open → retreat).
- Mobile manipulation: dog walks to object if not within reach, then
  picks (already-built `navigate` skill + arm coordination).
- Real hardware: Piper ROS2 driver + force-sensor feedback for grip
  closure instead of the position-only `is_holding` heuristic.
