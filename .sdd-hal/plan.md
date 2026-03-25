# Hardware Abstraction Layer — Technical Plan

## 1. Architecture

```
                    Agent
                      │
                      ▼
        ┌─────── SkillContext ───────┐
        │                            │
        │ arms: {"so101": arm}       │
        │ bases: {"go2": go2}        │
        │ grippers: {"g1": gripper}  │
        │ services: {"nav": client}  │
        │ world_model: wm            │
        └──┬──────────┬──────────┬───┘
           │          │          │
     ArmProtocol  BaseProtocol  NavStackClient
     (existing)   (NEW)         (NEW, optional)
           │          │          │
     ┌─────┘    ┌─────┘    ┌────┘
     │          │          │
  SO101Arm   MuJoCoGo2  vector_navigation_stack
  MuJoCoArm  (refactored) (ROS2 topics)
             UnitreeReal   /way_point
             (future)      /state_estimation
                           /goal_reached
```

## 2. Module Plan

### M1: core/types.py — Odometry + LaserScan
Add two frozen dataclasses. No deps on existing code.

### M2: hardware/base.py — BaseProtocol
New file, mirrors arm.py pattern. Runtime-checkable Protocol.

### M3: core/skill.py — SkillContext redesign
Replace flat fields with dict registries. Add backward-compat properties.
Critical: ALL existing skills use context.arm / context.base — properties preserve this.

### M4: core/agent.py — Agent constructor update
Change __init__ to accept arms/bases/grippers dicts OR single instances (backward compat).
Update _build_context() to produce new SkillContext.

### M5: hardware/sim/mujoco_go2.py — Background physics thread
- connect() starts _physics_thread (daemon)
- _physics_thread: while running, read _cmd_vel, MPC solve, mj_step, update odom/lidar
- set_velocity(): atomic write to _cmd_vel
- walk(): set_velocity + sleep + set_velocity(0,0,0)
- get_odometry(): read latest from physics thread
- get_lidar_scan(): mj_ray 360 rays, cached at 10Hz

### M6: core/nav_client.py — NavStackClient
Wraps ROS2 topics. Optional — only created if rclpy available + nav stack running.
- navigate_to(x, y): publishes /way_point, blocks until /goal_reached
- cancel(): publishes /cancel_goal
- get_state_estimation(): returns latest /state_estimation as Odometry
- is_available: True if /state_estimation topic has publisher

### M7: skills/navigate.py — Unified NavigateSkill
- Uses context.services.get("nav") for NavStackClient if available
- Falls back to dead-reckoning (existing go2/navigate.py logic)
- Hardware-agnostic: uses context.base (any BaseProtocol)
- Replaces skills/go2/navigate.py

## 3. Execution Waves

| Wave | Tasks | Files | Parallel |
|------|-------|-------|----------|
| 1 | M1, M2 | types.py, base.py | Yes (independent) |
| 2 | M3, M5 | skill.py, mujoco_go2.py | Yes (independent) |
| 3 | M4, M6 | agent.py, nav_client.py | Yes (M4 needs M3, M6 independent) |
| 4 | M7 | navigate.py | Needs M3, M4, M6 |
| Gate | All tests | | |

## 4. Risk Mitigations

| Risk | Mitigation |
|------|------------|
| MuJoCo not thread-safe | Physics thread owns ALL mj_* calls. Readers use Lock + snapshot. |
| SkillContext breaks existing skills | Properties provide exact same API. Run full test suite. |
| walk() timing changes | Physics thread steps at 1kHz regardless of sleep precision. |
| NavStackClient requires ROS2 | Optional. Lazy import rclpy. Dead-reckoning fallback always works. |
