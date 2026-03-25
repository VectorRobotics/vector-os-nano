# ADR-003: Hardware Abstraction Layer Redesign

- Status: proposed
- Date: 2026-03-25
- Context: CEO directives require hardware-agnostic framework. SO-101 and Go2 must be "adapter A" and "adapter B" plugging into the same Agent/Skill/LLM pipeline. Current system has ArmProtocol and GripperProtocol but no BaseProtocol, and SkillContext is accumulating ad-hoc fields.
- Decision: See below.
- Consequences: Breaking change to SkillContext. All existing skills need minor migration. New BaseProtocol and PerceptionProtocol formalized.
- Alternatives: Keep ad-hoc `base: Any` field and duck-typing. Rejected because it prevents capability queries and makes the system brittle as hardware variants multiply.

---

## A. Hardware Protocol Hierarchy

Three formal protocols plus one new one:

```
HardwareProtocol (base trait: connect/disconnect/stop/name)
  +-- ArmProtocol       (existing, unchanged)
  +-- GripperProtocol   (existing, unchanged)
  +-- BaseProtocol      (NEW)
  +-- PerceptionProtocol (NEW — formalize what skills assume)
```

### A.1 BaseProtocol

```python
@runtime_checkable
class BaseProtocol(Protocol):
    """Abstract interface for any mobile base (quadruped, wheeled, tracked)."""

    @property
    def name(self) -> str:
        """Identifier: 'go2', 'turtlebot', 'sim_diff_drive', etc."""
        ...

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def stop(self) -> None: ...

    # --- Blocking locomotion (for direct skill use) ---
    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        """Move at body velocity for duration seconds. Blocking.
        Returns True if completed without falling/error."""
        ...

    # --- Streaming velocity (for Nav2 cmd_vel) ---
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set target velocity. Non-blocking. Physics loop applies it.
        Calling set_velocity(0,0,0) stops motion.
        This is what the Nav2 cmd_vel subscriber calls."""
        ...

    # --- State queries ---
    def get_position(self) -> list[float]:
        """[x, y, z] in world frame (meters)."""
        ...

    def get_heading(self) -> float:
        """Yaw in radians, world frame."""
        ...

    def get_velocity(self) -> list[float]:
        """[vx, vy, vz] in world frame (m/s)."""
        ...

    def get_odometry(self) -> "Odometry":
        """Full odometry: pose + twist. For Nav2 /odom topic."""
        ...

    # --- Sensor data ---
    def get_lidar_scan(self) -> "LaserScan | None":
        """2D lidar scan. None if no lidar. For Nav2 /scan topic."""
        ...

    # --- Capability flags ---
    @property
    def supports_holonomic(self) -> bool:
        """True if base can strafe (omnidirectional). Go2=True, diff_drive=False."""
        ...

    @property
    def supports_lidar(self) -> bool:
        """True if get_lidar_scan() returns data."""
        ...
```

### A.2 Why Both walk() and set_velocity()

- `walk(vx, vy, vyaw, duration)` -- blocking, used by WalkSkill, TurnSkill, NavigateSkill (dead-reckoning waypoint following). Simple. Existing code uses this.
- `set_velocity(vx, vy, vyaw)` -- non-blocking, used by Nav2's cmd_vel bridge. The physics loop picks up the latest velocity each tick. Essential for closed-loop navigation.

Implementation in MuJoCoGo2:
- `walk()` calls `set_velocity()`, then runs the physics loop for `duration` seconds, then calls `set_velocity(0,0,0)`.
- `set_velocity()` stores the command in `self._cmd_vel`. The background physics thread reads `self._cmd_vel` each tick.

### A.3 Odometry and LaserScan Types

Lightweight frozen dataclasses in `core/types.py`. NOT ROS2 messages -- pure Python. The ROS2 bridge converts these to `nav_msgs/Odometry` and `sensor_msgs/LaserScan`.

```python
@dataclass(frozen=True)
class Odometry:
    """Odometry reading: pose + twist in a single snapshot."""
    timestamp: float          # seconds since epoch
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    vx: float = 0.0          # linear velocity, body frame
    vy: float = 0.0
    vz: float = 0.0
    vyaw: float = 0.0        # angular velocity

@dataclass(frozen=True)
class LaserScan:
    """2D laser scan (simulated via mj_ray or real lidar)."""
    timestamp: float
    angle_min: float          # start angle (radians)
    angle_max: float          # end angle (radians)
    angle_increment: float    # angular resolution
    range_min: float          # minimum range (meters)
    range_max: float          # maximum range (meters)
    ranges: tuple[float, ...]  # range readings (meters), inf = no return
```

---

## B. MuJoCoGo2 Architecture Change

### B.1 Current Problem

`MuJoCoGo2.walk()` is synchronous: it runs the MPC loop for N sim steps, blocking the caller. This works for direct skill use but is incompatible with Nav2, which sends continuous `cmd_vel` messages and expects the physics to run independently.

### B.2 New Architecture: Background Physics Thread

```
                        MuJoCoGo2
                    +------------------+
  set_velocity() -->| _cmd_vel (atomic)|
                    |                  |
  walk(dur)  ------>| calls set_vel,   |
                    | waits dur,       |
                    | calls set_vel(0) |
                    |                  |
  _physics_thread:  | while running:   |
                    |   read _cmd_vel  |
                    |   MPC solve      |
                    |   mj_step        |
                    |   update odom    |
                    |   update lidar   |
                    |   sync viewer    |
                    +------------------+
                           |
                   get_odometry() / get_lidar_scan()
```

Key design points:

1. **Physics thread** starts on `connect()`, stops on `disconnect()`. Runs at 1000 Hz (SIM_DT=0.001s). This thread owns `mj_step`.

2. **Thread safety**: `_cmd_vel` is a simple 3-float tuple protected by a threading.Lock (or use atomics). Odometry and lidar scan are written by the physics thread and read by callers -- use a Lock or double-buffering.

3. **walk() becomes a convenience wrapper**:
   ```python
   def walk(self, vx, vy, vyaw, duration):
       self.set_velocity(vx, vy, vyaw)
       time.sleep(duration)
       self.set_velocity(0, 0, 0)
       return self._check_upright()
   ```
   This is slightly less precise than the current synchronous loop (sleep vs sim-step counting), but much simpler and compatible with the streaming model. For simulation accuracy, the physics thread is stepping at 1000 Hz regardless.

4. **MPC controller** runs inside the physics thread at CTRL_HZ (200 Hz). The MPC solver runs at MPC_HZ (~48 Hz). Same algorithm as current `walk()`, just refactored into the thread loop.

5. **Lidar simulation**: `get_lidar_scan()` uses `mj_ray()` from the physics thread (or on-demand with a cached result). 360 rays at 1-degree resolution, 12m max range. Updated at 10 Hz (every 100 physics steps).

6. **Viewer sync** happens inside the physics thread (every 8 steps, same as now).

### B.3 Simulated Lidar via mj_ray

```python
def _update_lidar(self) -> None:
    """Cast 360 rays from base_link height, update _last_scan."""
    mj = _get_mujoco()
    pos = self._mj.data.qpos[0:3].copy()
    pos[2] = 0.15  # lidar height
    heading = self.get_heading()

    ranges = []
    n_rays = 360
    for i in range(n_rays):
        angle = heading + math.radians(i - 180)
        direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        geom_id = np.zeros(1, dtype=np.int32)
        dist = mj.mj_ray(
            self._mj.model, self._mj.data,
            pos, direction, None, 1,  # exclude base geom group
            -1, geom_id
        )
        ranges.append(dist if dist > 0 else float('inf'))

    self._last_scan = LaserScan(
        timestamp=float(self._mj.data.time),
        angle_min=-math.pi,
        angle_max=math.pi,
        angle_increment=math.radians(1.0),
        range_min=0.1,
        range_max=12.0,
        ranges=tuple(ranges),
    )
```

---

## C. Navigation Architecture

### C.1 The Two Navigation Modes

| Mode | When Used | Path Planning | Velocity Control |
|------|-----------|--------------|-----------------|
| **Dead-reckoning** | No Nav2 / No map / Quick demo | Waypoint graph (hardcoded room map) | walk(vx, vy, vyaw, duration) |
| **Nav2** | Full autonomy / Map available | Nav2 planner (global + local) | cmd_vel via set_velocity() |

Both modes are accessed through the same NavigateSkill interface. The skill detects which mode is available at runtime.

### C.2 NavigateSkill (Hardware-Agnostic)

```python
@skill(aliases=["navigate", "go to", "goto", "去", "到", "走到"])
class NavigateSkill:
    name = "navigate"
    parameters = {
        "target": {
            "type": "string",
            "required": True,
            "description": "Room name, coordinate, or semantic location",
        },
    }

    def execute(self, params, context):
        if context.base is None:
            return SkillResult(success=False, diagnosis_code="no_base")

        target = params["target"]

        # Try Nav2 first (if ROS2 navigation stack is running)
        nav2 = context.services.get("nav2")
        if nav2 is not None:
            goal_pose = self._resolve_target(target, context)
            return nav2.navigate_to_pose(goal_pose)

        # Fallback: dead-reckoning with room map
        return self._dead_reckoning_navigate(target, context)
```

### C.3 Nav2 Integration Flow (Simulation)

```
NavigateSkill
    |
    v
Nav2Client (Python, wraps action client)
    |
    v
/navigate_to_pose action (ROS2)
    |
    v
Nav2 BT Navigator
    |  (uses /map, /scan, /odom, /tf)
    v
Nav2 Controller (DWB / MPPI)
    |
    v
/cmd_vel topic (geometry_msgs/Twist)
    |
    v
Go2CmdVelBridge (ROS2 node)
    |  subscribes /cmd_vel
    |  calls base.set_velocity(vx, vy, vyaw)
    v
MuJoCoGo2._cmd_vel --> physics thread --> MPC --> mj_step
    |
    v  (publishes back)
Go2OdomBridge --> /odom topic
Go2LidarBridge --> /scan topic
Go2TfBridge --> /tf (odom -> base_link)
```

### C.4 ROS2 Bridge Nodes for Go2

Four lightweight nodes, all in one component container:

1. **Go2CmdVelBridge**: subscribes `/cmd_vel`, calls `base.set_velocity()`
2. **Go2OdomBridge**: timer at 50 Hz, calls `base.get_odometry()`, publishes `/odom` + `/tf`
3. **Go2LidarBridge**: timer at 10 Hz, calls `base.get_lidar_scan()`, publishes `/scan`
4. **Go2StateBridge**: timer at 10 Hz, publishes `/joint_states` for the 12 leg joints

These are the "Hardware Adapter B" bridge nodes. The existing `HardwareBridgeNode` is "Hardware Adapter A" (SO-101).

### C.5 Non-ROS2 Environments

When running without ROS2 (pure `python run.py --sim-go2`), NavigateSkill falls back to dead-reckoning. The room map and waypoint graph work without Nav2. This is the current behavior and remains the default.

For Nav2 mode, a launch file starts the bridge nodes + Nav2 stack. NavigateSkill detects Nav2 availability by checking `context.services["nav2"]`.

---

## D. Unified System Flow

### D.1 The Pipeline

```
User: "去厨房拿杯子"
    |
    v
Agent.execute("去厨房拿杯子")
    |
    v
LLM Planner (classify=task, plan steps):
    [
      {"skill": "navigate", "params": {"target": "kitchen"}},
      {"skill": "detect",   "params": {"query": "cup"}},
      {"skill": "pick",     "params": {"object_label": "cup"}},
    ]
    |
    v
TaskExecutor runs each step:
    |
    +-- NavigateSkill.execute({"target": "kitchen"}, ctx)
    |     ctx.base = MuJoCoGo2 (or any BaseProtocol)
    |     Uses Nav2 or dead-reckoning
    |
    +-- DetectSkill.execute({"query": "cup"}, ctx)
    |     ctx.perception = whatever is available on Go2
    |     (camera mounted on Go2, or room camera, or sim ground truth)
    |
    +-- PickSkill.execute({"object_label": "cup"}, ctx)
          ctx.arms["default"] = whatever arm is available
          (Go2 with arm attachment, or kitchen-mounted arm, or error)
```

### D.2 Multi-Robot Scenario

The unified flow supports a future where Go2 navigates and a stationary arm picks:

```
Agent has:
  ctx.bases = {"go2": MuJoCoGo2}
  ctx.arms = {"kitchen_arm": SO101Arm}
  ctx.grippers = {"kitchen_gripper": SO101Gripper}

LLM plans:
  [
    {"skill": "navigate", "params": {"target": "kitchen", "base": "go2"}},
    {"skill": "pick", "params": {"object_label": "cup", "arm": "kitchen_arm"}},
  ]
```

Skills use `context.get_arm(name)` to resolve which arm. Default = first/only arm.

### D.3 Capability-Gated Skill Selection

The LLM planner needs to know what hardware is available. The skill schema already includes preconditions. We add a `capabilities` dict to the planner context:

```python
capabilities = {
    "has_arm": ctx.has_arm(),
    "has_gripper": ctx.has_gripper(),
    "has_base": ctx.has_base(),
    "has_perception": ctx.has_perception(),
    "base_type": "quadruped",  # or "wheeled", "none"
    "arm_names": ["so101"],
    "base_names": ["go2"],
}
```

The LLM knows not to plan `pick` if `has_arm` is False. This is already partially handled by failure_modes/diagnosis_code, but putting it in the planner context prevents bad plans.

---

## E. SkillContext Redesign

### E.1 Current State (messy)

```python
@dataclass
class SkillContext:
    arm: Any              # single arm, duck-typed
    gripper: Any          # single gripper, duck-typed
    perception: Any       # single perception source
    world_model: Any
    calibration: Any
    config: dict
    arms: dict | None     # multi-arm (unused)
    base: Any | None      # mobile base (added ad-hoc)
```

### E.2 New SkillContext

```python
@dataclass(frozen=True)
class SkillContext:
    """Immutable execution context passed to every skill.

    Hardware is accessed via typed registries. Skills query capabilities
    before using hardware, enabling graceful degradation.
    """
    # --- Hardware registries (name -> Protocol instance) ---
    arms: dict[str, ArmProtocol] = field(default_factory=dict)
    grippers: dict[str, GripperProtocol] = field(default_factory=dict)
    bases: dict[str, BaseProtocol] = field(default_factory=dict)
    perception_sources: dict[str, PerceptionProtocol] = field(default_factory=dict)

    # --- Shared state ---
    world_model: WorldModel = field(default_factory=WorldModel)
    calibration: Any = None
    config: dict = field(default_factory=dict)

    # --- Service registry (e.g., nav2 action client) ---
    services: dict[str, Any] = field(default_factory=dict)

    # --- Convenience accessors (backward-compatible) ---
    @property
    def arm(self) -> ArmProtocol | None:
        """Default arm (first registered, or None)."""
        return next(iter(self.arms.values()), None)

    @property
    def gripper(self) -> GripperProtocol | None:
        """Default gripper (first registered, or None)."""
        return next(iter(self.grippers.values()), None)

    @property
    def base(self) -> BaseProtocol | None:
        """Default base (first registered, or None)."""
        return next(iter(self.bases.values()), None)

    @property
    def perception(self) -> PerceptionProtocol | None:
        """Default perception source (first registered, or None)."""
        return next(iter(self.perception_sources.values()), None)

    # --- Capability queries ---
    def has_arm(self, name: str | None = None) -> bool:
        return name in self.arms if name else bool(self.arms)

    def has_gripper(self, name: str | None = None) -> bool:
        return name in self.grippers if name else bool(self.grippers)

    def has_base(self, name: str | None = None) -> bool:
        return name in self.bases if name else bool(self.bases)

    def has_perception(self, name: str | None = None) -> bool:
        return name in self.perception_sources if name else bool(self.perception_sources)

    def get_arm(self, name: str | None = None) -> ArmProtocol | None:
        if name: return self.arms.get(name)
        return self.arm

    def get_gripper(self, name: str | None = None) -> GripperProtocol | None:
        if name: return self.grippers.get(name)
        return self.gripper

    def get_base(self, name: str | None = None) -> BaseProtocol | None:
        if name: return self.bases.get(name)
        return self.base

    def capabilities(self) -> dict[str, Any]:
        """Hardware capability summary for LLM planner context."""
        return {
            "has_arm": self.has_arm(),
            "has_gripper": self.has_gripper(),
            "has_base": self.has_base(),
            "has_perception": self.has_perception(),
            "arm_names": list(self.arms.keys()),
            "gripper_names": list(self.grippers.keys()),
            "base_names": list(self.bases.keys()),
            "perception_names": list(self.perception_sources.keys()),
        }
```

### E.3 Why frozen=True SkillContext

Skills should not mutate the context. Hardware registries are set once at Agent construction. World model mutations go through `world_model.update_*()` methods (which are internally mutable -- the frozen constraint is on the context wrapper, not the WorldModel itself).

Note: `frozen=True` means the dataclass fields cannot be reassigned. The dicts themselves are mutable (you can still call `world_model.add_object()`). This is intentional -- it prevents skills from doing `context.arm = None` but allows them to use the world model normally.

### E.4 Backward Compatibility

The `arm`, `gripper`, `base`, `perception` properties provide drop-in compatibility. Existing skills that do `context.arm.move_joints(...)` work without changes. Existing skills that do `context.base.walk(...)` work without changes.

The only breaking change: `Agent._build_context()` must construct the new SkillContext with dict-based registries instead of direct fields. This is a one-line change in one file.

---

## F. Module Structure

```
vector_os_nano/
  hardware/
    __init__.py
    arm.py              # ArmProtocol (existing, unchanged)
    gripper.py          # GripperProtocol (existing, unchanged)
    base.py             # BaseProtocol (NEW)
    perception.py       # PerceptionProtocol (NEW, formalize)
    sim/
      mujoco_arm.py     # MuJoCoArm (existing)
      mujoco_gripper.py # MuJoCoGripper (existing)
      mujoco_go2.py     # MuJoCoGo2 (REFACTORED: background physics thread)
      mujoco_perception.py  # (existing)
    so101/              # Hardware Adapter A (existing)
    # Future: unitree_real/ -- Hardware Adapter B (real Go2 SDK)
    # Future: turtlebot/   -- Hardware Adapter C
  core/
    types.py            # + Odometry, LaserScan types
    skill.py            # SkillContext redesigned
    agent.py            # _build_context() updated
  skills/
    navigate.py         # NEW: hardware-agnostic NavigateSkill
    go2/
      navigate.py       # DEPRECATED: replaced by skills/navigate.py
      walk.py           # unchanged (uses context.base.walk)
      turn.py           # unchanged
      stance.py         # unchanged
  ros2/
    nodes/
      hardware_bridge.py    # Hardware Adapter A bridge (existing)
      go2_bridge.py         # NEW: Hardware Adapter B bridge (cmd_vel, odom, scan, tf)
    launch/
      go2_nav2.launch.py    # NEW: Go2 + Nav2 launch file
```

---

## G. Migration Path

### Phase 1: Protocol + Types (non-breaking)
1. Add `BaseProtocol` to `hardware/base.py`
2. Add `Odometry`, `LaserScan` to `core/types.py`
3. Add `PerceptionProtocol` to `hardware/perception.py`
4. No existing code changes. All new files.

### Phase 2: SkillContext Redesign (breaking, but backward-compatible via properties)
1. Rewrite `SkillContext` in `core/skill.py` with dict registries + property accessors
2. Update `Agent._build_context()` to populate dicts
3. Run full test suite -- existing skills should pass via property accessors
4. Update `Agent.__init__` signature (arms=dict, bases=dict, etc.) with backward-compat wrapper

### Phase 3: MuJoCoGo2 Refactor (breaking for internal loop, API preserved)
1. Add background physics thread to MuJoCoGo2
2. Implement `set_velocity()` (new)
3. Refactor `walk()` to use set_velocity + sleep
4. Implement `get_odometry()`, `get_lidar_scan()`
5. Test: existing Go2 skills must still pass

### Phase 4: Hardware-Agnostic NavigateSkill
1. Create `skills/navigate.py` (replaces `skills/go2/navigate.py`)
2. NavigateSkill uses `context.base` (any BaseProtocol)
3. Nav2 integration via `context.services["nav2"]` (optional)
4. Dead-reckoning fallback uses room map (existing logic, moved)

### Phase 5: ROS2 Go2 Bridge Nodes
1. `Go2CmdVelBridge`, `Go2OdomBridge`, `Go2LidarBridge`, `Go2StateBridge`
2. Launch file for Go2 + Nav2 in simulation
3. Map generation from MuJoCo scene (or static map from go2_room.xml)

---

## H. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Background physics thread introduces race conditions in MuJoCo | HIGH | Single-writer (physics thread owns mj_step). Readers (get_position, get_odometry) read from double-buffered snapshots. MuJoCo is NOT thread-safe -- all mj_* calls must be on the physics thread. |
| walk() timing changes due to sleep vs sim-step counting | MEDIUM | Physics thread runs at 1000 Hz regardless. Sleep granularity (~1ms on Linux) is sufficient. For precise timing, physics thread counts steps internally. |
| SkillContext frozen=True breaks WorldModel mutations | LOW | WorldModel is a mutable object referenced by the frozen dataclass. Frozen only prevents field reassignment, not object mutation. Tested. |
| Nav2 requires valid map + tf tree, complex to set up | MEDIUM | Phase 5 is deferred. Dead-reckoning works today. Nav2 is additive, not required. |
| mj_ray lidar simulation is slow for 360 rays at 10 Hz | LOW | 360 raycasts per frame is ~0.5ms on modern hardware. Acceptable. Can reduce to 180 rays if needed. |
| Breaking SkillContext change affects all downstream tests | MEDIUM | Property accessors provide full backward compatibility. Phased rollout with full test suite at each phase. |

---

## I. Decisions Requiring CEO/CTO Approval

1. **New BaseProtocol interface** -- defines the contract for ALL future mobile bases. Once shipped, changing it is expensive.
2. **SkillContext redesign** -- touches every skill. Backward-compatible via properties, but the new canonical API uses dicts.
3. **NavigateSkill moved from go2/ to top-level skills/** -- signals that navigation is hardware-agnostic, not Go2-specific.
4. **Background physics thread in MuJoCoGo2** -- architectural change from synchronous to async simulation.

All four are reversible. None affect the external user API (`Agent.execute()`). The risk is contained to internal plumbing.
