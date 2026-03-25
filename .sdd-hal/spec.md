# Hardware Abstraction Layer Redesign — Specification

- Status: approved (CEO directive)
- Date: 2026-03-25
- Scope: BaseProtocol, Odometry/LaserScan types, SkillContext redesign, MuJoCoGo2 physics thread, NavStackClient, hardware-agnostic NavigateSkill

## 1. Overview

Redesign Vector OS Nano's hardware layer so that SO-101 arm, Go2 quadruped, Unity sim, and future robots are all interchangeable adapters behind the same Agent/Skill/LLM pipeline.

## 2. Goals

- MUST: Formal BaseProtocol with walk() (blocking) + set_velocity() (streaming)
- MUST: Odometry and LaserScan pure-Python types in core/types.py
- MUST: SkillContext uses dict registries (arms, bases, grippers) with backward-compatible properties
- MUST: MuJoCoGo2 runs physics in background thread, supports set_velocity() for Nav2 cmd_vel
- MUST: MuJoCoGo2 produces simulated lidar via mj_ray
- MUST: NavStackClient wraps vector_navigation_stack ROS2 interface (/way_point, /state_estimation, /goal_reached)
- MUST: NavigateSkill moved to top-level skills/, hardware-agnostic, uses NavStackClient when available
- MUST: All existing tests pass (backward compatibility)
- SHOULD: Python-only mode works without ROS2 (dead-reckoning fallback)

## 3. Non-Goals

- Real Go2 hardware driver (unitree_sdk2py) — future
- MuJoCo-to-Unity bridge — use Unity directly
- WebRTC — deferred
- New perception pipeline — existing works

## 4. Interface Definitions

### BaseProtocol (hardware/base.py)

```python
@runtime_checkable
class BaseProtocol(Protocol):
    @property
    def name(self) -> str: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def stop(self) -> None: ...

    # Blocking (for skills)
    def walk(self, vx: float, vy: float, vyaw: float, duration: float) -> bool: ...

    # Streaming (for Nav2 cmd_vel)
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None: ...

    # State
    def get_position(self) -> list[float]: ...
    def get_heading(self) -> float: ...
    def get_velocity(self) -> list[float]: ...
    def get_odometry(self) -> Odometry: ...
    def get_lidar_scan(self) -> LaserScan | None: ...

    # Capabilities
    @property
    def supports_holonomic(self) -> bool: ...
    @property
    def supports_lidar(self) -> bool: ...
```

### Odometry + LaserScan (core/types.py)

```python
@dataclass(frozen=True)
class Odometry:
    timestamp: float
    x: float; y: float; z: float
    qx: float; qy: float; qz: float; qw: float
    vx: float; vy: float; vz: float; vyaw: float

@dataclass(frozen=True)
class LaserScan:
    timestamp: float
    angle_min: float; angle_max: float; angle_increment: float
    range_min: float; range_max: float
    ranges: tuple[float, ...]
```

### SkillContext (core/skill.py)

```python
@dataclass
class SkillContext:
    arms: dict[str, Any] = field(default_factory=dict)
    grippers: dict[str, Any] = field(default_factory=dict)
    bases: dict[str, Any] = field(default_factory=dict)
    perception_sources: dict[str, Any] = field(default_factory=dict)
    world_model: Any = None
    calibration: Any = None
    config: dict = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)

    # Backward-compatible properties
    @property
    def arm(self): return next(iter(self.arms.values()), None)
    @property
    def gripper(self): return next(iter(self.grippers.values()), None)
    @property
    def base(self): return next(iter(self.bases.values()), None)
    @property
    def perception(self): return next(iter(self.perception_sources.values()), None)

    def has_arm(self) -> bool: return bool(self.arms)
    def has_base(self) -> bool: return bool(self.bases)
    def capabilities(self) -> dict: ...
```

### NavStackClient (core/nav_client.py)

```python
class NavStackClient:
    """Wraps vector_navigation_stack ROS2 topics."""
    def __init__(self, node): ...
    def navigate_to(self, x: float, y: float) -> bool: ...
    def cancel(self) -> None: ...
    def get_state_estimation(self) -> Odometry: ...
    @property
    def is_available(self) -> bool: ...
```

## 5. Test Contracts

### Unit Tests
- T1: BaseProtocol runtime checkable — MuJoCoGo2 satisfies it
- T2: Odometry/LaserScan to_dict/from_dict round-trip
- T3: SkillContext dict registries + backward-compat properties
- T4: SkillContext.capabilities() returns correct dict
- T5: MuJoCoGo2.set_velocity + physics thread runs
- T6: MuJoCoGo2.get_odometry returns valid Odometry
- T7: MuJoCoGo2.get_lidar_scan returns LaserScan with 360 ranges
- T8: NavigateSkill works with mock base (no Nav2)
- T9: NavigateSkill uses NavStackClient when in services
- T10: All existing Go2 tests still pass

### Integration Tests
- T11: Agent(bases={"go2": go2}) constructs correctly, skills work
- T12: MuJoCoGo2 set_velocity → walk via physics thread → position changes
