# HAL Redesign — Task List

## Execution Status
- Total: 7
- Completed: 0

## Wave 1 (parallel, no deps)

### T1: Odometry + LaserScan types
- **Agent**: gamma
- **File**: vector_os_nano/core/types.py
- **Test**: tests/unit/test_types_hal.py
- RED: Odometry/LaserScan frozen dataclass, to_dict/from_dict round-trip, field defaults
- GREEN: add to types.py (append, don't modify existing)

### T2: BaseProtocol
- **Agent**: alpha
- **File**: vector_os_nano/hardware/base.py (NEW)
- **Test**: tests/unit/test_base_protocol.py
- RED: runtime_checkable, MuJoCoGo2 satisfies protocol (isinstance check)
- GREEN: write base.py with Protocol class

## Wave 2 (parallel)

### T3: SkillContext redesign
- **Agent**: beta
- **Depends**: T1
- **File**: vector_os_nano/core/skill.py
- **Test**: tests/unit/test_skill_context.py
- RED: dict registries, backward-compat properties, has_arm/has_base, capabilities()
- GREEN: rewrite SkillContext, preserve all existing functionality

### T4: MuJoCoGo2 physics thread + set_velocity + lidar
- **Agent**: alpha
- **Depends**: T1, T2
- **File**: vector_os_nano/hardware/sim/mujoco_go2.py
- **Test**: tests/unit/test_mujoco_go2.py (extend existing)
- RED: set_velocity changes position, get_odometry returns Odometry, get_lidar_scan returns LaserScan
- GREEN: refactor to background thread, implement set_velocity/get_odometry/get_lidar_scan
- CRITICAL: all existing MuJoCoGo2 tests must still pass

## Wave 3 (parallel)

### T5: Agent constructor update
- **Agent**: gamma
- **Depends**: T3
- **File**: vector_os_nano/core/agent.py
- **Test**: tests/unit/test_agent.py (extend)
- RED: Agent(bases={"go2": go2}) works, _build_context returns new SkillContext
- GREEN: update __init__ + _build_context, backward compat for Agent(arm=x, base=y)

### T6: NavStackClient
- **Agent**: beta
- **Depends**: T1
- **File**: vector_os_nano/core/nav_client.py (NEW)
- **Test**: tests/unit/test_nav_client.py
- RED: mock ROS2 topics, navigate_to blocks until goal_reached, is_available check
- GREEN: implement with lazy rclpy import, fallback when no ROS2

## Wave 4

### T7: Unified NavigateSkill
- **Agent**: alpha
- **Depends**: T3, T5, T6
- **File**: vector_os_nano/skills/navigate.py (NEW, replaces skills/go2/navigate.py)
- **Test**: tests/unit/test_navigate_skill.py
- RED: uses NavStackClient when available, dead-reckoning fallback, hardware-agnostic
- GREEN: move logic from go2/navigate.py, add NavStackClient integration
- Update skills/go2/__init__.py to remove NavigateSkill, add import from skills/navigate.py

## Dependency Graph
```
T1 (types) ──┬──> T3 (SkillContext) ──> T5 (Agent) ──┐
             │                                        ├──> T7 (NavigateSkill)
T2 (BaseProto)──> T4 (MuJoCoGo2 thread)              │
             │                                        │
T1 ──────────┴──> T6 (NavStackClient) ────────────────┘
```
