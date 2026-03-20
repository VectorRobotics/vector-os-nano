# Vector OS Nano

Natural language robot arm control. `pip install` and go.

```bash
pip install vector-os-nano[all]
```

```python
from vector_os import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-...")
agent.execute("pick up the red cup")
```

## What is Vector OS Nano?

Vector OS Nano is a Python SDK that gives any robot arm a natural language brain. You describe what you want in plain English — "pick up the red cup and place it on the tray" — and the agent plans a skill sequence, handles perception, and executes the motion. It ships with a complete driver for the SO-101 open-source arm and an optional ROS2 integration layer for larger deployments.

The SDK works standalone with no ROS2 dependency. For hobbyists or researchers running a single arm on a laptop, a plain `pip install` is all that's needed. For production systems that need lifecycle management, TF2 transforms, multi-robot coordination, or Nav2 compatibility, the optional ROS2 mode launches a full node graph that exposes every capability as a service.

## Installation

**IMPORTANT:** Always use a dedicated virtual environment. Do NOT install into your system Python or an existing ROS2/conda environment — dependency conflicts will occur.

### Step 1: Create virtual environment

```bash
cd ~/Desktop/vector_os    # or wherever you cloned the repo
python3 -m venv vector_os_nano
source vector_os_nano/bin/activate
pip install --upgrade pip
```

On Windows:
```powershell
python -m venv vector_os_nano
vector_os_nano\Scripts\activate
pip install --upgrade pip
```

### Step 2: Install the SDK

**Core only** (no GPU, no simulation — works everywhere):

```bash
pip install -e "."
```

**Full** (GPU perception + IK + simulation + TUI):

```bash
pip install -e ".[all]"
```

**Individual extras** (pick what you need):

```bash
pip install -e ".[perception]"   # RealSense + VLM + EdgeTAM + pointcloud
pip install -e ".[ik]"           # Pinocchio FK/IK solver
pip install -e ".[tui]"          # Textual developer dashboard
pip install -e ".[sim]"          # PyBullet simulation
```

### ROS2 mode (Ubuntu 22.04 + ROS2 Humble)

```bash
# 1. Install ROS2 Humble: https://docs.ros.org/en/humble/Installation.html
# 2. ROS2 Python packages (rclpy, etc.) come from apt, not pip
# 3. Create venv WITH system packages so rclpy is accessible:
python3 -m venv vector_os_nano --system-site-packages
source vector_os_nano/bin/activate
pip install -e ".[all]"
```

### Windows

The core SDK (arm driver, skills, LLM providers, world model) is fully compatible with Windows 10/11. Use `COM3` instead of `/dev/ttyACM0` for the serial port. GPU perception requires an NVIDIA GPU with CUDA.

## Usage

### Without LLM (direct skill execution)

```python
from vector_os import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm)  # no LLM key

agent.execute("home")    # move to home position
agent.execute("scan")    # move to scan position
agent.execute("detect")  # run object detection (requires camera)
agent.execute("pick")    # pick the first detected object
```

Direct mode recognises skill names verbatim. No LLM call is made; the matching skill runs immediately.

### With LLM (natural language)

```python
from vector_os import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-ant-...")

result = agent.execute("pick up the red cup and place it on the tray")
if result.success:
    print("Done")
else:
    print("Failed:", result.error)
    print("Clarification:", result.clarification)
```

The LLM (Claude by default) decomposes the request into a sequence of built-in skills, checks preconditions against the world model, and re-plans on failure.

### Context manager (auto-connect and disconnect)

```python
from vector_os import Agent, SO101

with Agent(arm=SO101(port="/dev/ttyACM0"), llm_api_key="sk-...") as agent:
    agent.execute("home")
    agent.execute("pick the blue block")
# arm torque disabled and serial port closed on exit
```

### Custom skills

```python
from vector_os import Agent, SO101
from vector_os.core.skill import SkillContext
from vector_os.core.types import SkillResult


class WaveSkill:
    name = "wave"
    description = "Wave the arm back and forth"
    parameters = {"times": {"type": "integer", "default": 3}}
    preconditions = []
    postconditions = []
    effects = {}

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        times = int(params.get("times", 3))
        for _ in range(times):
            joints = list(context.arm.get_joint_positions())
            joints[0] = 0.5
            context.arm.move_joints(joints, duration=0.5)
            joints[0] = -0.5
            context.arm.move_joints(joints, duration=0.5)
        return SkillResult(success=True)


arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-...", skills=[WaveSkill()])
agent.execute("wave 5 times")
```

Custom skills are automatically made available to the LLM planner. The `skills=[...]` argument appends to the built-in set (pick, place, home, scan, detect).

### Interactive CLI

```bash
# Connect to real arm
vector-os --port /dev/ttyACM0 --llm-key sk-...

# No arm (planning + world model only)
vector-os --no-arm

# Verbose output (shows skill traces)
vector-os --port /dev/ttyACM0 --verbose
```

Available commands at the prompt: `pick`, `place`, `home`, `scan`, `detect`, `status`, `skills`, `world`, `help`, `quit`.

## ROS2 Integration

### Why ROS2?

The standalone SDK covers most use cases. Add ROS2 when you need:

- Lifecycle-managed node graph with automatic restart on crash
- TF2 coordinate frame transforms across multiple sensors or robots
- Nav2 integration (the arm publishes `/joint_states` compatible with standard controllers)
- RViz visualisation of the world model and planned trajectories
- Multi-robot coordination via standard ROS2 topics and services
- rosbag2 data recording for replay and debugging

### Launch the full node graph

```bash
ros2 launch vector_os nano.launch.py serial_port:=/dev/ttyACM0

# Custom port and RViz
ros2 launch vector_os nano.launch.py serial_port:=/dev/ttyACM1 use_rviz:=true
```

Startup order (staggered for reliable initialisation):

| t | Node | Role |
|---|------|------|
| 0 s | `hardware_bridge` | Connects to servos, publishes `/joint_states` at 30 Hz |
| 3 s | `perception_bridge` | Reads RGB-D camera, publishes `/perception/detections` |
| 5 s | `world_model_service` | Builds world model from joint states + detections |
| 6 s | `skill_server` | Exposes each skill as a `/skill/<name>` service |
| 7 s | `agent_node` | Accepts natural language via `/agent/execute` |

### Available ROS2 services

| Service | Type | Description |
|---------|------|-------------|
| `/skill/home` | `std_srvs/Trigger` | Move to home position |
| `/skill/scan` | `std_srvs/Trigger` | Move to scan position |
| `/skill/detect` | `std_srvs/Trigger` | Run VLM object detection |
| `/skill/pick` | `std_srvs/Trigger` | Pick first detected object |
| `/skill/place` | `std_srvs/Trigger` | Place held object |
| `/world_model/query` | `std_srvs/Trigger` | Return full world state as JSON |
| `/world_model/predicate` | `std_srvs/Trigger` | Evaluate a named predicate |
| `/agent/execute` | `std_srvs/Trigger` | Execute a natural language command |
| `/agent/plan` | `std_srvs/Trigger` | Return the planned skill sequence without executing |

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `serial_port` | `/dev/ttyACM0` | Serial port for SO-101 |
| `baudrate` | `1000000` | Serial baud rate |
| `publish_rate` | `30.0` | Joint state publish rate (Hz) |
| `color_topic` | `/camera/color/image_raw` | RGB image topic |
| `depth_topic` | `/camera/aligned_depth_to_color/image_raw` | Aligned depth topic |
| `use_rviz` | `false` | Launch RViz visualiser |

## Architecture

```
User input (text)
      |
      v
  [ Agent ]  <--- config, skills list, LLM provider
      |
      +---> [ LLM Planner ] <--- SkillRegistry schemas, WorldModel state
      |           |
      |           v
      |      [ TaskPlan ]  (ordered TaskStep list with dependencies)
      |           |
      +---> [ TaskExecutor ]
      |           |
      |     (for each step)
      |           |
      |           +---> precondition check (WorldModel)
      |           +---> skill.execute(params, SkillContext)
      |           +---> postcondition check + world model update
      |
      +---> [ Skills ]  (pick, place, home, scan, detect, custom...)
      |           |
      |           +---> [ ArmProtocol ]  (SO101Arm or SimulatedArm)
      |           +---> [ GripperProtocol ]  (SO101Gripper)
      |           +---> [ PerceptionProtocol ]  (PerceptionPipeline)
      |
      +---> [ WorldModel ]  (object states, robot state, predicates)

ROS2 mode wraps each layer in a lifecycle node:
  HardwareBridgeNode -> PerceptionBridgeNode -> WorldModelServiceNode
                     -> SkillServerNode -> AgentNode
```

## Hardware Setup

### SO-101 arm

The SO-101 is a 5-DOF open-source robot arm using Feetech STS3215 serial-bus servos. Assembly guides and CAD files are available at [huggingface.co/lerobot](https://huggingface.co/lerobot).

- Connect via USB (appears as `/dev/ttyACM0` on Linux, `COM3` on Windows)
- Default baud rate: 1 000 000 bps
- 5 arm joints + 1 gripper joint on the same serial bus

### Intel RealSense D405

- Mount on the gripper bracket (eye-in-hand configuration)
- USB 3.x port required (blue port)
- Streams aligned RGB + depth at up to 640x480 / 30 fps

### Calibration

Hand-eye calibration maps camera coordinates to robot base coordinates:

```python
from vector_os.skills.calibration import load_calibration
cal = load_calibration("my_calibration.npy")
```

Run the calibration routine once after mounting the camera and save the result. The pick skill uses it automatically.

## Configuration

The default configuration is in `config/default.yaml`. Override any key by passing a dict or a YAML file path to `Agent`:

```python
agent = Agent(
    arm=arm,
    config={
        "llm": {"provider": "openai", "model": "gpt-4o"},
        "skills": {"pick": {"max_retries": 3}},
    }
)
```

Or with a YAML file:

```python
agent = Agent(arm=arm, config="/path/to/my_config.yaml")
```

### Configuration reference

```yaml
agent:
  max_planning_retries: 3      # LLM replanning attempts on failure
  max_execution_retries: 2     # skill retry attempts
  planning_timeout_sec: 10.0

llm:
  provider: "claude"           # claude | openai | local
  model: "claude-sonnet-4-6"
  api_base: "https://openrouter.ai/api/v1"
  temperature: 0.0
  max_tokens: 2048

arm:
  type: "so101"
  port: "/dev/ttyACM0"         # COM3 on Windows
  baudrate: 1000000

camera:
  type: "realsense"
  serial: ""                   # empty = auto-detect
  resolution: [640, 480]
  fps: 30

perception:
  vlm_provider: "moondream"    # moondream | station | cloud
  vlm_model: "vikhyatk/moondream2"
  tracker: "edgetam"
  tracker_model: "yonigozlan/EdgeTAM-hf"

calibration:
  file: ""                     # path to .npy calibration file
  method: "affine"
  num_points: 25

skills:
  pick:
    z_offset: 0.12
    pre_grasp_height: 0.06
    max_retries: 2
  place:
    default_height: 0.05
  home:
    joint_values: [-0.014, -1.238, 0.562, 0.858, 0.311]
```

## API Reference

### `Agent`

The main entry point. Wires all components together.

```python
Agent(
    arm=None,           # ArmProtocol implementation (SO101Arm or SimulatedArm)
    gripper=None,       # GripperProtocol — auto-created from arm if omitted
    perception=None,    # PerceptionProtocol — enables vision-based skills
    llm=None,           # LLMProvider — enables natural language planning
    llm_api_key=None,   # Shorthand: creates ClaudeProvider automatically
    skills=None,        # Additional skills (appended to built-in defaults)
    config=None,        # dict, path to YAML, or None for defaults
)
```

Key methods: `execute(command) -> ExecutionResult`, `home() -> SkillResult`, `stop()`, `connect()`, `disconnect()`.

### `SO101`

Alias for `SO101Arm`. Implements `ArmProtocol`.

```python
SO101(port="/dev/ttyACM0", baudrate=1_000_000)
```

Key methods: `connect()`, `disconnect()`, `get_joint_positions() -> list[float]`, `move_joints(positions, duration) -> bool`, `stop()`.

### `Skill`

Runtime-checkable Protocol. Implement to add custom capabilities.

Required class attributes: `name`, `description`, `parameters`, `preconditions`, `postconditions`, `effects`.
Required method: `execute(params: dict, context: SkillContext) -> SkillResult`.

### `SkillResult`

Frozen dataclass returned by every skill.

```python
SkillResult(success=True)
SkillResult(success=False, error="IK failed: target out of reach")
```

### `ExecutionResult`

Returned by `Agent.execute()`.

```python
result.success        # bool
result.error          # str | None
result.clarification  # str | None — LLM question when it needs more info
result.steps          # list[StepTrace] — per-skill execution trace
```

### `WorldModel`

Tracks all known objects and robot state. Access via `agent.world`.

```python
agent.world.get_object("cup_01")           # ObjectState | None
agent.world.evaluate_predicate("gripper_empty")   # bool
agent.world.list_objects()                 # list[str]
```

## License

MIT License. Copyright 2024 Vector Robotics.

See [LICENSE](LICENSE) for the full text.
