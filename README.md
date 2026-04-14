<p align="center">
  <img src="images/ascii-art.png" width="800" alt="Vector Robotics">
</p>

<h1 align="center">Vector OS Nano</h1>

<p align="center">
  <b>Cross-embodiment robot OS: natural language control, industrial-grade navigation, sim-to-real.</b>
  <br>
  <b>No training. No fine-tuning. Just say what you want.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MuJoCo-3.6-green" alt="MuJoCo">
  <img src="https://img.shields.io/badge/ROS2_Jazzy-Navigation-blue?logo=ros&logoColor=white" alt="ROS2">
  <img src="https://img.shields.io/badge/Unitree_Go2-Quadruped-red" alt="Go2">
  <img src="https://img.shields.io/badge/Convex_MPC-1kHz_Control-orange" alt="MPC">
  <img src="https://img.shields.io/badge/Claude-LLM_Brain-blueviolet?logo=anthropic&logoColor=white" alt="Claude">
  <img src="https://img.shields.io/badge/LeRobot-SO--ARM100-black" alt="LeRobot">
  <img src="https://img.shields.io/badge/Intel_RealSense-D405-0071C5?logo=intel&logoColor=white" alt="RealSense">
</p>

<p align="center">
  <i>Being developed at <b>CMU Robotics Institute</b>. Nano is the grasping proof-of-value. Full stack coming soon.</i>
</p>

---

<h3 align="center">Demo</h3>

<p align="center">
  <a href="https://drive.google.com/file/d/1a0Y46zHZ9VNUqBVCpGbyP9m2getLlIio/view">
    <img src="images/compressed_demo.gif" width="700" alt="Click to watch full demo video">
  </a>
  <br>
  <i>Click to watch full demo video</i>
</p>

---

## Architecture

Two entry points, one engine:

```
vector-cli  ──┐
              ├─> VectorEngine ──> VGG / tool_use ──> skill.execute()
vector-os-mcp┘         |
                       ├── VGG cognitive layer (task decomposition + verification + retry)
                       ├── 39 tools (file ops, bash, ROS2 diag, skill wrappers)
                       ├── LLM backend (Anthropic / OpenRouter / local)
                       └── permission system + session + intent router
```

| Entry Point | Purpose |
|-------------|---------|
| `vector-cli` | Interactive REPL for humans. Natural language + slash commands. |
| `vector-os-mcp` | MCP server for Claude Code. Same engine, machine interface. |

## Quick Start

```bash
cd vector_os_nano
python3 -m venv .venv --prompt nano
source .venv/bin/activate
pip install -e ".[all]"

vector-cli                     # interactive AI agent
vector-cli --sim-go2           # Go2 quadruped in MuJoCo
```

LLM config: create `config/user.yaml` with your provider credentials, or type `/login` in vector-cli to authenticate with Claude subscription.

## Vector CLI

AI-powered terminal where you control robots, edit code, and diagnose issues from one prompt.

```
vector> explore the house
  > [1/1] explore_goal done 62.3s

vector> the dog hits walls at corners
  file_read("scripts/go2_vnav_bridge.py") ... ok
  file_edit(old="_MAX_SPEED = 0.6", new="_MAX_SPEED = 0.4") ... ok
  skill_reload("walk") ... ok

  Reduced turn speed from 0.6 to 0.4. Want me to re-run explore?

vector> go to the kitchen
  >> 距目标 5.2m, 已走 4s
  >> 距目标 2.1m, 已走 8s
  > [1/1] navigate_goal done 11.3s

vector> stop
  Stopped.
```

**39 tools** across 4 categories:

| Category | Tools |
|----------|-------|
| code | file_read, file_write, file_edit, bash, glob, grep |
| robot | 22 skills (walk, navigate, explore, pick, place...) + scene_graph_query |
| diag | ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status |
| system | robot_status, start_simulation, skill_reload, web_fetch, open_foxglove |

**Slash commands:** `/help` `/model` `/tools` `/status` `/login` `/compact` `/clear` `/copy` `/export`

<p align="center">
  <img src="images/agent.png" width="700" alt="vector-cli with Go2 simulation">
  <br>
  <i>vector-cli controlling Go2 in MuJoCo: natural language conversation with V (right), live simulation (left).</i>
</p>

## VGG: Verified Goal Graph

All actionable commands flow through the VGG cognitive layer. LLM decomposes complex tasks into verifiable sub-goal trees. Simple commands get 1-step GoalTrees without LLM call.

```
User input
  |
  should_use_vgg?
  |-- Action --> VGG
  |     |-- Simple (skill match) --> 1-step GoalTree (no LLM, <1ms)
  |     |-- Complex (multi-step) --> LLM decomposition --> GoalTree
  |     |
  |     VGG Harness: 3-layer feedback loop
  |       Layer 1: step retry (alt strategies)
  |       Layer 2: continue past failure
  |       Layer 3: re-plan with failure context
  |     |
  |     GoalExecutor --> verify --> trace --> stats
  |
  |-- Conversation --> tool_use path (LLM direct)
```

30 primitives across 4 categories: locomotion (8), navigation (5), perception (6), world (11).

## Go2 Navigation Stack

MuJoCo + CMU Vector Navigation Stack for autonomous indoor navigation.

```
TARE (frontier exploration)
  --> FAR V-Graph (global visibility-graph routing)
    --> localPlanner (terrain-aware obstacle avoidance)
      --> pathFollower --> Go2 MPC (1kHz control)
```

Sim-to-real: the nav stack is identical to what runs on real Unitree Go2 with Livox MID360 LiDAR. Switching sim to real requires only changing the bridge node.

<p align="center">
  <img src="images/go2nav.png" width="700" alt="Go2 navigation in MuJoCo">
  <br>
  <i>Go2 autonomous navigation: house environment (top), RViz with LiDAR + terrain analysis + path planning (bottom-left), first-person camera (bottom-right).</i>
</p>

Navigation parameters tunable via `config/nav.yaml`:
```yaml
navigation:
  ceiling_filter_height: 1.8
  arrival_radius: 0.8
  far_probe_timeout: 3.0
  waypoint_timeout: 30.0
  stall_timeout: 30.0
```

## SO-101 Arm

```python
from vector_os_nano import Agent, SO101
from vector_os_nano.core.skill import SkillContext

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm)
agent.execute_skill("pick", {"object_label": "red cup"})
```

No hardware? MuJoCo simulation:

```python
from vector_os_nano import Agent, MuJoCoArm

arm = MuJoCoArm(gui=True)
arm.connect()
agent = Agent(arm=arm)
agent.execute_skill("pick", {"object_label": "banana"})
```

<p align="center">
  <img src="images/setup.png" width="700" alt="SO-101 real hardware">
  <br>
  <i>Real hardware: SO-101 arm with RealSense D405, VLM object detection, natural language control.</i>
</p>

<p align="center">
  <img src="images/sim_setup.png" width="700" alt="MuJoCo arm simulation">
  <br>
  <i>MuJoCo simulation: SO-101 arm with graspable objects, CLI conversation with V.</i>
</p>

## SkillFlow -- Declarative Skill Routing

All routing via `@skill` decorator. No hard-coded if/else chains.

```python
from vector_os_nano.core.skill import skill, SkillContext
from vector_os_nano.core.types import SkillResult

@skill(
    aliases=["grab", "grasp", "抓", "拿"],
    direct=False,
    auto_steps=["scan", "detect", "pick"],
)
class PickSkill:
    name = "pick"
    description = "Pick up an object from the workspace"
    parameters = {"object_label": {"type": "string"}}
    preconditions = ["gripper_empty"]
    postconditions = ["gripper_holding_any"]
    effects = {"gripper_state": "holding"}

    def execute(self, params, context):
        return SkillResult(success=True)
```

## MCP Server

Claude Code controls the robot via Model Context Protocol:

```bash
vector-os-mcp --sim --stdio          # MuJoCo sim, stdio transport
vector-os-mcp --hardware --stdio     # real hardware
vector-os-mcp --sim                  # SSE on :8100
```

MCP tools: all 22 skills + natural_language + run_goal + diagnostics + debug_perception
MCP resources: world://state, world://objects, world://robot, camera://overhead, camera://front, camera://side, camera://live

<p align="center">
  <img src="images/mcp_claude.png" width="700" alt="Claude Code controlling robot via MCP">
  <br>
  <i>Claude Code operating the robot via MCP: terminal (top) + live camera RGB + depth (bottom).</i>
</p>

## Project Structure

```
vector_os_nano/
├── vcli/              VectorEngine + VGG cognitive layer + 39 tools
│   ├── engine.py      Core agent loop (run_turn, VGG decompose/execute)
│   ├── cognitive/     GoalDecomposer, GoalExecutor, GoalVerifier, VGGHarness,
│   │                  StrategySelector, ObjectMemory, VisualVerifier, abort
│   ├── tools/         CategorizedToolRegistry (code/robot/diag/system)
│   ├── primitives/    30 VGG primitives (locomotion/nav/perception/world)
│   └── backends/      LLM providers (Anthropic, OpenAI-compat)
├── mcp/               MCP server (uses VectorEngine, same as CLI)
├── core/
│   ├── agent.py       Hardware container (arm, gripper, base, perception)
│   ├── skill.py       @skill decorator, SkillRegistry, SkillContext
│   ├── types.py       Shared data types (SkillResult, GoalTree, etc.)
│   ├── world_model.py World state tracking
│   └── scene_graph.py Hierarchical spatial memory (rooms/doors/objects)
├── skills/
│   ├── go2/           walk, turn, stand, sit, explore, stop, navigate, patrol
│   └── ...            pick, place, home, scan, detect, gripper, wave, handover
├── hardware/
│   ├── so101/         SO-101 arm driver (Feetech serial, Pinocchio IK)
│   └── sim/           MuJoCo (arm, Go2, perception), ROS2 proxies, Isaac Sim
├── perception/        RealSense D405 + VLM + EdgeTAM tracker + pointcloud
└── ros2/              Optional ROS2 nodes (hardware_bridge, agent_node, etc.)

scripts/
├── go2_vnav_bridge.py  MuJoCo <-> ROS2 bridge
└── launch_*.sh         Nav stack launch scripts

config/
├── nav.yaml            Navigation parameters
├── user.yaml           LLM credentials
└── room_layout.yaml    Scene geometry
```

## Hardware (~$420 total)

| Component | Model | Cost |
|-----------|-------|------|
| Robot Arm | LeRobot SO-ARM100 (6-DOF, 3D-printed) | ~$150 |
| Camera | Intel RealSense D405 | ~$270 |
| GPU | Any NVIDIA with 8+ GB VRAM | (existing) |

<p align="center">
  <img src="images/skillgen.png" width="700" alt="Autonomous skill generation">
  <br>
  <i>Claude Agent autonomously designing, implementing, and executing new robot skills.</i>
</p>

## Nav Stack Dependencies

Go2 navigation requires the [Vector Navigation Stack](https://github.com/VectorRobotics/vector_navigation_stack):

```bash
sudo apt install ros-jazzy-desktop
cd ~/Desktop
git clone https://github.com/VectorRobotics/vector_navigation_stack.git
cd vector_navigation_stack && colcon build
```

Without the nav stack, basic Go2 skills (walk, turn, stand, dead-reckoning navigate) still work.

---

<p align="center">
  <i>CMU Robotics Institute. Star this repo and stay tuned.</i>
</p>
