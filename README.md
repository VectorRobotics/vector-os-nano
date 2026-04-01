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
  <img src="https://img.shields.io/badge/Claude_Haiku-LLM_Brain-blueviolet?logo=anthropic&logoColor=white" alt="Claude">
  <img src="https://img.shields.io/badge/LeRobot-SO--ARM100-black" alt="LeRobot">
  <img src="https://img.shields.io/badge/Intel_RealSense-D405-0071C5?logo=intel&logoColor=white" alt="RealSense">
</p>

<p align="center">
  <i>Vector OS: a cross-embodiment robot operating system with industrial-grade SLAM, navigation, generalized grasping, semantic mapping, long-chain task orchestration and explainable task execution.<br>
  Being developed at <b>CMU Robotics Institute</b>. Nano is the grasping proof-of-value. Full stack coming soon.</i>
</p>

<p align="center">
  <i>Vector OS: 跨本体通用机器人操作系统：工业级SLAM、导航、泛化抓取、语义建图、长链任务编排、可解释任务执行。<br>
  <b>CMU 机器人研究所</b> 全力开发中。Nano 是低成本硬件上的低门槛概念验证，完整系统即将分阶段开源。</i>
</p>

---

### Built-in Agentic CLI: Vibe-code your robot

Vector OS Nano ships with `vector-cli` -- an agentic terminal harness with V, the AI core. Talk to your robot, write code, run simulations, and explore environments, all from one prompt.

```bash
pip install -e .
vector-cli                     # auto-detects Claude subscription or OpenRouter key
```

```
vector> start the go2 simulation
  start_simulation(sim_type="go2", gui=true) ... ok 2.1s

vector> explore the house
  explore(duration=60) ... ok 62.3s

vector> list all python files here
  glob(pattern="**/*.py") ... ok 0.1s

vector> !git status                    # ! prefix for direct shell
```

**What V can do:**

| Capability | Tools |
|------------|-------|
| Robot control | start_simulation, walk, turn, stand, sit, explore, navigate, pick, place |
| Codebase work | file_read, file_write, file_edit, bash, glob, grep |
| Perception | world_query, robot_status, detect, describe |
| Web | web_fetch (documentation, API references) |

**Authentication:** Claude subscription (auto-detected from Claude Code), Anthropic API key, OpenRouter, or local models via `--base-url`.

**Slash commands:** `/help` `/model` `/tools` `/agent` `/status` `/login` `/compact` `/clear` `/copy` `/export` -- type `/` + Tab for auto-complete with descriptions.

**Architecture:** Multi-backend LLM engine (Anthropic, OpenRouter, local models via OpenAI-compatible API) with streaming, tool-use loop, 6-layer permission system, JSONL session persistence, and concurrent read-only tool execution. 19 robot skills auto-registered at runtime when hardware connects.

<p align="center">
  <img src="images/agent.png" width="700" alt="Vector CLI with Go2 simulation">
  <br>
  <i>vector-cli controlling Go2 quadruped in MuJoCo: natural language conversation with V (right), live simulation (left).</i>
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

## What is Vector OS Nano?

A **cross-embodiment robot SDK** with natural language control, industrial-grade navigation, and sim-to-real transfer. Supports both manipulation (SO-101 arm) and locomotion (Unitree Go2 quadruped) with a unified skill architecture.

### Go2 Quadruped -- Full Navigation Stack in Simulation

MuJoCo physics simulation with convex MPC locomotion (1kHz), integrated with the CMU/Ji Zhang [Vector Navigation Stack](https://github.com/VectorRobotics/vector_navigation_stack) for autonomous exploration and navigation. The same navigation stack runs on real Unitree robots -- sim-to-real ready.

```bash
python run.py --sim-go2                # Go2 in MuJoCo + natural language agent
```

Then talk to the robot:

```
you> explore the house
  TOOL explore()
  [EXPLORE] Starting ROS2 bridge... Nav stack launching... RViz opening...
  [EXPLORE] Entered room: hallway
  [EXPLORE] Entered room: kitchen
  ...

you> go to the bedroom
  TOOL navigate(room="master_bedroom")

you> where am i
  TOOL where_am_i()
```

**Navigation architecture:**

```
TARE planner (frontier exploration)
    --> FAR planner (global visibility-graph routing)
        --> localPlanner (terrain-aware local obstacle avoidance)
            --> pathFollower / pure-pursuit --> Go2 MPC locomotion
```

**Sim-to-real:** The navigation stack (localPlanner, FAR planner, TARE, terrainAnalysis) is identical to what runs on real Unitree Go2 robots with Livox MID360 LiDAR. The MuJoCo bridge simulates the same sensor topics (`/state_estimation`, `/registered_scan`, `/terrain_map`) at matching specs (10k+ points/scan, 30-degree tilt, asymmetric FOV). Switching from sim to real hardware requires only changing the bridge node.

<p align="center">
  <img src="images/go2nav.png" width="700" alt="Go2 Navigation in MuJoCo">
  <br>
  <i>Go2 autonomous navigation in MuJoCo: house environment (top), RViz with LiDAR point cloud + terrain analysis + path planning (bottom-left), first-person camera view (bottom-right).</i>
</p>

### SO-101 Arm -- Zero-Shot Grasping

```python
from vector_os_nano import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-...")
agent.execute("pick up the red cup and put it on the left")
```

No hardware? Full simulation:

```python
from vector_os_nano import Agent, MuJoCoArm

arm = MuJoCoArm(gui=True)
arm.connect()
agent = Agent(arm=arm, llm_api_key="sk-...")
agent.execute("put the duck in front")
agent.run_goal("clean the table")     # agent loop: observe-decide-act-verify
```

## SkillFlow — Declarative Skill Routing

All command routing is **declarative** via the `@skill` decorator. No hard-coded if/else chains. Skills describe themselves, the system routes automatically.

```python
from vector_os_nano.core.skill import skill, SkillContext
from vector_os_nano.core.types import SkillResult

@skill(
    aliases=["grab", "grasp", "抓", "拿", "抓起"],       # trigger words
    direct=False,                                          # needs planning
    auto_steps=["scan", "detect", "pick"],                 # default chain
)
class PickSkill:
    name = "pick"
    description = "Pick up an object from the workspace"
    parameters = {"object_label": {"type": "string"}}
    preconditions = ["gripper_empty"]
    postconditions = ["gripper_holding_any"]
    effects = {"gripper_state": "holding"}

    def execute(self, params, context):
        # ... pick implementation ...
        return SkillResult(success=True)
```

Routing logic:

```
"home"           → @skill alias match → direct=True → execute (zero LLM)
"close grip"     → @skill alias match → direct=True → execute (zero LLM)
"抓杯子"         → @skill alias match → auto_steps → scan→detect→pick (zero LLM)
"把鸭子放到左边"  → no simple match → LLM plans pick(hold) + place(left) + home
"你好"           → no match → LLM classify → chat response
```

See `docs/skill-protocol.md` for full SkillFlow specification.

## Agent Pipeline — Two Execution Modes

**One-shot** (simple commands) — plan once, execute all steps:

```
User Input → MATCH → CLASSIFY → PLAN → EXECUTE → SUMMARIZE
"抓杯子"      alias    (skip)    (skip)   scan→detect→pick   "Done"
"把X放左边"   no match  task     LLM plan  pick(hold)→place   "Done"
```

**Agent Loop** (iterative goals) — observe, decide one step, act, verify, repeat:

```
User Input → run_goal("clean the table")
                ↓
        ┌→ OBSERVE  (camera + world model)
        │  DECIDE   (LLM picks ONE action)
        │  ACT      (execute single skill)
        │  VERIFY   (re-detect: did it actually work?)
        └─ LOOP     (until goal achieved or max iterations)
```

The Agent Loop solves two problems:
1. **Iterative goals**: "clean the table" picks objects one-by-one until table is empty
2. **Hardware drift**: doesn't trust `success=True` — uses perception to verify actual outcomes

AI assistant **V** speaks before acting, shows live progress, and summarizes after:

```
vector> 把所有东西都随便乱放

╭─ V ──────────────────────────────────────────────────╮
│ 好的主人，我来把所有东西都随便乱放。                     │
╰──────────────────────────────────────────────────────╯
  Plan: 13 steps

  [1/13] pick ... OK 14.6s
  [2/13] place ... OK 11.5s
  ...
  [13/13] home ... OK 3.8s

  Done 13/13 steps, 180.0s

╭─ V ──────────────────────────────────────────────────╮
│ 主人，已把6个物体全部放到不同位置，手臂已回待命。       │
╰──────────────────────────────────────────────────────╯
```

## MuJoCo Simulation

**No robot? No camera? No problem.**

### Go2 Quadruped Simulation

```bash
python run.py --sim-go2              # Go2 + MuJoCo viewer + agent CLI
python run.py --sim-go2-headless     # headless Go2
python run.py --sim-go2 --explore    # Go2 + full nav stack + TARE + RViz
```

- Unitree Go2 with convex MPC locomotion (1kHz physics, dual-backend with sinusoidal fallback)
- Livox MID360 LiDAR simulation (10k+ points/scan, 30-degree tilt, asymmetric FOV)
- Full Vector Navigation Stack: localPlanner + FAR planner + TARE autonomous exploration
- Terrain analysis, obstacle avoidance, visibility-graph routing
- 9 agent skills: walk, turn, stand, sit, lie_down, navigate, explore, stop, where_am_i
- 26/26 locomotion tests passing (L0 physics through L4 navigation)

### SO-101 Arm Simulation

```bash
python run.py --sim              # MuJoCo viewer + interactive CLI
python run.py --sim-headless     # headless (no viewer)
```

- SO-101 arm with real STL meshes (13 parts from CAD)
- 6 graspable objects: banana, mug, bottle, screwdriver, duck, lego
- Weld-constraint grasping, smooth real-time motion
- Pick-and-place with named locations (front, left, right, center, etc.)
- Simulated perception with Chinese/English NL queries

<p align="center">
  <img src="images/sim_setup.png" width="700" alt="MuJoCo Simulation">
</p>

## Hardware (~$420 total)

| Component | Model | Cost |
|-----------|-------|------|
| Robot Arm | LeRobot SO-ARM100 (6-DOF, 3D-printed) | ~$150 |
| Camera | Intel RealSense D405 | ~$270 |
| GPU | Any NVIDIA with 8+ GB VRAM | (existing) |

<p align="center">
  <img src="images/setup.png" width="600" alt="Hardware Setup">
</p>

## Quick Start

```bash
# Setup
cd vector_os_nano
python3 -m venv .venv --prompt vector_os_nano
source .venv/bin/activate
pip install -e ".[all]"

# Launch (simulation — no hardware needed)
python run.py --sim

# Launch (real hardware)
python run.py
```

LLM config — create `config/user.yaml`:
```yaml
llm:
  api_key: "your-openrouter-api-key"
  model: "anthropic/claude-haiku-4-5"
  api_base: "https://openrouter.ai/api/v1"
```

## All Launch Modes

```bash
# Go2 quadruped (MuJoCo + agent mode)
python run.py --sim-go2                # Go2 sim + MuJoCo viewer + agent CLI
python run.py --sim-go2-headless       # Go2 sim headless
python run.py --sim-go2 --explore      # Go2 + nav stack launched externally via ROS2 proxy

# Go2 navigation stack (shell scripts, no agent)
./scripts/launch_vnav.sh               # Go2 + Vector Nav Stack + RViz (manual control)
./scripts/launch_explore.sh            # Go2 + VNav + TARE autonomous exploration + RViz
./scripts/launch_nav2.sh --rviz        # Go2 + Nav2 (AMCL + MPPI) alternative
./scripts/launch_slam.sh               # Go2 + SLAM real-time mapping

# SO-101 arm
python run.py                  # Real hardware + CLI
python run.py --sim            # MuJoCo sim + viewer + CLI
python run.py --sim-headless   # MuJoCo sim headless
python run.py --dashboard      # Textual TUI dashboard
python run.py --web --sim      # Web dashboard at localhost:8000

# Agent mode (LLM tool-calling -- works with both arm and Go2)
python run.py --agent                                     # Default: GPT-4o
python run.py --agent --agent-model anthropic/claude-sonnet-4-6  # Claude Sonnet
python run.py --sim --agent                               # Sim + agent mode
```

**Classic vs Agent mode**: Classic mode uses a rigid classify→plan→execute pipeline — fast for simple commands ("抓杯子") but can't handle multi-turn context. Agent mode uses LLM-native function calling — the LLM sees full conversation history, decides when to chat vs call tools, and understands context ("我饿了" → picks food proactively).

**Model selection**: Agent mode defaults to GPT-4o (most reliable tool calling + vision). Qwen2.5-VL-72B has better vision + Chinese but doesn't support tool calling on OpenRouter. GPT-5 Nano ($0.05/MTok) is cheapest but unstable (slow responses, frequent truncation, poor tool argument quality). Override with `--agent-model` or in `config/user.yaml` under `llm.models.agent`.

## REST API

When running with `--web`, these endpoints are available:

```bash
GET  /api/status              # Robot state (joints, gripper, objects)
GET  /api/skills              # Available skills with JSON schemas
GET  /api/world               # World model (objects + robot state)
GET  /api/camera              # Camera frame (JPEG)
POST /api/execute             # {"instruction": "pick the banana"}
POST /api/skill/{name}        # {"object_label": "banana", "mode": "drop"}
POST /api/run_goal            # {"goal": "clean the table", "max_iterations": 10}
```

Any agent framework (LangGraph, custom scripts, etc.) can control the robot via HTTP.

## Custom Skills

```python
from vector_os_nano.core.skill import skill, SkillContext
from vector_os_nano.core.types import SkillResult

@skill(aliases=["wave", "挥手", "打招呼"], direct=False, auto_steps=["wave"])
class WaveSkill:
    name = "wave"
    description = "Wave the arm as a greeting"
    parameters = {"times": {"type": "integer", "default": 3}}
    preconditions = []
    postconditions = []
    effects = {}

    def execute(self, params, context):
        for _ in range(params.get("times", 3)):
            joints = context.arm.get_joint_positions()
            joints[0] = 0.5
            context.arm.move_joints(joints, duration=0.5)
            joints[0] = -0.5
            context.arm.move_joints(joints, duration=0.5)
        return SkillResult(success=True)

agent.register_skill(WaveSkill())
agent.execute("wave")       # alias match → direct execute
agent.execute("挥手三次")    # alias match → LLM fills params
```

## Project Structure

```
vector_os_nano/
├── core/              SkillFlow, Agent, ToolAgent, Executor, WorldModel
├── llm/               LLM-agnostic providers (Claude, OpenAI, Ollama)
├── perception/        RealSense + Moondream VLM + EdgeTAM tracker
├── hardware/
│   ├── so101/         SO-101 arm driver (Feetech serial, Pinocchio IK)
│   └── sim/           MuJoCo simulation (arm, Go2, perception)
│       ├── mujoco_go2.py        Go2 physics (convex MPC, 1kHz)
│       └── go2_ros2_proxy.py    ROS2 topic proxy for nav stack integration
├── skills/
│   ├── go2/           walk, turn, stand, sit, explore, stop, where_am_i
│   ├── navigate.py    Room navigation (nav stack or dead-reckoning)
│   └── ...            pick, place, home, scan, detect, gripper, wave
├── cli/               Rich CLI + TUI dashboard
├── mcp/               MCP server (tools + resources)
├── web/               FastAPI + WebSocket dashboard
└── ros2/              Optional ROS2 integration

scripts/
├── go2_vnav_bridge.py     MuJoCo <-> ROS2 bridge (state_estimation, registered_scan, TF)
├── launch_vnav.sh         Full nav stack launch (bridge + localPlanner + FAR + terrain + RViz)
├── launch_explore.sh      Nav stack + TARE autonomous exploration
├── launch_nav_only.sh     Nav stack nodes only (bridge managed externally by run.py)
├── launch_nav2.sh         Nav2 alternative (AMCL + MPPI)
├── launch_slam.sh         SLAM real-time mapping
└── go2_demo.py            Visual locomotion demo (standalone)
```

## MCP Server -- Claude Code Controls the Robot

Vector OS Nano exposes all skills via the **Model Context Protocol (MCP)**. Claude Code connects directly and controls the robot -- sim or real hardware -- through natural language.

```bash
# Auto-connects when Claude Code starts (configured in .mcp.json)
# Or manual:
python -m vector_os_nano.mcp --sim --stdio        # sim + stdio (for .mcp.json)
python -m vector_os_nano.mcp --hardware --stdio    # real hardware + stdio
python -m vector_os_nano.mcp --sim                 # sim + SSE on :8100
```

**12 MCP tools**: pick, place, home, scan, detect, gripper_open, gripper_close, wave, natural_language, run_goal, diagnostics, debug_perception
**7 MCP resources**: world://state, world://objects, world://robot, camera://overhead, camera://front, camera://side, camera://live

<p align="center">
  <img src="images/mcp_claude.png" width="700" alt="Claude Code controlling robot via MCP">
  <br>
  <i>Claude Code operating the robot arm through MCP -- scan, detect, pick, place via natural conversation.</i>
</p>

### Autonomous Skill Generation

Claude Agent can autonomously design, implement, and test new skills with full reasoning -- then register and execute them immediately.

<p align="center">
  <img src="images/skillgen.png" width="700" alt="Claude Agent generating skills autonomously">
  <br>
  <i>Claude Agent designing a custom wave skill with full reasoning, code generation, and live execution.</i>
</p>

## Navigation Stack Dependencies

The Go2 navigation features require the [Vector Navigation Stack](https://github.com/VectorRobotics/vector_navigation_stack) (CMU/Ji Zhang group):

```bash
# 1. ROS2 Jazzy
sudo apt install ros-jazzy-desktop

# 2. Build the navigation stack
cd ~/Desktop
git clone https://github.com/VectorRobotics/vector_navigation_stack.git  # see repo for details
cd vector_navigation_stack
colcon build

# 3. Build TARE planner (autonomous exploration)
colcon build --packages-select tare_planner

# 4. Install Go2 MPC controller
cd ~/Desktop
git clone https://github.com/VectorRobotics/go2-convex-mpc.git
cd go2-convex-mpc
pip install -e .
```

Without the navigation stack, Go2 basic skills (walk, turn, stand, navigate via dead-reckoning) still work. The nav stack adds: terrain-aware obstacle avoidance, global path planning (FAR planner), and autonomous exploration (TARE planner).

## What's Coming

The full Vector OS stack under development at **CMU Robotics Institute**:

- **Semantic Mapping** -- 3D scene graphs, object permanence, spatial reasoning
- **Multi-Robot Coordination** -- fleet management, task allocation, shared world model
- **Mobile Manipulation** -- Go2 + arm integration, whole-body control
- **Humanoid Support** -- extending to bipedal platforms

**Star this repo and stay tuned.**

---

<details>
<summary><b>点击查看中文</b></summary>

## 什么是 Vector OS Nano？

一个 **Python SDK**，让任何机械臂拥有自然语言大脑。`pip install` 即可使用，不需要 ROS2。

```python
from vector_os_nano import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-...")
agent.execute("抓起红色杯子放到左边")
```

没有硬件？用 MuJoCo 仿真：

```bash
python run.py --sim
```

## SkillFlow 协议

所有命令路由通过 `@skill` 装饰器声明，零硬编码：

```python
@skill(aliases=["抓", "拿", "抓起"], auto_steps=["scan", "detect", "pick"])
class PickSkill: ...
```

简单命令（home, open, close）零 LLM 调用，常见模式（抓X）自动编排，复杂任务才用 LLM 规划。

## 快速开始

```bash
cd vector_os_nano
python3 -m venv .venv --prompt vector_os_nano
source .venv/bin/activate
pip install -e ".[all]"
python run.py --sim
```

</details>

---

## License

MIT License

---

<p align="center"><i>Built by Vector Robotics at CMU Robotics Institute with Claude Code</i></p>
