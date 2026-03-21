<p align="center">
  <img src="images/vector.jpg" width="800" alt="Vector Robotics">
</p>

<h1 align="center">Vector OS Nano</h1>

<p align="center">
  <b>Zero-shot, natural language generalized grasping on a $150 robot arm.</b>
  <br>
  <b>No training. No fine-tuning. Just say what you want.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ROS2-Optional-blue?logo=ros&logoColor=white" alt="ROS2">
  <img src="https://img.shields.io/badge/Moondream2-VLM-purple" alt="Moondream2">
  <img src="https://img.shields.io/badge/EdgeTAM-Tracking-orange" alt="EdgeTAM">
  <img src="https://img.shields.io/badge/Claude_Haiku-LLM_Brain-blueviolet?logo=anthropic&logoColor=white" alt="Claude">
  <img src="https://img.shields.io/badge/Pinocchio-IK_Solver-yellow" alt="Pinocchio">
  <img src="https://img.shields.io/badge/Intel_RealSense-D405-0071C5?logo=intel&logoColor=white" alt="RealSense">
  <img src="https://img.shields.io/badge/LeRobot-SO--ARM100-black" alt="LeRobot">
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

A **Python SDK** that gives any robot arm a natural language brain. `pip install` and go. No ROS2 required.

```python
from vector_os_nano import Agent, SO101

arm = SO101(port="/dev/ttyACM0")
agent = Agent(arm=arm, llm_api_key="sk-...")
agent.execute("pick up the red cup")
```

**10 lines of code. From unboxing to natural language control.**

## System Architecture

```
User (natural language, Chinese/English)
  |
  v
+---------------------------------------------+
|              LLM Brain Layer                 |
|  Claude Haiku (via OpenRouter API)           |
|  - Intent parsing & task decomposition       |
|  - Tool calling / function execution         |
|  - Multi-step planning                       |
|  - Bilingual: Chinese + English              |
+---------------------------------------------+
|              Skill Layer                     |
|  - pick(object)     detect_all()             |
|  - home() / scan()  describe_scene()         |
|  - place(x,y,z)     track(object)            |
+---------------------------------------------+
|           Perception Layer                   |
|  - Moondream2 VLM (local, ~4GB GPU)         |
|  - EdgeTAM real-time tracking (20fps)        |
|  - D405 depth camera (640x480 RGB+D @30fps) |
|  - Workspace calibration (camera->base)      |
+---------------------------------------------+
|            Control Layer                     |
|  - Pinocchio FK/IK solver                    |
|  - Joint trajectory interpolation            |
|  - Gripper command with retry logic          |
|  - Dynamic position compensation             |
+---------------------------------------------+
|           Hardware Layer                     |
|  - SO-ARM100 (6-DOF, STS3215 servos)        |
|  - Intel RealSense D405 (USB 3.x)           |
|  - Total cost: ~$420                         |
+---------------------------------------------+
```

## Capabilities

| Capability | Status |
|-----------|--------|
| Zero-shot natural language grasping | Working |
| Real-time object tracking (20fps) | Working |
| Scene description via VLM | Working |
| Chinese + English commands | Working |
| LLM-powered task interpretation | Working |
| Workspace calibration (14-point) | Working |
| Auto-retry on pick failure | Working |
| Dynamic gripper compensation | Working |
| Place skill | Working |
| Multi-step task planning | Working |
| Textual TUI dashboard | Working |
| ROS2 integration (optional) | Working |
| PyBullet simulation | Working |

## Hardware (~$420 total)

| Component | Model | Cost |
|-----------|-------|------|
| Robot Arm | LeRobot SO-ARM100 (6-DOF, 3D-printed) | ~$150 |
| Camera | Intel RealSense D405 | ~$270 |
| GPU | Any NVIDIA with 8+ GB VRAM | (existing) |
| Computer | Ubuntu 22.04 / Windows / macOS | (existing) |

<p align="center">
  <img src="images/setup.png" width="600" alt="Hardware Setup">
</p>

## Quick Start

### 1. Create virtual environment

```bash
python3 -m venv vector_os_nano
source vector_os_nano/bin/activate
pip install --upgrade pip
```

### 2. Install SDK

```bash
# Core only (no GPU)
pip install -e "."

# Full (GPU perception + IK + TUI + simulation)
pip install -e ".[all]"

# GPU: RTX 5080/Blackwell requires nightly PyTorch
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Configure LLM

Create `config/user.yaml` (do NOT commit this file):
```yaml
llm:
  api_key: "your-openrouter-api-key"
  model: "anthropic/claude-haiku-4-5"
  api_base: "https://openrouter.ai/api/v1"
```
Get your key at: https://openrouter.ai/keys

### 4. Run

```bash
python run.py
```

### 5. Use

```
vector> pick battery              # Pick up a battery
vector> grab the red cup          # Natural language pick
vector> home                      # Return to home position
vector> scan                      # Move to scan position
vector> open                      # Open gripper (instant, no LLM)
vector> close                     # Close gripper (instant, no LLM)
vector> detect all objects        # Detect everything on table
vector> world                     # Show world model state
```

Chinese works too:
```
vector> 抓电池
vector> 看看桌上有什么
vector> 抓蛋白棒
```

## Custom Skills

```python
from vector_os_nano import Agent, SO101
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult

class WaveSkill:
    name = "wave"
    description = "Wave the arm back and forth"
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

agent = Agent(arm=SO101(port="/dev/ttyACM0"), skills=[WaveSkill()])
agent.execute("wave 5 times")  # LLM discovers and uses the skill
```

## ROS2 Mode (Optional)

ROS2 is **not required** for basic operation. For advanced features (lifecycle management, TF2, multi-robot):

```bash
python3 -m venv vector_os_nano --system-site-packages
source vector_os_nano/bin/activate
pip install -e ".[all]"
ros2 launch vector_os_nano nano.launch.py serial_port:=/dev/ttyACM0
```

| Service | Description |
|---------|-------------|
| `/skill/pick` | Pick an object |
| `/skill/place` | Place held object |
| `/skill/home` | Move to home |
| `/skill/scan` | Move to scan |
| `/skill/detect` | Detect objects |
| `/world_model/query` | Query world state |
| `/agent/execute` | Execute NL command |

## What's Coming

Vector OS Nano is a proof of value for the grasping module. The full Vector OS stack under development at **CMU Robotics Institute** includes:

- **SLAM + Navigation** -- LiDAR/visual SLAM, Nav2 integration, multi-floor mapping
- **Semantic Mapping** -- 3D scene graphs, object permanence, spatial reasoning
- **Multi-Robot Coordination** -- fleet management, task allocation, shared world model
- **Mobile Manipulation** -- wheeled, legged, and humanoid platforms
- **Explainable Planning** -- neuro-symbolic task decomposition with reasoning traces
- **Visual Servoing** -- sub-millimeter closed-loop precision manipulation
- **Multi-Modal HRI** -- voice, gesture, gaze-aware human-robot interaction

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
agent.execute("抓起红色杯子")
```

## 快速开始

```bash
python3 -m venv vector_os_nano
source vector_os_nano/bin/activate
pip install -e ".[all]"
python run.py
```

## 硬件（总计约 $420）

| 组件 | 型号 | 成本 |
|------|------|------|
| 机械臂 | LeRobot SO-ARM100 | 约 $150 |
| 相机 | Intel RealSense D405 | 约 $270 |
| GPU | 任意 NVIDIA 8GB+ 显存 | （已有） |

## 即将到来

**CMU 机器人研究所** 正在开发完整 Vector OS 栈：SLAM、导航、语义建图、多机协调、移动操作、可解释规划、视觉伺服。

**Star 这个仓库，敬请关注。**

</details>

---

## License

MIT License

---

<p align="center"><i>Built by Vector Robotics at CMU Robotics Institute with Claude Code</i></p>
