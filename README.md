<!-- ⚠️ STALE — under redesign (2026-06-18). Entry points below (--sim-go2, habitat apartment,
     deepseek-v4-flash) are outdated and the project direction is being restarted from scratch.
     See docs/REDESIGN-BRIEF.md and docs/agent-kernel-STATUS.md. Will be rewritten post-redesign. -->

<p align="center">
  <img src="images/ascii-art.png" width="800" alt="Vector Robotics">
</p>

<h1 align="center">Vector OS Nano</h1>

<p align="center">
  <b>Cross-embodiment robot OS: natural language controls everything.</b>
  <br>
  <b>No training. No fine-tuning. Just say what you want.</b>
  <br>
  <b>The agent decomposes any NL goal into a verified plan and executes long-chain tasks end-to-end.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MuJoCo-3.6-green" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Claude-LLM_Brain-blueviolet?logo=anthropic&logoColor=white" alt="Claude">
  <img src="https://img.shields.io/badge/ROS2_Jazzy-Navigation-blue?logo=ros&logoColor=white" alt="ROS2">
</p>

<p align="center">
  <i>Being developed at <b>CMU Robotics Institute</b>.</i>
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

## Quick Start

**Install** (requires Python 3.10+ and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/VectorRobotics/vector-os-nano.git
cd vector-os-nano
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

**Configure API key** — copy the example and add your key:

```bash
cp .env.example .env
# then open .env and fill in one provider key:
#   DEEPSEEK_API_KEY=sk-...      # default (deepseek-v4-flash)
#   OPENROUTER_API_KEY=sk-or-... # multi-model fallback
#   ANTHROPIC_API_KEY=sk-ant-... # direct Anthropic
```

`.env` is git-ignored. One provider key is enough. DeepSeek `deepseek-v4-flash` is the default; OpenRouter is the multi-model fallback.

**Run:**

```bash
vector-cli                  # interactive AI agent REPL
vector-cli --sim            # SO-101 arm in MuJoCo (natural language: "wave", "pick up the mug")
vector-cli --sim-go2        # Go2 quadruped in MuJoCo
```

---

## Vector CLI

AI-powered terminal: control robots, edit code, and diagnose issues from one prompt. Powered by the VGG (Verified Goal Graph) cognitive layer — complex goals are decomposed into verified sub-plans; simple commands skip LLM and execute in <1 ms.

```
vector> explore the house
  > [1/1] explore_goal done 62.3s

vector> the dog hits walls at corners
  file_read("scripts/go2_vnav_bridge.py") ... ok
  file_edit(old="_MAX_SPEED = 0.6", new="_MAX_SPEED = 0.4") ... ok
  skill_reload("walk") ... ok

vector> go to the kitchen
  > [1/1] navigate_goal done 11.3s
```

**Slash commands:** `/help` `/model` `/tools` `/status` `/login` `/compact` `/clear` `/copy` `/export`

<p align="center">
  <img src="images/agent.png" width="700" alt="vector-cli with Go2 simulation">
  <br>
  <i>vector-cli controlling Go2 in MuJoCo: natural language conversation with V (right), live simulation (left).</i>
</p>

---

## Arm Simulation (MuJoCo)

Drive the SO-101 arm by natural language — no hardware required:

```bash
vector-cli --sim
# "wave", "go home", "pick up the mug"
```

<p align="center">
  <img src="images/sim_setup.png" width="700" alt="MuJoCo arm simulation">
  <br>
  <i>MuJoCo simulation: SO-101 arm with graspable objects, CLI conversation with V.</i>
</p>

Manipulation (`scan → detect → top-down pick`) is validated in MuJoCo. Enable with `VECTOR_ENABLE_MANIPULATION=1`.

---

## MCP Server

Let Claude Code control the robot via [Model Context Protocol](https://modelcontextprotocol.io):

```bash
vector-os-mcp --sim --stdio      # MuJoCo sim, stdio transport (add to Claude Code MCP config)
vector-os-mcp --hardware --stdio # real hardware
vector-os-mcp --sim              # SSE on :8100
```

Exposes all skills as MCP tools plus camera and world-state resources (`world://state`, `world://objects`, `camera://live`, …).

<p align="center">
  <img src="images/mcp_claude.png" width="700" alt="Claude Code controlling robot via MCP">
  <br>
  <i>Claude Code operating the robot via MCP: terminal (top) + live camera RGB + depth (bottom).</i>
</p>

<p align="center">
  <img src="images/skillgen.png" width="700" alt="Autonomous skill generation">
  <br>
  <i>Claude Agent autonomously designing, implementing, and executing new robot skills.</i>
</p>

---

## Contributing

PRs welcome. By submitting a pull request you agree your contribution is licensed under Apache 2.0 (Section 5). Include the SPDX header on new files:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
```

---

## License

Copyright 2024-2026 Vector Robotics.

Licensed under the **Apache License, Version 2.0**; you may not use this file except in compliance with the License. You may obtain a copy at

> http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an **"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or implied. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for the full terms, including attribution and patent-grant terms, and a list of bundled third-party components.

### Trademarks

"Vector Robotics" and "Vector OS Nano" are trademarks of Vector Robotics. The Apache License does not grant permission to use these names except as required for describing the origin of the Work and reproducing the contents of the NOTICE file.

---

<p align="center">
  <i>CMU Robotics Institute. Star this repo and stay tuned.</i>
</p>
