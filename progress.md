# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-23  
**Current version:** v0.2.0 (stable)  
**In development:** v0.3.0 — Claude Code integration + enhanced perception

## v0.2.0 — COMPLETE (2026-03-23)

### Phase 1: LLM Memory + Model Router — DONE
- [x] `vector_os_nano/core/memory.py` — SessionMemory class (50-entry bounded history)
  - MemoryEntry frozen dataclass (role, content, timestamp, entry_type, metadata)
  - add_user_message / add_assistant_message / add_task_result
  - get_llm_history(max_turns) — formats for Claude API
  - get_last_task_context() — anaphora resolution
  - 44 unit tests (tests/unit/test_memory.py)

- [x] `vector_os_nano/llm/router.py` — ModelRouter class (complexity-driven model selection)
  - ModelSelection frozen dataclass (model, reason)
  - for_classify / for_plan / for_chat / for_summarize methods
  - Complexity scoring: spatial words, multi-action patterns (EN + ZH)
  - Defaults to safe model on missing config
  - 34 unit tests (tests/unit/test_router.py)

- [x] `vector_os_nano/core/agent.py` — Integrated SessionMemory + ModelRouter
  - Replaced `_conversation_history` list with `SessionMemory(max_entries=50)`
  - Added `ModelRouter(config)` for per-stage model selection
  - Fixed bug: task history no longer resets on each command
  - All 6 LLM call sites pass `model_override=` from router
  - `add_task_result()` called after execution for anaphora context
  - 42 integration tests pass

- [x] LLM provider updates (`llm/claude.py`, `llm/base.py`, `llm/openai_compat.py`)
  - Added `model_override` parameter to all LLM call methods
  - Backwards compatible (optional, defaults to config)

- [x] Config updates (`config/default.yaml`)
  - New `models` section: classify, plan, chat, summarize (defaults: haiku, haiku, haiku, haiku)
  - New `mcp` section with optional server config

### Phase 2: MCP Server — DONE
- [x] `vector_os_nano/mcp/__init__.py` — Package init
- [x] `vector_os_nano/mcp/tools.py` — Skill-to-MCP tool conversion (7 tools: pick, place, home, etc.)
  - SkillRegistry integration
  - handle_tool_call for execution
  - Full parameter validation
  - 34 unit tests (tests/unit/test_mcp_tools.py)

- [x] `vector_os_nano/mcp/resources.py` — World state + camera resources (6 resources)
  - world://state, world://objects (JSON)
  - camera://overhead, camera://left, camera://right (PNG)
  - BGR→RGB conversion, PIL/cv2 fallback
  - 20 unit tests (tests/unit/test_mcp_resources.py)

- [x] `vector_os_nano/mcp/server.py` — VectorMCPServer + create_sim_agent
  - Wire tools + resources via Server.set_tool_handlers / .set_resource_handlers
  - create_sim_agent() mirrors run.py _init_sim
  - stdio entry point via stdio_server()
  - 21 unit tests (tests/unit/test_mcp_server.py)

- [x] `vector_os_nano/mcp/__main__.py` — `python -m vector_os_nano.mcp` entry point

- [x] `pyproject.toml` — MCP optional dependency + console script
  - `mcp>=1.0` optional dependency
  - `vector-os-mcp` console script (stdio server)

### Test Results v0.2.0
- Phase 1 unit tests: 78 pass (memory + router)
- Phase 1 integration tests: 42 pass (agent cross-task memory)
- Phase 2 unit tests: 75 pass (tools + resources + server)
- Pre-existing passing tests: 671 pass (v0.1.0 features)
- Pre-existing skipped: 10 skip (ROS2 conditional)
- **Total: 866 tests passing**

---

## v0.1.0 — Stable

Complete and working:
- Full NL pipeline: "抓杯子" → classify → plan → execute → summarize
- Multi-stage Agent Pipeline: MATCH → CLASSIFY → PLAN → EXECUTE → ADAPT → SUMMARIZE
- AI Chat (V): multi-turn conversation with Claude Haiku, context-aware
- SkillFlow protocol: declarative @skill decorator routing
- MuJoCo simulation: SO-101 with 13 STL meshes, 6 graspable objects
- Simulated perception: ground-truth object detection, Chinese/English NL queries
- Web Dashboard: localhost:8000, real-time WebSocket chat
- Direct commands (zero LLM): home, scan, open, close
- 671 unit tests + 61 integration tests passing
- ROS2 integration layer (optional, 5 nodes)
- Textual TUI dashboard (5 tabs)
- SO-101 hardware driver (Feetech STS3215 serial + Pinocchio IK)

---

## Architecture

### Multi-Stage Agent Pipeline

```
User Input
    |
    v
[Stage 1: MATCH]   — @skill alias matching (zero LLM)
[Stage 2: CLASSIFY] — Haiku intent detection (chat/task/query)
[Stage 3: PLAN]     — ModelRouter selects Haiku/Sonnet
[Stage 4: EXECUTE]  — Deterministic step execution
[Stage 5: ADAPT]    — Retry/explain on failure
[Stage 6: SUMMARIZE] — Haiku result report
```

### SkillFlow Protocol

All command routing via `@skill` decorator:

```python
@skill(aliases=["grab", "抓"], auto_steps=["scan", "detect", "pick"])
class PickSkill: ...

@skill(aliases=["close", "夹紧"], direct=True)
class GripperCloseSkill: ...
```

### System Layers

```
vector_os_nano/
├── core/           Agent, Executor, WorldModel, SessionMemory, Skill protocol
├── llm/            Claude/OpenAI providers, ModelRouter, prompts
├── perception/     RealSense, Moondream VLM, EdgeTAM tracker
├── hardware/
│   ├── so101/      SO-101 arm driver (Feetech serial, Pinocchio)
│   └── sim/        MuJoCo simulation
├── skills/         pick, place, home, scan, detect, gripper
├── cli/            Interactive CLI with Rich + prompt_toolkit
├── web/            FastAPI + WebSocket dashboard
├── mcp/            MCP tools + resources (v0.2.0)
└── ros2/           Optional ROS2 nodes
```

---

## Launcher Commands

Real hardware:
```bash
python run.py                  # CLI mode
python run.py --dashboard      # TUI dashboard
python run.py -v               # Verbose
```

MuJoCo simulation:
```bash
python run.py --sim            # With viewer
python run.py --sim-headless   # Headless
python run.py --sim -d         # With TUI
python run.py --web --sim      # Web dashboard
```

MCP server (Claude Desktop integration):
```bash
vector-os-mcp                  # Stdio server
python run.py --mcp            # Stdio server from launcher
```

Testing:
```bash
python run.py --no-arm         # No hardware
python run.py --no-perception  # No camera
```

---

## CLI Commands

```
vector> 你好                    # Chat
vector> 桌上有什么              # Query + detection
vector> 抓杯子                  # Task (LLM plan)
vector> home                    # Direct command
vector> open / close            # Gripper
vector> scan / detect           # Move / perceive
vector> status                  # Show state
vector> world                   # Show objects
vector> help / q                # Help / quit
```

---

## MuJoCo Simulation

- SO-101 arm: 13 real STL meshes from CAD
- 6 graspable objects: banana, mug, bottle, screwdriver, duck, lego
- Weld-constraint grasping (reliable)
- Smooth real-time motion + 60fps viewer sync
- Simulated perception: ground-truth positions
- Jacobian IK solver (< 2mm accuracy)
- Camera rendering for MCP resources

---

## AI Agent (V)

- Name: V, calls user "主人"
- System prompt: config/agent.md
- SessionMemory: cross-task conversation continuity (50 entries max)
- ModelRouter: auto-select Haiku (simple) vs Sonnet (complex tasks)
- Context-aware: knows robot mode, gripper state, visible objects
- Task planning: decomposes to skill sequences
- Post-execution summarization
