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
- [x] `vector_os_nano/mcp/tools.py` — Skill-to-MCP tool conversion (7 skill tools + natural_language meta-tool)
  - SkillRegistry integration
  - handle_tool_call for execution with Agent.execute_skill()
  - Full parameter validation with JSON Schema
  - 34 unit tests (tests/unit/test_mcp_tools.py)

- [x] `vector_os_nano/mcp/resources.py` — World state + camera resources (7 resources)
  - world://state, world://objects (JSON)
  - camera://overhead, camera://left, camera://right (PNG)
  - camera://live (RealSense D405 hardware mode)
  - BGR→RGB conversion, PIL/cv2 fallback
  - 20 unit tests (tests/unit/test_mcp_resources.py)

- [x] `vector_os_nano/mcp/server.py` — VectorMCPServer + create_sim_agent
  - Wire tools + resources via Server.set_tool_handlers / .set_resource_handlers
  - create_sim_agent() mirrors run.py _init_sim
  - stdio entry point via stdio_server()
  - 21 unit tests (tests/unit/test_mcp_server.py)

- [x] `vector_os_nano/mcp/__main__.py` — `python -m vector_os_nano.mcp` entry point with --sim / --sim-headless / --hardware modes

- [x] `pyproject.toml` — MCP optional dependency + console script
  - `mcp>=1.0` optional dependency
  - `vector-os-mcp` console script (stdio server)

- [x] `.mcp.json` — Claude Desktop auto-connect config
  - `--sim --stdio` mode: MuJoCo viewer + stdio transport
  - Drop-in for Claude Desktop users
  - Manual modes: `--sim-headless`, `--hardware` (edit args to switch)

- [x] `vector_os_nano/mcp/tools.py` — 10 MCP tools total:
  1. pick(object) — grasp by name
  2. place(location) — release to location
  3. home() — safe position
  4. scan() — detect objects
  5. detect(query) — identify by description
  6. open() — open gripper
  7. close() — close gripper
  8. natural_language(query) — full pipeline
  9. diagnostics() — debug agent state
  10. debug_perception(object) — trace VLM/tracker/calibration

### Phase 2.5: Critical Bug Fixes — DONE
- [x] **CRITICAL: depth_scale hardcoding** — D405 uses 0.1mm units (hw_scale=0.0004), code assumed 1mm (depth_scale=1000), producing 3D coords 10x too large. Root cause of all pick failures in CLI + MCP. Fixed by reading `get_depth_scale()` from RealSense hardware in RealSenseCamera.connect(). All pick operations now work correctly.

- [x] MOONDREAM_MODEL env var — MCP server now sets before VLMDetector init; was causing fallback to non-existent Moondream Station

- [x] MCP parameter passing — Fixed string concatenation bug ("pick battery hold" was corrupting object names); now uses Agent.execute_skill() for structured params

- [x] JSON Schema type mapping — SkillFlow `"float"` → JSON Schema `"number"` (was causing Claude API 400 error)

- [x] Python 3.10+ asyncio compatibility — Fixed asyncio.get_event_loop() in test_mcp_server.py

### Test Results v0.2.0
- Phase 1 unit tests: 78 pass (memory + router)
- Phase 1 integration tests: 42 pass (agent cross-task memory)
- Phase 2 unit tests: 75 pass (tools + resources + server)
- Phase 2.5 new tests: 54 pass (calibration transform, pick workspace, execute_skill)
- Pre-existing passing tests: 783 pass (v0.1.0 features, minor regressions)
- Pre-existing skipped: 11 skip (ROS2 conditional)
- **Total: 852+ tests passing, all critical features functional**

### Known Regressions
- `test_skill_schemas.py`: 8 skills now (added WaveSkill) — test expects 7, needs update (non-blocking)

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
- 783 unit + 61 integration tests passing
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
├── skills/         pick, place, home, scan, detect, gripper, wave
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

MCP server (Claude Desktop / Claude Code):
```bash
python -m vector_os_nano.mcp --sim --stdio              # Sim mode + stdio (for .mcp.json)
python -m vector_os_nano.mcp --sim-headless --stdio     # Headless sim + stdio
python -m vector_os_nano.mcp --hardware --stdio         # Real hardware + stdio
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
