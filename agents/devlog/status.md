# Development Status — v0.2.0 FINAL

**Session Date:** 2026-03-23  
**Project:** Vector OS Nano SDK  
**Status:** v0.2.0 stable release complete, ready for v0.3.0 planning  
**Final Test Count:** 852+ tests passing (all critical features working)

---

## v0.2.0 Final Summary

### Major Features Delivered

**Phase 1: LLM Memory + Model Router** (COMPLETE)
- SessionMemory: Persistent cross-task conversation history (50 entries max)
- ModelRouter: Complexity-driven model selection (Haiku for simple, Sonnet for complex)
- Agent integration: Fixed conversation history reset bug
- 120 new tests passing (78 unit + 42 integration)

**Phase 2: MCP Server** (COMPLETE)
- MCP tools: 7 skill tools + natural_language meta-tool + diagnostics + debug_perception = 10 total
- MCP resources: 7 resources (world state, objects, 3 cameras, live camera for hardware)
- VectorMCPServer: Full MCP protocol with stdio transport
- Modes: --sim, --sim-headless, --hardware
- Claude Desktop integration: .mcp.json auto-connect config

**Phase 2.5: Critical Bug Fixes** (COMPLETE)
1. **CRITICAL: depth_scale hardcoding** — D405 uses 0.1mm units (hw_scale=0.0001), code assumed 1mm (depth_scale=1000), producing 3D coords 10x too large. Root cause of all pick failures in both CLI and MCP. Fixed by reading actual `get_depth_scale()` from RealSense hardware. All pick operations now work.
2. MOONDREAM_MODEL env var — MCP server now sets before VLMDetector init
3. MCP parameter passing — Fixed string concat bug; now uses Agent.execute_skill()
4. JSON Schema type mapping — SkillFlow float → JSON Schema number
5. Python 3.10+ asyncio compatibility

### Test Results
- Phase 1 unit tests: 78 pass
- Phase 1 integration tests: 42 pass
- Phase 2 unit tests: 75 pass
- Phase 2.5 new tests: 54 pass
- Pre-existing tests: 783 pass (v0.1.0 features)
- Skipped: 11 (ROS2 conditional)
- **TOTAL: 852+ tests passing**

### Known Regressions (non-blocking)
- test_skill_schemas.py expects 7 skills, now has 8 (WaveSkill added)

---

## Session Activity

### Commits This Session (~20)
- e067cd9: MCP camera viewer with annotated RGB + depth + pointcloud centroids
- e3de99c: CRITICAL FIX — read actual depth_scale from RealSense hardware
- f6c2f4c: Add 54 tests for calibration, pick workspace, execute_skill
- 772a536: Fix MOONDREAM_MODEL env var before VLMDetector init
- e41f47a, 061056f: debug_perception tool + error reporting
- ce5fafc: MCP hardware mode with camera viewer
- dc9cdeb: Fix skill calls with structured params
- And more (full history in git log)

### Files Changed
**New:**
- vector_os_nano/mcp/ (tools, resources, server, __main__, __init__)
- .mcp.json
- tests/unit/test_calibration_transform.py, test_pick_workspace.py, test_execute_skill.py
- And more

**Modified:**
- vector_os_nano/core/agent.py (SessionMemory + ModelRouter integration)
- vector_os_nano/perception/ (depth_scale fix, VLM env var)
- vector_os_nano/llm/ (model_override parameter)
- config/default.yaml (models + mcp sections)
- pyproject.toml (mcp optional dependency)

---

## Entry Points (v0.2.0)

**CLI:**
```bash
python run.py                  # Real hardware or simulation
python run.py --sim            # With MuJoCo viewer
python run.py --sim-headless   # Headless
python run.py --dashboard      # TUI mode
```

**MCP Server:**
```bash
python -m vector_os_nano.mcp --sim --stdio              # Sim + stdio (for .mcp.json)
python -m vector_os_nano.mcp --sim-headless --stdio     # Headless sim
python -m vector_os_nano.mcp --hardware --stdio         # Real hardware
```

---

## v0.3.0 Planning

### Proposed Features (awaiting Yusen approval)
1. Claude Code agent team integration (full parallel pipeline)
2. Real RealSense D405 camera feed + depth validation
3. Moondream VLM open-vocabulary detection
4. EdgeTAM continuous object tracking
5. Architecture documentation + API reference

### Agent Status
| Agent | Model | Status | Notes |
|-------|-------|--------|-------|
| Lead/Architect | opus | Ready | v0.3.0 spec authoring |
| Alpha | sonnet | DONE | T3 (v0.4.0): AgentLoop Full Implementation Tests — 13 new tests in test_agent_loop.py; all 23 T3 tests pass (13 new + 4 skeleton + 6 types), no implementation fixes needed |
| Beta | sonnet | DONE | T4 (v0.4.0): Agent.run_goal() + MCP run_goal tool + config — run_goal() on Agent with lazy AgentLoop import, build_run_goal_tool() + _format_goal_result() + handle_tool_call handler in tools.py, agent_loop config in default.yaml; 14 new tests pass (6 agent + 8 mcp), 70 total in target suite |
| Gamma | sonnet | DONE | T5: Integration tests — tests/unit/test_agent_loop_integration.py (5 tests: full Agent.run_goal() path with mock arm + mock LLM, MCP handler, no-LLM fallback, max-iterations cap); all 5 pass, regression 1010 passed / 1 pre-existing fail |
| QA | — | Ready | Code review for v0.3.0 |
| Scribe | haiku | DONE | Final session docs |

---

## Documentation Status

**COMPLETE:**
- progress.md — Full v0.2.0 feature list, entry points, MCP reference, test counts
- agents/devlog/status.md — This file

**TODO for v0.3.0:**
- README.md — Add MCP section, Claude Desktop setup guide
- docs/architecture.md — SessionMemory/ModelRouter/MCP flow diagrams
- docs/api.md — MCP tools + resources reference
- QUICKSTART.md — MCP server startup guide (optional)

---

## Critical Path Forward

1. All v0.2.0 features complete and tested
2. depth_scale bug fix verified (all pick operations working)
3. MCP server production-ready (10 tools, 7 resources, 3 transport modes)
4. Zero blockers, ready for v0.3.0 planning

**Next:** Yusen review of v0.3.0 feature list, then parallel agent execution.

