# Architecture Plan: LLM Enhancements + MCP Server

Status: proposed
Date: 2026-03-23
Author: Lead Architect (Opus)

---

## Executive Summary

Two features for Vector OS Nano SDK v0.2.0, both simulation-focused:

1. **LLM Memory + Model Routing** -- Persistent cross-task conversation memory
   and automatic model selection (Haiku for classify, Sonnet for complex plans).
2. **MCP Server** -- Expose skills, world state, and camera renders via the
   Model Context Protocol so Claude Desktop (or any MCP client) can drive the
   simulated robot.

No new heavy dependencies. Feature 1 adds zero deps. Feature 2 adds one:
the official `mcp` Python package (~50 KB, pure Python).

---

## Feature 1: LLM Memory + Model Routing

### 1.1 Problem Analysis

The current agent has THREE separate conversation flows with broken continuity:

```
Chat mode:   _conversation_history persists (30-turn window)
Task mode:   _conversation_history RESETS to [user msg] on every task (line 419)
Query mode:  appends to _conversation_history
```

When a user says "pick the red cup" (task), then "now put it on the left" (task),
the second call hits `_handle_task` which does:

    self._conversation_history = [{"role": "user", "content": instruction}]

This destroys all prior context. The LLM planner has no idea what "it" refers to.

Additionally, the system uses a single model for all LLM calls:
- classify() uses the same model as plan()
- Haiku is fine for classify (single-word response)
- Haiku struggles with complex multi-object planning
- No way to specify different models for different stages

### 1.2 Design Decision: Conversation Memory

**Decision**: Replace the raw `_conversation_history` list with a `SessionMemory`
class that maintains a unified conversation log across all modes (chat, task, query)
and enriches it with structured task execution records.

**Alternatives considered**:

| Option | Pros | Cons |
|--------|------|------|
| A. Keep list, stop resetting | Minimal change | No structure, LLM sees raw noise |
| B. SessionMemory class | Structured, testable, bounded | New file (~150 lines) |
| C. Full vector DB/RAG | Rich retrieval | Way overkill, adds deps |

**Chosen**: Option B. A `SessionMemory` frozen-dataclass-based log that:
- Stores conversation turns AND task execution summaries
- Provides `get_context(max_turns)` that returns a compact context window
- Automatically summarizes completed tasks into single entries
- Has a bounded size (configurable, default 50 entries)

**Key insight**: The world model already tracks what happened (objects moved, gripper
state changed). SessionMemory just needs to record WHICH instruction caused which
world model change, so the LLM can resolve anaphora ("it", "that one", "the same").

### 1.3 Design Decision: Model Routing

**Decision**: Add a `ModelRouter` that maps pipeline stages to model identifiers.

**Alternatives considered**:

| Option | Pros | Cons |
|--------|------|------|
| A. Config-only (model per stage in YAML) | Simple | No runtime adaptation |
| B. ModelRouter with complexity scoring | Adapts to task | More code, heuristic |
| C. Always use best model | Simple | Expensive for simple tasks |

**Chosen**: Option A with a lightweight complexity heuristic. The router reads
model assignments from config and the agent can override for a specific call:

```yaml
llm:
  models:
    classify: "anthropic/claude-haiku-4-5"    # fast, cheap
    plan_simple: "anthropic/claude-haiku-4-5"  # 1-2 step tasks
    plan_complex: "anthropic/claude-sonnet-4-6" # 3+ step or multi-object
    chat: "anthropic/claude-haiku-4-5"
    summarize: "anthropic/claude-haiku-4-5"
  # Fallback: use this if a stage-specific model is not configured
  model: "anthropic/claude-haiku-4-5"
```

The "simple vs complex" split is determined by a lightweight classifier:
- Single object + single action = simple
- Multiple objects, spatial reasoning, or multi-step = complex

This runs AFTER classify (which already exists) and BEFORE plan. It adds ~zero
latency since it is a local heuristic, not an LLM call.

### 1.4 Interface Definitions

#### SessionMemory (new file: `core/memory.py`, ~200 lines)

```python
@dataclass(frozen=True)
class MemoryEntry:
    """Single entry in session memory."""
    role: str                    # "user" | "assistant" | "system" | "task_result"
    content: str                 # Text content
    timestamp: float             # time.time()
    entry_type: str = "chat"     # "chat" | "task" | "query" | "task_result"
    metadata: dict = field(default_factory=dict)
    # metadata examples:
    #   task_result: {"skill": "pick", "object": "mug", "success": True, "world_diff": {...}}
    #   task: {"intent": "task", "instruction": "pick the red cup"}

class SessionMemory:
    """Bounded, structured conversation memory across all agent modes."""

    def __init__(self, max_entries: int = 50) -> None: ...

    def add_user_message(self, content: str, entry_type: str = "chat") -> None: ...
    def add_assistant_message(self, content: str, entry_type: str = "chat") -> None: ...
    def add_task_result(self, instruction: str, result: ExecutionResult,
                        world_diff: dict) -> None: ...

    def get_llm_history(self, max_turns: int = 20) -> list[dict[str, str]]:
        """Return conversation history formatted for LLM API.

        Task results are condensed into single assistant messages like:
        "I picked the mug and placed it on the left. The mug is now at (0.1, -0.2)."

        Returns list of {"role": "user"|"assistant", "content": str}.
        """
        ...

    def get_last_task_context(self) -> dict | None:
        """Return metadata of the most recent task execution.

        Used by the planner to resolve references like "it", "that one".
        Returns None if no task has been executed yet.
        """
        ...

    def clear(self) -> None: ...

    @property
    def entries(self) -> list[MemoryEntry]: ...
```

#### ModelRouter (new file: `llm/router.py`, ~100 lines)

```python
@dataclass(frozen=True)
class ModelSelection:
    """Which model to use for a given LLM call."""
    model: str
    reason: str  # For logging: "simple_task", "complex_task", "classify", etc.

class ModelRouter:
    """Select the appropriate LLM model for each pipeline stage."""

    def __init__(self, config: dict) -> None:
        """Read model assignments from config['llm']['models']."""
        ...

    def for_classify(self) -> ModelSelection: ...
    def for_plan(self, instruction: str, world_state: dict) -> ModelSelection: ...
    def for_chat(self) -> ModelSelection: ...
    def for_summarize(self) -> ModelSelection: ...

    @staticmethod
    def estimate_complexity(instruction: str, world_state: dict) -> str:
        """Heuristic complexity estimation: 'simple' or 'complex'.

        Complex if:
        - Multiple objects mentioned
        - Spatial reasoning words (left, right, front, behind)
        - Multi-action words (then, and then, after that)
        - World state has 4+ objects (planning space is larger)
        """
        ...
```

#### Changes to ClaudeProvider

```python
class ClaudeProvider:
    # NEW: accept model override per call
    def plan(self, goal, world_state, skill_schemas, history=None,
             model_override: str | None = None) -> TaskPlan:
        ...

    def classify(self, user_message: str,
                 model_override: str | None = None) -> str:
        ...

    def chat(self, user_message, system_prompt, history=None,
             model_override: str | None = None) -> str:
        ...

    def summarize(self, original_request, execution_trace,
                  model_override: str | None = None) -> str:
        ...
```

The `model_override` parameter overrides `self.model` for that single request.
This is backward-compatible: existing callers pass no override and get the
default model.

### 1.5 Data Flow: Cross-Task Memory

```
User: "pick the red cup"
  |
  v
Agent.execute("pick the red cup")
  |-- memory.add_user_message("pick the red cup", entry_type="task")
  |-- _handle_task(...)
  |     |-- router.for_plan(instruction, world_state) -> Haiku (simple task)
  |     |-- llm.plan(..., model_override="haiku") -> TaskPlan
  |     |-- executor.execute(plan) -> ExecutionResult(success=True)
  |     |-- memory.add_task_result("pick the red cup", result, world_diff)
  |     |   (condensed: "Picked the red cup. Gripper now empty, cup removed.")
  |     '-- return result
  |
User: "now put it on the left"
  |
  v
Agent.execute("now put it on the left")
  |-- memory.add_user_message("now put it on the left", entry_type="task")
  |-- _handle_task(...)
  |     |-- history = memory.get_llm_history(max_turns=20)
  |     |   [
  |     |     {"role": "user", "content": "pick the red cup"},
  |     |     {"role": "assistant", "content": "Picked the red cup successfully."},
  |     |     {"role": "user", "content": "now put it on the left"},
  |     |   ]
  |     |-- router.for_plan(...) -> Sonnet (spatial reasoning detected)
  |     |-- llm.plan("now put it on the left", world_state, ..., history=history)
  |     |   LLM sees prior context, resolves "it" = "red cup"
  |     '-- Plans: pick(red_cup, hold) -> place(left) -> home
```

### 1.6 Files to Create / Modify

| File | Action | Lines (est.) |
|------|--------|-------------|
| `core/memory.py` | **CREATE** | ~200 |
| `llm/router.py` | **CREATE** | ~120 |
| `core/agent.py` | MODIFY | ~30 lines changed |
| `llm/claude.py` | MODIFY | ~15 lines (model_override param) |
| `llm/base.py` | MODIFY | ~5 lines (model_override in Protocol) |
| `llm/openai_compat.py` | MODIFY | ~10 lines (model_override param) |
| `config/default.yaml` | MODIFY | ~8 lines (models section) |
| `tests/unit/test_memory.py` | **CREATE** | ~200 |
| `tests/unit/test_router.py` | **CREATE** | ~150 |

### 1.7 Integration Points

1. **Agent.__init__**: Create `self._memory = SessionMemory()` and
   `self._router = ModelRouter(self._config)`.
2. **Agent._handle_task**: Replace `self._conversation_history = [...]` with
   `self._memory.add_user_message(instruction, "task")` and pass
   `self._memory.get_llm_history()` to the planner. After execution, call
   `self._memory.add_task_result(...)`.
3. **Agent._handle_chat**: Replace direct history manipulation with
   `self._memory.add_user_message(...)` / `self._memory.add_assistant_message(...)`.
4. **Agent._handle_query**: Same pattern as chat.
5. **ClaudeProvider**: Each method gains `model_override` kwarg. Internal
   `_chat_completion` uses `model_override or self.model`.

### 1.8 Test Strategy

- **test_memory.py**: Unit tests for SessionMemory
  - add messages, verify get_llm_history output
  - task result condensation (verify structured summary)
  - bounded size (add 100 entries, verify max 50 kept)
  - get_last_task_context after 0, 1, 2 tasks
  - cross-mode continuity (chat -> task -> chat -> task)
  - MemoryEntry frozen dataclass validation

- **test_router.py**: Unit tests for ModelRouter
  - config parsing (with and without models section)
  - complexity estimation (simple vs complex inputs)
  - model selection for each stage
  - fallback to default model when stage not configured
  - edge cases: empty instruction, empty world state

- **test_agent.py** (modify existing integration test):
  - Cross-task memory: execute("pick mug"), then execute("put it left"),
    verify second plan has context from first

### 1.9 Implementation Order

```
Wave 1 (independent, parallelizable):
  [alpha] core/memory.py + tests/unit/test_memory.py
  [beta]  llm/router.py + tests/unit/test_router.py

Wave 2 (depends on Wave 1):
  [alpha] Modify llm/claude.py, llm/base.py, llm/openai_compat.py for model_override
  [beta]  Modify core/agent.py to use SessionMemory + ModelRouter
  [gamma] Modify config/default.yaml, update tests/integration/test_agent.py

Wave 3 (integration):
  [any]   End-to-end test: python run.py --sim, execute cross-task commands
```

---

## Feature 2: MCP Server

### 2.1 Problem Analysis

Skills are currently accessible only through:
- CLI (`run.py` interactive prompt)
- Web dashboard (FastAPI + WebSocket at :8000)
- Direct Python API (`agent.execute("pick mug")`)

There is no standard protocol for external AI agents to:
- Discover available skills and their parameters
- Invoke skills programmatically
- Query world state (objects, robot position)
- Get camera renders from simulation

The Model Context Protocol (MCP) is the emerging standard for this.
Claude Desktop, Cursor, and other AI tools support MCP natively.

### 2.2 Design Decision: MCP Transport

**Decision**: Use the official `mcp` Python SDK with SSE (Server-Sent Events)
transport over HTTP, reusing the existing FastAPI app.

**Alternatives considered**:

| Option | Pros | Cons |
|--------|------|------|
| A. stdio transport | Simple, works with Claude Desktop | Can't share with web app |
| B. SSE over FastAPI | Reuses existing server, HTTP-accessible | Slightly more code |
| C. Separate MCP server process | Clean isolation | Extra process, port, complexity |

**Chosen**: Option B. Mount the MCP SSE endpoint on the existing FastAPI app
at `/mcp`. This means `python run.py --sim --web` serves BOTH the web dashboard
AND the MCP endpoint on the same port (8000). Claude Desktop connects to
`http://localhost:8000/mcp`.

Fallback: Also support stdio transport for direct `claude_desktop_config.json`
integration (a separate entry point: `python -m vector_os_nano.mcp`).

### 2.3 Design Decision: MCP Tool Mapping

**Decision**: Map each skill in `SkillRegistry` to an MCP tool automatically,
plus add a `natural_language` meta-tool for free-form commands.

```
Skill Registry                    MCP Tools
--------------                    ---------
pick                         -->  pick(object_label, mode)
place                        -->  place(location)
home                         -->  home()
scan                         -->  scan()
detect                       -->  detect(query)
gripper_open                 -->  gripper_open()
gripper_close                -->  gripper_close()

Agent.execute()              -->  natural_language(instruction)
                                  ^ This is the key tool: user types anything
                                    and the full agent pipeline handles it
```

The `natural_language` tool is critical: it lets Claude Desktop users type
free-form commands like "pick up all the objects and sort them by color"
and the full agent pipeline (classify -> plan -> execute -> summarize) handles it.

Individual skill tools are exposed for programmatic use by other MCP clients
that want to bypass LLM planning (e.g., another agent that already has a plan).

### 2.4 Design Decision: MCP Resources

**Decision**: Expose three resource types:

| Resource URI | Type | Description |
|-------------|------|-------------|
| `world://state` | application/json | Full world model serialized |
| `world://objects` | application/json | Just the objects list |
| `world://robot` | application/json | Just the robot state |
| `camera://overhead` | image/png | MuJoCo overhead camera render |
| `camera://front` | image/png | MuJoCo front camera render |
| `camera://side` | image/png | MuJoCo side camera render |

Camera resources return base64-encoded PNG images. The MuJoCo renderer
already supports named cameras (`arm.render(camera_name="overhead")`).

### 2.5 Interface Definitions

#### MCP Server (new file: `mcp_server.py`, ~250 lines)

```python
"""MCP Server for Vector OS Nano.

Exposes robot skills as MCP tools and world/camera state as MCP resources.
Supports both SSE (over FastAPI) and stdio transports.
"""

class VectorMCPServer:
    """MCP server backed by a Vector OS Nano Agent instance."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._mcp = Server("vector-os-nano")
        self._register_tools()
        self._register_resources()

    def _register_tools(self) -> None:
        """Register all skills as MCP tools + the natural_language meta-tool."""
        ...

    def _register_resources(self) -> None:
        """Register world state and camera resources."""
        ...

    def mount_sse(self, app: FastAPI) -> None:
        """Mount SSE transport on an existing FastAPI app at /mcp."""
        ...

    async def run_stdio(self) -> None:
        """Run as stdio transport (for claude_desktop_config.json)."""
        ...
```

#### MCP Tool Definitions (derived from SkillRegistry.to_schemas())

```python
# Auto-generated from skill registry:
Tool(
    name="pick",
    description="Pick up an object from the workspace",
    inputSchema={
        "type": "object",
        "properties": {
            "object_label": {"type": "string", "description": "Name of the object to pick"},
            "mode": {"type": "string", "enum": ["drop", "hold"], "default": "drop"},
        },
        "required": ["object_label"],
    },
)

# Meta-tool:
Tool(
    name="natural_language",
    description="Execute a natural language robot command through the full agent pipeline. "
                "Use this for complex or multi-step instructions.",
    inputSchema={
        "type": "object",
        "properties": {
            "instruction": {"type": "string", "description": "Natural language command"},
        },
        "required": ["instruction"],
    },
)
```

#### MCP Resource Definitions

```python
Resource(
    uri="world://state",
    name="World State",
    description="Complete world model: objects, robot state, spatial relations",
    mimeType="application/json",
)

Resource(
    uri="camera://overhead",
    name="Overhead Camera",
    description="RGB image from the overhead MuJoCo camera (640x480 PNG)",
    mimeType="image/png",
)
```

### 2.6 Data Flow: Claude Desktop -> MCP -> Sim

```
Claude Desktop                    Vector OS Nano (localhost:8000)
--------------                    --------------------------------
User: "pick up the banana"
  |
  v
Claude selects tool:
  natural_language(instruction="pick up the banana")
  |
  | HTTP SSE POST /mcp
  v
VectorMCPServer._handle_tool_call("natural_language", {...})
  |
  v
agent.execute("pick up the banana")
  |-- classify -> "task"
  |-- plan -> [scan, detect, pick, home]
  |-- execute -> MuJoCo sim runs pick sequence
  |-- summarize -> "Picked the banana"
  v
MCP Response:
  {
    "content": [
      {"type": "text", "text": "Picked the banana successfully in 4.2s."},
      {"type": "text", "text": "Steps: scan(ok) -> detect(ok) -> pick(ok) -> home(ok)"}
    ]
  }
  |
  v
Claude reads resource:
  camera://overhead
  |
  | HTTP GET /mcp (resource read)
  v
VectorMCPServer._handle_resource_read("camera://overhead")
  |
  v
arm.render(camera_name="overhead") -> numpy array
  |-- encode as PNG -> base64
  v
MCP Response:
  {"contents": [{"uri": "camera://overhead", "mimeType": "image/png",
                 "blob": "iVBORw0KGgo..."}]}
```

### 2.7 Data Flow: MCP + Memory Interaction

When MCP is used, the same Agent instance is shared. This means:
- MCP tool calls go through `agent.execute()` which uses SessionMemory
- Conversation context persists across MCP calls
- Claude Desktop can say "pick the banana" then "now put it on the left"
  and the agent remembers what was picked (Feature 1 enables this)

```
MCP call 1: natural_language("pick the banana")
  -> SessionMemory records task + result
  -> World model updated

MCP call 2: natural_language("put it on the left")
  -> SessionMemory provides history to planner
  -> LLM resolves "it" = banana from memory
  -> Plans: pick(banana, hold) -> place(left) -> home
```

This is why Feature 1 (memory) should be built BEFORE Feature 2 (MCP).

### 2.8 Files to Create / Modify

| File | Action | Lines (est.) |
|------|--------|-------------|
| `mcp_server.py` | **CREATE** | ~250 |
| `mcp_tools.py` | **CREATE** | ~150 (skill-to-tool conversion) |
| `mcp_resources.py` | **CREATE** | ~120 (world state + camera resources) |
| `web/app.py` | MODIFY | ~10 lines (mount MCP SSE endpoint) |
| `run.py` | MODIFY | ~15 lines (--mcp flag, stdio entry point) |
| `pyproject.toml` | MODIFY | ~5 lines (add mcp optional dep) |
| `config/default.yaml` | MODIFY | ~3 lines (mcp section) |
| `tests/unit/test_mcp_server.py` | **CREATE** | ~200 |
| `tests/unit/test_mcp_tools.py` | **CREATE** | ~150 |
| `tests/unit/test_mcp_resources.py` | **CREATE** | ~120 |

Note: MCP files are placed in the package root (`vector_os_nano/mcp_server.py`)
rather than a subdirectory, because the MCP SDK server is a single-file pattern
and three files is not enough to justify a `mcp/` package.

**Correction**: To keep the package organized, create `vector_os_nano/mcp/`:

| File | Action | Lines (est.) |
|------|--------|-------------|
| `mcp/__init__.py` | **CREATE** | ~5 |
| `mcp/server.py` | **CREATE** | ~250 |
| `mcp/tools.py` | **CREATE** | ~150 |
| `mcp/resources.py` | **CREATE** | ~120 |

### 2.9 Claude Desktop Configuration

After Feature 2 is complete, users can add this to their
`~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vector-os-nano": {
      "command": "python",
      "args": ["-m", "vector_os_nano.mcp", "--sim"],
      "env": {
        "OPENROUTER_API_KEY": "sk-..."
      }
    }
  }
}
```

Or for SSE transport (when the web server is already running):

```json
{
  "mcpServers": {
    "vector-os-nano": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### 2.10 Config Changes

```yaml
# config/default.yaml additions:
mcp:
  enabled: true
  transport: "sse"    # "sse" (FastAPI mount) or "stdio"
  # Tool exposure: which skills to expose as individual MCP tools
  expose_individual_skills: true  # false = only natural_language tool
  # Resource exposure
  expose_cameras: true
  expose_world_state: true
```

### 2.11 Dependency Addition

```toml
# pyproject.toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.0",
]
```

The `mcp` package is the official Model Context Protocol SDK. It is pure Python,
lightweight (~50 KB), and has minimal transitive dependencies.

### 2.12 Test Strategy

- **test_mcp_tools.py**: Unit tests for skill-to-tool conversion
  - Convert each built-in skill to MCP Tool, verify schema
  - natural_language tool schema validation
  - Round-trip: skill schema -> MCP tool -> call -> verify params passed correctly
  - Edge cases: skill with no params, skill with optional params

- **test_mcp_resources.py**: Unit tests for resource handlers
  - world://state returns valid JSON matching WorldModel.to_dict()
  - world://objects returns just the objects list
  - camera://overhead returns PNG-encoded bytes (mock MuJoCoArm.render)
  - Unknown resource URI returns appropriate error

- **test_mcp_server.py**: Integration tests
  - Create server with mock Agent
  - List tools -> verify all skills + natural_language present
  - Call natural_language tool -> verify agent.execute() called
  - Call individual skill tool -> verify skill executed
  - Read world://state resource -> verify JSON response
  - Read camera resource -> verify PNG blob returned

### 2.13 Implementation Order

```
Wave 1 (independent, parallelizable -- AFTER Feature 1 Wave 2):
  [alpha] mcp/tools.py + tests/unit/test_mcp_tools.py
  [beta]  mcp/resources.py + tests/unit/test_mcp_resources.py

Wave 2 (depends on Wave 1):
  [gamma] mcp/server.py + tests/unit/test_mcp_server.py

Wave 3 (integration):
  [alpha] Modify web/app.py to mount MCP SSE endpoint
  [beta]  Modify run.py for --mcp flag and stdio entry point
  [gamma] Modify pyproject.toml and config/default.yaml

Wave 4 (manual validation):
  [any]   Test with Claude Desktop: configure, send commands, verify sim responds
```

---

## Combined Implementation Order

Both features should be built in this sequence because Feature 2 (MCP)
benefits from Feature 1 (memory) -- MCP clients get cross-task context.

```
Phase 1: Memory + Router (Feature 1)
  Wave 1.1: core/memory.py, llm/router.py (parallel)
  Wave 1.2: Modify llm providers for model_override (parallel)
  Wave 1.3: Modify core/agent.py to integrate memory + router

Phase 2: MCP Server (Feature 2)
  Wave 2.1: mcp/tools.py, mcp/resources.py (parallel)
  Wave 2.2: mcp/server.py
  Wave 2.3: Mount on FastAPI, add CLI flags, update config

Phase 3: Integration Testing
  Wave 3.1: End-to-end sim test (run.py --sim)
  Wave 3.2: Claude Desktop manual test
```

Estimated total: ~1,500 new lines of code + ~800 lines of tests.

---

## Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| MCP SDK API instability (pre-1.0?) | Medium | Pin version, wrap in our types |
| Memory bloat with long sessions | Low | Bounded to 50 entries, oldest dropped |
| Model routing heuristic wrong | Low | Conservative: defaults to same model |
| SSE transport not supported by Claude Desktop version | Medium | stdio fallback always works |
| Agent.execute() is synchronous, MCP is async | Medium | Use run_in_executor (already done in web/app.py) |
| Camera render slow in CI (no GPU) | Low | Mock in tests, skip render in CI |

---

## Open Questions for CEO/CTO Approval

1. **Model cost**: Sonnet is ~15x more expensive than Haiku per token. The router
   will use Sonnet only for complex tasks (estimated 20% of calls). Is this
   acceptable, or should Sonnet be opt-in only?

2. **MCP scope**: Should we expose ALL skills as individual MCP tools, or just
   the `natural_language` meta-tool? Individual tools give programmatic control
   but increase the tool list Claude Desktop sees (currently 7 skills + 1 meta).

3. **Session persistence**: Should SessionMemory survive across `run.py` restarts?
   Current design is in-memory only (lost on restart). Persistent storage would
   require file-based storage (YAML/JSON) -- easy to add later but increases scope.

---

## Appendix: File Tree After Implementation

```
vector_os_nano/
  core/
    agent.py          (modified: use SessionMemory + ModelRouter)
    memory.py          (NEW: SessionMemory, MemoryEntry)
    executor.py        (unchanged)
    skill.py           (unchanged)
    types.py           (unchanged)
    world_model.py     (unchanged)
  llm/
    base.py            (modified: model_override in Protocol)
    claude.py          (modified: model_override param)
    openai_compat.py   (modified: model_override param)
    local.py           (unchanged)
    prompts.py         (unchanged)
    router.py          (NEW: ModelRouter, ModelSelection)
  mcp/
    __init__.py        (NEW)
    server.py          (NEW: VectorMCPServer)
    tools.py           (NEW: skill-to-tool conversion)
    resources.py       (NEW: world state + camera resources)
  hardware/            (unchanged)
  web/
    app.py             (modified: mount MCP SSE endpoint)
  ...

tests/
  unit/
    test_memory.py     (NEW)
    test_router.py     (NEW)
    test_mcp_server.py (NEW)
    test_mcp_tools.py  (NEW)
    test_mcp_resources.py (NEW)
  ...

config/
  default.yaml         (modified: models section, mcp section)
```
