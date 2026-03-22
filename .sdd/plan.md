# Vector OS Nano — Web Dashboard Technical Plan

**Date:** 2026-03-22
**Prereq:** spec.md

---

## 1. Architecture

```
Browser (localhost:8000)
  │
  ├── GET /           → index.html (single-page app)
  ├── WS /ws/chat     → AI chat (bidirectional, async)
  └── WS /ws/status   → robot status (server push, 2Hz)
         │
    FastAPI (uvicorn, async)
         │
    ├── ChatManager    → conversation history + LLM calls
    ├── Agent bridge   → execute commands via Agent.execute()
    └── StatusBroadcaster → poll arm/objects, push to all WS clients
```

## 2. Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Web framework | FastAPI | Async, WebSocket native, minimal |
| ASGI server | uvicorn | Standard, fast |
| Frontend | Single HTML file | No build step, embedded CSS/JS |
| Styling | CSS custom properties + glassmorphism | Modern dark theme |
| Real-time | WebSocket | Bidirectional, low latency |
| LLM | httpx async | Reuse existing ClaudeProvider pattern |

## 3. File Structure

```
vector_os_nano/
└── web/
    ├── __init__.py
    ├── app.py          # FastAPI app, routes, WebSocket handlers
    ├── chat.py         # ChatManager: conversation + LLM calls
    └── static/
        └── index.html  # Single-file frontend (HTML + CSS + JS)
```

## 4. Implementation Modules

### Module A: web/app.py — FastAPI application
- Mount static files
- WebSocket /ws/chat endpoint
- WebSocket /ws/status endpoint (broadcast loop)
- Startup/shutdown lifecycle (create Agent, start status broadcaster)
- Reference to Agent instance for command execution

### Module B: web/chat.py — Chat + LLM manager
- Conversation history (list of {role, content})
- Async LLM call via httpx (reuse ClaudeProvider pattern but async)
- System prompt: "You are Vector OS Nano assistant controlling a robot arm..."
- Detect robot commands vs general chat
- Route commands to Agent.execute() in a thread pool

### Module C: web/static/index.html — Frontend
- Single HTML file with embedded CSS and JS
- Layout: 60% chat panel (left), 40% status panel (right)
- Chat: message bubbles, input box, auto-scroll
- Status: joint angles, gripper state, objects list, command history
- WebSocket connection management with auto-reconnect
- Dark theme with teal accent, glassmorphism cards
- Responsive (min-width 768px)

## 5. Key Design Decisions

### Chat + Command Routing
The AI decides if a message is a robot command or general chat:
1. User message → send to LLM with system prompt including available skills
2. If LLM returns a task plan (JSON with steps) → execute via Agent
3. If LLM returns plain text → display as chat response
4. During execution, stream status updates via WebSocket

### Async Architecture
- FastAPI runs in uvicorn's async event loop
- Agent.execute() is synchronous (blocks during arm motion) → run in thread pool
- Status broadcaster runs as background task, polls arm state every 500ms
- Multiple browser tabs supported (broadcast to all connected WS clients)

## 6. Dependencies

New: `fastapi`, `uvicorn[standard]` (2 packages)
Already available: `httpx` (for LLM calls)
