# Vector OS Nano — Web Dashboard + AI Chat Specification

**Version:** 0.1.0
**Date:** 2026-03-22
**Author:** Lead Architect (Opus)
**Status:** DRAFT

---

## 1. Overview

A beautiful localhost web dashboard for Vector OS Nano with real-time AI chat, robot control, and status monitoring. Users interact with the robot via natural language in a modern chat interface, see live robot status, and control the arm — all through a browser.

---

## 2. Background & Motivation

### Problem
- The current CLI (readline) is functional but plain
- The Textual TUI dashboard is terminal-limited — no images, no rich chat, no modern styling
- Users expect modern, polished interfaces (reference: DIMOS, RobotStudio, Isaac Sim)
- No persistent AI conversation — each CLI command is stateless

### Solution
A localhost web dashboard that provides:
- Professional dark-theme UI with glassmorphism styling
- Real-time AI chat with Claude Haiku (multi-turn conversation)
- Live robot status (joint angles, gripper, objects)
- Command execution with visual feedback
- Works with both real hardware and MuJoCo simulation

---

## 3. Goals

### MUST (v0.1)
- M1: Localhost web server (`python run.py --web`) on port 8000
- M2: Real-time AI chat panel with Claude Haiku via OpenRouter
- M3: Multi-turn conversation memory within a session
- M4: Robot command execution via chat ("抓杯子" triggers scan→detect→pick)
- M5: Live robot status panel (joint angles, gripper state, arm position)
- M6: Object list panel showing detected/known objects
- M7: Command history with execution results (success/fail, timing)
- M8: Beautiful dark theme inspired by DIMOS aesthetic
- M9: Chinese + English chat support
- M10: Works with MuJoCo sim and real hardware

### SHOULD (v0.2)
- S1: MuJoCo camera render displayed in dashboard (live sim view)
- S2: 3D joint visualization (Three.js or canvas)
- S3: Settings panel (LLM model, API key, sim config)
- S4: Session persistence (chat history saved/loaded)
- S5: Mobile-responsive layout

### MAY (future)
- F1: Multi-user support
- F2: Remote control (not just localhost)
- F3: Voice input (microphone → ASR → chat)
- F4: Camera feed display (RealSense live view)

---

## 4. Non-Goals

- NG1: NOT replacing the CLI or Textual dashboard — this is a third interface option
- NG2: NOT a cloud service — localhost only, no authentication
- NG3: NOT a general-purpose web framework — single-purpose robotics dashboard
- NG4: NOT real-time control (< 10ms) — chat-based, human-speed interaction

---

## 5. User Scenarios

### Scenario 1: First-time user with MuJoCo sim
- **Actor**: Developer without hardware
- **Trigger**: `python run.py --web --sim`
- **Expected Behavior**:
  1. Browser opens to `http://localhost:8000`
  2. Dashboard shows: chat panel (left), status panel (right)
  3. Status shows: sim mode, 6 objects on table, arm at home
  4. User types: "抓杯子"
  5. Chat shows AI thinking → plan → execution steps in real-time
  6. Status panel updates: joint angles change, gripper opens/closes
  7. AI responds: "已成功抓取杯子并放置到侧面"

### Scenario 2: Multi-turn conversation
- **Actor**: User exploring capabilities
- **Trigger**: Sequential chat messages
- **Expected Behavior**:
  1. User: "桌上有什么？"
  2. AI: "桌上有6个物体：香蕉、杯子、瓶子、螺丝刀、鸭子、乐高积木"
  3. User: "把红色的抓起来"
  4. AI remembers context → picks the mug (red)
  5. User: "现在抓蓝色的"
  6. AI: picks the bottle (blue)

### Scenario 3: Real hardware monitoring
- **Actor**: User with SO-101 arm
- **Trigger**: `python run.py --web`
- **Expected Behavior**:
  1. Dashboard shows real arm status (joint angles from serial)
  2. Camera feed from RealSense (if available)
  3. Chat commands execute on real hardware

---

## 6. Technical Constraints

- **Python**: 3.10+
- **Web framework**: FastAPI (async, WebSocket, lightweight)
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
- **Real-time**: WebSocket for chat + status updates
- **LLM**: Claude Haiku via OpenRouter (existing config)
- **Dependencies**: fastapi, uvicorn, websockets (pip-installable)
- **Port**: 8000 (configurable)
- **Browser**: Modern browsers (Chrome, Firefox, Safari, Edge)

---

## 7. Interface Definitions

### HTTP Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve dashboard HTML |
| GET | `/api/status` | Robot status JSON |
| GET | `/api/objects` | Detected objects list |
| GET | `/api/history` | Command execution history |
| POST | `/api/command` | Execute a robot command |

### WebSocket Endpoints

| Path | Purpose | Message Format |
|------|---------|---------------|
| `/ws/chat` | AI chat (bidirectional) | `{type, content, role}` |
| `/ws/status` | Live status updates (server→client) | `{joints, gripper, objects, mode}` |

### WebSocket Chat Protocol

```json
// Client → Server
{"type": "message", "content": "抓杯子"}

// Server → Client (AI response, streamed)
{"type": "thinking", "content": "正在规划..."}
{"type": "plan", "content": "scan → detect → pick"}
{"type": "executing", "step": "scan", "status": "running"}
{"type": "executing", "step": "pick", "status": "complete"}
{"type": "response", "content": "已成功抓取杯子"}

// Server → Client (error)
{"type": "error", "content": "IK failed for target position"}
```

### WebSocket Status Protocol

```json
// Server → Client (pushed every 500ms)
{
  "type": "status",
  "mode": "sim",
  "arm": {
    "connected": true,
    "joints": [0.0, -0.5, 0.3, 0.2, 0.0],
    "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    "gripper": "closed"
  },
  "objects": [
    {"name": "mug", "position": [0.22, 0.05, 0.018], "color": "red"},
    {"name": "banana", "position": [0.11, 0.12, 0.023], "color": "yellow"}
  ],
  "last_command": {"text": "抓杯子", "status": "success", "duration": 16.4}
}
```

---

## 8. UI Layout

```
+------------------------------------------------------------------+
|  VECTOR OS NANO                              [SIM] [Connected]    |
+------------------------------------------------------------------+
|                          |                                        |
|     CHAT PANEL           |         STATUS PANEL                   |
|                          |                                        |
|  [AI] Welcome! I can     |   Mode: MuJoCo Simulation              |
|  control the robot arm.  |   Arm: Connected                       |
|                          |   ┌─────────────────────┐              |
|  [You] 抓杯子            |   │  Joint Angles        │              |
|                          |   │  shoulder_pan:  0.00  │              |
|  [AI] Planning...        |   │  shoulder_lift: -0.50 │              |
|  scan → detect → pick    |   │  elbow_flex:    0.30  │              |
|                          |   │  wrist_flex:    0.20  │              |
|  [AI] ✓ Pick complete!   |   │  wrist_roll:    0.00  │              |
|  Grasped mug at          |   │  gripper:     closed  │              |
|  (22.0, 5.0) cm          |   └─────────────────────┘              |
|                          |                                        |
|                          |   ┌─────────────────────┐              |
|                          |   │  Objects on Table     │              |
|                          |   │  ● banana    (0.11,0.12)│           |
|                          |   │  ● mug       (0.22,0.05)│           |
|                          |   │  ● bottle    (0.30,0.12)│           |
|                          |   │  ● screwdriver(0.30,-0.10)│         |
|                          |   │  ● duck      (0.21,-0.16)│          |
|                          |   │  ● lego      (0.24,-0.08)│          |
|                          |   └─────────────────────┘              |
|                          |                                        |
| +---------------------+  |   ┌─────────────────────┐              |
| | Type a message...   |  |   │  Command History     │              |
| +---------------------+  |   │  ✓ 抓杯子   16.4s    │              |
|                          |   │  ✓ scan      3.0s    │              |
+------------------------------------------------------------------+
```

---

## 9. Visual Design

### Color Palette (Dark Theme)
- Background: `#0a0a0f` (near-black)
- Surface: `#12121a` (dark card)
- Surface hover: `#1a1a2e`
- Primary: `#00b4b4` (teal, matching Vector OS brand)
- Primary glow: `rgba(0, 180, 180, 0.15)`
- Success: `#00c853`
- Error: `#ff5252`
- Text primary: `#e0e0e0`
- Text secondary: `#888888`
- Border: `rgba(255, 255, 255, 0.06)`

### Styling
- Glassmorphism cards: `backdrop-filter: blur(10px); background: rgba(18,18,26,0.8)`
- Subtle border glow on focus
- Smooth transitions (200ms ease)
- Monospace font for technical data (joint angles, positions)
- Sans-serif for chat text
- Chat bubbles: user messages right-aligned (teal), AI messages left-aligned (dark)

---

## 10. Test Contracts

### Unit Tests
- [ ] `test_chat_message_parsing`: WebSocket message format validation
- [ ] `test_status_json_format`: Status endpoint returns valid schema
- [ ] `test_command_execution`: POST /api/command triggers Agent.execute
- [ ] `test_conversation_memory`: Multi-turn context preserved
- [ ] `test_llm_streaming`: Haiku response streams via WebSocket

### Integration Tests
- [ ] `test_websocket_chat_flow`: Client sends message, receives AI response
- [ ] `test_websocket_status_updates`: Client receives periodic status
- [ ] `test_sim_mode_web`: --web --sim launches both MuJoCo and web server
- [ ] `test_command_via_chat`: Chat message "home" executes home skill

### System Tests
- [ ] `test_full_pick_via_web`: Chat "抓杯子" completes pick in sim

---

## 11. Acceptance Criteria

- [ ] AC1: `python run.py --web --sim` opens browser to localhost:8000
- [ ] AC2: Chat with AI in Chinese and English, receives meaningful responses
- [ ] AC3: Robot commands execute via chat ("抓杯子" triggers pick pipeline)
- [ ] AC4: Multi-turn conversation works (AI remembers context)
- [ ] AC5: Status panel shows live joint angles updated every 500ms
- [ ] AC6: Objects panel shows all scene objects with positions
- [ ] AC7: Dark theme with teal accent, visually polished
- [ ] AC8: Works with MuJoCo sim (--sim) and without (real hardware)
- [ ] AC9: Clean shutdown on Ctrl+C or browser close
- [ ] AC10: < 3 new pip dependencies added

---

## 12. Open Questions

1. **Streaming vs batch LLM response**: Stream tokens via WebSocket for typewriter effect, or wait for full response?
   → Recommend streaming for better UX
2. **Auto-open browser**: Should `--web` auto-open the browser?
   → Yes, with `webbrowser.open()`
3. **Concurrent commands**: Allow multiple commands or queue them?
   → Queue (one at a time), show "busy" indicator
