# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-22
**Status:** v0.1.0 + MuJoCo sim + Web Dashboard — full NL pick-and-place, AI chat, web UI

## What Works

- Full NL pipeline: "抓杯子" → LLM → scan → detect → pick → place (rotate + drop) → home
- **Web Dashboard**: `python run.py --web` — localhost:8000, real-time AI chat with Claude Haiku, live robot status, dark glassmorphism theme, WebSocket bidirectional, multi-turn conversation
- **MuJoCo simulation**: `python run.py --sim-gui` — SO-101 arm with real STL meshes, 6 mesh objects (banana, mug, bottle, screwdriver, duck, lego), weld-based grasping, smooth real-time motion, pick-and-place with rotate-and-drop
- **Simulated perception**: ground-truth object detection from MuJoCo state, Chinese/English NL queries
- Direct commands without LLM: home, scan, open, close (instant, no API call)
- Chinese + English natural language
- Live camera viewer: RGB + depth side-by-side, EdgeTAM tracking overlay
- 733 unit tests passing
- ROS2 integration layer (optional, 5 nodes + launch file)
- Textual TUI dashboard (5 tabs, real-time status, camera preview)
- SO-101 arm driver (Feetech STS3215 serial)
- Calibration wizard (TUI + readline)

## Current Release: v0.1.0

### Unified Launcher: run.py

Single entry point with four modes:

1. **CLI Mode** (default): `python run.py`
   - Interactive readline shell, natural language commands

2. **Dashboard Mode**: `python run.py --dashboard` or `-d`
   - Rich TUI with 5 tabs, real-time visualization

3. **Simulation Mode**: `python run.py --sim` or `--sim-gui`
   - MuJoCo physics with real SO-101 STL meshes
   - 6 mesh objects: banana, mug, bottle, screwdriver, duck, lego
   - Weld-constraint grasping (reliable, no contact/friction issues)
   - Smooth real-time motion with linear interpolation
   - Pick sequence: open → approach → grasp → lift → rotate 90deg → drop → home
   - Sim perception: ground-truth positions, NL queries (Chinese + English)

4. **Testing Modes**: `--no-arm`, `--no-perception`

## Architecture

| Layer | Component | Status |
|-------|-----------|--------|
| LLM | Claude Haiku via OpenRouter | Working |
| Planning | Task decomposition + executor | Working |
| Perception | VLM + EdgeTAM + RealSense D405 (real) / MuJoCo ground-truth (sim) | Working |
| Control | Pinocchio FK/IK + MuJoCo Jacobian IK | Working |
| Hardware | SO-101 arm + gripper (real) / MuJoCo sim | Working |
| Skills | pick, place, home, scan, detect | Working |
| CLI | Readline shell + TUI dashboard | Working |
| ROS2 | 5 nodes + launch file (optional) | Working |

## TODO (Next Priorities)

### 1. ~~MuJoCo Simulation~~ DONE
- SO-101 with real STL meshes, 6 mesh objects, weld grasping
- Sim perception, smooth motion, pick-and-place with rotate-drop
- 26 unit tests + manual verification

### 2. LLM Agent Brain Upgrade
- Skill Manifest Protocol (ADR-002), multi-step planning
- Multi-turn memory, model auto-select (Haiku/Sonnet)

### 3. Pick Accuracy
- Re-calibration, hand-eye calibration, grasp detection

### 4. Merge & Release
- Merge feat/vector-os-nano-python-sdk → main
- Tag v0.1.0 release, PyPI publish
