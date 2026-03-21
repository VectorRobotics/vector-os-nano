# Vector OS Nano SDK — Progress

**Last updated:** 2026-03-21 16:30  
**Status:** v0.1.0 functional, pick pipeline working, TUI improvements in progress

## What Works

- Full NL pipeline: "抓电池" → LLM → scan → detect(VLM) → track(EdgeTAM) → 3D → calibrate → IK → pick → drop
- Direct commands without LLM: home, scan, open, close (instant, no API call)
- Chinese + English natural language
- Live camera viewer: RGB + depth side-by-side, EdgeTAM tracking overlay
- 696 unit tests passing
- ROS2 integration layer (optional, 5 nodes + launch file)
- Textual TUI dashboard
- PyBullet simulation
- SO-101 arm driver (Feetech STS3215 serial)
- Calibration wizard (TUI + readline)

## TUI Improvements (In Progress)

### Alpha: Core Dashboard Enhancements
- ASCII art logo at dashboard top
- Fixed command input handling (focus management)
- Status indicator dots (connection status, hardware state)
- Joint angle progress bars (real-time joint visualization)
- Skill execution progress indicator

### Beta: Camera Tab Implementation
- Camera frame renderer (Unicode half-block compression: 60x60 pixel equivalent)
- New "Camera" tab in 5-tab dashboard (Dashboard, Log, Skills, World, Camera)
- 2Hz refresh rate (only active when Camera tab is in focus)
- Camera preview with grayscale rendering

### Dashboard Navigation
- F1-F5: Tab switching (Dashboard, Log, Skills, World, Camera)
- F6: Fullscreen camera view
- `/` : Focus command input bar

## Current Limitations

### Pick Accuracy
- Empirical XY offsets tuned for specific workspace region
- Calibration matrix Z-row collapsed (all objects at Z=0.005m)
- Gripper asymmetry compensation is position-dependent (left/right/center)
- No look-then-move correction yet (calibration is pose-dependent)
- URDF model doesn't perfectly match real arm (3D-printed, servo backlash)

### Perception
- VLM detection depends on lighting conditions
- EdgeTAM tracking can lose objects if they move fast or get occluded
- Camera serial number hardcoded (335122270413)

### LLM
- Haiku sometimes over-plans (scan→detect even when just told to pick)
- Conversation context reset after each command (no multi-turn memory)

### Architecture
- Calibration only valid at home/scan pose (eye-in-hand, pose-dependent)
- World model cleared after each pick (conservative but loses history)
- No grasp success detection (servo current feedback not implemented)

## Tuning History

| Parameter | Value | Notes |
|-----------|-------|-------|
| z_offset | 10cm | Gripper link to table surface |
| pre_grasp_height | 6cm | Above grasp target |
| X offset | +2cm | Uniform forward compensation |
| Y left | +3cm + 50% proportional | Gripper asymmetry |
| Y right | +1cm | Gripper asymmetry |
| Y center | +2cm | Gripper asymmetry |

## Next Steps

1. TUI improvements completion (Alpha + Beta)
2. Skill Manifest Protocol (ADR-002) — alias-based command routing
3. Re-calibration with more points + Z variation
4. Hand-eye calibration for pose-independent transforms
5. Grasp success detection via servo current/load
6. Multi-object pick-and-place sequences
