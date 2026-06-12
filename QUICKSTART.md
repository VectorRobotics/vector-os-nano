# QUICKSTART — the House World in 30 minutes

Drive a Unitree G1 through a furnished photoreal apartment with natural
language. No robot hardware needed; an NVIDIA GPU + Linux is.

## What you get

- A large multi-room scene (ReplicaCAD, CC-BY-4.0) rendered by habitat-sim,
  with a visible G1 humanoid and a third-person follow camera.
- Natural-language control via `vector-cli`: decompose → plan → execute →
  **deterministic verify** (navigation goals are judged by measured
  distance, never by the LLM's opinion).
- Optional: the real CMU-style sensor navigation stack (terrain analysis +
  local planner) and SysNav semantic perception ("走到sofa那里").

## Prerequisites

- Ubuntu 22.04/24.04, NVIDIA GPU + driver (EGL headless rendering)
- [miniconda](https://docs.conda.io/en/latest/miniconda.html), `git-lfs`,
  python 3.12, [uv](https://docs.astral.sh/uv/) (or plain venv+pip)
- An LLM API key in `.env` (deepseek or any OpenAI-compatible endpoint —
  see `.env.example`)

## 1. One-click setup

```bash
git clone <this-repo> && cd vector_os_nano
uv sync                                # the repo venv (.venv)
./scripts/setup_house_world.sh         # everything else, self-checking
```

The script is idempotent and verifies each step: habitat-sim 0.3.3 conda
env, the numpy 1.26 pin (habitat is numpy-1 ABI — pip likes to silently
break it), ReplicaCAD download (~300 MB, CC-BY-4.0), and the G1 body GLB
(composed from MuJoCo Menagerie, BSD-3). Re-run any time;
`--check` audits without changing anything. `RESULT: READY` means go.

## 2. Launch

```bash
.venv/bin/vector-cli --scenario house
```

A third-person window opens with the G1 standing in the apartment
(`VECTOR_HABITAT_VIEWER=first` switches to the robot's eye view;
`VECTOR_HABITAT_VIEWER_SIZE=1200` for a bigger window, default 800;
`VECTOR_HABITAT_GUI=0` for headless). Or start bare `vector-cli` and just
say **启动habitat模拟**.

Try:

```
走到厨房
走到门口
走到坐标 (-5.3, -3.4)
向前走0.5米然后右转90度
机器人状态
```

The house's named rooms (kitchen/entryway/living_room/dining/tv_corner)
are built-in navigation targets — no perception needed. Navigation runs
on the navmesh oracle out of the box, animated at real speed in the
window; every goal is verified by measured geodesic distance.

## 3. Optional: the REAL sensor navigation stack

With the sibling `vector_navigation_stack` workspace built (separate
repo), the same instructions run through terrain analysis + local
planner + path follower — the robot navigates from its own sensors and
honestly fails when a goal is unreachable:

```bash
# terminal A (nav stack):
./scripts/launch_habitat_nav.sh
# terminal B (no sourcing needed — vector-cli auto-sources ROS/SysNav
# overlays when they exist on disk and re-launches itself once):
.venv/bin/vector-cli --scenario house
```

The navigate skill auto-detects the running stack (`transport:
nav_stack` in results) and falls back to the oracle when it's down.

## 4. Optional: semantic goals (SysNav)

With the SysNav sibling workspace provisioned (PolyForm-NC license —
obtain it yourself) — no terminal sourcing needed, any plain shell works:

```
启动sysnav
走到sofa那里
```

Detected objects (sofa, chair, desk, …) flow into the live world model
AND appear as green labels in the viewer window as they are found; object
goals park at a ~1.5 m standoff and verify by euclidean distance.

## Troubleshooting

- **Black/blank window**: needs a real DISPLAY; over SSH use
  `VECTOR_HABITAT_GUI=0` and the `viewer_frame` artifacts instead.
- **habitat-sim import errors after installing anything**: re-run
  `./scripts/setup_house_world.sh` — it re-pins numpy 1.26.4.
- **Robot stops ~0.5 m short of a goal near furniture**: that's the real
  controller's clearance behavior (path ends at the obstacle boundary,
  follower parks within 0.4 m of the path end). Open-floor goals arrive
  within ~0.4 m.
- **`navigate_to requires a label OR numeric x and y`**: include explicit
  coordinates or a known object label; `机器人状态` lists live objects.
