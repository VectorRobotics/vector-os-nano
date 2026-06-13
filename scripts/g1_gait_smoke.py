# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""One-command G1 real-gait smoke (campaign #5 batch 3).

Boots the g1_flat scenario on the real unitree_rl_gym policy, drives a
forward walk then a turn-while-walking, captures chase-camera frames, and
grades DETERMINISTICALLY from physics state: did the robot actually
DISPLACE (real gait, not the habitat glide)? Frames are owner evidence —
no pixel grading. Distinct from e2e_gui_smoke.py (habitat tmux/window
capture); G1 is in-process MuJoCo with on-demand control-thread renders.

Usage: MUJOCO_GL=egl .venv/bin/python scripts/g1_gait_smoke.py
Exit 0 = both phases produced real displacement + valid frames; 1 = a
phase failed to move / render; 2 = assets missing (run setup_g1_gait.sh).
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PHASES = [
    {"name": "walk_forward", "cmd": (0.5, 0.0, 0.0), "dur": 4.0,
     "min_move": 0.6},
    {"name": "turn_while_walking", "cmd": (0.3, 0.0, 0.4), "dur": 4.0,
     "min_move": 0.4},
]


def main() -> int:
    from vector_os_nano.hardware.sim.mujoco_g1 import g1_assets_ready
    if not g1_assets_ready():
        print("PREFLIGHT FAIL: g1_gait assets missing — run "
              "scripts/setup_g1_gait.sh")
        return 2

    from vector_os_nano.playground.catalog import get_scenario
    from vector_os_nano.playground.world import PlaygroundWorld
    from vector_os_nano.vcli.g1_runtime import boot_g1_agent

    art = REPO / "artifacts" / f"g1-smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    art.mkdir(parents=True, exist_ok=True)

    agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")))
    base = agent._base
    results = []
    rc = 0
    try:
        for ph in PHASES:
            start = base.get_position()
            walk_t = threading.Thread(
                target=lambda c=ph["cmd"], d=ph["dur"]: base.walk(*c, duration=d),
                daemon=True)
            walk_t.start()
            shots = []
            n_shots = 4
            for i in range(n_shots):
                time.sleep(ph["dur"] / n_shots)
                png = base.viewer_frame_png()
                fn = art / f"{ph['name']}_{i}.png"
                fn.write_bytes(png)
                shots.append({"frame": fn.name, "pos": base.get_position()[:2],
                              "valid_png": png[:8] == b"\x89PNG\r\n\x1a\n"})
            walk_t.join()
            end = base.get_position()
            moved = math.hypot(end[0] - start[0], end[1] - start[1])
            upright = end[2] > 0.45
            ok = moved >= ph["min_move"] and upright and all(
                s["valid_png"] for s in shots)
            results.append({"phase": ph["name"], "moved_m": round(moved, 2),
                            "upright": upright, "verdict": "ok" if ok else "fail",
                            "frames": [s["frame"] for s in shots]})
            print(f"[{ph['name']}] moved={moved:.2f}m upright={upright} "
                  f"-> {'OK' if ok else 'FAIL'}")
            if not ok:
                rc = 1
            base.stop()
            time.sleep(0.8)   # settle before the next phase
    finally:
        base.disconnect()    # instance hygiene

    (art / "summary.json").write_text(
        json.dumps({"results": results, "exit_code": rc}, indent=2))
    print(f"\nartifacts: {art}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
