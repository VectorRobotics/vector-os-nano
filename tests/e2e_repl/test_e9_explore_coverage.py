# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E9 explore without a nav stack, bare REPL go2: '探索一下房子' -> the coverage tour.

Before WP08 explore 'started' and did nothing (no TARE) while reporting success.
Now the tour must visit >= 6 of the 8 rooms, graded twice from outside the product:
the tour's ground-truth log is re-checked against room boxes read from the scene's
own floor geoms (not the layout file the product uses), and the turn's verify is a
GT at_position on the tour's last room, which only fires once explore succeeded.
Lockstep (WP06): physics advances only inside the tour's sim-time waits (~90 s wall).
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from tests.harness.kin_base import KinBase
from tests.harness.pty_cli import run_repl_session
from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.sim.nav.coverage import LOG_NAME, CoverageExplorer, room_layout

_ROOM_XML = Path(__file__).resolve().parents[2] / "vector_os_nano/hardware/sim/go2_room.xml"
_FLOORS = {"f_living": "living_room", "f_dining": "dining_room", "f_kitchen": "kitchen",
           "f_study": "study", "f_master": "master_bedroom", "f_guest": "guest_bedroom",
           "f_bath": "bathroom"}


def _scene_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Room floor boxes from the scene XML's f_* geoms; the hallway is the open centre
    span x 6..14 y 0..10 and the laundry (x 10..12) is merged into the bathroom."""
    boxes = {}
    for m in re.finditer(r'<geom name="(f_\w+)" type="box" pos="([\d.]+) ([\d.]+) [\d.]+" '
                         r'size="([\d.]+) ([\d.]+)', _ROOM_XML.read_text()):
        name, x, y, hx, hy = m.group(1), *map(float, m.groups()[1:])
        boxes[_FLOORS[name]] = (x - hx, y - hy, x + hx, y + hy)
    x0, y0, _x1, y1 = boxes["bathroom"]
    boxes["bathroom"] = (x0, y0, 12.0, y1)
    boxes["hallway"] = (6.0, 0.0, 14.0, 10.0)
    return boxes


def _last_room() -> tuple[float, float]:
    """The tour's final room point: the same nearest-first order from the spawn."""
    explorer = CoverageExplorer(KinBase(*load_embodiment_config("go2").spawn.xy), log_path="/dev/null")
    return room_layout()["rooms"][explorer._order()[-1]]


_LX, _LY = _last_room()
_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "go2", "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
    {"when": "探索一下房子", "tool_calls": [{"name": "explore", "input": {}}]},
    {"when": "Skill 'explore' succeeded", "tool_calls": [
        {"name": "verify", "input": {"expr": f"at_position({_LX}, {_LY})"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_e9_explore_without_a_nav_stack_tours_the_house() -> None:
    run_dir = Path(tempfile.mkdtemp(prefix="vos_e9_"))
    r = run_repl_session(["启动 go2 仿真 无窗口", "探索一下房子"], tool_script=_SCRIPT,
                         answers=["y"], timeout_sec=900,
                         extra_env={"VECTOR_RUN_DIR": str(run_dir), "VECTOR_SIM_LOCKSTEP": "1"})
    text = r.text
    assert not r.timed_out, text[-4000:]
    tour = text[text.rfind("vector> 探索一下房子"):]
    assert "Exploration started" in tour and "Exploration complete" in tour, tour[-3000:]
    assert f"verify at_position({_LX}, {_LY}) ✓ (actor=CAUSED)" in tour, tour[-3000:]
    assert "verdict GROUNDED verified=True" in tour, tour[-3000:]

    rows = [json.loads(line) for line in (run_dir / LOG_NAME).read_text().splitlines()]
    boxes = _scene_boxes()

    def inside(room: str, x: float, y: float) -> bool:
        x0, y0, x1, y1 = boxes[room]
        return x0 <= x <= x1 and y0 <= y <= y1

    entered = {row["room"] for row in rows if inside(row["room"], *row["gt_pos"])}
    assert len(entered) >= 6, rows
    assert {row["room"] for row in rows if row["in_room"]} == entered, rows  # product agrees with GT
