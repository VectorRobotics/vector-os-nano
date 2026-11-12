# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""scripts/gt_grade.py — the independent per-turn GT grader, and the session's
VECTOR_GT_DUMP lines it reads (written at every turn start / end)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "scripts" / "gt_grade.py"
sys.path.insert(0, str(_TOOL.parent))
import gt_grade  # noqa: E402

sys.path.remove(str(_TOOL.parent))


def test_the_grader_shares_no_code_with_the_oracles() -> None:
    mods = set()
    for node in ast.walk(ast.parse(_TOOL.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "argparse", "json", "math", "sys", "typing"}, mods


def _line(phase: str, turn: int, pos, welded=False, user=(8.0, 3.0)) -> dict:
    return {
        "phase": phase, "turn": turn, "goal": "把绿色瓶子拿过来",
        "objects": {"b": {"pos": list(pos), "welded": welded}},
        "surfaces": {"t": {"box": [0.0, 0.0, 1.0, 1.0], "height": 0.5}},
        "user": list(user),
    }


def test_grade_rederives_delivered_and_object_on(tmp_path) -> None:
    dump = tmp_path / "gt.jsonl"
    lines = [
        _line("start", 1, (10.9, 3.0, 0.32)), _line("end", 1, (8.2, 3.1, 0.6), welded=True),  # carried to the user
        _line("start", 2, (8.2, 3.1, 0.6)), _line("end", 2, (8.3, 3.1, 0.6)),  # already there: no delivery
        _line("start", 3, (5, 5, 0.3)), _line("end", 3, (0.5, 0.5, 0.56)),  # placed on t
        _line("start", 4, (0.5, 0.5, 0.56)),  # unfinished turn: dropped
    ]
    dump.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = gt_grade.grade(str(dump), "b", "t")
    assert [r["turn"] for r in rows] == [1, 2, 3]
    assert [r["delivered"] for r in rows] == [True, False, False]
    assert [r["object_on"] for r in rows] == [False, False, True]  # turn 1 still welded
    assert rows[0]["moved_m"] == pytest.approx(2.70, abs=0.01)


def test_object_on_rejects_beside_and_above_the_surface() -> None:
    end = _line("end", 1, (1.5, 0.5, 0.56))
    assert not gt_grade.object_on(end, "b", "t")  # outside the footprint
    end = _line("end", 1, (0.5, 0.5, 0.9))
    assert not gt_grade.object_on(end, "b", "t")  # floating well above it
    end = _line("end", 1, (0.5, 0.5, 0.3))
    assert not gt_grade.object_on(end, "b", "t")  # under the top (on the floor below)


def test_held_lifted_needs_a_new_weld_and_a_rise() -> None:
    start = _line("start", 1, (1.0, 1.0, 0.50))
    assert gt_grade.held_lifted(start, _line("end", 1, (1.0, 1.0, 0.60), welded=True), "b")
    # adversarial: attached but still resting on the table (a weld alone is no pick)
    assert not gt_grade.held_lifted(start, _line("end", 1, (1.0, 1.0, 0.51), welded=True), "b")
    # risen but not held (knocked up, tossed)
    assert not gt_grade.held_lifted(start, _line("end", 1, (1.0, 1.0, 0.60)), "b")
    # already held at the start: nothing was picked up in this turn
    held = _line("start", 1, (1.0, 1.0, 0.50), welded=True)
    assert not gt_grade.held_lifted(held, _line("end", 1, (1.0, 1.0, 0.60), welded=True), "b")
    assert not gt_grade.held_lifted(start, _line("end", 1, (1.0, 1.0, 0.60), welded=True), "other")


def _base(phase: str, xy) -> dict:
    return {"phase": phase, "turn": 1, "goal": "去厨房", "objects": {}, "surfaces": {}, "user": None,
            "base": None if xy is None else [*xy, 0.3, 0.0]}


def test_base_in_and_base_at_need_the_base_to_arrive_in_this_turn(tmp_path) -> None:
    kitchen, point = [14.0, 0.0, 20.0, 5.0], [3.0, 2.5]
    assert gt_grade.base_in(_base("start", (10, 3)), _base("end", (17, 1.7)), kitchen)
    assert not gt_grade.base_in(_base("start", (10, 3)), _base("end", (13.9, 1.7)), kitchen)  # at the door
    assert not gt_grade.base_in(_base("start", (15, 2)), _base("end", (17, 1.7)), kitchen)  # already there
    assert not gt_grade.base_in(_base("start", None), _base("end", None), kitchen)  # a fixed base never arrives
    assert gt_grade.base_at(_base("start", (10, 3)), _base("end", (3.3, 2.2)), point)
    assert not gt_grade.base_at(_base("start", (10, 3)), _base("end", (3.0, 3.1)), point)  # 0.6 m off
    assert not gt_grade.base_at(_base("start", (3.1, 2.5)), _base("end", (3.0, 2.5)), point)  # already there
    dump = tmp_path / "gt.jsonl"
    dump.write_text("\n".join(json.dumps(x) for x in (_base("start", (10, 3)), _base("end", (17, 1.7)))) + "\n")
    assert gt_grade.grade(str(dump), box=kitchen, at=point) == [
        {"turn": 1, "goal": "去厨房", "base_in": True, "base_at": False}]


def test_the_session_dumps_ground_truth_at_turn_start_and_end(tmp_path, monkeypatch) -> None:
    mj = pytest.importorskip("mujoco")
    from vector_os_nano.embodiments import registry
    from vector_os_nano.hardware.sim.scene_builder import room_scene
    from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld
    from vector_os_nano.sim.runtime import SimRuntime
    from vector_os_nano.sim.session import SimSession, SimSpec
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    world = MuJoCoWorld(room_scene(registry.load("go2_piper"), load_world("apartment")))
    m = world.model
    rt = SimRuntime(world, threaded=False, paced=False, name="dump_test")
    try:
        rt.call(lambda w: mj.mj_forward(w.model, w.data))
        s = SimSession(SimSpec("go2_piper", gui=False), registry.load("go2_piper"))
        s.agent = SimpleNamespace(_base=SimpleNamespace(), _arm=None, _config={})
        s.runtime = rt  # what a builder hands the session; the agent never carries it
        s._bind_ground_truth()
        assert s.agent._config["world"] == "apartment"  # skills resolve names by world id
        assert not any(v is s.gt for v in vars(s.agent).values())  # GT never rides on the agent

        dump = tmp_path / "gt.jsonl"
        monkeypatch.setenv("VECTOR_GT_DUMP", str(dump))
        s.mark_turn("start", "把绿色瓶子拿过来")
        a = int(m.jnt_qposadr[m.body("pickable_bottle_green").jntadr[0]])
        rt.call(lambda w: (w.data.qpos.__setitem__(slice(a, a + 3), [8.2, 3.0, 0.4]), mj.mj_forward(w.model, w.data)))
        s.mark_turn("end", "把绿色瓶子拿过来")

        lines = gt_grade.load(str(dump))
        assert [ln["phase"] for ln in lines] == ["start", "end"]
        assert "base_link" not in lines[0]["objects"]
        assert lines[0]["user"] == [8.0, 3.0] and "pick_table" in lines[0]["surfaces"]
        (row,) = gt_grade.grade(str(dump), "pickable_bottle_green")
        assert row["delivered"] is True and row["turn"] == 1
        # the oracles see the same turn start (kept_holding / localized freshness)
        assert s._turn[0] == lines[0]["sim_time"]
        assert "kept_holding" in s.oracles() and "localized" in s.oracles()
    finally:
        rt.close()
