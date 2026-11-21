# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""scripts/gt_grade.py — the independent per-turn GT grader, on hand-built dumps.

Hermetic on purpose: the pin that the grader shares no code with the product's oracles
runs in the hermetic tier and the CI workflow. The session side of the dump (what
SimSession.mark_turn writes) is pinned in tests/unit/sim/test_sim_session.py."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

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


def _line(phase: str, turn: int, pos, welded=False, user=(8.0, 3.0), goal="把绿色瓶子拿过来") -> dict:
    return {
        "phase": phase, "turn": turn, "goal": goal,
        "objects": {"b": {"pos": list(pos), "welded": welded}},
        "surfaces": {"t": {"box": [0.0, 0.0, 1.0, 1.0], "height": 0.5}},
        "user": list(user),
    }


def _write(path: Path, lines: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    return str(path)


def test_grade_rederives_delivered_and_object_on(tmp_path) -> None:
    dump = _write(tmp_path / "gt.jsonl", [
        _line("start", 1, (10.9, 3.0, 0.32)), _line("end", 1, (8.2, 3.1, 0.6), welded=True),  # carried to the user
        _line("start", 2, (8.2, 3.1, 0.6)), _line("end", 2, (8.3, 3.1, 0.6)),  # already there: no delivery
        _line("start", 3, (5, 5, 0.3)), _line("end", 3, (0.5, 0.5, 0.56)),  # placed on t
        _line("start", 4, (0.5, 0.5, 0.56)),  # unfinished turn: dropped
    ])
    rows = gt_grade.grade(dump, "b", "t")
    assert [r["turn"] for r in rows] == [1, 2, 3]
    assert [r["delivered"] for r in rows] == [True, False, False]
    assert [r["object_on"] for r in rows] == [False, False, True]  # turn 1 still welded
    assert rows[0]["moved_m"] == pytest.approx(2.70, abs=0.01)


def test_object_on_rejects_beside_above_and_already_there() -> None:
    start = _line("start", 1, (5.0, 5.0, 0.3))
    assert gt_grade.object_on(start, _line("end", 1, (0.5, 0.5, 0.56)), "b", "t")
    assert not gt_grade.object_on(start, _line("end", 1, (1.5, 0.5, 0.56)), "b", "t")  # outside the footprint
    assert not gt_grade.object_on(start, _line("end", 1, (0.5, 0.5, 0.9)), "b", "t")  # floating well above it
    assert not gt_grade.object_on(start, _line("end", 1, (0.5, 0.5, 0.3)), "b", "t")  # under the top
    # adversarial: resting on the table at the start already (a pre-placed object was never placed)
    on = _line("start", 1, (0.5, 0.5, 0.56))
    assert not gt_grade.object_on(on, _line("end", 1, (0.6, 0.5, 0.56)), "b", "t")


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


def _base(phase: str, xy, turn: int = 1, goal: str = "去厨房") -> dict:
    return {"phase": phase, "turn": turn, "goal": goal, "objects": {}, "surfaces": {}, "user": None,
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
    dump = _write(tmp_path / "gt.jsonl", [_base("start", (10, 3)), _base("end", (17, 1.7))])
    assert gt_grade.grade(dump, box=kitchen, at=point) == [
        {"turn": 1, "goal": "去厨房", "base_in": True, "base_at": False}]


def test_a_dump_of_several_sim_sessions_pairs_each_end_with_its_own_start(tmp_path) -> None:
    """A body switch starts a new sim session, which counts its turns from 1 again: the
    old session's turn 2 is left open by the switch, the new one's turn 1 closes. Paired
    by number over the whole file, the new end took the old session's turn-1 start."""
    dump = _write(tmp_path / "gt.jsonl", [
        _base("start", (10, 3), 1, "启动 go2"), _base("end", (10, 3), 1, "启动 go2"),
        _base("start", (10, 3), 2, "去厨房"), _base("end", (17, 1.7), 2, "去厨房"),
        _base("start", (17, 1.7), 3, "切换到 g1"),  # the switch: the old body never ends it
        _base("start", (10, 3), 1, "切换到 g1"), _base("end", (10, 3), 1, "切换到 g1"),
        _base("start", (15, 2), 2, "去厨房"), _base("end", (16, 2), 2, "去厨房"),  # already in it
    ])
    kitchen = [14.0, 0.0, 20.0, 5.0]
    rows = gt_grade.grade(dump, box=kitchen)
    assert [(r["goal"], r["base_in"]) for r in rows] == [
        ("启动 go2", False), ("去厨房", True), ("切换到 g1", False), ("去厨房", False)]
    # one chosen turn: the nth finished turn of a sentence
    assert gt_grade.grade(dump, box=kitchen, goal="去厨房", nth=1) == [
        {"turn": 2, "goal": "去厨房", "base_in": True}]
    assert gt_grade.grade(dump, box=kitchen, goal="去厨房", nth=2) == [
        {"turn": 2, "goal": "去厨房", "base_in": False}]
    assert gt_grade.grade(dump, box=kitchen, goal="去厨房", nth=3) == []
    assert gt_grade.grade(dump, box=kitchen, goal="去书房") == []


def test_the_cli_grades_one_chosen_turn(tmp_path, capsys) -> None:
    dump = _write(tmp_path / "gt.jsonl", [_base("start", (10, 3)), _base("end", (17, 1.7)),
                                          _base("start", (17, 1.7), 2), _base("end", (10, 3), 2)])
    assert gt_grade.main(["--dump", dump, "--box", "14", "0", "20", "5", "--goal", "去厨房", "--nth", "2"]) == 0
    assert [json.loads(x) for x in capsys.readouterr().out.splitlines()] == [
        {"turn": 2, "goal": "去厨房", "base_in": False}]
    with pytest.raises(SystemExit):
        gt_grade.main(["--dump", dump, "--box", "14", "0", "20", "5", "--goal", "去厨房", "--nth", "0"])
