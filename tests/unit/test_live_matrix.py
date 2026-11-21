# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""tests/harness/live_matrix.py — the live matrix instrument (A10), graded from
hand-built files: the transcript, the GT dump and the session a repetition leaves.
The instrument never trusts the verdict: the GT dump decides what happened."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.harness import live_matrix as lm

_CELL = {"id": "so101-grasp", "body": "so101", "backend": "mujoco", "launch": "启动 so101 机械臂仿真 无窗口",
         "task": "抓起香蕉", "kind": "grasp", "expect": {"held_lifted": "banana"}, "n": 1}
_READY = "  ▸ Started so101 simulation, ready in 2.1s: MuJoCoArm, sim_time=3.0s\n"


def _transcript(verified: bool, goal: str = "proven", ready: str = _READY) -> str:
    ev = "GROUNDED" if verified else "FAILED"
    return (f"  Model: deepseek-v4 | Provider: deepseek | Arm: none\n"
            f"vector> 启动 so101 机械臂仿真 无窗口\n{ready}vector> 抓起香蕉\n"
            f"  ▸ pick → verify holding_object('banana') ✓ (actor=CAUSED)\n"
            f"  verdict {ev} verified={verified} (1/1 grounded) goal={goal}\nvector> quit\n")


def _dump(z_end: float, welded_end: bool) -> str:
    def line(phase: str, z: float, welded: bool) -> dict:
        return {"phase": phase, "turn": 1, "goal": "抓起香蕉", "base": None, "surfaces": {}, "user": None,
                "objects": {"banana": {"pos": [0.3, 0.0, z], "welded": welded}}}
    return "\n".join(json.dumps(x, ensure_ascii=False)
                     for x in (line("start", 0.02, False), line("end", z_end, welded_end))) + "\n"


def _rec(tmp_path: Path, transcript: str, dump: str | None, session: list[dict] | None = None) -> dict:
    (tmp_path / "t.txt").write_text(transcript, encoding="utf-8")
    if dump is not None:
        (tmp_path / "gt.jsonl").write_text(dump, encoding="utf-8")
    if session is not None:
        (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in session))
    return {"cell": _CELL["id"], "rep": 0, "wall_s": 1.0, "timed_out": False, "exit_code": 0,
            "transcript": str(tmp_path / "t.txt"), "dump": str(tmp_path / "gt.jsonl"),
            "session": str(tmp_path / "s.jsonl")}


def test_the_shipped_matrix_loads_and_every_cell_is_runnable_or_says_why() -> None:
    cells = lm.load_cells()
    run = [c for c in cells if "not_yet" not in c]
    assert {c["body"] for c in run} == {"go2", "go2_piper", "piper", "so101", "g1"}
    assert {c["kind"] for c in run} >= {"navigate", "grasp", "fetch"}
    assert all(c["backend"] == "mujoco" for c in run)
    assert any(c["kind"] == "place" for c in cells if "not_yet" in c)
    assert any(c["backend"] == "pybullet" for c in cells if "not_yet" in c)


@pytest.mark.parametrize("bad, words", [
    ({"expect": {"held_lifted": "a", "delivered": "a"}}, "exactly one"),
    ({"expect": {"teleported": "a"}}, "unknown expect"),
    ({"kind": "dance"}, "kind"),
    ({"not_yet": " "}, "needs a reason"),
    ({"task": None}, None),
])
def test_a_bad_cell_fails_loud(tmp_path, bad, words) -> None:
    cell = {**_CELL, **bad}
    if words is None:
        del cell["task"]
        words = "missing"
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"schema": 1, "cells": [cell]}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=words):
        lm.load_cells(p)


def test_a_verified_turn_with_the_banana_still_on_the_table_is_false_grounded(tmp_path) -> None:
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True), _dump(0.02, False)))
    assert r["status"] == "ran" and r["verdict"]["verified"] is True
    assert r["grade"] is False and r["outcome"] == "FALSE_GROUNDED"
    s = lm.summarize([_CELL], [r])
    assert s["false_grounded"] == 1 and s["cells"][0]["false_grounded"] == [r["transcript"]]
    assert lm.exit_code(s) == 1


def test_a_weld_with_no_lift_is_no_pick_either(tmp_path) -> None:
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True), _dump(0.025, True)))
    assert r["outcome"] == "FALSE_GROUNDED"


def test_a_pick_that_happened_while_the_turn_failed_is_false_reject(tmp_path) -> None:
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(False, "failed"), _dump(0.12, True)))
    assert r["grade"] is True and r["outcome"] == "FALSE_REJECT"
    s = lm.summarize([_CELL], [r])
    c = s["cells"][0]
    assert (c["success"], c["verified"], c["goal_proven"], len(c["false_reject"])) == (1, 0, 0, 1)
    assert lm.exit_code(s) == 0  # a false reject is reported, it does not fail the run


def test_a_turn_with_no_verdict_line_is_not_verified(tmp_path) -> None:
    text = _transcript(True).replace("verdict GROUNDED", "the model only talked")
    r = lm.judge(_CELL, _rec(tmp_path, text, _dump(0.12, True)))
    assert r["verdict"] is None and r["outcome"] == "FALSE_REJECT"


def test_a_verified_turn_with_no_gt_is_ungraded_never_a_success(tmp_path) -> None:
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True), None))
    assert r["grade"] is None and r["outcome"] == "ungraded" and "no GT dump" in r["grade_note"]
    s = lm.summarize([_CELL], [r])
    c = s["cells"][0]
    assert c["success"] == 0 and c["ungraded"] == [r["transcript"]]
    # L1-V03: a run in which nothing was graded proved nothing: never exit 0
    assert (s["reps_run"], s["graded"], lm.exit_code(s)) == (1, 0, 2)


def test_the_report_names_the_model_that_ran(tmp_path) -> None:
    """L1-V02: a scripted model leaking into a 'live' run was invisible in the report."""
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True), _dump(0.12, True)))
    assert r["model"] == "deepseek-v4 (deepseek)"
    s = lm.summarize([_CELL], [r])
    assert s["models"] == ["deepseek-v4 (deepseek)"] and "model: deepseek-v4 (deepseek)" in lm.markdown(s)
    blind = lm.judge(_CELL, _rec(tmp_path, _transcript(True).split("\n", 1)[1], _dump(0.12, True)))
    assert lm.summarize([_CELL], [blind])["models"] == ["unknown"]


def test_a_proven_goal_that_the_grade_denies_is_false_grounded_too(tmp_path) -> None:
    """I4D: done is read from goal_check == 'proven', not from verified alone."""
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(False, "proven"), _dump(0.02, False)))
    assert r["verdict"]["verified"] is False and r["outcome"] == "FALSE_GROUNDED"
    agree = lm.judge(_CELL, _rec(tmp_path, _transcript(False, "proven"), _dump(0.12, True)))
    assert agree["outcome"] == "agree"


def test_success_agrees_and_reads_the_task_turn_tools(tmp_path) -> None:
    session = [
        {"type": "user", "content": "启动 so101 机械臂仿真 无窗口", "ts": "2026-11-12T06:23:00Z"},
        {"type": "assistant", "tool_use": [{"id": "a", "name": "start_simulation", "input": {}}], "ts": "2026-11-12T06:23:01Z"},
        {"type": "user", "content": "抓起香蕉", "ts": "2026-11-12T06:23:24Z"},
        {"type": "assistant", "tool_use": [{"id": "b", "name": "pick_top_down", "input": {}}], "ts": "2026-11-12T06:23:26Z"},
        {"type": "tool_result", "results": [{"tool_use_id": "b", "is_error": True}], "ts": "2026-11-12T06:23:26Z"},
        {"type": "assistant", "tool_use": [{"id": "c", "name": "pick", "input": {}}], "ts": "2026-11-12T06:23:30Z"},
        {"type": "tool_result", "results": [{"tool_use_id": "c", "is_error": False}], "ts": "2026-11-12T06:23:50Z"},
        {"type": "meta", "ts": "2026-11-12T06:24:30Z"},
    ]
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True), _dump(0.12, True), session))
    assert r["outcome"] == "agree" and r["grade"] is True
    assert r["tools"] == ["pick_top_down!", "pick"] and r["turn_s"] == 26.0 and r["model_calls"] == 3
    assert lm.exit_code(lm.summarize([_CELL], [r])) == 0


@pytest.mark.parametrize("ready, why", [
    ("  ▸ Failed to start so101 simulation: assets not provisioned: run scripts/fetch_assets.py\n",
     "Failed to start so101 simulation: assets not provisioned"),
    ("  ▸ Started piper simulation, ready in 2.0s: Arm\n", "the launch started piper, the cell asks for so101"),
    ("I can't do that.\n", "no 'Started so101 simulation, ready' line"),
])
def test_a_cell_whose_launch_fails_is_not_run_with_its_reason(tmp_path, ready, why) -> None:
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True, ready=ready), _dump(0.02, False)))
    assert r["status"] == "not_run" and why in r["reason"]
    s = lm.summarize([_CELL], [r])
    assert s["cells"][0]["status"] == "not_run" and s["false_grounded"] == 0
    assert lm.exit_code(s) == 2  # nothing ran: never green
    assert "not_run" in lm.markdown(s)


def test_no_model_key_runs_nothing_and_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lm, "missing_key", lambda: "No DEEPSEEK_API_KEY set")
    monkeypatch.setattr(lm, "run_rep", lambda *a, **k: pytest.fail("a session started without a key"))
    cells = [_CELL, {**_CELL, "id": "later", "not_yet": "no backend flag"}]
    s = lm.run(cells, tmp_path)
    assert [(c["status"], c["reason"]) for c in s["cells"]] == [
        ("not_run", "No DEEPSEEK_API_KEY set"), ("not_yet", "no backend flag")]
    assert lm.exit_code(s) == 2
    assert (tmp_path / "live_matrix.json").is_file() and "No DEEPSEEK_API_KEY" in (tmp_path / "live_matrix.md").read_text()


def test_markdown_rows_have_the_header_s_columns(tmp_path) -> None:
    r = lm.judge(_CELL, _rec(tmp_path, _transcript(True), _dump(0.12, True)))
    later = {**_CELL, "id": "later", "not_yet": "no backend flag"}
    md = lm.markdown(lm.summarize([_CELL, later], [r]))
    tables = [[ln for ln in block.splitlines() if ln.startswith("|")] for block in md.split("\n\n")]
    for rows in filter(None, tables):
        assert {ln.count("|") for ln in rows} == {rows[0].count("|")}, rows
