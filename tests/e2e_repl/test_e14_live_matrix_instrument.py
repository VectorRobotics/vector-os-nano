# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E14 the live matrix instrument (A10) end to end, with the scripted model.

The instrument's own cell (so101 '抓起香蕉') through a real bare REPL session and the
real GT dump: a scripted pick that verifies is a success; the same turn graded
against a hand-built dump in which the banana never left the table is FALSE
GROUNDED (the instrument never trusts the verdict); a pick that happens in a turn
that does not verify (the model then checks the wrong object) is FALSE REJECT; a
launch that fails is NOT RUN with its reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness import live_matrix as lm

_CELL = next(c for c in lm.load_cells() if c["id"] == "so101-mujoco-grasp")
_OBJ = _CELL["expect"]["held_lifted"]


def _script(*, verify: str = _OBJ, embodiment: str = "so101") -> dict:
    """Launch, pick the banana, verify ``holding_object(<verify>)``, finish."""
    return {"steps": [
        {"when": _CELL["launch"], "tool_calls": [
            {"name": "start_simulation", "input": {"embodiment": embodiment, "gui": False}}]},
        {"when": "", "text": "ok", "tool_calls": []},
        {"when": _CELL["task"], "tool_calls": [{"name": "pick", "input": {"object_label": _OBJ, "mode": "hold"}}]},
        {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": f"holding_object('{verify}')"}}]},
        {"when": "", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_e14_a_scripted_pick_is_a_success_and_a_dump_with_the_banana_on_the_table_is_false_grounded(tmp_path) -> None:
    rec = lm.run_rep(_CELL, 0, tmp_path, tool_script=_script())
    text = Path(rec["transcript"]).read_text(encoding="utf-8")
    r = lm.judge(_CELL, rec)
    assert not rec["timed_out"] and r["status"] == "ran", (r.get("reason"), text[-3000:])
    assert r["verdict"]["verified"] is True and r["verdict"]["goal_check"] == "proven", text[-3000:]
    assert r["grade"] is True and r["outcome"] == "agree", (r, text[-3000:])
    assert r["tools"] == ["pick", "verify", "finish"] and r["model_calls"] >= 4, r
    assert lm.exit_code(lm.summarize([_CELL], [r])) == 0

    # The same turn and verdict; the dump rewritten by hand: the banana never moved.
    dump = Path(rec["dump"])
    lines = [json.loads(ln) for ln in dump.read_text(encoding="utf-8").splitlines()]
    start = next(ln for ln in lines if ln["phase"] == "start" and ln["goal"] == _CELL["task"])
    for ln in lines:
        if ln["phase"] == "end" and ln["turn"] == start["turn"]:
            ln["objects"][_OBJ] = dict(start["objects"][_OBJ])
    dump.write_text("\n".join(json.dumps(ln, ensure_ascii=False) for ln in lines) + "\n", encoding="utf-8")
    fg = lm.judge(_CELL, rec)
    assert fg["verdict"]["verified"] is True and fg["grade"] is False and fg["outcome"] == "FALSE_GROUNDED", fg
    s = lm.summarize([_CELL], [fg])
    assert s["cells"][0]["false_grounded"] == [rec["transcript"]] and lm.exit_code(s) == 1


@pytest.mark.sim
@pytest.mark.cli_main
def test_e14_a_pick_that_happened_in_a_turn_that_did_not_verify_is_false_reject(tmp_path) -> None:
    # the banana is picked; the model then checks the wrong object: a FAIL the spine rejects
    rec = lm.run_rep(_CELL, 0, tmp_path, tool_script=_script(verify="mug"))
    text = Path(rec["transcript"]).read_text(encoding="utf-8")
    r = lm.judge(_CELL, rec)
    assert not rec["timed_out"] and r["status"] == "ran", (r.get("reason"), text[-3000:])
    assert r["verdict"] is not None and r["verdict"]["verified"] is False, text[-3000:]
    assert r["grade"] is True and r["outcome"] == "FALSE_REJECT", (r, text[-3000:])
    c = lm.summarize([_CELL], [r])["cells"][0]
    assert (c["success"], c["verified"], c["false_reject"]) == (1, 0, [rec["transcript"]])


@pytest.mark.cli_main
def test_e14_a_cell_whose_launch_fails_is_not_run_with_its_reason(tmp_path) -> None:
    rec = lm.run_rep(_CELL, 0, tmp_path, tool_script=_script(embodiment="no_such_body"))
    r = lm.judge(_CELL, rec)
    assert r["status"] == "not_run" and "no_such_body" in r["reason"], r
    s = lm.summarize([_CELL], [r])
    assert s["cells"][0]["status"] == "not_run" and lm.exit_code(s) == 2
