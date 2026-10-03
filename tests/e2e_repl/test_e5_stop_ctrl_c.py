# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E5 stop, bare REPL go2: the owner can always stop the robot, and nothing runs twice.

One session, every line typed into bare ``vector-cli``:
  * a long navigate, then Ctrl-C -> the REPL survives and the base is stopped;
  * a long walk, then a typed '停' mid-turn -> stopped, with no LLM call;
  * '停下' at the prompt -> the P0 stop, < 100 ms, no LLM call;
  * the next '往前走一米' grounds (no sticky abort from the stops above);
  * an LLM outage injected after one dispatched walk -> the partial verdict, and the
    legacy planner never re-runs the walk.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from tests.harness.pty_cli import run_repl_session

_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "go2", "gui": False}}]},
    {"when": "ready", "text": "go2 仿真已启动", "tool_calls": []},
    # ~5 m through the kitchen door: well over 10 sim-s of walking.
    {"when": "导航到厨房", "tool_calls": [{"name": "navigate", "input": {"x": 15.0, "y": 3.0}}]},
    {"when": "往前走五米", "tool_calls": [
        {"name": "walk", "input": {"direction": "forward", "distance": 5.0, "speed": 0.3}}]},
    {"when": "往前走一米", "tool_calls": [
        {"name": "walk", "input": {"direction": "forward", "distance": 1.0, "speed": 0.3}}]},
    {"when": "Skill 'walk' succeeded", "tool_calls": [
        {"name": "verify", "input": {"expr": "get_position()[0] > 10.0"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": "后退半米", "tool_calls": [
        {"name": "walk", "input": {"direction": "backward", "distance": 0.5, "speed": 0.3}}]},
    {"when": "Skill 'walk' succeeded", "raise": "injected LLM outage (503)"},
]}

# The zero command goes out in the reported ms; the legged body then decelerates
# (measured from a 0.4 m/s cruise: 0.1-0.5 m/s left at 0.2 sim-s depending on box
# load, ~0 by 1.0 sim-s), so the REPL reports the body speed at both marks.
_STOPPED = re.compile(r"■ stopped \((\d+) ms, no LLM call\)")
_SPEED = re.compile(r"base ([\d.]+) m/s at 0\.2 sim-s, ([\d.]+) m/s at 1\.0 sim-s")


@pytest.fixture(scope="module")
def e5():
    log = Path(tempfile.mkstemp(prefix="vos_e5_llm_", suffix=".jsonl")[1])
    r = run_repl_session(
        ["启动 go2 仿真 无窗口",
         "导航到厨房 (15, 3)", (4.0, "\x03"),
         "往前走五米", (3.0, "停"),
         "停下",
         "往前走一米",
         "后退半米"],
        tool_script=_SCRIPT, answers=["y"], timeout_sec=420,
        extra_env={"VECTOR_FAKE_LLM_LOG": str(log)},
    )
    calls = [json.loads(line)["obs"] for line in log.read_text(encoding="utf-8").splitlines()]
    log.unlink(missing_ok=True)
    return r, r.text, calls


def _section(text: str, start: str, end: str | None = None) -> str:
    i = text.rfind(f"vector> {start}")
    assert i != -1, f"{start!r} was never typed at a prompt:\n{text[-3000:]}"
    j = text.find(f"vector> {end}", i) if end else -1
    return text[i: j if j != -1 else None]


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_ctrl_c_mid_navigate_keeps_the_repl_and_stops_the_base(e5) -> None:
    r, text, _calls = e5
    assert not r.timed_out, text[-3000:]
    nav = _section(text, "导航到厨房", "往前走五米")
    assert "Interrupted" in nav, nav
    m, v = _STOPPED.search(nav), _SPEED.search(nav)
    assert m and v, nav
    assert int(m.group(1)) < 100, m.group(0)  # the zero command went out at once
    assert float(v.group(2)) < 0.05, f"base still moving 1.0 sim-s after Ctrl-C: {v.group(0)}"
    assert "aborted" in nav and "verdict FAILED" in nav, nav
    assert r.exit_code == 0  # the session went on and quit normally


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_typed_stop_mid_walk_and_p0_stop_call_no_llm(e5) -> None:
    _r, text, calls = e5
    walk = _section(text, "往前走五米", "停下")
    assert _STOPPED.search(walk) and _SPEED.search(walk) and "aborted" in walk, walk
    p0 = _section(text, "停下", "往前走一米")
    m = _STOPPED.search(p0)
    assert m and int(m.group(1)) < 100, p0
    assert not any(o.strip() in ("停", "停下") for o in calls), calls  # zero LLM calls for a stop


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_next_walk_after_the_stops_grounds(e5) -> None:
    _r, text, _calls = e5
    after = _section(text, "往前走一米", "后退半米")
    assert "verify get_position()[0] > 10.0 ✓" in after, after
    assert "verdict GROUNDED verified=True" in after, after


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_llm_outage_after_a_walk_is_owned_not_rerun(e5) -> None:
    _r, text, calls = e5
    tail = _section(text, "后退半米")
    assert "LLM unavailable" in tail and "injected LLM outage" in tail, tail
    assert "verdict FAILED" in tail, tail
    assert "VGG" not in tail, f"the legacy planner re-ran the turn:\n{tail}"
    assert sum("后退半米" == o.strip() for o in calls) == 1, calls  # one producer, one attempt
