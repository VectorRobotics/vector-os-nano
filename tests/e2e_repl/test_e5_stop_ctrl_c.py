# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E5 stop, bare REPL go2: the owner can always stop the robot, and nothing runs twice.

One session, every line typed into bare ``vector-cli``:
  * '停下' at the prompt -> the P0 stop, < 100 ms, no LLM call; a panicked Ctrl-C
    during its speed report never exits the REPL;
  * the next '往前走一米' grounds on a literal at_position from the spawn (no sticky
    abort from the stop above);
  * a long navigate, then Ctrl-C -> the REPL survives and the base is stopped: the
    body brakes (MPC to zero velocity) instead of being dropped onto the stance
    mid-stride, so it is near-still from 0.2 sim-s on (measured, peak included);
  * a long walk, then a typed '停' mid-turn -> stopped, with no LLM call;
  * a '停' typed while the model is still thinking -> nothing runs for that line (the
    legacy planner never gets the stopped command);
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
    # One metre forward of the spawn (9.99, 3) facing +x is (11, 3): a graded base
    # predicate, false before the walk. The wide tol is the open-loop walk's
    # wall-clock budget: under load it covers 0.4-1.0 m (walks move onto sim time in
    # WP06; tighten to the default 0.5 then). The planner calls (11, 3) unreachable
    # (inflated obstacle), so a closed-loop navigate is no substitute.
    {"when": "Skill 'walk' succeeded", "tool_calls": [
        {"name": "verify", "input": {"expr": "at_position(11.0, 3.0, 0.9)"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    # A slow model: the owner's '停' lands while this reply is still in flight.
    {"when": "向左走一米", "delay": 2.5, "tool_calls": [
        {"name": "walk", "input": {"direction": "left", "distance": 1.0, "speed": 0.3}}]},
    {"when": "后退半米", "tool_calls": [
        {"name": "walk", "input": {"direction": "backward", "distance": 0.5, "speed": 0.3}}]},
    {"when": "Skill 'walk' succeeded", "raise": "injected LLM outage (503)"},
    # The body can stumble on one of the stops above (the known red below): the walk is
    # still dispatched and the outage still lands right after it.
    {"when": "Walk command failed", "raise": "injected LLM outage (503)"},
]}

# The zero command goes out in the reported ms; the legged body then decelerates,
# so the REPL reports the measured body speed at 0.2 sim-s, its peak from there to
# 1.0 sim-s, and the speed at 1.0 sim-s.
_STOPPED = re.compile(r"■ stopped \((\d+) ms, no LLM call\)")
_SPEED = re.compile(
    r"base ([\d.]+) m/s at 0\.2 sim-s, peak ([\d.]+) m/s to 1\.0 sim-s, ([\d.]+) m/s at 1\.0 sim-s"
)
# "Stopped", the measured bound (the brake: the MPC trots the body to zero velocity,
# then hands over to the stance with all four feet down). In this session (3 runs x
# 2 stops): 0.03-0.07 m/s at 0.2 sim-s, peak 0.07-0.19 m/s to 1.0 sim-s, 0.02 at 1.0.
# Before the brake: 0.12-0.19 at 0.2 on the navigate stop, up to 0.58 at 1.0 when the
# body stumbled; a 0.4 m/s trot read 0.35-0.46 at 0.2 sim-s in-process.
# Lead decision (A4, 2026-10-04): a legged body cannot be at 0 m/s 0.2 sim-s after a
# trot; "stopped" = the zero command goes out < 100 ms AND the body is < 0.05 m/s at
# 1.0 sim-s. The 0.2 s / peak bounds only rule out a runaway or a stop lurch (the
# verifier measured up to 0.30 m/s at 0.2 sim-s on free-moving lateral/0.5 m/s stops).
_AT_02_MPS, _PEAK_MPS, _SETTLED_MPS = 0.35, 0.40, 0.05


def _assert_stopped(section: str) -> None:
    v = _SPEED.search(section)
    assert v, section
    at_02, peak, at_10 = (float(g) for g in v.groups())
    assert at_02 < _AT_02_MPS and peak < _PEAK_MPS and at_10 < _SETTLED_MPS, v.group(0)


@pytest.fixture(scope="module")
def e5():
    log = Path(tempfile.mkstemp(prefix="vos_e5_llm_", suffix=".jsonl")[1])
    r = run_repl_session(
        ["启动 go2 仿真 无窗口",
         "停下", (0.3, "\x03"),
         "往前走一米",
         "导航到厨房 (15, 3)", (4.0, "\x03"),
         "往前走五米", (3.0, "停"),
         "向左走一米", (0.8, "停"),
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
    m = _STOPPED.search(nav)
    assert m and int(m.group(1)) < 100, nav  # the zero command went out at once
    _assert_stopped(nav)  # ...and the body actually stopped
    assert "navigate → aborted" in nav and "verdict FAILED" in nav, nav
    assert r.exit_code == 0  # the session went on and quit normally


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_typed_stop_mid_walk_and_p0_stop_call_no_llm(e5) -> None:
    _r, text, calls = e5
    walk = _section(text, "往前走五米", "向左走一米")
    # The walk right after the Ctrl-C above ran (no sticky abort), then the typed stop
    # stopped it, and the body.
    assert _STOPPED.search(walk) and "walk → aborted" in walk, walk
    _assert_stopped(walk)
    p0 = _section(text, "停下", "往前走一米")
    m = _STOPPED.search(p0)
    assert m and int(m.group(1)) < 100, p0
    # The Ctrl-C landed during the stop's speed report: it stopped again, never exited.
    assert "Interrupted" in p0 and "Traceback" not in p0, p0
    assert not any(o.strip() in ("停", "停下") for o in calls), calls  # zero LLM calls for a stop


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_next_walk_after_the_stop_grounds(e5) -> None:
    _r, text, _calls = e5
    after = _section(text, "往前走一米", "导航到厨房")
    assert "verify at_position(11.0, 3.0, 0.9) ✓ (actor=CAUSED)" in after, after
    assert "verdict GROUNDED verified=True" in after, after


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_a_stop_while_the_model_thinks_runs_nothing_after_it(e5) -> None:
    _r, text, calls = e5
    left = _section(text, "向左走一米", "后退半米")
    assert _STOPPED.search(left), left
    assert "(no action) → aborted" in left and "verdict FAILED" in left, left
    assert "VGG" not in left and "walk_goal" not in left, (
        f"the legacy planner ran the stopped command:\n{left}"
    )
    assert sum("向左走一米" == o.strip() for o in calls) == 1, calls  # no second producer


@pytest.mark.sim
@pytest.mark.cli_main
def test_e5_llm_outage_after_a_walk_is_owned_not_rerun(e5) -> None:
    _r, text, calls = e5
    tail = _section(text, "后退半米")
    assert "LLM unavailable" in tail and "injected LLM outage" in tail, tail
    assert "verdict FAILED" in tail, tail
    assert "VGG" not in tail, f"the legacy planner re-ran the turn:\n{tail}"
    assert sum("后退半米" == o.strip() for o in calls) == 1, calls  # one producer, one attempt
