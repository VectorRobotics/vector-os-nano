# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E1 + E3a — bare ``vector-cli`` + NL launches go2+Piper and grounds a perception grasp.

The owner's acceptance face, driven on a PTY with NO flags (the D97 failure mode
was this test hand-building the rig instead of using the product launcher):

    vector> 启动 go2 仿真 带机械臂 无窗口
      ▸ sim start go2_piper ok
      ▸ Started go2_piper simulation, ready in …            (E1: within 90 s)
    vector> 拿起绿色的瓶子
      ▸ perception_grasp → verify holding_object('pickable_bottle_green') ✓
      verdict GROUNDED verified=True                         (E3a)

Only the network LLM is replaced (``VECTOR_FAKE_LLM_TOOLS`` replays the model's tool
calls); the REPL is driven by the shared tests/harness/pty_cli.run_repl_session.
Everything else is the product: SimSession (MuJoCoGo2 + MuJoCoPiper +
MuJoCoPiperGripper), rebind_world, the native producer, perception_grasp (colour
resolve -> rendered depth -> IK -> weld, never ground truth) and the untouched
verify spine.

The grasp itself is physics (measured ~0.83 reliable, D94/D95): a red here is
first a grasp miss — read the transcript before blaming the launcher.
"""
from __future__ import annotations

import os
import re

import pytest

from tests.harness.pty_cli import run_repl_session

pytest.importorskip("mujoco")

_LAUNCH = "启动 go2 仿真 带机械臂 无窗口"
_READY = re.compile(r"Started go2_piper simulation, ready in ([0-9.]+)s: MuJoCoGo2 \+ MuJoCoPiper")

_SCRIPT = {"turns": [
    # turn 1 (legacy tool_use path: "仿真" bypasses the planner)
    {"tool_calls": [{"name": "start_simulation",
                     "input": {"embodiment": "go2_piper", "gui": False}}]},
    {"text": "go2_piper is up."},
    # turn 2 (native producer): perceive + grasp, then prove possession by GT
    {"tool_calls": [{"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
    {"tool_calls": [{"name": "verify",
                     "input": {"expr": "holding_object('pickable_bottle_green')"}}]},
    {"text": "holding the green bottle."},
]}


def _launch_then_grasp(grasp_line: str, timeout_sec: float, *, live: bool = False,
                       tool_script: dict | None = None) -> tuple[str, str, str]:
    """E1 launch line -> ready line (<90 s), then the grasp line -> (evidence, verified, text).

    The optional TARE stack is kept off (VECTOR_NAV_STACK blank, live too): E1/E3a
    must not depend on it, and its bridge follows a machine-global nav flag.
    """
    r = run_repl_session([_LAUNCH, grasp_line], tool_script=tool_script, live=live,
                         extra_env={"VECTOR_NAV_STACK": ""}, timeout_sec=timeout_sec)
    text = " ".join(r.text.split())
    ready = _READY.search(text)
    assert ready, f"no go2_piper ready line:\n{text[-3000:]}"
    assert float(ready.group(1)) < 90, "E1: the ready line must arrive within 90 s"
    line = text[ready.start():ready.start() + 300]
    assert "gait=" in line and "explore: coverage (VECTOR_NAV_STACK unset)" in line, line
    verdicts = re.findall(r"verdict (\w+) verified=(\w+)", text)
    evidence, verified = verdicts[-1] if verdicts else ("NO_VERDICT", "")
    return evidence, verified, text


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_bare_cli_nl_launch_go2_piper_then_grasp_grounds() -> None:
    evidence, verified, t = _launch_then_grasp("拿起绿色的瓶子", 420, tool_script=_SCRIPT)
    assert "sim start go2_piper ok" in t
    assert "perception_grasp → verify holding_object('pickable_bottle_green')" in t
    assert (evidence, verified) == ("GROUNDED", "True"), t[-3000:]


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
@pytest.mark.live_llm
@pytest.mark.skipif(os.environ.get("VECTOR_LIVE_LLM") != "1",
                    reason="live DeepSeek (billed): set VECTOR_LIVE_LLM=1")
def test_live_bare_cli_launch_and_grasp_two_of_three(tmp_path) -> None:
    """The owner's exact lines on the real .env model, N=3 fresh REPLs: >= 2/3 GROUNDED."""
    results: list[tuple[str, str]] = []
    for i in range(int(os.environ.get("VECTOR_LIVE_N", "3"))):
        try:
            evidence, verified, text = _launch_then_grasp("把绿色瓶子拿起来", 540, live=True)
            results.append((evidence, verified))
        except AssertionError as exc:
            text = str(exc)
            results.append(("NO_VERDICT", text[-1500:]))
        (tmp_path / f"transcript_{i}.txt").write_text(text)  # why a trial was RAN
        print(f"[live trial {i}] {results[-1][0]} verified={results[-1][1][:200]}")
    grounded = sum(1 for e, v in results if (e, v) == ("GROUNDED", "True"))
    assert grounded * 3 >= 2 * len(results), results
