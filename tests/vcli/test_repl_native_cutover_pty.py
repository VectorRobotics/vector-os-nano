# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""REPL CUTOVER acceptance — drive the REAL interactive vector-cli REPL by NL on go2 sim.

The owner's ONLY test interface is bare ``vector-cli`` + natural language. This pins
the cutover ON THE REAL SIM through the ACTUAL interactive REPL (not ``-p``, not a
hand-built engine script): a natural-language nav command must route to the NATIVE
tool-use producer, drive the real go2 in MuJoCo, and render the HONEST moat verdict in
the conversation. Deterministic via the ``VECTOR_FAKE_LLM_TOOLS`` seam (the real
walk skill + GoalVerifier + actor-causation + verdict still run on real sim state; only
the network LLM is replaced). The real-LLM end-to-end is covered by the live PTY tests
+ the orchestrator's manual REPL drive; this is the regression guard for the WIRING.

Asserts on the rendered transcript (the REPL prints a human verdict, not the JSON
sentinel). Exit code is NOT asserted: the go2 MuJoCo teardown SIGABRTs on quit (a known
shutdown artifact AFTER the verdict has already been rendered).
"""
from __future__ import annotations

import re

import pytest

# go2 starts at (10, 3); a generous forward walk reaches within at_position tol of
# (11, 3) (the same honest-walk script the trichotomy / native-first -p tests use).
_WALK_SCRIPT = {
    "turns": [
        {"tool_calls": [
            {"name": "walk", "input": {"direction": "forward", "distance": 2.5, "speed": 0.3}}
        ]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(11.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}

@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_repl_native_walk_routes_to_native_on_go2_sim(sim_cleanup) -> None:
    """`vector-cli` + "走到坐标 (11,3)" routes to native on the real go2 sim, verifies honestly."""
    pytest.importorskip("mujoco")
    from tests.harness.pty_cli import run_repl_session

    result = run_repl_session(
        ["走到坐标 (11.0,3.0)"], sim_go2=True, tool_script=_WALK_SCRIPT, native=True,
        timeout_sec=240,
    )
    text = result.text
    assert not result.timed_out, text[-2500:]

    # The NL command reached the REPL.
    assert "走到坐标" in text, f"command not echoed; transcript:\n{result.transcript[:2000]}"
    # The CUTOVER engaged: native (not the legacy planner) handled the turn.
    assert "native" in text and "working" in text, "native producer was not attempted in the REPL"
    # Native dispatched the WALK skill (routing contract: walk, not navigate) and the
    # deterministic verify read real sim state.
    assert "walk" in text and "verify at_position" in text, (
        f"native walk/verify not rendered; transcript:\n{text[-2500:]}"
    )
    # The HONEST moat verdict surfaced in the conversation: GROUNDED + actor CAUSED +
    # verified True (the robot actually reached the target on the real sim).
    assert "GROUNDED" in text, f"verdict not GROUNDED; transcript:\n{text[-2500:]}"
    assert "actor=CAUSED" in text, f"actor-causation not CAUSED; transcript:\n{text[-2500:]}"
    assert "verified=True" in text, f"verdict not verified; transcript:\n{text[-2500:]}"


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_repl_legacy_env_does_not_attempt_native(sim_cleanup) -> None:
    """VECTOR_REPL_NATIVE=0 forces the pure-legacy REPL — native is NEVER attempted.

    The reversible escape hatch: with the cutover disabled, a nav command must NOT show
    the native producer's "working…" spinner / "actor=CAUSED" verdict (it runs the
    legacy VGG planner instead). Proves the cutover is gated and reversible on the real
    product, not just in unit tests.
    """
    pytest.importorskip("mujoco")
    from tests.harness.pty_cli import run_repl_session

    result = run_repl_session(
        ["走到坐标 (11.0,3.0)"], sim_go2=True, tool_script=_WALK_SCRIPT,
        native=False,  # VECTOR_REPL_NATIVE=0
        timeout_sec=240,
    )
    text = result.text
    assert not result.timed_out, text[-2500:]
    # POSITIVE legacy marker: the legacy planner rendered its plan for the command
    # (an absence-only check also passed when the REPL hung at boot).
    assert re.search(r"VGG .*走到坐标", text), f"legacy plan not rendered:\n{text[-2500:]}"
    # The native cutover path's signatures must be ABSENT (legacy planner handled it).
    assert "actor=CAUSED" not in text, "native verdict rendered despite VECTOR_REPL_NATIVE=0"
