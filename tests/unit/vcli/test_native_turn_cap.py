# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""I4 (producers:turn-cap-silent): a native turn that hits the round-trip cap ends on
the record as an aborted step that says so — never as a silent, unexplained stop."""
from __future__ import annotations

from tests.harness.fake_backend import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import _make_agent, _make_engine, _session, _verdict


def test_hitting_the_turn_cap_is_an_aborted_step_that_says_so() -> None:
    from vector_os_nano.vcli.native_loop import run_turn_native

    walk = tool_turn(("walk", {"distance": 0.2, "speed": 0.5}))
    agent, _ = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, FakeToolScriptBackend.from_tool_script([walk, walk, walk, walk]))
    trace = run_turn_native(eng, "walk forward", agent=agent, session=_session(), max_turns=2)
    last = trace.steps[-1]
    assert (last.success, last.error) == (False, "turn cap reached (2 model round-trips)")
    assert _verdict(agent, eng, trace).evidence == "FAILED"
