# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""I4 G4 (spine:trailing-read-grades-ran): a read-only tool after the last verified step
must not downgrade a verified turn — with or without a task contract — while a turn
that only read, or a read that failed, still never verifies."""
from __future__ import annotations

from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import _Skill, _make_agent, _make_engine, _session, _verdict


def _read(name: str, fail: str = ""):
    s = _Skill(name, lambda: None, fail=fail)
    s.__skill_actuates__ = False
    return s


def _run(goal: str, script, *skills):
    agent, _ = _make_agent(0.0, 0.0)
    for sk in skills:
        agent._skill_registry.register(sk)
    backend = FakeToolScriptBackend.from_tool_script(
        [*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    return trace, _verdict(agent, eng, trace)


_WALK = ("walk", {"distance": 1.0, "speed": 0.5})
_AT = ("verify", {"expr": "at_position(1.0, 0.0, 0.5)"})


def test_a_trailing_read_does_not_downgrade_a_verified_turn() -> None:
    trace, rep = _run("walk forward 1 m then tell me where you are", [_WALK, _AT, ("where_fake", {})],
                      _read("where_fake"))
    assert trace.contract == ()  # no contract: the old exemption never applied here
    assert [(s.strategy, s.housekeeping) for s in trace.steps] == [("walk", False), ("where_fake", True)]
    assert rep.verified is True, rep.reject_reason


def test_a_turn_that_only_read_never_verifies() -> None:
    _trace, rep = _run("where are you", [("where_fake", {})], _read("where_fake"))
    assert rep.verified is False and rep.reject_reason == "no checked step"


def test_a_failed_trailing_read_is_still_a_failed_step() -> None:
    _trace, rep = _run("walk forward 1 m then tell me where you are", [_WALK, _AT, ("where_fake", {})],
                       _read("where_fake", fail="localization lost"))
    assert rep.verified is False and rep.evidence == "FAILED"


def test_an_unproven_actuating_step_after_a_verified_one_still_fails_the_turn() -> None:
    _trace, rep = _run("walk forward 1 m then turn left", [_WALK, _AT, ("turn_fake", {})],
                       _Skill("turn_fake", lambda: None))
    assert rep.verified is False and rep.reject_reason.startswith("native_step_1:")
