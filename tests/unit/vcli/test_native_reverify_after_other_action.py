# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""I4 / WP07 verifier round a14-6: a re-verify of an already-proven predicate after a
DIFFERENT, FAILED or no-op action must never ground (spine G1).

Before the fix the re-verify joined the proven predicate's attempt group and was
graded against that group's first baseline, so it inherited the navigate's motion,
graded CAUSED and superseded the real row; and a failed actuating tool followed by a
read recorded success=True, because only the chain's LAST tool was looked at.
"""
from __future__ import annotations

from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import (
    _Skill, _make_agent, _make_engine, _nav_agent, _session, _verdict,
)


def _ro(name):
    s = _Skill(name, lambda: None)
    s.__skill_actuates__ = False
    return s


def _run(goal, script, *skills, agent=None):
    agent = agent or _nav_agent(*skills)
    backend = FakeToolScriptBackend.from_tool_script(
        [*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    rep = _verdict(agent, eng, trace)
    rows = [(s.strategy, s.success, s.verify_result, s.actor_caused.value, s.superseded, sg.verify)
            for s, sg in zip(trace.steps, trace.goal_tree.sub_goals)]
    return rep, rows, trace


V = ("verify", {"expr": "at_position(9.0, 3.0)"})
NAV = ("navigate", {"x": 9.0, "y": 3.0})


def test_noop_turn_reverifies_the_proven_target() -> None:
    rep, rows, _tr = _run("走到坐标 (9.0,3.0) 然后左转90度", [NAV, V, ("turn_fake", {}), V],
                          _Skill("turn_fake", lambda: None))
    assert rep.verified is False, rows
    assert rows[-1][3] == "UNCAUSED", rows  # graded on its own step: the turn moved nothing


def test_failed_turn_then_read_then_reverify() -> None:
    # Charter 3.5: the turn erred before commanding motion — REFUSED, not an action — so
    # the re-verify after only a read is a probe (refused unevaluated), and the read's
    # row carries the refusal. Its failed-after-motion twin still rejects
    # (test_native_failed_attempts).
    rep, rows, _tr = _run("走到坐标 (9.0,3.0) 然后左转90度",
                          [NAV, V, ("turn_fake", {}), ("where_fake", {}), V],
                          _Skill("turn_fake", lambda: None, fail="Unknown turn direction"), _ro("where_fake"))
    assert rep.verified is True, (rep.reject_reason, rows)
    assert rows[-1] == ("where_fake", True, False, "NOT_GRADED", False, ""), rows
    assert rep.per_step[-1].refused == ("turn_fake: Unknown turn direction",), rep.per_step


def test_failed_turn_then_reverify_no_contract() -> None:
    # Charter 3.5: a refused turn is not an action, with or without a contract (the
    # failed-after-motion twin rejects: test_native_failed_attempts).
    rep, rows, _tr = _run("go to the kitchen then turn left",
                          [NAV, V, ("turn_fake", {}), ("where_fake", {}), V],
                          _Skill("turn_fake", lambda: None, fail="Unknown turn direction"), _ro("where_fake"))
    assert rep.verified is True and rep.goal_check == "unchecked", (rep.reject_reason, rows)


def test_walk_variant_noop_turn() -> None:
    # walk (in-process, CAUSED at base too) instead of navigate
    agent, _ = _make_agent(0.0, 0.0)
    agent._skill_registry.register(_Skill("turn_fake", lambda: None))
    w = ("verify", {"expr": "at_position(1.0, 0.0, 0.5)"})
    rep, rows, _tr = _run("walk forward 1 m then turn left 90 degrees",
                          [("walk", {"distance": 1.0, "speed": 0.5}), w, ("turn_fake", {}), w], agent=agent)
    assert rep.verified is False, rows


def test_a_stop_recheck_cannot_launder_the_unproven_attempt() -> None:
    """A stop (declared housekeeping) re-checks the group's LATEST attempt: after the
    no-op turn re-verify it is graded against the turn's baseline, never the navigate's."""
    stop = _Skill("stop", lambda: None)
    stop.housekeeping = True
    rep, rows, _tr = _run("走到坐标 (9.0,3.0) 然后左转90度", [NAV, V, ("turn_fake", {}), V, ("stop", {}), V],
                          _Skill("turn_fake", lambda: None), stop)
    assert rep.verified is False, rows
    assert rows[-1][3] == "UNCAUSED", rows


def test_a_failed_action_before_an_auto_verified_one_fails_its_row() -> None:
    """Charter 3.5: the turn erred before commanding motion (REFUSED), so it fails no
    row: two facts, a refused call and a proven goal. (Erring AFTER motion it would be
    its own FAILED row: test_native_failed_attempts.)"""
    rep, rows, _tr = _run("走到坐标 (9.0,3.0)", [("turn_fake", {}), NAV],
                          _Skill("turn_fake", lambda: None, fail="Unknown turn direction"))
    assert [r[1] for r in rows] == [True], rows
    assert rep.verified is True and rep.evidence == "GROUNDED" and rep.goal_check == "proven", rows
    assert rep.per_step[0].refused == ("turn_fake: Unknown turn direction",), rep.per_step


def test_re_dispatching_the_same_action_still_recovers() -> None:
    """The positive twin: a retry that re-runs the group's own action (navigate) is
    graded against the group baseline, so FAIL -> navigate -> PASS verifies."""
    class _Short:
        n = 0

    agent = _nav_agent()
    real = agent._base.navigate_to

    def _nav(x, y, timeout=None, **kwargs):  # noqa: ANN001, ANN202 — the first try lands short
        _Short.n += 1
        return real(5.0, 3.0) if _Short.n == 1 else real(x, y)

    agent._base.navigate_to = _nav
    rep, rows, _tr = _run("走到坐标 (9.0,3.0)", [NAV, V, NAV, V], agent=agent)
    assert rep.verified is True, (rep.reject_reason, rows)
