# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 STEP 4 (A) — native producer SUBSUMES a THIRD capability class: ARM/GRIPPER.

Steps 1-3 proved the native producer routes + grades the go2 BASE capability
(walk->at_position, turn->facing, multi-leg journeys). The legacy planner's job
is broader still: it routes the MANIPULATION class too. Before any legacy deletion
we must show the native producer covers that surface as well. This module is the
arm/gripper proof on the REAL SO-101 arm sim:

    [pick banana, mode=hold]  -> verify(holding_object('banana'))  -> GROUNDED + CAUSED
    finish

A SCRIPTED ``FakeToolScriptBackend`` replays the single (action -> verify) pair (the
same deterministic seam the multiskill test uses), so the test is network-free, but
the PICK runs on the REAL so101 session (the generic Arm + Gripper, contact-gated
grasp) and the trace is graded by the UNTOUCHED honest spine
(``VerdictReport.from_trace`` + ``verify_oracle_names``). The (skill ``pick`` ->
predicate ``holding_object``) pair is what proves the producer subsumes the
arm/gripper class the base acceptances never exercised:

  - ROUTING: the step's strategy is the producing skill — ``pick`` (a manipulation
    skill, never a base motion skill);
  - PREDICATE: the verify names the registry-derived GRIPPER oracle that MEASURES
    the goal — ``holding_object`` (both jaws touch the object, it is attached, it is
    lifted), not a base position/heading oracle;
  - CAUSATION: a real grasp attaches the banana 0->1, which its
    ``@oracle(channel="gripper", transition="rise")`` tag grades CAUSED (a fresh
    attachment this step) — the gripper channel, distinct from the base's.

The arm channel too: ``home`` -> ``arm_at_home`` is GROUNDED and CAUSED on the
so101 (the generic Arm counts its commanded motion); it used to grade UNCAUSED
because the SO-101 driver had no ``ctrl_motion``.

SIM DISCIPLINE: serialized (one sim), headless, caller-stepped; the session is
stopped after each case. Reproduce::

    scripts/gate.sh sim  (GATE_PATHS=tests/vcli/test_native_loop_arm_pty.py)
"""
from __future__ import annotations


import pytest

from tests.harness.sim_session import started

_PICK_TARGET: str = "banana"


def _turn(s, script, text):
    from pathlib import Path as _Path

    from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    eng = VectorEngine(
        backend=FakeToolScriptBackend.from_tool_script(script), registry=CategorizedToolRegistry(),
        permissions=PermissionContext(no_permission=True),  # no REPL: grant asks
    )
    eng.bind(s.agent, RobotWorld())
    session = Session(session_id="native-arm", created_at="t", updated_at="t", path=_Path("/tmp/native_arm.jsonl"))
    return eng, eng.run_turn_native(text, session=session)


def _grounded_and_caused(s, eng, trace, skill: str, oracle: str):
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence, verify_oracle_names
    from vector_os_nano.vcli.verdict import VerdictReport

    assert len(trace.steps) == 1, [(st.strategy, sg.verify) for st, sg in zip(trace.steps, trace.goal_tree.sub_goals)]
    names = verify_oracle_names(s.agent, eng)
    (sub0,) = trace.goal_tree.sub_goals
    (step0,) = trace.steps
    assert step0.strategy == skill, f"step strategy must be {skill}; got {step0.strategy!r}"
    assert sub0.verify.startswith(oracle), f"verify={sub0.verify!r}"
    assert step0.verify_result is True
    assert step0.actor_caused is ActorCaused.CAUSED, f"{skill} must grade CAUSED; got {step0.actor_caused.value}"
    assert classify_step_evidence(step0, sub0, names) == "GROUNDED"
    report = VerdictReport.from_trace(trace, names)
    assert report.verified is True, f"a grounded step must verify; {report.evidence}"
    assert report.evidence == "GROUNDED" and report.exit_code() == 0
    return report


@pytest.mark.sim
def test_native_arm_pick_routes_and_grounds_holding(sim_cleanup) -> None:
    """pick->holding_object, GROUNDED+CAUSED through the UNTOUCHED honest spine."""
    from vector_os_nano.vcli.backends.fake import tool_turn

    with started("so101") as s:
        # The arm STARTS not holding anything: a grasp is a genuine, caused 0->1.
        assert s.oracles()["holding_object"]() is False
        eng, trace = _turn(s, [
            tool_turn(("grasp", {"query": _PICK_TARGET})),
            tool_turn(("verify", {"expr": f"holding_object('{_PICK_TARGET}')"})),
            tool_turn(end=True),
        ], "把香蕉抓起来")  # a closed form: a manipulation verifies only through a proven contract
        report = _grounded_and_caused(s, eng, trace, "grasp", "holding_object")
        assert report.goal_check == "proven", report.reject_reason


@pytest.mark.sim
def test_native_arm_home_grounds_and_is_caused(sim_cleanup) -> None:
    """home->arm_at_home: GROUNDED and CAUSED on the so101 (its Arm counts ctrl_motion)."""
    from vector_os_nano.vcli.backends.fake import tool_turn

    with started("so101") as s:
        assert s.oracles()["arm_at_home"]() is False  # the arm starts stretched out (qpos0)
        eng, trace = _turn(s, [
            tool_turn(("home", {})),
            tool_turn(("verify", {"expr": "arm_at_home()"})),
            tool_turn(end=True),
        ], "回到初始位置")
        _grounded_and_caused(s, eng, trace, "home", "arm_at_home")
