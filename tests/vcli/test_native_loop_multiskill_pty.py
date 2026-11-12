# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 STEP 3 (A) — native producer SUBSUMES a SECOND distinct (skill, predicate).

Step 1 (``test_native_loop_trichotomy_pty.py``) proved ONE motor skill + ONE
predicate route through ``run_turn_native`` (walk -> at_position). The legacy
planner's job is broader: it routes MANY distinct skills to MANY distinct verify
predicates. Before any legacy deletion we must show the native producer covers more
of that surface. This module is the multi-skill proof on the REAL go2 sim:

    [walk forward ~2m]  -> verify(at_position(9.0, 3.0))   -> GROUNDED + CAUSED
    [turn left]         -> verify(facing(<start+90deg>))     -> GROUNDED + CAUSED
    finish

A SCRIPTED ``FakeToolScriptBackend`` replays the two action->verify pairs (the same
deterministic seam the trichotomy teleport case uses), so the test is network-free,
but the ACTIONS run on a REAL ``MuJoCoGo2`` base and the trace is graded by the
UNTOUCHED honest spine (``VerdictReport.from_trace`` + ``verify_oracle_names``). The
second pair (skill ``turn`` -> predicate ``facing``) is what proves the producer
subsumes a distinct (skill, predicate) the first acceptance never exercised:

  - ROUTING: per-step strategy is the producing skill — ``walk`` then ``turn`` (never
    ``navigate``, which is gated out of actor-causation and not offered as a tool);
  - PREDICATE: each verify names the registry-derived oracle that MEASURES that
    goal — ``at_position`` for a position, ``facing`` for a heading;
  - CAUSATION: a real forward walk grades the ``at_position`` step CAUSED (planar
    channel); a real yaw command + yaw displacement grades the ``facing`` step CAUSED
    (``facing`` carries the ``@oracle(channel="base")`` tag, graded PER-CHANNEL on the heading
    delta — a forward walk's planar move does NOT count for facing). [STEP-12 fix]

GAIT UNDER-ROTATION: the MPC gait achieves only about two thirds of a commanded
open-loop yaw, so this script used to command 135deg to land +90deg. TurnSkill now
closes the loop on the measured heading (WP08), so the script commands exactly the
90deg it verifies. The verify target is a clean start+90deg heading computed from
the base's ACTUAL heading read BEFORE the run (never the achieved heading).

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after the
case via fixtures (the generated scene xml is untracked). Reproduce::

    MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python -m pytest \\
      tests/vcli/test_native_loop_multiskill_pty.py -v -s -m sim
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("mujoco")

# Left turn (counter-clockwise) -> positive yaw; closed-loop, so command what is verified.
_COMMANDED_TURN_DEG: float = 90.0
_TARGET_TURN_RAD: float = math.radians(90.0)  # the heading the step must PROVE


@pytest.mark.sim
def test_native_multiskill_walk_then_turn_routes_and_grounds(sim_cleanup) -> None:
    """walk->at_position THEN turn->facing, both GROUNDED+CAUSED through the spine.

    Mirrors the trichotomy teleport case's setup (real MuJoCoGo2 + a scripted
    backend + ``eng.run_turn_native``), but drives a SECOND distinct (skill,
    predicate) pair to prove the native producer subsumes more of the planner's
    routing — not a hand-built engine loop, the real ``run_turn_native``.
    """
    from pathlib import Path as _Path

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.skills.go2 import get_go2_skills
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import (
        classify_step_evidence,
        verify_oracle_names,
    )
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn

    base = MuJoCoGo2(gui=False, room=True, backend="auto")
    base.connect()
    try:
        base.stand()
        agent = Agent(base=base, llm_api_key="x", config={})
        from tests.harness.fake_gt import bind_real_sim

        bind_real_sim(agent, "go2")  # the GT oracles of a live session
        for s in get_go2_skills():
            agent._skill_registry.register(s)

        # The facing target is computed from the ACTUAL pre-run heading + 90deg, so
        # the predicate is the goal (face start+90deg) — never the outcome.
        start_heading = float(base.get_heading())
        target_rad = start_heading + _TARGET_TURN_RAD

        backend = FakeToolScriptBackend.from_tool_script([
            # PAIR 1: walk 1 m back to (9, 3) (forward is blocked by the pick table).
            tool_turn(("walk", {"direction": "backward", "distance": 1.0, "speed": 0.3})),
            tool_turn(("verify", {"expr": "at_position(9.0, 3.0)"})),
            # PAIR 2: turn left (over-command angle so the gait lands in tol).
            tool_turn(("turn", {"direction": "left", "angle": _COMMANDED_TURN_DEG})),
            tool_turn(("verify", {"expr": f"facing({target_rad:.5f})"})),
            tool_turn(end=True),
        ])
        eng = VectorEngine(
            backend=backend, registry=CategorizedToolRegistry(),
            permissions=PermissionContext(no_permission=True),  # a hand-built rig is no SimSession: grant asks
        )
        eng.bind(agent, RobotWorld())

        session = Session(
            session_id="native-multiskill", created_at="t", updated_at="t",
            path=_Path("/tmp/native_multiskill.jsonl"),
        )
        trace = eng.run_turn_native(
            "走到坐标 (9.0,3.0)，然后向左转 90 度", session=session
        )

        # EXACTLY two StepRecords — one per (action-chain -> verify) pair.
        assert len(trace.steps) == 2, (
            f"expected 2 native steps (walk->at_position, turn->facing); "
            f"got {len(trace.steps)}: "
            f"{[(s.strategy, sg.verify) for s, sg in zip(trace.steps, trace.goal_tree.sub_goals)]}"
        )

        names = verify_oracle_names(agent, eng)
        sub0, sub1 = trace.goal_tree.sub_goals
        step0, step1 = trace.steps

        # STEP 0 — walk -> at_position: GROUNDED + CAUSED (the fix-6 routing contract).
        assert step0.strategy == "walk", f"step0 strategy must be walk; got {step0.strategy!r}"
        assert sub0.verify.startswith("at_position"), f"step0 verify={sub0.verify!r}"
        assert step0.verify_result is True, "step0 walk should reach at_position tol"
        assert step0.actor_caused is ActorCaused.CAUSED, (
            f"step0 walk must grade CAUSED; got {step0.actor_caused.value}"
        )
        assert classify_step_evidence(step0, sub0, names) == "GROUNDED"

        # STEP 1 — turn -> facing: the SECOND distinct (skill, predicate). This is the
        # subsume proof — a turn's yaw command + yaw displacement grades the facing
        # predicate CAUSED through the UNTOUCHED actor-causation spine.
        assert step1.strategy == "turn", f"step1 strategy must be turn; got {step1.strategy!r}"
        assert sub1.verify.startswith("facing"), f"step1 verify={sub1.verify!r}"
        assert step1.verify_result is True, (
            "step1 turn should land within facing tol of start+90deg "
            f"(achieved heading={math.degrees(base.get_heading()):.1f}deg, "
            f"target={math.degrees(target_rad):.1f}deg)"
        )
        assert step1.actor_caused is ActorCaused.CAUSED, (
            f"step1 turn must grade CAUSED (yaw command + yaw displacement); "
            f"got {step1.actor_caused.value}"
        )
        assert classify_step_evidence(step1, sub1, names) == "GROUNDED"

        # SAME verdict gate the PTY harness asserts -> both grounded -> verified.
        report = VerdictReport.from_trace(trace, names)
        assert report.verified is True, f"both grounded steps must verify; {report.evidence}"
        assert report.evidence == "GROUNDED"
        assert report.exit_code() == 0
    finally:
        try:
            base.disconnect()
        except Exception:  # noqa: BLE001
            pass
