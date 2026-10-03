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
the PICK runs on a REAL ``MuJoCoArm`` + ``MuJoCoGripper`` base and the trace is
graded by the UNTOUCHED honest spine (``VerdictReport.from_trace`` +
``verify_oracle_names``). The (skill ``pick`` -> predicate ``holding_object``) pair
is what proves the producer subsumes the arm/gripper class the base acceptances
never exercised:

  - ROUTING: the step's strategy is the producing skill — ``pick`` (a manipulation
    skill, never a base motion skill);
  - PREDICATE: the verify names the registry-derived GRIPPER oracle that MEASURES
    the goal — ``holding_object`` (was the requested object grasped), not a base
    position/heading oracle;
  - CAUSATION: a real grasp toggles the banana's weld constraint 0->1, which
    R2b's ``_GRIPPER_PREDICATES`` grades CAUSED via ``gripper.weld_is_active()`` (a
    fresh weld transition this step). This is a DISTINCT causation channel from the
    base steps (commanded |cmd| + pose displacement) — the gripper-weld channel.

MOAT-HONEST CHOICE — pick, not home (a real, verified finding):
The SO-101 ``MuJoCoArm`` exposes NO ``ctrl_motion`` instrumentation (only the
go2-mounted ``MuJoCoPiper`` does), so an ``arm_at_home`` step grades UNCAUSED on
this sim (``ActorBaseline.arm_ctrl_motion`` is None -> ``grade`` fails closed). The
GRIPPER channel does NOT depend on ``ctrl_motion`` — it reads the live weld state —
so PICK -> ``holding_object`` is the cleanly-CAUSED capability here. The arm boots
NOT holding anything (every ``<weld ... active="false">`` in so101_mujoco.xml), so a
real grasp is a genuine 0->1 transition BY THE ACTOR: a true caused state change,
not a satisfied-at-baseline NO-OP.

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after the
case via the fixture. The arm sim does NOT write its scene xml (verified), so no
scene restore is needed. Reproduce::

    MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python -m pytest \\
      tests/vcli/test_native_loop_arm_pty.py -v -s -m sim
"""
from __future__ import annotations


import pytest

pytest.importorskip("mujoco")

# The graspable object the script targets — present in so101_mujoco.xml with a
# pre-defined weld constraint, and the historic 30/30 grasp target.
_PICK_TARGET: str = "banana"


@pytest.mark.sim
def test_native_arm_pick_routes_and_grounds_holding(sim_cleanup) -> None:
    """pick->holding_object, GROUNDED+CAUSED through the UNTOUCHED honest spine.

    Mirrors the multiskill test's setup (a real sim base + an ``Agent`` with the
    default skills + a scripted backend + the real ``eng.run_turn_native``), but
    drives the arm/gripper (skill, predicate) pair to prove the native producer
    subsumes a THIRD distinct capability class — not a hand-built engine loop, the
    real ``run_turn_native``.
    """
    from pathlib import Path as _Path

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.skills import get_default_skills
    from vector_os_nano.skills.pick import SIM_PICK_CONFIG
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

    from tests.harness.fake_backend import FakeToolScriptBackend, tool_turn

    # Build the arm agent EXACTLY as the --sim CLI branch does (cli.py:543-557):
    # MuJoCoArm + MuJoCoGripper + MuJoCoPerception + SIM_PICK_CONFIG, default skills.
    arm = MuJoCoArm(gui=False)
    arm.connect()
    try:
        gripper = MuJoCoGripper(arm)
        perception = MuJoCoPerception(arm)
        agent = Agent(
            arm=arm,
            gripper=gripper,
            perception=perception,
            config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
        )
        for s in get_default_skills():
            agent._skill_registry.register(s)

        # The arm must START not holding anything — that is what makes a successful
        # grasp a genuine, actor-CAUSED 0->1 weld transition (never a NO-OP).
        from vector_os_nano.vcli.worlds.arm_sim_oracle import make_holding_object
        assert make_holding_object(agent)() is False, (
            "arm must boot NOT holding so the pick is genuinely caused"
        )

        backend = FakeToolScriptBackend.from_tool_script([
            # The ONLY (action -> verify) pair: grasp + hold, then prove the grasp.
            tool_turn(("pick", {"object_label": _PICK_TARGET, "mode": "hold"})),
            tool_turn(("verify", {"expr": f"holding_object('{_PICK_TARGET}')"})),
            tool_turn(end=True),
        ])
        eng = VectorEngine(
            backend=backend, registry=CategorizedToolRegistry(),
            permissions=PermissionContext(),
        )
        eng._world = RobotWorld()
        eng.init_vgg(agent=agent, skill_registry=agent._skill_registry, world=RobotWorld())
        eng._vgg_agent = agent
        eng._backend = backend

        session = Session(
            session_id="native-arm", created_at="t", updated_at="t",
            path=_Path("/tmp/native_arm.jsonl"),
        )
        trace = eng.run_turn_native(
            "把香蕉抓起来拿在手里", session=session
        )

        # EXACTLY one StepRecord — one (action-chain -> verify) pair.
        assert len(trace.steps) == 1, (
            f"expected 1 native step (pick->holding_object); got {len(trace.steps)}: "
            f"{[(s.strategy, sg.verify) for s, sg in zip(trace.steps, trace.goal_tree.sub_goals)]}"
        )

        names = verify_oracle_names(agent, eng)
        (sub0,) = trace.goal_tree.sub_goals
        (step0,) = trace.steps

        # ROUTING — the producing skill is the manipulation skill ``pick`` (fix-6).
        assert step0.strategy == "pick", f"step strategy must be pick; got {step0.strategy!r}"
        # PREDICATE — the registry-derived GRIPPER oracle that MEASURES the goal.
        assert sub0.verify.startswith("holding_object"), f"verify={sub0.verify!r}"
        # GROUNDING — a real grasp lifted + welded the requested object.
        assert step0.verify_result is True, "pick should leave holding_object True"
        # CAUSATION — the THIRD distinct causation channel: a fresh weld 0->1 grades
        # the gripper predicate CAUSED through the UNTOUCHED actor-causation spine
        # (NOT the base commanded-motion channel the walk/turn steps use).
        assert step0.actor_caused is ActorCaused.CAUSED, (
            f"pick must grade CAUSED (fresh weld 0->1); got {step0.actor_caused.value}"
        )
        assert classify_step_evidence(step0, sub0, names) == "GROUNDED"

        # SAME verdict gate the PTY harness asserts -> grounded -> verified.
        report = VerdictReport.from_trace(trace, names)
        assert report.verified is True, f"a grounded grasp must verify; {report.evidence}"
        assert report.evidence == "GROUNDED"
        assert report.exit_code() == 0
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001
            pass
