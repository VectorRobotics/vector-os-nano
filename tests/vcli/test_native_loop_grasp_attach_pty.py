# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S3b SEAL — the native producer grounds a PERCEPTION grasp on the go2+Piper ATTACH scene.

``test_native_loop_arm_pty.py`` proves the native producer routes + grades the
arm/gripper class — but only on the STANDALONE SO-101 arm, whose ``pick`` reads
the simulator's omniscient object table. This module is the analogue on the
combined **go2 + Piper attach scene** (the scene S3b's ONE ``MjSpec.attach``
builder assembles), driven by the honest, perception-first skill the North Star
calls for — ``perception_grasp``:

    [抓绿色的瓶子]  -> verify(holding_object('pickable_bottle_green'))  -> GROUNDED + CAUSED
    finish

A SCRIPTED ``FakeToolScriptBackend`` replays the single (action -> verify) pair,
so the test is NETWORK-FREE (no deepseek multi-turn REPL — 5 prior live runs
stalled on model REPL variance: network drop / cold-load timeout / a faked grasp
/ bash-wander / launch-wander, NEVER the grasp code). But the grasp runs the FULL
real pipeline on a REAL ``MuJoCoGo2`` + ``MuJoCoPiper`` + ``MuJoCoPiperGripper``
base — VLM/colour resolve -> EdgeTAM segment -> rendered-depth pointcloud -> 3D
grasp point (NEVER ground truth) -> approach walk -> Piper top-down IK -> weld ->
lift — and the trace is graded by the UNTOUCHED honest spine
(``VerdictReport.from_trace`` + ``verify_oracle_names``) PLUS the D69 object-goal
turn gate (the goal "把前面的绿色瓶子抓起来" carries grasp intent, so the turn passes
ONLY if a step is GROUNDED via a NECESSARY ``holding_object`` conjunct).

What this proves that the SO-101 ``pick`` template cannot:
  - ROUTING: the producing skill is the perception-first manipulation skill
    ``perception_grasp`` (never ``pick``, which reads GT pose);
  - PREDICATE: the verify names the registry-derived GRIPPER oracle that MEASURES
    the goal — ``holding_object('pickable_bottle_green')`` — TARGET-AWARE (the
    green bottle, not "holding anything");
  - CAUSATION: a real Piper grasp toggles the green bottle's weld 0->1, graded
    CAUSED via ``MuJoCoPiperGripper.weld_is_active()`` — the gripper-weld channel
    on the go2 physics thread (distinct from the SO-101's standalone arm);
  - HONESTY: the 3D grasp point comes ONLY from rendered depth + the EdgeTAM/colour
    mask; the skill FAILS LOUD if perception fails and NEVER substitutes a GT pose.

Why GREEN (a real, verified choice):
``pickable_bottle_green`` sits dead-ahead on the dog's spawn-y line (y=3.0), the
D47-proven head-on "前面的东西" target. A colour query ("抓绿色的瓶子") routes through
the classical HSV ``front_object`` resolver (no grounding-dino network) and resolves
the verify LABEL to ``pickable_bottle_green`` (``_COLOR_TO_SCENE``), so the
target-aware oracle matches the body the weld pins — the faithful analogue of the
arm template's ``holding_object('banana')``. The dog boots NOT holding (all welds
``active="false"``), so a successful grasp is a genuine actor-CAUSED 0->1 transition.

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after the
case via the fixture. ``VECTOR_SIM_WITH_ARM=1`` MUST be set before the scene builds
so ``MuJoCoGo2`` assembles the go2_piper attach scene (cylinders + welds). Reproduce::

    MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python -m pytest \\
      tests/vcli/test_native_loop_grasp_attach_pty.py -v -s -m sim
"""
from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

# Colour query -> classical HSV front_object resolver (no network); resolves the
# verify LABEL to the green bottle's scene name. Dead-ahead on the spawn-y line.
_GRASP_QUERY: str = "抓绿色的瓶子"
_GRASP_TARGET: str = "pickable_bottle_green"
# The NL goal carries grasp INTENT (抓) -> the D69 object-goal turn gate applies,
# so the turn passes ONLY if a step is GROUNDED via a NECESSARY holding_object.
_GOAL: str = "把前面的绿色瓶子抓起来"


@pytest.mark.sim
def test_native_grasp_attach_routes_and_grounds_holding(sim_cleanup, monkeypatch) -> None:
    """perception_grasp -> holding_object, GROUNDED+CAUSED on the go2+Piper attach scene.

    Mirrors ``test_native_loop_arm_pty.py`` (a real sim base + an ``Agent`` with the
    go2+arm skill set + a scripted backend + the real ``eng.run_turn_native``), but
    drives the PERCEPTION grasp on the COMBINED dog+arm attach scene to seal S3b: the
    grasp pipeline routes to the real ``perception_grasp`` and grounds through the
    untouched spine + the D69 gate, with NO live LLM in the loop.
    """
    # Select the go2_piper attach scene (cylinders + INACTIVE welds) at scene-build
    # time — MuJoCoGo2._build_room_scene_xml reads this env var.
    monkeypatch.setenv("VECTOR_SIM_WITH_ARM", "1")

    from pathlib import Path as _Path

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
    from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception
    from vector_os_nano.skills.go2 import get_go2_skills
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.mobile_place import MobilePlaceSkill
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import (
        classify_step_evidence,
        evidence_passed,
        verify_oracle_names,
    )
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds.arm_sim_oracle import make_holding_object
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    from tests.harness.fake_backend import FakeToolScriptBackend, tool_turn

    # Build the go2+Piper world EXACTLY as sim_tool._start_go2(with_arm=True) does,
    # but IN-PROCESS (no ROS2 proxy) — the analogue of test_mujoco_piper.py's fixture.
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc")
    go2.connect()
    piper = gripper = None
    try:
        piper = MuJoCoPiper(go2)
        piper.connect()
        gripper = MuJoCoPiperGripper(go2)
        gripper.connect()
        perception = Go2GraspPerception(go2, width=320, height=240)
        agent = Agent(base=go2, arm=piper, gripper=gripper, perception=perception, config={})

        # Skill set EXACTLY as sim_tool registers for go2+arm: base skills + the
        # Piper manipulation skills + perception_grasp LAST (wins the 抓/grab aliases).
        for s in get_go2_skills():
            agent._skill_registry.register(s)
        for sk in (PickTopDownSkill(), PlaceTopDownSkill(), MobilePickSkill(), MobilePlaceSkill()):
            agent._skill_registry.register(sk)
        agent._skill_registry.register(PerceptionGraspSkill())

        # The dog must START not holding anything — that is what makes a successful
        # grasp a genuine, actor-CAUSED 0->1 weld transition (never a NO-OP).
        assert make_holding_object(agent)() is False, (
            "the go2+Piper rig must boot NOT holding so the grasp is genuinely caused"
        )

        backend = FakeToolScriptBackend.from_tool_script([
            # The ONLY (action -> verify) pair: perceive+grasp the green bottle, then
            # prove possession via the target-aware GT oracle.
            tool_turn(("perception_grasp", {"query": _GRASP_QUERY})),
            tool_turn(("verify", {"expr": f"holding_object('{_GRASP_TARGET}')"})),
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
            session_id="native-grasp", created_at="t", updated_at="t",
            path=_Path("/tmp/native_grasp_attach.jsonl"),
        )
        trace = eng.run_turn_native(_GOAL, session=session)

        # EXACTLY one StepRecord — one (perception_grasp -> holding_object) pair.
        assert len(trace.steps) == 1, (
            f"expected 1 native step (perception_grasp->holding_object); got "
            f"{len(trace.steps)}: "
            f"{[(s.strategy, sg.verify) for s, sg in zip(trace.steps, trace.goal_tree.sub_goals)]}"
        )

        names = verify_oracle_names(agent, eng)
        (sub0,) = trace.goal_tree.sub_goals
        (step0,) = trace.steps

        # ROUTING — the producing skill is the perception-first grasp, never `pick`.
        assert step0.strategy == "perception_grasp", (
            f"step strategy must be perception_grasp; got {step0.strategy!r}"
        )
        # PREDICATE — the registry-derived GRIPPER oracle that MEASURES the goal,
        # TARGET-AWARE on the green bottle.
        assert sub0.verify.startswith("holding_object"), f"verify={sub0.verify!r}"
        # GROUNDING — a real perceived grasp lifted + welded the green bottle.
        assert step0.verify_result is True, (
            f"perception_grasp should leave holding_object True; "
            f"result_data={getattr(step0, 'result_data', None)}"
        )
        # CAUSATION — a fresh Piper weld 0->1 grades CAUSED through the UNTOUCHED
        # actor-causation spine (gripper-weld channel on the go2 physics thread).
        assert step0.actor_caused is ActorCaused.CAUSED, (
            f"grasp must grade CAUSED (fresh weld 0->1); got {step0.actor_caused.value}"
        )
        assert classify_step_evidence(step0, sub0, names) == "GROUNDED"

        # SAME verdict gate the PTY harness asserts -> grounded -> verified.
        report = VerdictReport.from_trace(trace, names)
        assert report.verified is True, f"a grounded grasp must verify; {report.evidence}"
        assert report.evidence == "GROUNDED"
        assert report.exit_code() == 0

        # D69 OBJECT-GOAL GATE (explicit): the goal carries grasp intent and the world
        # exposes holding_object, so the turn passes the gate ONLY because a step is
        # GROUNDED via a NECESSARY holding_object conjunct (a faked file-based grasp
        # would downgrade to RAN here).
        assert evidence_passed(trace, names) is True
    finally:
        for dev in (gripper, piper, go2):
            if dev is None:
                continue
            try:
                dev.disconnect()
            except Exception:  # noqa: BLE001
                pass
