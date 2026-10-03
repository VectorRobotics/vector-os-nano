# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2b acceptance — actor-causation grading through the REAL go2 sim + verdict gate.

The verify-moat acceptance for R2b: a GROUNDED robot-predicate step is verified
ONLY when the ACTOR caused the state change. Exercised on the REAL MuJoCo go2 sim
(starts at (10,3)) through the SAME ``evidence_passed`` / ``classify_step_evidence``
gate that the VECTOR_VERDICT carries — never a re-derivation.

Three cases, each on the real sim, each landing on the real verdict:

- HONEST walk  (verify=at_position(9.0,3.0), strategy walk_skill -> base.walk()):
      the actor COMMANDS motion (instrumented set_velocity) AND displaces ~1m ->
      CAUSED -> GROUNDED -> verified True / exit 0.  Driven through the R2a PTY
      harness (the REAL ``python -m vector_os_nano.vcli.cli -p ... --json --sim-go2``).

- NO-OP        (verify=at_position(10.0,3.0) already true @ start, stand_skill):
      predicate true at baseline + zero commanded motion -> UNCAUSED -> downgraded
      to RAN -> verified False / exit 2.  Also driven through the PTY harness.

- TELEPORT     (a fake/test sub_goal pokes the root x=9.0 with NO set_velocity):
      pose jumps but zero commanded motion -> UNCAUSED -> RAN -> verified False /
      exit 2.  Modeled by a TEST-ONLY primitive (NO production teleport tool — the
      qpos poke cannot cross the cli subprocess boundary without one, which is
      forbidden), driven against the REAL go2 sim + REAL engine + the SAME verdict
      gate the PTY harness asserts (VerdictReport.from_trace + verify_oracle_names).

SCOPE (honest, per the actor_causation docstring): the live ``navigate`` route is
OUT OF SCOPE — it drives via a ROS2 bridge whose cmd_vel is GATED OUT before the
instrumented counter (mujoco_go2.set_velocity ``_gated`` guard), so an honest
navigate would false-FAIL. ``walk_skill`` (-> base.walk(), which sets
``_skill_ctrl_tid``) is the honest, instrumented route.

SIM DISCIPLINE: serialized (one sim at a time), headless (no GLFW window — the
viewer's glXSwapBuffers crashes under a PTY-spawned subprocess), MuJoCo closed +
its own process group reaped after each case (sim_cleanup).
"""
from __future__ import annotations


import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

# go2 starts at (10, 3); a 1.0 m backward walk_skill (SIM-timed) must reach within
# at_position's 0.5m tol of (9, 3) (forward is blocked by the pick table at x~10.47).
_HONEST_PLAN = {
    "goal": "走到坐标 (9.0,3.0)",
    "sub_goals": [
        {
            "name": "walk_to_target",
            "description": "向后走到 (9,3)",
            "verify": "at_position(9.0, 3.0)",
            "strategy": "walk_skill",  # validator-kept; -> WalkSkill -> base.walk()
            "strategy_params": {"direction": "backward", "distance": 1.0, "speed": 0.3},
        }
    ],
}

# NO-OP: verify the START position (10,3) with a non-moving strategy. The predicate
# is already true at baseline; the actor commands no motion -> UNCAUSED -> RAN.
_NOOP_PLAN = {
    "goal": "走到坐标 (10.0,3.0)",
    "sub_goals": [
        {
            "name": "stay_put",
            "description": "原地待命",
            "verify": "at_position(10.0, 3.0)",
            "strategy": "stand_skill",  # non-moving, validator-kept
            "strategy_params": {},
        }
    ],
}

_SIM_TIMEOUT_SEC = 220.0


# ---------------------------------------------------------------------------
# HONEST + NO-OP — through the REAL cli.main + PTY harness on the real go2 sim
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_honest_walk_is_caused_and_verified(sim_cleanup) -> None:
    """Honest walk_skill -> base.walk() displaces ~1m -> CAUSED -> verified/exit 0."""
    r = run_cli_turn(
        "走到坐标 (9.0,3.0)",
        fake_plan=_HONEST_PLAN,
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
    )
    assert r.verified is True, f"honest walk should verify; got {r.verdict}"
    assert r.exit_code == 0, f"verified honest walk must exit 0; got {r.exit_code}"
    assert r.evidence == "GROUNDED", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "GROUNDED"
    assert step["success"] is True and step["verify_result"] is True


@pytest.mark.sim
def test_noop_is_uncaused_and_not_verified(sim_cleanup) -> None:
    """Predicate true at baseline + no commanded motion -> UNCAUSED -> RAN/exit 2."""
    r = run_cli_turn(
        "走到坐标 (10.0,3.0)",
        fake_plan=_NOOP_PLAN,
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
    )
    assert r.verified is False, f"a no-op must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    # The step ran and its predicate is true, but the moat downgrades it to RAN.
    assert step["success"] is True and step["verify_result"] is True
    assert step["evidence"] == "RAN"


# ---------------------------------------------------------------------------
# TELEPORT — fake sub_goal on the REAL go2 sim + REAL engine + the verdict gate
# (no production teleport tool; the poke cannot cross the cli subprocess boundary)
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_teleport_is_uncaused_and_not_verified(sim_cleanup) -> None:
    """A fake sub_goal pokes qpos with NO set_velocity -> UNCAUSED -> RAN/exit 2.

    Models the teleport on the REAL go2 sim with a TEST-ONLY primitive (NO
    production teleport tool — explicitly forbidden), executed by the REAL
    GoalExecutor against the REAL go2 base, then graded by the SAME verdict gate
    (VerdictReport.from_trace + verify_oracle_names) the PTY harness asserts.
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.skills.go2 import get_go2_skills
    from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
    from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    base = MuJoCoGo2(gui=False, room=True, backend="auto")
    base.connect()
    try:
        base.stand()
        agent = Agent(base=base, llm_api_key="x", config={})
        for s in get_go2_skills():
            agent._skill_registry.register(s)
        eng = VectorEngine(
            backend=None, registry=CategorizedToolRegistry(),
            permissions=PermissionContext(),
        )
        eng._world = RobotWorld()
        eng.init_vgg(agent=agent, skill_registry=agent._skill_registry, world=RobotWorld())

        # TEST-ONLY teleport: poke the root x at a step boundary (runtime.call),
        # NO set_velocity. The commanded-motion counter never advances, so the
        # grader sees a large displacement with zero commanded motion -> UNCAUSED.
        adr = base._mj.layout.root_qpos_adr

        def teleport_primitive(**_):
            base.runtime.call(lambda _m, d: d.qpos.__setitem__(adr, 9.0))
            base.wait_sim(0.15)
            return True

        ex = GoalExecutor(
            strategy_selector=StrategySelector(has_base=True),
            verifier=eng._goal_executor._verifier,
            primitives={"teleport_skill": teleport_primitive},
            agent=agent,
        )
        sg = SubGoal(
            name="tp", description="瞬移(fake)",
            verify="at_position(9.0, 3.0)", strategy="teleport_skill",
            timeout_sec=20.0,
        )
        start = list(base.get_position()[:2])
        trace = ex.execute(GoalTree(goal="走到坐标 (9.0,3.0)", sub_goals=(sg,)))
        step = trace.steps[0]

        # The teleport "succeeded" and the predicate is TRUE (pose jumped to ~9),
        # but no command was issued — the grader catches it.
        assert step.success is True
        assert step.verify_result is True
        from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
        assert step.actor_caused is ActorCaused.UNCAUSED
        assert abs(base.cmd_motion()) < 1e-9, "teleport issued zero commanded motion"
        assert base.get_position()[0] < start[0] - 0.5, "qpos actually jumped"

        # The SAME verdict gate the PTY harness asserts -> verified False / exit 2.
        report = VerdictReport.from_trace(trace, verify_oracle_names(agent, eng))
        assert report.verified is False
        assert report.evidence == "RAN"
        assert report.exit_code() == 2
    finally:
        try:
            base.disconnect()
        except Exception:  # noqa: BLE001
            pass
