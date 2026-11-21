# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 acceptance — the R2b trichotomy reproduced THROUGH run_turn_native.

The non-negotiable moat proof (next-prompt.md): reproduce honest / no-op / teleport
through the NEW native tool-use PRODUCER on the REAL go2 sim, with the honest verify
spine UNMODIFIED. The biggest risk is a producer false-green — so the no-op and
teleport cases MUST flip ``verified=False`` through ``run_turn_native``, not the
legacy plan.

- HONEST  [walk -> verify('at_position(x,y)')] -> CAUSED -> GROUNDED -> verified
          True / exit 0. Driven through the REAL ``cli.main -p ... --json
          --sim-go2`` (the R2a PTY harness) with the native tool-
          script seam (VECTOR_FAKE_LLM_TOOLS).
- NO-OP   [verify only, no walk], predicate true at start -> UNCAUSED -> RAN ->
          verified False / exit 2. Same PTY path.
- TELEPORT [a test-only skill pokes qpos, NO set_velocity] -> UNCAUSED -> RAN ->
          verified False / exit 2. Modeled in the TEST (NO production teleport
          tool — forbidden), driven through run_turn_native against the REAL go2
          base + the SAME verdict gate the PTY harness asserts.

ROUTING (review fix 6): the native motor tool-set excludes ``navigate`` (its
cmd_vel is gated out of the actor-causation counter); per-step strategy is asserted
to be ``walk`` (never navigate). The verify expr is asserted to start with
``at_position`` (single-sourced from the registry-derived vocab).

SIM DISCIPLINE: serialized (one sim), headless, MuJoCo closed + the child process group reaped after
each case via fixtures (the generated scene xml is untracked).
"""
from __future__ import annotations


import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

_SIM_TIMEOUT_SEC = 240.0

# go2 starts at (10, 3). A 1.0 m backward walk (SIM-timed) reaches within
# at_position's tol of (9, 3); forward is blocked by the pick table at x~10.47.
_HONEST_SCRIPT = {
    "turns": [
        {"tool_calls": [
            {"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}
        ]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}

# NO-OP: verify the START position (10,3) with NO walk -> predicate true at
# baseline, zero commanded motion -> UNCAUSED -> RAN.
_NOOP_SCRIPT = {
    "turns": [
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(10.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


# ---------------------------------------------------------------------------
# HONEST + NO-OP — THROUGH run_turn_native via the REAL cli.main + PTY harness
# ---------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_honest_walk_is_caused_and_verified(sim_cleanup) -> None:
    r = run_cli_turn(
        "走到坐标 (9.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_HONEST_SCRIPT,
    )
    assert r.verified is True, f"native honest walk should verify; got {r.verdict}"
    assert r.exit_code == 0, f"verified honest walk must exit 0; got {r.exit_code}"
    assert r.evidence == "GROUNDED", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "GROUNDED"
    assert step["success"] is True and step["verify_result"] is True
    # ROUTING contract (review fix 6): walk, not navigate; verify starts at_position.
    assert step["strategy"] == "walk", f"per-step strategy must be walk; got {step['strategy']}"
    assert step["verify"].startswith("at_position"), f"got verify={step['verify']}"


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_noop_is_uncaused_and_not_verified(sim_cleanup) -> None:
    r = run_cli_turn(
        "声称已在 (10.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_NOOP_SCRIPT,
    )
    assert r.verified is False, f"a native no-op must NOT verify; got {r.verdict}"
    assert r.exit_code == 2, f"ran-not-verified must exit 2; got {r.exit_code}"
    assert r.evidence == "RAN", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["success"] is True and step["verify_result"] is True
    assert step["evidence"] == "RAN"


# ---------------------------------------------------------------------------
# TELEPORT — test-only skill on the REAL go2 sim, graded THROUGH run_turn_native
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_native_teleport_is_uncaused_and_not_verified(sim_cleanup) -> None:
    """A test-only teleport SKILL pokes qpos with NO set_velocity, driven by the
    native loop -> UNCAUSED -> RAN -> verified False / exit 2.

    NO production teleport tool (forbidden). The skill is registered ONLY in this
    test and invoked by the SAME run_turn_native producer; the trace is graded by
    the SAME VerdictReport.from_trace + verify_oracle_names the PTY harness asserts.
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.types import SkillResult
    from vector_os_nano.skills import get_skills
    from vector_os_nano.embodiments.registry import load as load_robot
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    from pathlib import Path as _Path

    # A TEST-ONLY teleport skill: pokes the root x, NO set_velocity. Wrapped as a
    # native motor tool so the model can "call" it — but the commanded-motion
    # counter never advances -> the grader sees displacement w/ zero command.
    class _TeleportSkill:
        name = "teleport"
        description = "TEST-ONLY: jump the base to a coordinate (moves the base)."
        parameters = {"x": {"type": "number", "required": True}}
        preconditions = ["base"]
        effects = {"is_moving": False}

        def execute(self, params, context):
            base, rt = context.base, dog.rt  # the root freejoint is NOT qpos[0] once pickables precede it
            sc = rt.world.scene
            adr, x = sc.free_q(sc.robot_root).start, float(params.get("x", 9.0))
            rt.call(lambda w: w.data.qpos.__setitem__(adr, x))  # at a step boundary
            base.wait_sim(0.15)
            return SkillResult(success=True, result_data={"x": params.get("x")})

    from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn

    from tests.hardware.sim.dog import stand_up

    dog = stand_up("go2")
    base = dog.base
    try:
        agent = Agent(base=base, config={})
        from tests.harness.fake_gt import bind_real_sim

        bind_real_sim(agent, "go2", dog.rt)  # the GT oracles of a live session
        for s in get_skills(load_robot("go2").capabilities):
            agent._skill_registry.register(s)
        agent._skill_registry.register(_TeleportSkill())

        backend = FakeToolScriptBackend.from_tool_script([
            tool_turn(("teleport", {"x": 9.0})),
            tool_turn(("verify", {"expr": "at_position(9.0, 3.0)"})),
            tool_turn(end=True),
        ])
        eng = VectorEngine(
            backend=backend, registry=CategorizedToolRegistry(),
            permissions=PermissionContext(no_permission=True),  # a hand-built rig is no SimSession: grant asks
        )
        eng.bind(agent, RobotWorld())

        session = Session(
            session_id="native-teleport", created_at="t", updated_at="t",
            path=_Path("/tmp/native_teleport.jsonl"),
        )
        start = list(base.get_position()[:2])
        trace = eng.run_turn_native("走到坐标 (9.0,3.0)", session=session)
        step = trace.steps[0]

        # The teleport "succeeded" and the predicate is TRUE (pose jumped to ~9),
        # but no command was issued — the native producer's grader catches it.
        assert step.success is True
        assert step.verify_result is True
        assert step.actor_caused is ActorCaused.UNCAUSED
        assert abs(base.cmd_motion()) < 1e-9, "teleport issued zero commanded motion"
        assert base.get_position()[0] < start[0] - 0.5, "qpos actually jumped"
        # The per-step strategy is the teleport skill (NOT navigate) — routing held.
        assert step.strategy == "teleport"

        # SAME verdict gate the PTY harness asserts -> verified False / exit 2.
        report = VerdictReport.from_trace(trace, verify_oracle_names(agent, eng))
        assert report.verified is False
        assert report.evidence == "RAN"
        assert report.exit_code() == 2
    finally:
        dog.close()
