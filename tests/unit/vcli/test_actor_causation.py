# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2b unit tests — actor-causation grading (deterministic fakes, no MuJoCo).

Pin the actor-causation mechanism in isolation, on duck-typed fakes (no sim):

(snapshot)  capture(agent) reads ONLY the commanded-motion counters + pose; a
            (0,0,0) "write" (zero magnitude) does NOT satisfy causation.
(grade)     honest / no-op / teleport / missing-baseline(fail-closed) -> the right
            CAUSED/UNCAUSED grade.
(gate)      a robot at_position step is GROUNDED only when CAUSED (single-source
            through classify_step_evidence).
(holding)   weld eq_active 0->1 grades CAUSED; no transition grades UNCAUSED.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.vcli.cognitive.actor_causation import (
    ActorCaused,
    capture,
    from_name,
    grade,
    is_robot_predicate,
)
from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence
from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal

ROBOT_ORACLES = frozenset(
    {"at_position", "facing", "visited", "holding_object", "arm_at_home"}
)


# ---------------------------------------------------------------------------
# Fakes — duck-typed to the actor-causation capture surface
# ---------------------------------------------------------------------------


class _FakeBase:
    """A go2-base stand-in exposing the instrumented surface + pose."""

    def __init__(self, x: float, y: float, yaw: float = 0.0, cmd_motion: float = 0.0) -> None:
        self._x, self._y, self._yaw = float(x), float(y), float(yaw)
        self._cmd_motion = float(cmd_motion)
        self._connected = True

    def get_position(self) -> list[float]:
        return [self._x, self._y, 0.0]

    def get_heading(self) -> float:
        return self._yaw

    def cmd_motion(self) -> float:
        return self._cmd_motion

    # --- test helpers to simulate commands / motion ---
    def command(self, mag: float) -> None:
        self._cmd_motion += float(mag)

    def teleport(self, x: float, y: float) -> None:
        self._x, self._y = float(x), float(y)


def _agent(base=None, arm=None, gripper=None) -> SimpleNamespace:
    return SimpleNamespace(_base=base, _arm=arm, _gripper=gripper)


# ---------------------------------------------------------------------------
# (snapshot) capture reads ungated counters; a (0,0,0) write doesn't count
# ---------------------------------------------------------------------------


def test_capture_snapshots_counter_and_pose() -> None:
    base = _FakeBase(10.0, 3.0, cmd_motion=1.5)
    b = capture(_agent(base=base))
    assert b.base_cmd_motion == 1.5
    assert b.base_pos == (10.0, 3.0, 0.0)


def test_capture_none_agent_is_all_none() -> None:
    b = capture(None)
    assert b.base_cmd_motion is None and b.base_pos is None and b.arm_joints is None


def test_zero_command_does_not_advance_motion() -> None:
    # The go2 contract: set_velocity(0,0,0) adds a write but ZERO magnitude. Model
    # that here — a base whose cmd_motion is unchanged after a stop -> UNCAUSED.
    base = _FakeBase(10.0, 3.0, cmd_motion=2.0)
    before = capture(_agent(base=base))
    base.command(0.0)  # a (0,0,0) stop: magnitude 0
    after = capture(_agent(base=base))
    assert after.base_cmd_motion == before.base_cmd_motion  # no advance
    g = grade(before, after, "at_position(10.0, 3.0)", ROBOT_ORACLES)
    assert g == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# (grade) honest / no-op / teleport / missing-baseline
# ---------------------------------------------------------------------------


def test_grade_honest_walk_is_caused() -> None:
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    # The skill commands motion AND the base displaces ~1m.
    base.command(0.3)
    base.teleport(11.0, 3.0)
    after = capture(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_grade_noop_is_uncaused() -> None:
    # The target is already satisfied at baseline; no command, no displacement.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    after = capture(_agent(base=base))  # nothing happened
    assert grade(before, after, "at_position(10.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_grade_teleport_is_uncaused() -> None:
    # The pose JUMPS far, but NO command was issued (qpos poked) -> UNCAUSED.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.teleport(11.0, 3.0)  # large displacement, zero commanded motion
    after = capture(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_grade_missing_baseline_fails_closed() -> None:
    base = _FakeBase(11.0, 3.0, cmd_motion=5.0)
    post = capture(_agent(base=base))
    assert grade(None, post, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_grade_command_without_displacement_is_uncaused() -> None:
    # Commanded motion but the robot did NOT move (e.g. wall, fell) -> UNCAUSED:
    # both conjuncts (command AND pose change) are required.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.command(0.5)  # commanded, but no teleport => no displacement
    after = capture(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# (gate) robot at_position step GROUNDED only when CAUSED (single-source)
# ---------------------------------------------------------------------------


def _grounded_step(actor_caused: ActorCaused) -> tuple[StepRecord, SubGoal]:
    sg = SubGoal(name="walk", description="d", verify="at_position(11.0, 3.0)", strategy="walk_forward")
    step = StepRecord(
        sub_goal_name="walk",
        strategy="walk_forward",
        success=True,
        verify_result=True,
        duration_sec=0.1,
        actor_caused=actor_caused,
    )
    return step, sg


def test_gate_caused_is_grounded() -> None:
    step, sg = _grounded_step(ActorCaused.CAUSED)
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "GROUNDED"


def test_gate_uncaused_downgrades_to_ran() -> None:
    step, sg = _grounded_step(ActorCaused.UNCAUSED)
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "RAN"


def test_gate_not_graded_stays_grounded_legacy() -> None:
    # The legacy/default path: a step the executor never graded must classify
    # EXACTLY as before R2b (GROUNDED) — zero regression.
    step, sg = _grounded_step(ActorCaused.NOT_GRADED)
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "GROUNDED"


def test_gate_failed_step_unaffected_by_grade() -> None:
    sg = SubGoal(name="walk", description="d", verify="at_position(11.0, 3.0)", strategy="walk_forward")
    step = StepRecord(
        sub_goal_name="walk", strategy="walk_forward", success=False,
        verify_result=False, duration_sec=0.1, actor_caused=ActorCaused.CAUSED,
    )
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "FAILED"


# ---------------------------------------------------------------------------
# (holding) weld eq_active 0->1 -> CAUSED; no transition -> UNCAUSED
# ---------------------------------------------------------------------------


class _FakeGripper:
    def __init__(self, welds: dict[str, bool]) -> None:
        self._welds = dict(welds)

    def weld_is_active(self) -> dict[str, bool]:
        return dict(self._welds)


def test_holding_fresh_grasp_is_caused() -> None:
    grip = _FakeGripper({"mug": False})
    before = capture(_agent(gripper=grip))
    grip._welds["mug"] = True  # weld goes 0 -> 1 this step
    after = capture(_agent(gripper=grip))
    assert grade(before, after, "holding_object('mug')", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_holding_no_transition_is_uncaused() -> None:
    # Already welded at baseline (no fresh grasp) -> UNCAUSED.
    grip = _FakeGripper({"mug": True})
    before = capture(_agent(gripper=grip))
    after = capture(_agent(gripper=grip))
    assert grade(before, after, "holding_object('mug')", ROBOT_ORACLES) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# misc — predicate detection + enum (de)serialization round-trip
# ---------------------------------------------------------------------------


def test_is_robot_predicate_detection() -> None:
    assert is_robot_predicate("at_position(11.0, 3.0)", ROBOT_ORACLES) is True
    assert is_robot_predicate("arm_at_home()", ROBOT_ORACLES) is True
    # A dev predicate (not a robot oracle) is not graded.
    assert is_robot_predicate("file_exists('/tmp/x')", ROBOT_ORACLES) is False
    # A robot predicate ABSENT from the live oracle set is not graded.
    assert is_robot_predicate("at_position(1, 2)", frozenset()) is False


def test_from_name_roundtrip_and_legacy() -> None:
    for c in ActorCaused:
        assert from_name(c.value) is c
        assert from_name(c) is c
    # Unknown / missing (legacy trace) -> NOT_GRADED.
    assert from_name(None) is ActorCaused.NOT_GRADED
    assert from_name("garbage") is ActorCaused.NOT_GRADED


# ---------------------------------------------------------------------------
# trace_store serialization round-trips the actor_caused enum (v4 schema)
# ---------------------------------------------------------------------------


def test_trace_serialization_roundtrips_actor_caused(tmp_path) -> None:
    from vector_os_nano.vcli.cognitive.trace_store import load_trace, save_trace
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree

    sg = SubGoal(name="walk", description="d", verify="at_position(11.0, 3.0)", strategy="walk_forward")
    step = StepRecord(
        sub_goal_name="walk", strategy="walk_forward", success=True,
        verify_result=True, duration_sec=0.1, actor_caused=ActorCaused.UNCAUSED,
    )
    trace = ExecutionTrace(
        goal_tree=GoalTree(goal="g", sub_goals=(sg,)), steps=(step,),
        success=True, total_duration_sec=0.1,
    )
    path = save_trace(trace, tmp_path / "t.json")
    reloaded = load_trace(path)
    assert reloaded.steps[0].actor_caused is ActorCaused.UNCAUSED


def test_legacy_trace_without_field_loads_as_not_graded(tmp_path) -> None:
    # An older on-disk trace (no actor_caused key) must load NOT_GRADED -> legacy.
    import json

    from vector_os_nano.vcli.cognitive.trace_store import load_trace

    payload = {
        "schema_version": 3,
        "goal_tree": {"goal": "g", "sub_goals": [
            {"name": "walk", "description": "d", "verify": "at_position(11.0, 3.0)",
             "strategy": "walk_forward"}]},
        "steps": [{"sub_goal_name": "walk", "strategy": "walk_forward",
                   "success": True, "verify_result": True, "duration_sec": 0.1}],
        "success": True, "total_duration_sec": 0.1,
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(payload))
    reloaded = load_trace(p)
    assert reloaded.steps[0].actor_caused is ActorCaused.NOT_GRADED


# ---------------------------------------------------------------------------
# Executor integration — the executor threads the grade into the StepRecord
# ---------------------------------------------------------------------------


class _VerifierOverFrozenAgent:
    """A minimal GoalVerifier: evaluate() reads at_position over the LIVE agent.

    Exposes ``_namespace`` (so the executor single-sources oracle names from it)
    and ``evaluate`` / ``verify`` using the real go2 oracle bound to the live base.
    """

    def __init__(self, agent) -> None:
        from tests.harness.fake_gt import fake_namespace

        self._namespace = {"at_position": fake_namespace(agent, "go2")["at_position"]}

    def evaluate(self, expr: str):
        val = eval(expr, {"__builtins__": {}}, dict(self._namespace))  # noqa: S307 - test only
        return bool(val), val

    def verify(self, expr: str) -> bool:
        return self.evaluate(expr)[0]


class _MovingPrimitives(dict):
    """Primitive namespace whose 'walk_forward' commands + displaces the base."""

    def __init__(self, base) -> None:
        super().__init__()
        self._base = base
        self["walk_forward"] = self._walk

    def _walk(self, **_):
        self._base.command(0.3)
        self._base.teleport(11.0, 3.0)
        return True


def _executor_for(agent, primitives) -> "object":
    from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
    from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector

    return GoalExecutor(
        strategy_selector=StrategySelector(),
        verifier=_VerifierOverFrozenAgent(agent),
        primitives=primitives,
        agent=agent,
    )


def test_executor_threads_caused_on_honest_walk() -> None:
    from vector_os_nano.vcli.cognitive.types import GoalTree

    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    agent = _agent(base=base)
    ex = _executor_for(agent, _MovingPrimitives(base))
    sg = SubGoal(name="walk", description="向前走", verify="at_position(11.0, 3.0)", strategy="walk_forward")
    trace = ex.execute(GoalTree(goal="g", sub_goals=(sg,)))
    assert trace.steps[0].success is True
    assert trace.steps[0].actor_caused is ActorCaused.CAUSED
    # And the gate agrees it is GROUNDED.
    assert classify_step_evidence(trace.steps[0], sg, ROBOT_ORACLES) == "GROUNDED"


def test_executor_threads_uncaused_on_noop() -> None:
    from vector_os_nano.vcli.cognitive.types import GoalTree

    # Base already AT (10,3); verify targets (10,3) -> true@baseline, no command.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    agent = _agent(base=base)
    noop = dict(walk_forward=lambda **_: True)  # a primitive that does NOTHING
    ex = _executor_for(agent, noop)
    sg = SubGoal(name="stand", description="原地", verify="at_position(10.0, 3.0)", strategy="walk_forward")
    trace = ex.execute(GoalTree(goal="g", sub_goals=(sg,)))
    assert trace.steps[0].success is True  # verify passes (already there)
    assert trace.steps[0].actor_caused is ActorCaused.UNCAUSED
    # The moat downgrades it: a no-op is NOT grounded.
    assert classify_step_evidence(trace.steps[0], sg, ROBOT_ORACLES) == "RAN"


def test_executor_no_agent_stays_not_graded() -> None:
    from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
    from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
    from vector_os_nano.vcli.cognitive.types import GoalTree

    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    agent = _agent(base=base)
    # agent=None disables grading -> the step stays NOT_GRADED (legacy).
    ex = GoalExecutor(
        strategy_selector=StrategySelector(),
        verifier=_VerifierOverFrozenAgent(agent),
        primitives=dict(walk_forward=lambda **_: True),
        agent=None,
    )
    sg = SubGoal(name="stand", description="原地", verify="at_position(10.0, 3.0)", strategy="walk_forward")
    trace = ex.execute(GoalTree(goal="g", sub_goals=(sg,)))
    assert trace.steps[0].actor_caused is ActorCaused.NOT_GRADED


# ---------------------------------------------------------------------------
# (cross-channel, STEP-12 fix) a base predicate grades on the dimension it
# MEASURES: a forward walk must NOT launder a satisfied-at-baseline facing(h)
# no-op to CAUSED, and a pure turn must NOT launder at_position. The old
# max(planar, yaw) metric accepted the irrelevant channel.
# ---------------------------------------------------------------------------


def test_facing_uncaused_after_forward_walk_no_turn() -> None:
    # Forward walk: position moves ~1.5m, heading UNCHANGED. facing(h0) was already
    # true at baseline; the planar move must NOT grade the heading no-op CAUSED.
    base = _FakeBase(10.0, 3.0, yaw=0.5, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.command(0.3)
    base.teleport(11.5, 3.0)  # dxy ~1.5, dyaw 0
    after = capture(_agent(base=base))
    assert grade(before, after, "facing(0.5)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_facing_caused_after_real_turn() -> None:
    # A real turn: heading changes ~0.5 rad (>_YAW_CAUSE_EPS), position unchanged.
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.command(0.3)
    base._yaw = 0.5  # ~29deg intentional turn
    after = capture(_agent(base=base))
    assert grade(before, after, "facing(0.5)", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_facing_uncaused_on_drift_below_yaw_threshold() -> None:
    # Gait yaw-drift below the ~10deg threshold must NOT count as an intentional turn
    # for facing — even alongside a planar move (a forward walk that wobbles).
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.command(0.3)
    base.teleport(11.5, 3.0)
    base._yaw = 0.05  # ~2.9deg drift < _YAW_CAUSE_EPS (~10deg)
    after = capture(_agent(base=base))
    assert grade(before, after, "facing(0.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_at_position_still_caused_after_forward_walk() -> None:
    # The planar channel is unaffected by the fix: a forward walk grades at_position
    # CAUSED exactly as before.
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.command(0.3)
    base.teleport(11.0, 3.0)
    after = capture(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_at_position_uncaused_after_pure_turn() -> None:
    # STEP-12 bonus tightening: a pure turn (no planar move) must NOT grade an
    # at_position predicate CAUSED — the position channel didn't move.
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = capture(_agent(base=base))
    base.command(0.3)
    base._yaw = 1.0  # turned in place, position unchanged
    after = capture(_agent(base=base))
    assert grade(before, after, "at_position(10.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# WP10: the GT oracle set is causation-graded on its channel (a new predicate is
# never NOT_GRADED, which would let a pre-existing state ground for free).
# ---------------------------------------------------------------------------

_WP10 = ROBOT_ORACLES | {
    "in_room", "near_object", "in_view", "lifted", "kept_holding", "released",
    "object_at", "object_on", "localized",
}


@pytest.mark.parametrize("expr", [
    "in_room('kitchen')", "near_object('user')", "in_view('red can')", "lifted('mug')",
    "kept_holding('mug')", "released('mug')", "object_at('mug', 1, 2)", "object_on('mug', 'table')",
])
def test_wp10_predicates_are_graded(expr) -> None:
    assert is_robot_predicate(expr, _WP10)


def test_object_on_already_true_with_no_release_is_uncaused() -> None:
    grip = _FakeGripper({"mug": False})  # the mug was resting on the table all along
    before = capture(_agent(gripper=grip))
    after = capture(_agent(gripper=grip))
    assert grade(before, after, "object_on('mug', 'table')", _WP10) == ActorCaused.UNCAUSED
    assert grade(before, after, "released('mug')", _WP10) == ActorCaused.UNCAUSED


def test_object_on_after_a_release_is_caused() -> None:
    grip = _FakeGripper({"mug": True})
    before = capture(_agent(gripper=grip))
    grip._welds["mug"] = False  # placed: weld 1 -> 0 this step
    after = capture(_agent(gripper=grip))
    for expr in ("object_on('mug', 'table')", "object_at('mug', 1, 2)", "released('mug')"):
        assert grade(before, after, expr, _WP10) == ActorCaused.CAUSED, expr
    # a release never proves a grasp
    assert grade(before, after, "lifted('mug')", _WP10) == ActorCaused.UNCAUSED


def test_carry_needs_base_motion_kept_holding_alone_is_uncaused() -> None:
    base, grip = _FakeBase(10.0, 3.0), _FakeGripper({"mug": True})
    before = capture(_agent(base=base, gripper=grip))
    after = capture(_agent(base=base, gripper=grip))
    assert grade(before, after, "kept_holding('mug')", _WP10) == ActorCaused.UNCAUSED
    base.command(0.3)
    base.teleport(8.0, 3.0)
    after = capture(_agent(base=base, gripper=grip))
    expr = "kept_holding('mug') and near_object('user', 0.8)"
    assert grade(before, after, expr, _WP10) == ActorCaused.CAUSED


def test_in_view_is_caused_by_a_turn_or_a_walk_not_by_looking() -> None:
    base = _FakeBase(10.0, 3.0)
    before = capture(_agent(base=base))
    assert grade(before, capture(_agent(base=base)), "in_view('red can')", _WP10) == ActorCaused.UNCAUSED
    base.command(0.3)
    base._yaw = 1.0  # turned in place
    assert grade(before, capture(_agent(base=base)), "in_view('red can')", _WP10) == ActorCaused.CAUSED


# ---------------------------------------------------------------------------
# WP11: localized is graded on the PERCEPTION channel — caused only by a localized
# observation written this step, never by memory (adversarial cases first).
# ---------------------------------------------------------------------------


def _perceiver():
    from vector_os_nano.core.scene_graph import SceneGraph

    agent = _agent()
    agent._spatial_memory = SceneGraph()
    return agent


def test_localized_from_memory_alone_is_uncaused() -> None:
    agent = _perceiver()
    agent._spatial_memory.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    before = capture(agent)  # the step starts with the object already on the map
    assert is_robot_predicate("localized('green bottle')", _WP10)
    assert grade(before, capture(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED
    agent._spatial_memory.merge_object("sofa", "hallway", "vp")  # a name with no position
    assert grade(before, capture(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED
    agent._spatial_memory.clear()
    agent._spatial_memory.load()  # nothing persisted: still nothing perceived this step
    assert grade(before, capture(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED


def test_localized_after_a_perceive_wrote_the_belief_is_caused() -> None:
    agent = _perceiver()
    before = capture(agent)
    agent._spatial_memory.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    assert grade(before, capture(agent), "localized('green bottle')", _WP10) == ActorCaused.CAUSED


def test_localized_without_a_belief_or_a_baseline_fails_closed() -> None:
    assert grade(None, capture(_perceiver()), "localized('x')", _WP10) == ActorCaused.UNCAUSED
    assert grade(capture(_agent()), capture(_agent()), "localized('x')", _WP10) == ActorCaused.UNCAUSED
