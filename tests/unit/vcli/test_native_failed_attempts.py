# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""SP1 — failed attempts and recovery (charter 3.5) through the native runner.

The live case: "把绿色的瓶子拿起来" on go2_piper; the model's first call (pick_top_down)
was refused before the robot moved, then it detected, grasped and verified, and the
verdict said FAILED because the refused call failed the step the grasp then shared. A
REFUSED call (an error with no commanded motion) is not an action; a FAILED ATTEMPT (an
error after commanded motion) is its own FAILED row; a proven contract is not rejected
by the attempts that failed on the way. Everything else still rejects.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from rich.console import Console

from tests.unit.vcli.test_native_loop import _NavBase, _Skill, _make_agent, _make_engine, _session, _verdict
from tests.unit.vcli.test_native_safety import _dev_engine, _script
from tests.unit.vcli.test_task_contract_binding import GREEN
from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from vector_os_nano.vcli.cognitive.actor_causation import ActorBaseline, commanded_nothing
from vector_os_nano.vcli.render import render_turn
from vector_os_nano.vcli.tools.base import PermissionResult, ToolResult

GOAL = "把绿色的瓶子拿起来"
HOLD = f"holding_object('{GREEN}')"
HINT = "holding_object('green bottle')"  # perception_grasp's bound hint: backs HOLD by another name
_POS = {"rest": [10.9, 3.0, 0.71], "held": [10.9, 3.0, 0.95], "floor": [10.0, 3.0, 0.02]}


class _Arm:
    """The Piper stand-in with the commanded-motion counter the drivers keep."""

    _connected = True
    joint_names = ("joint1",)

    def __init__(self) -> None:
        self.state, self.q, self.motion = "rest", 0.0, 0.0

    def get_object_positions(self):
        return {GREEN: _POS[self.state]}

    def get_joint_positions(self):
        return [self.q]

    def ctrl_motion(self) -> float:
        return self.motion

    def move(self) -> None:
        self.q += 0.5
        self.motion += 0.5


class _Grip:
    def __init__(self, arm: _Arm) -> None:
        self._arm = arm

    def is_holding(self) -> bool:
        return self._arm.state == "held"

    def weld_is_active(self) -> dict:
        return {GREEN: self._arm.state == "held"}


class _Act(_Skill):
    """Every call commands arm motion, leaves the bottle in the next scripted state and
    reports the next scripted error ("" = success)."""

    def __init__(self, name: str, arm: _Arm, *outcomes: tuple[str, str], hint: str = "holding_object({object})"):
        super().__init__(name, None, hint)
        self._arm, self._outcomes = arm, list(outcomes)

    def execute(self, params, context):
        from vector_os_nano.core.types import SkillResult

        state, fail = self._outcomes.pop(0)
        self._arm.move()
        self._arm.state = state
        return SkillResult(success=not fail, error_message=fail)


def _read(name: str):
    s = _Skill(name, lambda: None)
    s.__skill_actuates__ = False
    return s


REFUSED = ("pick_top_down", {"object": GREEN})
DETECT = ("detect", {"object": "green bottle"})
GRASP = ("perception_grasp", {"object": "green bottle"})
VERIFY = ("verify", {"expr": HOLD, "goal_id": "grasp_green_bottle"})


def _run(goal: str, script: list[tuple[str, dict]], *, grasp=(("held", ""),), other=(), arm_counter=True):
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _ = _make_agent(10.0, 3.0)
    agent._base = _NavBase(10.0, 3.0)
    arm = _Arm()
    if not arm_counter:
        arm.ctrl_motion = None  # a body without the counter: no motion evidence
    agent._arm, agent._gripper = arm, _Grip(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper", support="pick_table")
    for sk in (_Skill("pick_top_down", lambda: None, "holding_object({object_label})",  # the real skill's hint
                      fail="Cannot locate target object (object_not_found)"),
               _read("detect"), _Act("perception_grasp", arm, *grasp), _Act("other_grasp", arm, *other),
               _Act("drop", arm, ("floor", "lift failed: the object slipped"), hint=""),
               _Act("wave", arm, ("held", ""), hint="")):
        agent._skill_registry.register(sk)
    backend = FakeToolScriptBackend.from_tool_script([*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    console = Console(record=True, width=240)
    render_turn(console, eng, trace, "", [])
    rows = [(s.strategy, s.success, s.failed_attempt, s.superseded) for s in trace.steps]
    return trace, _verdict(agent, eng, trace), rows, console.export_text()


# ---------------------------------------------------------------------------
# Recovery: verified, goal proven, the failed and refused calls stay in the report
# ---------------------------------------------------------------------------


def test_the_live_case_a_refused_call_then_a_proven_grasp_verifies() -> None:
    trace, rep, rows, text = _run(GOAL, [REFUSED, DETECT, GRASP, VERIFY])
    assert trace.contract == (HOLD,)
    assert rep.verified is True and rep.evidence == "GROUNDED" and rep.goal_check == "proven", rep.reject_reason
    assert rows == [("perception_grasp", True, False, False)], rows
    (row,) = rep.per_step
    assert row.evidence == "GROUNDED" and row.actor_caused == "CAUSED" and not row.failed_attempt
    assert row.refused == ("pick_top_down: Cannot locate target object (object_not_found)",), row.refused
    assert '"refused": ["pick_top_down: ' in rep.to_sentinel_line()
    assert "refused pick_top_down: " in text and "verdict GROUNDED verified=True" in text, text
    assert f"goal proven: {HOLD}" in text, text


@pytest.mark.parametrize("retry", ["perception_grasp", "other_grasp"])
def test_a_miss_after_motion_then_a_retry_verifies_with_two_rows(retry) -> None:
    grasp = (("rest", "grasp missed: nothing between the fingers"), ("held", ""))
    trace, rep, rows, text = _run(GOAL, [GRASP, (retry, GRASP[1]), VERIFY],
                                  grasp=grasp if retry == "perception_grasp" else grasp[:1], other=grasp[1:])
    assert rep.verified is True and rep.goal_check == "proven", rep.reject_reason
    assert rows == [("perception_grasp", False, True, True), (retry, True, False, False)], rows
    assert [(s.evidence, s.failed_attempt) for s in rep.per_step] == [("FAILED", True), ("GROUNDED", False)]
    assert rep.per_step[0].reject_reason.startswith("the action failed: ")
    assert rep.n_steps == 1 and rep.n_grounded == 1 and trace.success is True  # the miss is audit only
    assert "perception_grasp → failed attempt " in text and "verdict GROUNDED verified=True" in text, text


# ---------------------------------------------------------------------------
# Still rejected
# ---------------------------------------------------------------------------


def test_a_failed_attempt_and_nothing_after_it_is_failed() -> None:
    trace, rep, rows, _ = _run(GOAL, [GRASP], grasp=(("rest", "grasp missed"),))
    assert rows == [("perception_grasp", False, True, False)], rows
    assert rep.verified is False and rep.evidence == "FAILED" and rep.goal_check == "failed"
    assert trace.success is False


def test_a_failed_attempt_then_a_verify_already_true_is_uncaused() -> None:
    """The grasp welded the bottle and then reported an error: a verify after it has no
    step of its own (no baseline) and grades UNCAUSED, never GROUNDED."""
    _trace, rep, rows, _ = _run(GOAL, [GRASP, VERIFY], grasp=(("held", "lift failed"),))
    assert rows == [("perception_grasp", False, True, True), ("", True, False, False)], rows
    assert rep.per_step[1].evidence == "RAN" and rep.per_step[1].reject_reason.startswith("UNCAUSED")
    assert rep.verified is False and rep.goal_check == "failed"


@pytest.mark.parametrize("goal", ["go to the kitchen then turn left", "go to (9,3) then turn left"])
def test_a_failed_turn_after_motion_is_never_recovered_from(goal) -> None:
    """No contract, or a coordinate contract that names only part of the sentence and was
    proven BEFORE the turn: the failed turn rejects the turn."""
    trace, rep, rows, _ = _run(goal, [("navigate", {"x": 9.0, "y": 3.0}), ("verify", {"expr": "at_position(9.0, 3.0)"}),
                                      ("drop", {})])  # "drop": commands motion, then fails
    assert rows[-1] == ("drop", False, True, False), rows
    assert rep.verified is False and rep.evidence == "FAILED", rep.reject_reason
    assert rep.reject_reason == "the turn did not succeed" and trace.success is False, rep.reject_reason
    assert rep.per_step[1].reject_reason.startswith("the action failed: "), rep.per_step


def test_an_unverified_success_after_a_recovered_contract_still_rejects() -> None:
    grasp = (("rest", "grasp missed"), ("held", ""))
    _trace, rep, rows, _ = _run(GOAL, [GRASP, GRASP, VERIFY, ("wave", {})], grasp=grasp)
    assert rows[-1] == ("wave", True, False, False), rows
    assert rep.goal_check == "proven" and rep.verified is False
    assert rep.reject_reason == "native_step_2: no verify", rep.reject_reason


@pytest.mark.parametrize("calls, grasp", [
    ([("wave", {}), GRASP, GRASP, VERIFY], (("rest", "grasp missed"), ("held", ""))),
    ([GRASP, ("wave", {}), GRASP, GRASP, VERIFY], (("rest", "miss"), ("rest", "miss"), ("held", ""))),
])
def test_an_unverified_success_is_never_folded_into_a_later_failed_attempt(calls, grasp) -> None:
    """SP1V-02: a successful, never-verified action is its own unproven row; the failed
    attempt after it holds only the failing call, so recovery never excuses the action."""
    _trace, rep, rows, _ = _run(GOAL, calls, grasp=grasp)
    assert ("wave", True, False, False) in rows, rows
    assert all(r[0] == "perception_grasp" for r in rows if r[2]), rows
    assert [s.description.split(" | ")[0] for s in _trace.goal_tree.sub_goals if "wave" in s.description] == [
        "native: wave"], _trace.goal_tree.sub_goals
    assert rep.goal_check == "proven" and rep.verified is False, (rep.reject_reason, rows)
    assert rep.reject_reason.endswith(": no verify"), rep.reject_reason


def test_a_contract_not_met_at_turn_end_does_not_excuse_the_failed_attempts() -> None:
    """The retry grasped; a later failed action dropped the bottle."""
    grasp = (("rest", "grasp missed"), ("held", ""))
    trace, rep, rows, _ = _run(GOAL, [GRASP, GRASP, VERIFY, ("drop", {})], grasp=grasp)
    assert [r[2] for r in rows] == [True, False, True], rows
    assert trace.contract_end == (False,) and rep.goal_check == "failed"
    assert rep.verified is False and rep.evidence == "FAILED"


def test_a_refused_call_followed_by_nothing_is_not_verified() -> None:
    _trace, rep, rows, text = _run(GOAL, [REFUSED])
    assert rows == [("", True, False, False)], rows
    assert rep.per_step[0].refused[0].startswith("pick_top_down: ")
    assert rep.verified is False and rep.goal_check == "failed"
    assert "refused pick_top_down: " in text, text


def test_a_refused_call_proves_nothing_and_a_verify_after_it_is_a_probe() -> None:
    """A refused call is not an action: a verify that follows only it after an earlier
    verify is refused unevaluated (no GT bit leaks), and it fails nothing either."""
    trace, rep, rows, text = _run(GOAL, [GRASP, VERIFY, REFUSED, VERIFY])
    assert rows == [("perception_grasp", True, False, False), ("", True, False, False)], rows
    assert trace.goal_tree.sub_goals[1].verify == ""  # the second verify never ran
    assert trace.steps[1].housekeeping and rep.per_step[1].refused[0].startswith("pick_top_down: ")
    assert rep.verified is True and rep.goal_check == "proven", rep.reject_reason
    assert "refused pick_top_down: " in text, text


@pytest.mark.parametrize("params, verified", [
    ({"object_label": "green bottle"}, True),  # its hint binds: an attempt of the contract item
    ({"object_id": GREEN}, False),             # the live call's params: no hint, no attempt of an item
])
def test_without_motion_evidence_an_error_is_a_failed_attempt(params, verified) -> None:
    """A body without its arm counter: the refused-looking call is a failed attempt
    (fail closed), so it is its own FAILED row; the contract proven after it excuses it
    only when the call's bound hint backs the contract item (SP1V-01)."""
    trace, rep, rows, _ = _run(GOAL, [("pick_top_down", params), GRASP, VERIFY], arm_counter=False)
    assert rows[0] == ("pick_top_down", False, True, verified), rows  # a bound hint: the proof supersedes it
    assert rep.goal_check == "proven" and rep.verified is verified, rep.reject_reason


def test_a_call_the_owner_stopped_is_the_aborted_row_never_a_failed_attempt() -> None:
    """A stop lands while the walk runs and it then reports an error after moving: the
    abort row records it ("walk -> aborted", FAILED, never excused)."""
    from vector_os_nano.vcli.cognitive import abort

    agent, base = _make_agent()
    real_walk = base.walk

    def walk_stopped(*a: Any) -> bool:
        real_walk(*a)
        abort.request_abort()
        return False  # the walk skill then fails (walk_failed)

    base.walk = walk_stopped
    backend = _script(tool_turn(("walk", {"distance": 1.0})), tool_turn(end=True))
    trace = _make_engine(agent, backend).run_turn_native("walk", session=_session())
    assert [(s.strategy, s.success, s.failed_attempt, s.error) for s in trace.steps] == [
        ("walk", False, False, "aborted")]


class _Flaky:
    """A dev-world side-effecting tool: errors on its first call, then succeeds."""

    name, description = "bash", "flaky bash"
    input_schema = {"type": "object", "properties": {"command": {"type": "string"}}}

    def __init__(self) -> None:
        self.calls = 0

    def check_permissions(self, params: dict, ctx: Any) -> PermissionResult:
        return PermissionResult("allow", "scripted")

    def is_read_only(self, params: dict) -> bool:
        return False

    def execute(self, params: dict, ctx: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(content="exit 1: boom", is_error=True) if self.calls == 1 else ToolResult(content="ok")


def test_in_the_dev_world_an_erroring_bash_still_fails_its_step() -> None:
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.verdict import VerdictReport

    eng = _dev_engine(_script(tool_turn(("bash", {"command": "make"})), tool_turn(("bash", {"command": "make"})),
                              tool_turn(("verify", {"expr": "file_exists('pyproject.toml')"})),
                              tool_turn(("finish", {}))), _Flaky())
    trace = eng.run_turn_native("build it", session=_session())
    assert [(s.strategy, s.success, s.failed_attempt) for s in trace.steps] == [
        ("bash", False, True), ("bash", True, False)]
    rep = VerdictReport.from_trace(trace, verify_oracle_names(None, eng))
    assert rep.verified is False and rep.evidence == "FAILED" and rep.goal_check == "unchecked"


# ---------------------------------------------------------------------------
# The gate on hand-built traces (adversarial): only a FAILED row the producer marked a
# failed attempt, followed by a proof of every contract item, under a proven goal
# ---------------------------------------------------------------------------


def _trace(goal: str, rows: list[tuple[str, bool, bool]], *, contract=(HOLD,), end=(True,)):
    """*rows*: (verify, success, failed_attempt); a successful row is CAUSED and PASSes."""
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal

    sgs = tuple(SubGoal(name=f"s{i}", description="", verify=v) for i, (v, _ok, _fa) in enumerate(rows))
    steps = tuple(StepRecord(f"s{i}", "skill", ok, ok, 0.0, error="" if ok else "boom", attempt_of="native",
                             actor_caused=ActorCaused.CAUSED if ok else ActorCaused.NOT_GRADED, failed_attempt=fa)
                  for i, (_v, ok, fa) in enumerate(rows))
    return ExecutionTrace(GoalTree(goal, sgs), steps, False, 0.0, contract=contract,
                          contract_start=tuple(False for _ in contract), contract_end=end, world="apartment")


@pytest.mark.parametrize("rows, contract, end, verified", [
    ([(HINT, False, True), (HOLD, True, False)], (HOLD,), (True,), True),          # recovered from
    ([("", False, True), (HOLD, True, False)], (HOLD,), (True,), False),           # no hint: not an attempt of the item
    ([("at_position(1.0, 2.0)", False, True), (HOLD, True, False)], (HOLD,), (True,), False),  # another predicate
    ([(HINT, False, False), (HOLD, True, False)], (HOLD,), (True,), False),        # not a failed attempt (an abort)
    ([(HOLD, True, False), (HINT, False, True)], (HOLD,), (True,), False),         # failed AFTER the proof
    ([(HINT, False, True), (HOLD, True, False)], (HOLD,), (False,), False),        # not met at turn end
    ([(HINT, False, True)], (HOLD,), (True,), False),                              # no proof at all
    ([("", True, True), (HOLD, True, False)], (HOLD,), (True,), False),            # "failed attempt" that ran: unproven
])
def test_the_gate_excuses_only_a_failed_attempt_a_proven_goal_recovered_from(rows, contract, end, verified) -> None:
    from tests.unit.vcli.test_task_contract_binding import APT
    from vector_os_nano.vcli.cognitive.trace_store import turn_reject_reason, turn_succeeded

    t = _trace(GOAL, rows, contract=contract, end=end)
    t = replace(t, success=turn_succeeded(t))
    assert (turn_reject_reason(t, APT) == "") is verified, turn_reject_reason(t, APT)


def test_the_gate_never_excuses_without_a_contract() -> None:
    """No contract: a failed attempt that no later row of the same predicate supersedes
    rejects, whatever proof follows it."""
    from tests.unit.vcli.test_task_contract_binding import APT
    from vector_os_nano.vcli.cognitive.trace_store import turn_reject_reason, turn_succeeded

    t = _trace("move the green bottle", [("at_position(1.0, 2.0)", False, True), (HOLD, True, False)],
               contract=(), end=())
    assert turn_succeeded(t) is False
    t = replace(t, success=True)  # even a producer that claims success
    assert turn_reject_reason(t, APT).startswith("s0: the action failed")


_RED_HOLD = "holding_object('pickable_can_red')"


@pytest.mark.parametrize("rows, attempts, verified, superseded", [
    ([(HINT, False, True), (HOLD, True, False)], ("native", "grasp_green"), True, True),         # another goal_id, alias
    ([(HOLD, False, True), (HOLD, True, False)], ("native", "grasp_green"), True, True),         # another goal_id
    ([(HINT, False, True), (HOLD, True, False)], ("native", "native"), True, True),              # alias, same goal_id
    ([(HINT, False, True), (_RED_HOLD, True, False)], ("native", "g"), False, False),         # another object
    ([(HINT, False, True), (f"{HOLD} or True", True, False)], ("native", "g"), False, False),  # not necessarily asserted
    ([(HINT, False, True), (HOLD, True, False)], ("native", ""), False, False),               # later row not opted in
    ([("", False, True), (HOLD, True, False)], ("native", "g"), False, False),                # failed row with no predicate
    ([(HOLD, True, False), (HINT, False, True)], ("g", "native"), False, False),              # failed AFTER the proof
    ([(HINT, False, False), (HOLD, True, False)], ("native", "g"), False, False),             # a FAILED row, not an attempt
    ([(HINT, False, True), (HOLD, False, False)], ("native", "g"), False, True),            # the retry failed too
])
def test_without_a_contract_a_later_attempt_of_the_same_predicate_supersedes_a_failed_one(
        rows, attempts, verified, superseded) -> None:
    """SP1F1V-01 (charter 3.5): the failed attempt is superseded by the next opted-in row
    that necessarily asserts its predicate (an object by any name the world resolves),
    whatever goal_id each carries; it never supersedes a row before it."""
    from tests.unit.vcli.test_task_contract_binding import APT
    from vector_os_nano.vcli.cognitive.trace_store import graded_indices, turn_reject_reason, turn_succeeded

    t = _trace("move the green bottle", rows, contract=(), end=())
    t = replace(t, steps=tuple(replace(s, attempt_of=a) for s, a in zip(t.steps, attempts)))
    t = replace(t, success=turn_succeeded(t))
    assert (turn_reject_reason(t, APT) == "") is verified, turn_reject_reason(t, APT)
    assert graded_indices(t) == ([1] if superseded else [0, 1]), graded_indices(t)


class _FlakyNav(_NavBase):
    """The first navigate walks part of the way, then the nav stack dies (an error after
    commanded motion: a failed attempt); the next one arrives."""

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)
        self.calls = 0

    def navigate_to(self, x, y, timeout=None):
        self.calls += 1
        if self.calls == 1:
            self.walk(0.5, 0.0, 0.0, 1.0)
            raise RuntimeError("nav stack died mid-way")
        return super().navigate_to(x, y)


@pytest.mark.parametrize("goal", ["go to the kitchen", "走到坐标 (9.0,3.0)"])
@pytest.mark.parametrize("gid", ["", "reach_kitchen"])
def test_a_failed_navigate_then_a_retry_verifies_whatever_the_goal_id(goal, gid) -> None:
    """SP1F1V-01: the failed attempt is filed under its bound hint; the model's verify of
    the same coordinate under its own goal_id supersedes it (base behaviour, no contract)."""
    agent, _ = _make_agent(0.0, 0.0)
    agent._base = _FlakyNav(0.0, 0.0)
    nav, at = ("navigate", {"x": 9.0, "y": 3.0}), ("verify", {"expr": "at_position(9.0, 3.0)", "goal_id": gid})
    backend = FakeToolScriptBackend.from_tool_script([tool_turn(c) for c in (nav, at, nav, at, ("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    rep = _verdict(agent, eng, trace)
    assert [(s.strategy, s.success, s.failed_attempt, s.superseded) for s in trace.steps] == [
        ("navigate", False, True, True), ("", True, False, True), ("navigate", True, False, False)]
    assert rep.verified is True and trace.success is True, rep.reject_reason


def test_a_missed_grasp_then_a_retry_verified_by_another_name_verifies_without_a_contract() -> None:
    """SP1F1V-01 (fuzz seed 11973): the failed grasp's hint names the bottle one way, the
    model's verify another; the world resolves both to the same object."""
    grasp = (("rest", "grasp missed"), ("held", ""))
    trace, rep, rows, _ = _run("move the green bottle", [GRASP, GRASP, VERIFY], grasp=grasp)
    assert trace.contract == () and rows == [("perception_grasp", False, True, True),
                                             ("perception_grasp", True, False, False)], rows
    assert rep.verified is True and rep.goal_check == "unchecked", rep.reject_reason


# ---------------------------------------------------------------------------
# The refused-call predicate itself (adversarial)
# ---------------------------------------------------------------------------


def _agent(**devices: Any) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(**{"_base": None, "_arm": None, "_gripper": None, **{f"_{k}": v for k, v in devices.items()}})


def test_commanded_nothing_needs_every_device_read_and_unchanged() -> None:
    b = ActorBaseline(base_cmd_motion=1.0, arm_ctrl_motion=2.0, gripper_welds={}, session="s")
    body = _agent(base=object(), arm=object(), gripper=object())
    assert commanded_nothing(body, b, b)
    assert not commanded_nothing(body, b, ActorBaseline(base_cmd_motion=1.0 + 1e-9, arm_ctrl_motion=2.0,
                                                        gripper_welds={}, session="s"))  # any base motion
    assert not commanded_nothing(body, b, ActorBaseline(base_cmd_motion=1.0, arm_ctrl_motion=2.5,
                                                        gripper_welds={}, session="s"))  # arm motion
    assert not commanded_nothing(body, b, ActorBaseline(base_cmd_motion=1.0, arm_ctrl_motion=2.0,
                                                        gripper_welds={GREEN: True}, session="s"))  # a weld
    assert not commanded_nothing(body, b, ActorBaseline(base_cmd_motion=1.0, arm_ctrl_motion=2.0,
                                                        gripper_welds={}, session="t"))  # another session
    unread = ActorBaseline(base_cmd_motion=1.0, arm_ctrl_motion=None, gripper_welds={}, session="s")
    assert not commanded_nothing(body, unread, unread)  # an arm without its counter: no evidence
    assert not commanded_nothing(_agent(), ActorBaseline(), ActorBaseline())  # no body (the dev world)
    assert not commanded_nothing(None, b, b)
    assert commanded_nothing(_agent(base=object()), b, b)  # a go2: its one device read, unchanged


def test_a_refused_call_is_not_a_permission_refusal() -> None:
    """trace.refused stays the permission gate's list; the turn still owns the call (no
    other producer re-runs it)."""
    trace, _rep, _rows, _ = _run(GOAL, [REFUSED, DETECT, GRASP, VERIFY])
    assert trace.refused == () and trace.actions_dispatched == 2


# ---------------------------------------------------------------------------
# The failed-after-motion twins of the refused-turn tests whose assertions charter 3.5
# changed (test_native_reverify_after_other_action, test_native_loop)
# ---------------------------------------------------------------------------


class _TurnFails(_Skill):
    """Commands base motion, then reports 'Unknown turn direction': a failed attempt."""

    def __init__(self, base: Any) -> None:
        super().__init__("turn_fake", None)
        self._base = base

    def execute(self, params, context):
        from vector_os_nano.core.types import SkillResult

        self._base.walk(0.0, 0.0, 1.0, 1.0)
        return SkillResult(success=False, error_message="Unknown turn direction")


def _nav_run(goal: str, script: list[tuple[str, dict]], *extra: Any):
    from tests.unit.vcli.test_native_loop import _nav_agent
    from tests.unit.vcli.test_native_reverify_after_other_action import _run as _reverify_run

    agent = _nav_agent(*extra)
    agent._skill_registry.register(_TurnFails(agent._base))
    return _reverify_run(goal, script, agent=agent)


_NAV, _AT = ("navigate", {"x": 9.0, "y": 3.0}), ("verify", {"expr": "at_position(9.0, 3.0)"})


@pytest.mark.parametrize("goal", ["走到坐标 (9.0,3.0) 然后左转90度", "go to the kitchen then turn left"])
def test_a_turn_failed_after_motion_then_a_read_and_a_reverify_rejects(goal) -> None:
    rep, rows, _tr = _nav_run(goal, [_NAV, _AT, ("turn_fake", {}), ("where_fake", {}), _AT], _read("where_fake"))
    assert rows[1][:2] == ("turn_fake", False), rows
    assert rep.verified is False and rep.evidence == "FAILED", (rep.reject_reason, rows)


@pytest.mark.parametrize("expr", ["True", "len(get_position()) == 3", "facing(0.0)"])
def test_a_stop_never_launders_a_turn_failed_after_motion(expr) -> None:
    stop = _Skill("stop", lambda: None)
    stop.housekeeping = True
    rep, rows, _tr = _nav_run("走到坐标 (9.0,3.0) 然后左转90度",
                              [_NAV, _AT, ("turn_fake", {}), ("verify", {"expr": expr}), ("stop", {}),
                               ("verify", {"expr": expr})], stop)
    assert [r[0] for r in rows] == ["navigate", "turn_fake", "", "stop"], rows
    assert rep.per_step[1].failed_attempt and rep.verified is False and rep.evidence == "FAILED", rep.reject_reason


@pytest.mark.parametrize("goal", ["走到坐标 (9.0,3.0) 然后左转90度", "go to (9,3) then turn left", "走到坐标 (9.0,3.0)"])
@pytest.mark.parametrize("script", [[("turn_fake", {}), _NAV], [("turn_fake", {}), _NAV, _AT]])
def test_a_failed_turn_before_a_proving_navigate_is_never_recovered_from(goal, script) -> None:
    """SP1V-01: the turn has no hint, so it is no attempt of the coordinate item; a proof
    of that item after it excuses nothing, whatever part of the sentence the contract covers."""
    rep, rows, tr = _nav_run(goal, script)
    assert tr.contract == ("at_position(9.0, 3.0)",), tr.contract
    assert [(r[0], r[1]) for r in rows][:2] == [("turn_fake", False), ("navigate", True)], rows
    assert rep.goal_check == "proven" and rep.verified is False and rep.evidence == "FAILED", rep.reject_reason
    assert rep.per_step[0].reject_reason.startswith("the action failed: "), rep.per_step
