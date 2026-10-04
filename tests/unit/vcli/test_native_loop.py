# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 unit tests — native tool-use producer (run_turn_native + NativeStepRunner).

Deterministic, no MuJoCo, no network: a ``FakeToolScriptBackend`` replays a SCRIPT
of native tool_use turns through the REAL ``run_turn_native``, against a duck-typed
fake agent (a base exposing the actor-causation surface + a registered walk skill).

Pinned here (the RED-0 tripwire + the review fixes):
- (RED-0) the tool-script backend drives run_turn_native to the EXPECTED ordered
  tool calls (walk -> verify -> finish), and the legacy single-plan FakeBackend
  constructor is byte-identical (unchanged).
- (fix 2 granularity) N bare walks then ONE verify -> exactly ONE StepRecord whose
  strategy is the last walk; intermediate walks are NOT each a checked sub-goal.
- (fix 3 timing + spine parity) an HONEST walk -> CAUSED -> GROUNDED -> verified;
  a verify-only NO-OP (predicate true at baseline, no walk) -> UNCAUSED -> RAN ->
  NOT verified; verified == evidence_passed (the spine, not a re-derivation).
- (replan-via-model) a False verify is followed by ANOTHER model-issued walk +
  verify — the runner holds NO replan/iteration state; the trace has both pairs.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.harness.fake_backend import (
    FakeBackend,
    FakeToolScriptBackend,
    tool_turn,
)
from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.trace_store import (
    evidence_passed,
    verify_oracle_names,
)
from vector_os_nano.vcli.session import Session
from vector_os_nano.vcli.verdict import VerdictReport


# ---------------------------------------------------------------------------
# Fakes — a base on the actor-causation surface + a registered walk skill
# ---------------------------------------------------------------------------


class _FakeBase:
    """Go2-base stand-in: instrumented commanded-motion counter + pose.

    ``walk`` advances the commanded-motion counter AND displaces the pose (an
    honest, actor-caused move). ``cmd_motion`` / ``get_position`` / ``get_heading``
    are the SAME accessors actor_causation.capture reads.
    """

    # mark as a sim adapter so SkillWrapperTool auto-allows the motor skill.
    __module__ = "vector_os_nano.hardware.sim.fake"

    def __init__(self, x: float, y: float) -> None:
        self._x, self._y, self._yaw = float(x), float(y), 0.0
        self._cmd_motion = 0.0
        self._connected = True

    def get_position(self) -> list[float]:
        return [self._x, self._y, 0.0]

    def get_heading(self) -> float:
        return self._yaw

    def cmd_motion(self) -> float:
        return self._cmd_motion

    def walk(self, vx: float, vy: float, vyaw: float, duration: float) -> bool:
        # Command magnitude + a real displacement -> CAUSED.
        self._cmd_motion += abs(vx) + abs(vy) + abs(vyaw)
        self._x += vx * duration
        self._y += vy * duration
        return True


class _FakeWalkSkill:
    """Minimal Skill-protocol walk: drives base.walk, no auto_steps."""

    name = "walk"
    description = "Walk the quadruped forward by a given distance (moves the base)."
    parameters = {
        "distance": {"type": "number", "required": False, "default": 1.0},
        "speed": {"type": "number", "required": False, "default": 0.3},
    }
    preconditions = ["base"]
    effects = {"is_moving": False}

    def execute(self, params, context):
        from vector_os_nano.core.types import SkillResult

        base = context.base
        if base is None:
            return SkillResult(success=False, error_message="no base", diagnosis_code="no_base")
        distance = float(params.get("distance", 1.0))
        speed = float(params.get("speed", 0.3))
        duration = distance / speed if speed > 0 else 0.0
        base.walk(speed, 0.0, 0.0, duration)
        return SkillResult(success=True, result_data={"distance": distance})


def _make_agent(x: float = 0.0, y: float = 0.0):
    """A duck-typed agent: a fake base + a real SkillRegistry with the walk skill."""
    from vector_os_nano.core.skill import SkillRegistry

    base = _FakeBase(x, y)
    reg = SkillRegistry()
    reg.register(_FakeWalkSkill())
    agent = SimpleNamespace(
        _base=base,
        _arm=None,
        _gripper=None,
        _spatial_memory=None,
        _skill_registry=reg,
    )

    def _build_context():
        return SimpleNamespace(base=base, arm=None, gripper=None, services=None)

    def _sync_robot_state():
        return None

    agent._build_context = _build_context
    agent._sync_robot_state = _sync_robot_state
    return agent, base


def _make_engine(agent, backend):
    """A REAL VectorEngine wired for VGG over the fake agent + a scripted backend."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    eng = VectorEngine(
        backend=backend,
        registry=CategorizedToolRegistry(),
        permissions=PermissionContext(no_permission=True),  # harness grants asks (the gate: test_native_safety)
    )
    eng._world = RobotWorld()
    eng.init_vgg(agent=agent, skill_registry=agent._skill_registry, world=RobotWorld())
    eng._vgg_agent = agent
    eng._backend = backend
    return eng


def _session() -> Session:
    return Session(
        session_id="native-test",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        path=Path("/tmp/native_test_session.jsonl"),
    )


# ---------------------------------------------------------------------------
# RED-0 — the tool-script backend + run_turn_native ordered drive
# ---------------------------------------------------------------------------


def test_legacy_fake_backend_constructor_unchanged() -> None:
    """The single-plan FakeBackend constructor is byte-identical (legacy seam)."""
    fb = FakeBackend({"goal": "g", "sub_goals": []})
    resp = fb.call(messages=[], tools=[], system=[], max_tokens=10)
    assert resp.stop_reason == "end_turn"
    assert resp.tool_calls == []
    assert "sub_goals" in resp.text


def test_tool_script_replays_ordered_turns() -> None:
    """from_tool_script replays the SEQUENCE; run_turn_native dispatches in order."""
    calls: list[str] = []

    class _Recorder(FakeToolScriptBackend):
        def call(self, **kw):  # type: ignore[override]
            # record the order the loop drives the model
            r = super().call(**kw)
            calls.append(",".join(tc.name for tc in r.tool_calls) or "end")
            return r

    backend = _Recorder.from_tool_script(
        [
            tool_turn(("walk", {"distance": 2.0, "speed": 0.3})),
            tool_turn(("verify", {"expr": "at_position(0.0, 0.0, 5.0)"})),
            tool_turn(end=True),
        ]
    )
    agent, _base = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("walk somewhere then verify", session=_session())

    # Ordered drive: walk turn, then verify turn, then terminal end_turn.
    assert calls == ["walk", "verify", "end"]
    # Exactly ONE recorded (action-chain -> verify) step.
    assert len(trace.steps) == 1
    assert trace.steps[0].strategy == "walk"


# ---------------------------------------------------------------------------
# fix 2 — ONE StepRecord per (action-chain -> verify) pair, not per walk
# ---------------------------------------------------------------------------


def test_three_walks_then_one_verify_is_one_step() -> None:
    """3 bare walks then 1 verify -> a single GROUNDED step (intermediate walks
    are NOT each a checked sub-goal; no sentinel-verify of intermediates)."""
    backend = FakeToolScriptBackend.from_tool_script(
        [
            tool_turn(("walk", {"distance": 1.0, "speed": 0.3})),
            tool_turn(("walk", {"distance": 1.0, "speed": 0.3})),
            tool_turn(("walk", {"distance": 1.0, "speed": 0.3})),
            tool_turn(("verify", {"expr": "at_position(3.0, 0.0, 1.0)"})),
            tool_turn(("finish", {})),
        ]
    )
    agent, base = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("walk 3m then verify", session=_session())

    assert len(trace.steps) == 1, "3 walks + 1 verify must be ONE checked step"
    step = trace.steps[0]
    assert step.verify_result is True
    assert step.actor_caused is ActorCaused.CAUSED
    # The honest-spine verdict: GROUNDED + verified.
    oracle_names = verify_oracle_names(agent, eng)
    report = VerdictReport.from_trace(trace, oracle_names)
    assert report.verified is True
    assert report.evidence == "GROUNDED"
    assert report.verified == evidence_passed(trace, oracle_names)


# ---------------------------------------------------------------------------
# fix 3 + spine parity — HONEST CAUSED vs NO-OP UNCAUSED through the producer
# ---------------------------------------------------------------------------


def test_honest_walk_is_caused_and_verified() -> None:
    backend = FakeToolScriptBackend.from_tool_script(
        [
            tool_turn(("walk", {"distance": 2.0, "speed": 0.3})),
            tool_turn(("verify", {"expr": "at_position(2.0, 0.0, 1.0)"})),
            tool_turn(("finish", {})),
        ]
    )
    agent, base = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("go to (2,0)", session=_session())
    oracle_names = verify_oracle_names(agent, eng)
    report = VerdictReport.from_trace(trace, oracle_names)

    assert trace.steps[0].actor_caused is ActorCaused.CAUSED
    assert report.verified is True and report.evidence == "GROUNDED"
    assert report.verified == evidence_passed(trace, oracle_names)
    assert base.get_position()[0] > 1.5  # actually moved


def test_noop_verify_only_is_uncaused_and_not_verified() -> None:
    """verify-only (no walk), predicate true at baseline -> UNCAUSED -> RAN."""
    backend = FakeToolScriptBackend.from_tool_script(
        [
            tool_turn(("verify", {"expr": "at_position(0.0, 0.0, 1.0)"})),
            tool_turn(("finish", {})),
        ]
    )
    agent, base = _make_agent(0.0, 0.0)  # already at (0,0) -> predicate true
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("claim you're at (0,0)", session=_session())
    oracle_names = verify_oracle_names(agent, eng)
    report = VerdictReport.from_trace(trace, oracle_names)

    step = trace.steps[0]
    assert step.verify_result is True, "predicate is true at baseline"
    assert step.actor_caused is ActorCaused.UNCAUSED, "no commanded motion"
    assert report.verified is False and report.evidence == "RAN"
    assert report.exit_code() == 2
    assert report.verified == evidence_passed(trace, oracle_names)


# ---------------------------------------------------------------------------
# replan-via-MODEL — the runner holds NO replan state; the model re-issues
# ---------------------------------------------------------------------------


def test_replan_is_model_driven_not_runner_state() -> None:
    """A False verify is followed by ANOTHER model walk+verify. Both pairs are
    recorded; the runner computed no 'retry' state. WP07 attempt semantics: the
    re-verify of the SAME predicate supersedes the FAIL (kept as audit) and, being
    actor-CAUSED, verifies the recovered turn (spine:retry-poisons-turn)."""
    backend = FakeToolScriptBackend.from_tool_script(
        [
            tool_turn(("walk", {"distance": 0.3, "speed": 0.3})),  # lands short
            tool_turn(("verify", {"expr": "at_position(3.0, 0.0, 0.5)"})),  # FAIL
            tool_turn(("walk", {"distance": 3.0, "speed": 0.3})),  # the model retries
            tool_turn(("verify", {"expr": "at_position(3.0, 0.0, 0.5)"})),  # PASS
            tool_turn(("finish", {})),
        ]
    )
    agent, base = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("go to (3,0), retry if short", session=_session())

    # TWO (action-chain -> verify) pairs recorded — the MODEL re-issued, the runner
    # did not loop internally.
    assert len(trace.steps) == 2
    assert trace.steps[0].verify_result is False  # the short attempt
    assert trace.steps[1].verify_result is True   # the corrected attempt
    assert trace.steps[0].superseded is True and trace.steps[1].superseded is False
    assert trace.steps[1].actor_caused is ActorCaused.CAUSED
    oracle_names = verify_oracle_names(agent, eng)
    report = VerdictReport.from_trace(trace, oracle_names)
    assert report.verified is True, report.reject_reason
    assert [s.superseded for s in report.per_step] == [True, False]
    assert report.per_step[0].reject_reason == "verify FAIL"  # the FAIL row stays as audit
    assert (report.n_steps, report.n_grounded) == (1, 1)


def test_empty_script_yields_unverified_empty_trace() -> None:
    """No verify ever called -> empty trace -> fail closed (not verified)."""
    backend = FakeToolScriptBackend.from_tool_script([tool_turn(end=True)])
    agent, _ = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("do nothing", session=_session())
    oracle_names = verify_oracle_names(agent, eng)
    report = VerdictReport.from_trace(trace, oracle_names)
    assert len(trace.steps) == 0
    assert report.verified is False


# ---------------------------------------------------------------------------
# WP07 (h) — the prompt carries no ground-truth scene names. The old STEP-7 object
# vocab listed the arm's GT object bodies; world context may only come from the
# actor's belief (the grasp skill reports the object it resolved).
# ---------------------------------------------------------------------------


def test_prompt_never_lists_gt_scene_names() -> None:
    seen: list[str] = []

    class _Recorder(FakeToolScriptBackend):
        def call(self, **kw):  # type: ignore[override]
            seen.append(" ".join(b.get("text", "") for b in kw.get("system") or []))
            return super().call(**kw)

    class _Arm:
        def get_object_positions(self):
            return {"pickable_bottle_green": [1.0, 1.0, 0.3]}

    agent, _base = _make_agent(0.0, 0.0)
    agent._arm = _Arm()
    eng = _make_engine(agent, _Recorder.from_tool_script([tool_turn(end=True)]))
    eng.run_turn_native("grab the green bottle", session=_session())
    assert seen and "pickable_bottle_green" not in seen[0]
    assert "graspable objects are" not in seen[0]


def test_oracle_stays_strict_wrong_canonical_name_is_false() -> None:
    """STEP 7 moat guard: the fix did NOT loosen the oracle's exact match.

    While genuinely holding the banana, a WRONG canonical name (apple / mug) still
    returns False — verifying that the vocab-expose approach left ``make_holding_object``
    strictly canonical (case-insensitive EXACT on the scene name, no fuzzy/loose match).
    """
    from vector_os_nano.vcli.worlds.arm_sim_oracle import make_holding_object

    class _Arm:
        _connected = True

        def get_object_positions(self):
            # banana lifted at the EE; mug resting far away.
            return {"banana": [0.0, 0.0, 0.26], "mug": [0.5, 0.5, 0.0]}

        def get_joint_positions(self):
            return [0.0]

        def fk(self, _joints):
            return ([0.0, 0.0, 0.26], None)

    class _Gripper:
        def is_holding(self):
            return True

    agent = SimpleNamespace(_arm=_Arm(), _gripper=_Gripper())
    holding_object = make_holding_object(agent)
    assert holding_object("banana") is True  # the real, canonical match
    assert holding_object("apple") is False  # wrong name -> NOT loosened
    assert holding_object("mug") is False    # present but resting, not held
    assert holding_object("香蕉") is False    # localized name is NOT a canonical key


# ---------------------------------------------------------------------------
# D17 Prong-1 — a ROBOT world drops the MUTATING code tools (file_write/file_edit/
# bash) from the action surface so a physical robot goal cannot be "accomplished" by
# writing a marker file; read-only diagnostics (file_read/glob/grep) are KEPT. The
# DEV world (no robot agent) keeps the full code-tool set unchanged.
# ---------------------------------------------------------------------------


class _CodeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.input_schema = {"type": "object", "properties": {}}

    def execute(self, params, context):  # pragma: no cover - never dispatched here
        return None


class _CodeRegistry:
    def list_categories(self):
        return {"code": ["file_read", "file_write", "file_edit", "bash", "glob", "grep"]}

    def get(self, name):
        return _CodeTool(name)


def _engine_with_code_tools():
    return SimpleNamespace(_registry=_CodeRegistry(), _goal_executor=None)


def test_robot_world_drops_mutating_code_tools() -> None:
    from vector_os_nano.vcli.native_loop import _build_motor_tools

    robot_agent = SimpleNamespace(_base=None, _arm=None, _perception=None, _skill_registry=None)
    tools = _build_motor_tools(robot_agent, _engine_with_code_tools())
    # Prong-1: no file/shell WRITE is offered as an action in a robot world.
    assert "file_write" not in tools
    assert "file_edit" not in tools
    assert "bash" not in tools
    # Read-only diagnostics stay — the persona inspects code with them.
    assert "file_read" in tools
    assert "glob" in tools
    assert "grep" in tools


def test_dev_world_keeps_full_code_tool_set() -> None:
    from vector_os_nano.vcli.native_loop import _build_motor_tools

    tools = _build_motor_tools(None, _engine_with_code_tools())  # dev world: no agent
    for name in ("file_read", "file_write", "file_edit", "bash", "glob", "grep"):
        assert name in tools, f"dev world must keep {name}"


# ---------------------------------------------------------------------------
# WP07 — the runner's attempt groups, auto post-verify, probe refusal, the task
# contract, failure recording and dispatched-skill intents (fakes, no sim).
# ---------------------------------------------------------------------------


class _NavBase(_FakeBase):
    """A base with the in-process ``navigate_to`` the native navigate tool drives."""

    def navigate_to(self, x: float, y: float, timeout: float | None = None) -> bool:
        self.walk(abs(x - self._x) + abs(y - self._y), 0.0, 0.0, 1.0)
        self._x, self._y = float(x), float(y)
        return True


class _HeldArm:
    """An arm whose one object 'can' rises to the EE once grasped."""

    _connected = True

    def __init__(self) -> None:
        self.held = False

    def get_object_positions(self):
        return {"can": [0.0, 0.0, 0.3] if self.held else [0.6, 0.6, 0.02]}

    def get_joint_positions(self):
        return [0.0] * 6

    def fk(self, _joints):
        return ([0.0, 0.0, 0.3], None)


class _WeldGripper:
    def __init__(self, arm: _HeldArm) -> None:
        self._arm = arm

    def is_holding(self) -> bool:
        return self._arm.held

    def weld_is_active(self) -> dict:
        return {"can": self._arm.held}


class _Skill:
    """A Skill-protocol stub: ``run()`` is its effect; ``fail`` makes it error."""

    preconditions: list = []
    effects: dict = {}

    def __init__(self, name: str, run, verify_hint: str = "", fail: str = "") -> None:
        self.name, self.description, self.verify_hint = name, name, verify_hint
        self.parameters = {"object": {"type": "string", "required": False}}
        self._run, self._fail = run, fail

    def execute(self, params, context):
        from vector_os_nano.core.types import SkillResult

        if self._fail:
            return SkillResult(success=False, error_message=self._fail)
        self._run()
        return SkillResult(success=True, result_data={"object": "can"})


def _make_arm_agent(*skills_fn):
    agent, _base = _make_agent(0.0, 0.0)
    arm = _HeldArm()
    agent._arm, agent._gripper = arm, _WeldGripper(arm)
    for make in skills_fn:
        agent._skill_registry.register(make(arm))
    return agent, arm


def _grasp(arm):
    return _Skill("grasp_fake", lambda: setattr(arm, "held", True), verify_hint="holding_object()")


def _stop(arm):
    return _Skill("stop", lambda: None)


def _verdict(agent, eng, trace):
    return VerdictReport.from_trace(trace, verify_oracle_names(agent, eng))


def _tool_results(session) -> list[str]:
    return [str(c.get("content")) for m in session.to_messages() for c in (m.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "tool_result"]


def test_reverify_after_a_housekeeping_action_is_graded_against_the_group_baseline() -> None:
    """The live WP03 shape through the runner: grasp -> verify holding -> stop ->
    verify holding. The second verify is an attempt of the same sub-goal, graded
    against the baseline before the grasp, so it is CAUSED and the turn verifies."""
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "can"})),
        tool_turn(("verify", {"expr": "holding_object('can')"})),
        tool_turn(("stop", {})),
        tool_turn(("verify", {"expr": "holding_object('can')"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp, _stop)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起那个罐子", session=_session())
    assert trace.contract == ("holding_object()",)
    assert (trace.contract_start, trace.contract_end) == ((False,), (True,))
    assert [s.superseded for s in trace.steps] == [True, False]
    assert trace.steps[1].actor_caused is ActorCaused.CAUSED
    assert trace.intents == ("grasp",)  # a dispatched gripper skill is a manipulation
    report = _verdict(agent, eng, trace)
    assert report.verified is True, report.reject_reason
    assert report.contract == ("holding_object()",)


def test_auto_post_verify_proves_an_action_the_model_did_not_verify() -> None:
    """Graft (c): the skill's verify_hint runs right after it; the finish is accepted
    without a nudge and the auto-verify is the step's GROUNDED, CAUSED evidence."""
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "can"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp)
    eng = _make_engine(agent, backend)
    session = _session()
    trace = eng.run_turn_native("拿起那个罐子", session=session)
    (step,) = trace.steps
    assert trace.goal_tree.sub_goals[0].verify == "holding_object()"
    assert step.channels[0] == "auto-verify" and step.actor_caused is ActorCaused.CAUSED
    assert _verdict(agent, eng, trace).verified is True
    assert any("auto-verify holding_object() -> PASS" in r for r in _tool_results(session))


def test_a_model_verify_takes_the_place_of_the_auto_verify() -> None:
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "can"})),
        tool_turn(("verify", {"expr": "holding_object('can')"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起那个罐子", session=_session())
    (step,) = trace.steps
    assert trace.goal_tree.sub_goals[0].verify == "holding_object('can')"
    assert step.result_data["auto_verify"] == {"expr": "holding_object()", "result": True}


def _probe_script():
    return FakeToolScriptBackend.from_tool_script([
        tool_turn(("walk", {"distance": 0.3, "speed": 0.3})),
        tool_turn(("verify", {"expr": "at_position(3.0, 0.0, 0.5)"})),
        tool_turn(("verify", {"expr": "at_position(0.3, 0.0, 0.5)"})),
        tool_turn(("finish", {})),
    ])


def test_a_verify_probe_with_no_action_since_the_last_verify_is_refused() -> None:
    """Graft (d): the second verify reads nothing — no GT bit leaks to the actor."""
    agent, _base = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, _probe_script())
    session = _session()
    trace = eng.run_turn_native("go to (3,0)", session=session)
    assert [sg.verify for sg in trace.goal_tree.sub_goals] == ["at_position(3.0, 0.0, 0.5)"]
    assert any("no action ran since your last verify" in r for r in _tool_results(session))


def test_mutation_probe_refusal(monkeypatch) -> None:
    """Graft (g): with the refusal switched off the same script reads GT twice."""
    from vector_os_nano.vcli import native_loop

    real = native_loop.NativeStepRunner.handle_verify

    def leaky(self, expr, goal_id=""):  # noqa: ANN001, ANN202
        self._acted_since_verify = True
        return real(self, expr, goal_id)

    monkeypatch.setattr(native_loop.NativeStepRunner, "handle_verify", leaky)
    agent, _base = _make_agent(0.0, 0.0)
    trace = _make_engine(agent, _probe_script()).run_turn_native("go to (3,0)", session=_session())
    assert len(trace.steps) == 2


def test_a_failed_skill_is_recorded_with_its_message() -> None:
    """producers:native-step-collapses-result: the ToolResult is kept (rule 4)."""
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "can"})),
        tool_turn(("verify", {"expr": "holding_object('can')"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(
        lambda arm: _Skill("grasp_fake", lambda: None, "holding_object()", fail="ik_unreachable"))
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起那个罐子", session=_session())
    (step,) = trace.steps
    assert step.success is False and "ik_unreachable" in step.error
    assert step.result_data["is_error"] is True and "ik_unreachable" in step.result_data["content"]
    assert _verdict(agent, eng, trace).evidence == "FAILED"


def test_navigate_retry_recovers_and_the_contract_is_printed_first() -> None:
    """The unit analogue of the e2e: navigate short -> FAIL -> navigate -> PASS ->
    GROUNDED; the task contract is announced before any tool runs."""
    events: list[str] = []
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("navigate", {"x": 1.0, "y": 0.0})),
        tool_turn(("verify", {"expr": "at_position(3.0, 0.0)"})),
        tool_turn(("navigate", {"x": 3.0, "y": 0.0})),
        tool_turn(("verify", {"expr": "at_position(3.0, 0.0)"})),
        tool_turn(("finish", {})),
    ])
    agent, _base = _make_agent(0.0, 0.0)
    agent._base = _NavBase(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("走到坐标 (3.0,0.0)", session=_session(), on_progress=events.append)
    assert events[0] == "Task contract: at_position(3.0, 0.0)"
    assert [s.verify_result for s in trace.steps] == [False, True]
    assert [s.superseded for s in trace.steps] == [True, False]
    assert trace.steps[1].actor_caused is ActorCaused.CAUSED
    assert _verdict(agent, eng, trace).verified is True


def test_navigate_auto_verifies_its_bound_target() -> None:
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("navigate", {"x": 3.0, "y": 0.0})),
        tool_turn(("finish", {})),
    ])
    agent, _base = _make_agent(0.0, 0.0)
    agent._base = _NavBase(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("走到坐标 (3.0,0.0)", session=_session())
    assert trace.goal_tree.sub_goals[0].verify == "at_position(3.0, 0.0)"
    assert trace.steps[0].actor_caused is ActorCaused.CAUSED
    assert _verdict(agent, eng, trace).verified is True
