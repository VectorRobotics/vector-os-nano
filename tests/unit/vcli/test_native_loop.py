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
    from tests.harness.fake_gt import bind_fake_sim

    bind_fake_sim(agent, "go2")  # the GT oracles read the fake base
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
    recorded; the runner computed no 'retry' state. The final trace verifies only
    when EVERY checked step is GROUNDED, so a False first step keeps it unverified."""
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
    oracle_names = verify_oracle_names(agent, eng)
    report = VerdictReport.from_trace(trace, oracle_names)
    # A trace with ANY non-GROUNDED checked step is NOT verified (all-must-pass).
    assert report.verified is False


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
# WP14 — the native prompt is the kernel loop rules + the world's persona (task
# policy composed from the manifests) + the actor's belief ([Robot State]).
# ---------------------------------------------------------------------------


class _Gripper:
    def is_holding(self) -> bool:
        return False


def _go2_piper_like_agent():
    """A go2+arm-shaped agent in the apartment: base, gripper, scene graph, skills."""
    from vector_os_nano.core.scene_graph import ObjectNode, SceneGraph
    from vector_os_nano.core.skill import SkillRegistry

    agent, base = _make_agent(10.0, 3.0)
    reg = SkillRegistry()
    for name in ("walk", "look", "explore", "navigate", "navigate_to_object", "perception_grasp", "mobile_place"):
        reg.register(SimpleNamespace(name=name, description=name, parameters={}))
    sg = SceneGraph()
    sg.add_object(ObjectNode(object_id="o1", category="bottle", x=10.86, y=2.8, z=0.31,
                             attributes={"color": "green"}))
    agent._skill_registry, agent._spatial_memory = reg, sg
    agent._gripper, agent._config = _Gripper(), {"world": "apartment"}
    return agent


def _prompt_text(agent, world) -> str:
    from vector_os_nano.vcli.native_loop import _native_system_prompt

    blocks = _native_system_prompt(SimpleNamespace(_world=world), frozenset({"holding_object"}), agent)
    return "\n".join(b["text"] for b in blocks)


def test_kernel_prompt_has_no_grasp_first_rule() -> None:
    """docs-vs-code:prompt-forbids-fetch-composition: nothing forbids look / navigate
    before a grasp any more; the kernel rules are the verify/finish loop only."""
    text = _prompt_text(None, None)
    assert "verify(expr)" in text and "Never finish while" in text
    for banned in ("FIRST action", "do NOT navigate", "do NOT look", "obstacle-aware"):
        assert banned not in text, banned


def test_prompt_names_no_ground_truth_object() -> None:
    """The GT object-name leak (arm.get_object_positions keys) is gone."""
    agent = _go2_piper_like_agent()
    agent._arm = SimpleNamespace(get_object_positions=lambda: {"pickable_bottle_green": [10.9, 2.8, 0.3]})
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    assert "pickable_" not in _prompt_text(agent, RobotWorld())


def test_robot_world_policy_and_belief_state_reach_the_native_turn() -> None:
    """producers:native-no-context: the turn sees its body's task policy (composed from
    the manifests: the user's spot, the surfaces) and its belief (pose, room, gripper,
    the objects it localized)."""
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    text = _prompt_text(_go2_piper_like_agent(), RobotWorld())
    assert "look (it localizes" in text and "navigate_to_object first" in text
    assert "grasp it with perception_grasp" in text
    assert "navigate(8.0, 3.0)" in text and "near_object('user') and kept_holding" in text
    assert "dining_table centre (3.00, 7.50) top 0.66 m" in text and "object_on(" in text
    assert "go_to_room(room)" in text and "kitchen" in text
    assert "Position: (10.0, 3.0" in text and "Current room: hallway" in text
    assert "Gripper: empty" in text
    assert "green bottle at (10.86, 2.80, 0.31), seen" in text


def test_policy_lines_follow_the_body() -> None:
    """An arm-only body gets no base lines; a base without an arm no grasp lines."""
    from vector_os_nano.vcli.worlds.robot import task_policy

    arm_only = _go2_piper_like_agent()
    arm_only._base = None
    text = task_policy(arm_only)
    assert "perception_grasp" in text and "navigate(" not in text and "go_to_room" not in text
    base_only = _go2_piper_like_agent()
    base_only._gripper = None
    text = task_policy(base_only)
    assert "go_to_room" in text and "holding_object" not in text
    assert task_policy(None) == ""


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
# WP14 — sim lifecycle inside a native turn, real-session memory, usage
# ---------------------------------------------------------------------------


class _FakeStartSim:
    """start_simulation stand-in: brings up a body and rebinds the REPL to it."""

    name = "start_simulation"
    description = "Start a sim."
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, engine_ref: list) -> None:
        self._engine = engine_ref

    def check_permissions(self, params, context):
        from vector_os_nano.vcli.tools.base import PermissionResult

        return PermissionResult("allow")

    def is_read_only(self, params) -> bool:
        return False

    def execute(self, params, context):
        from vector_os_nano.vcli.tools.base import ToolResult
        from vector_os_nano.vcli.worlds.robot import RobotWorld

        agent, _ = _make_agent(0.0, 0.0)
        context.app_state["agent"] = agent  # what rebind_world does
        self._engine[0].init_vgg(agent=agent, skill_registry=agent._skill_registry, world=RobotWorld())
        return ToolResult(content="Started go2 simulation, ready in 1.0s")


def test_lifecycle_tool_rebinds_the_turn_to_the_new_body() -> None:
    """A dev-world turn starts a sim, then walks and verifies on it in the SAME turn: the
    runner rebuilds tools, verifier and oracle names after the lifecycle tool."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.worlds.dev import DevWorld

    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("start_simulation", {})),
        tool_turn(("walk", {"distance": 2.0, "speed": 0.3})),
        tool_turn(("verify", {"expr": "at_position(2.0, 0.0, 1.0)"})),
        tool_turn(("finish", {})),
    ])
    ref: list = []
    registry = CategorizedToolRegistry()
    registry.register(_FakeStartSim(ref), category="sim")
    eng = VectorEngine(backend=backend, registry=registry, permissions=PermissionContext(no_permission=True))
    ref.append(eng)
    eng.init_vgg(agent=None, world=DevWorld())
    app_state: dict = {"agent": None}
    trace = eng.run_turn_native("start go2 and walk 2 m", session=_session(), app_state=app_state)

    new_agent = app_state["agent"]
    assert new_agent is not None and eng._vgg_agent is new_agent
    assert trace.actions_dispatched == 2  # the start + the walk: nothing re-runs either
    assert [s.strategy for s in trace.steps] == ["walk"]  # the start opens no graded step
    assert trace.steps[0].actor_caused is ActorCaused.CAUSED
    report = VerdictReport.from_trace(trace, verify_oracle_names(new_agent, eng))
    assert report.verified is True


def test_turns_share_the_real_session() -> None:
    """cli-entry:native-scratch-session-no-memory: the second turn's model call sees the
    first turn ('it' has a referent), and the usage lands on the session."""
    from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
    from vector_os_nano.vcli.session import TokenUsage

    seen: list[list] = []

    class _Recorder:
        def __init__(self, turns):
            self._turns = list(turns)

        def call(self, messages, **_kw):
            seen.append(list(messages))
            return self._turns.pop(0) if self._turns else LLMResponse(text="", stop_reason="end_turn")

    used = TokenUsage(input_tokens=100, output_tokens=7)
    walk = LLMResponse(text="", tool_calls=[LLMToolCall(id="a", name="walk", input={"distance": 1.0})],
                       stop_reason="tool_use", usage=used)
    check = LLMResponse(text="", tool_calls=[LLMToolCall(id="b", name="verify", input={"expr": "at_position(1.0, 0.0, 1.0)"})],
                        stop_reason="tool_use", usage=used)
    done = LLMResponse(text="到了", stop_reason="end_turn", usage=used)
    agent, _ = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, _Recorder([walk, check, done, done]))
    session = _session()
    eng.run_turn_native("往前走一米", session=session)
    eng.run_turn_native("再走一次", session=session)

    second_turn_first_call = seen[3]
    assert second_turn_first_call[0] == {"role": "user", "content": "往前走一米"}
    assert any(b.get("name") == "walk" for m in second_turn_first_call if m["role"] == "assistant"
               for b in m["content"] if isinstance(b, dict))
    assert session.token_usage.input_tokens == 400 and session.token_usage.output_tokens == 28
