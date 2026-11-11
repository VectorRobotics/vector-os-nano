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

import time
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


def _make_agent(x: float = 0.0, y: float = 0.0, **gt_kwargs):
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

    bind_fake_sim(agent, "go2", **gt_kwargs)  # the GT oracles read the fake base
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
# INT2 — localized is caused by the look that observed it this step (the look is
# read-only: it never actuates, so the no-action rule alone graded it UNCAUSED);
# a right map entry the step did not write never grounds, whatever tool ran.
# ---------------------------------------------------------------------------

_BOTTLE_XY = (10.9, 3.0)


class _FakeLookSkill:
    """A read-only look that maps the green bottle (or maps nothing)."""

    name = "look"
    description = "Look around and record what the camera sees on the map."
    parameters: dict = {}
    preconditions: list = []
    effects: dict = {}

    def __init__(self, sg, sees: bool) -> None:
        self._sg, self._sees = sg, sees

    def execute(self, params, context):
        from vector_os_nano.core.types import SkillResult

        if self._sees:  # a localized observation (what observation_count counts)
            self._sg.merge_object("green bottle", "", "", x=_BOTTLE_XY[0], y=_BOTTLE_XY[1])
        return SkillResult(success=True)


def _localize_turn(first_tool: str, *, sees: bool, on_map: bool):
    from vector_os_nano.core.scene_graph import ObjectNode, SceneGraph
    from vector_os_nano.core.skill import skill

    agent, _base = _make_agent(10.0, 3.0, objects={"pickable_bottle_green": [*_BOTTLE_XY, 0.8]})
    agent._spatial_memory = sg = SceneGraph()
    if on_map:  # mapped by an earlier step of this turn (stamped after the turn start)
        sg.add_object(ObjectNode(object_id="obj_old", category="bottle", x=_BOTTLE_XY[0],
                                 y=_BOTTLE_XY[1], attributes={"color": "green"},
                                 first_seen=time.time() + 3600))
    agent._skill_registry.register(skill(actuates=False)(_FakeLookSkill)(sg, sees))
    params = {"distance": 0.3, "speed": 0.3} if first_tool == "walk" else {}
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn((first_tool, params)),
        tool_turn(("verify", {"expr": "localized('green bottle')"})),
        tool_turn(("finish", {})),
    ])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("找到绿色瓶子", session=_session())
    return trace, VerdictReport.from_trace(trace, verify_oracle_names(agent, eng))


def test_a_look_that_maps_the_object_localizes_it_grounded() -> None:
    trace, report = _localize_turn("look", sees=True, on_map=False)
    assert trace.steps[0].verify_result is True
    assert trace.steps[0].actor_caused is ActorCaused.CAUSED
    assert report.verified is True and report.evidence == "GROUNDED"


@pytest.mark.parametrize("tool", ["look", "walk"])
def test_a_map_entry_the_step_did_not_write_never_grounds(tool) -> None:
    trace, report = _localize_turn(tool, sees=False, on_map=True)
    assert trace.steps[0].verify_result is True  # the map IS right...
    assert trace.steps[0].actor_caused is ActorCaused.UNCAUSED  # ...but not this step's work
    assert report.verified is False and report.evidence == "RAN"


# ---------------------------------------------------------------------------
# STEP 7 — cross-language grasp vocab: the native loop hands the MODEL the
# world's CANONICAL object names so a non-English command verifies the STRICT
# object-specific predicate, while the oracle stays untouched-strict.
# ---------------------------------------------------------------------------


class _FakeArmWithObjects:
    """Duck-typed arm exposing get_object_positions (the single-source vocab)."""

    def __init__(self, objects):
        self._objects = objects

    def get_object_positions(self):
        return self._objects


def _arm_agent(objects):
    return SimpleNamespace(_arm=_FakeArmWithObjects(objects))


def test_scene_object_names_single_sources_canonical_names() -> None:
    """``_scene_object_names`` returns the arm's object-position keys, sorted.

    This is the SAME ground truth ``holding_object`` matches against, so the names
    the model is taught are exactly the names the oracle accepts (no split source).
    """
    from vector_os_nano.vcli.native_loop import _scene_object_names

    agent = _arm_agent({"banana": [0.1, 0.1, 0.06], "mug": [0.2, 0.0, 0.06]})
    assert _scene_object_names(agent) == ("banana", "mug")


def test_scene_object_names_fail_safe_empty() -> None:
    """No agent / no arm / no getter / raising getter -> EMPTY (pre-step-7 prompt)."""
    from vector_os_nano.vcli.native_loop import _scene_object_names

    assert _scene_object_names(None) == ()
    assert _scene_object_names(SimpleNamespace(_arm=None)) == ()
    assert _scene_object_names(SimpleNamespace(_arm=SimpleNamespace())) == ()

    class _Raises:
        def get_object_positions(self):
            raise RuntimeError("boom")

    assert _scene_object_names(SimpleNamespace(_arm=_Raises())) == ()


def test_system_prompt_lists_object_vocab_when_present() -> None:
    """The prompt teaches the canonical names + the translate-to-scene-name rule."""
    from vector_os_nano.vcli.native_loop import _native_system_prompt

    blocks = _native_system_prompt(None, frozenset({"holding_object"}), ("banana", "mug"))
    text = blocks[0]["text"]
    assert "graspable objects are: banana, mug" in text
    # The rule that closes the cross-language gap: translate the user's wording to
    # the canonical scene name (the model does 香蕉->banana itself).
    assert "translating the user's wording" in text
    assert "holding_object('banana')" in text


def test_system_prompt_omits_object_vocab_when_empty() -> None:
    """An EMPTY object set yields the EXACT pre-step-7 prompt (no object list)."""
    from vector_os_nano.vcli.native_loop import _native_system_prompt

    with_names = _native_system_prompt(None, frozenset({"holding_object"}), ("banana",))
    without = _native_system_prompt(None, frozenset({"holding_object"}), ())
    assert "graspable objects are" not in without[0]["text"]
    # Default arg also omits it (back-compat with any caller passing only 2 args).
    default = _native_system_prompt(None, frozenset({"holding_object"}))
    assert default[0]["text"] == without[0]["text"]
    assert with_names[0]["text"] != without[0]["text"]


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
