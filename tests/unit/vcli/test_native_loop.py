# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 unit tests — native tool-use producer (run_turn_native + NativeStepRunner).

Deterministic, no MuJoCo, no network: a ``FakeToolScriptBackend`` replays a SCRIPT
of native tool_use turns through the REAL ``run_turn_native``, against a duck-typed
fake agent (a base exposing the actor-causation surface + a registered walk skill).

Pinned here (the RED-0 tripwire + the review fixes):
- (RED-0) the tool-script backend drives run_turn_native to the EXPECTED ordered
  tool calls (walk -> verify -> finish).
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

from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
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
    """Minimal Skill-protocol walk: drives base.walk."""

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
    """A REAL VectorEngine bound to the fake agent + a scripted backend."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    eng = VectorEngine(
        backend=backend,
        registry=CategorizedToolRegistry(),
        permissions=PermissionContext(no_permission=True),  # harness grants asks (the gate: test_native_safety)
    )
    eng.bind(agent, RobotWorld())
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
    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.core.scene_nodes import ObjectNode
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
# WP14 — the native prompt is the kernel loop rules + the world's persona (task
# policy composed from the manifests) + the actor's belief ([Robot State]).
# ---------------------------------------------------------------------------


class _Gripper:
    def is_holding(self) -> bool:
        return False


def _go2_piper_like_agent():
    """A go2+arm-shaped agent in the apartment: base, gripper, scene graph, skills."""
    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.core.scene_nodes import ObjectNode
    from vector_os_nano.core.skill import SkillRegistry

    agent, base = _make_agent(10.0, 3.0)
    reg = SkillRegistry()
    for name in ("walk", "look", "explore", "navigate", "navigate_to_object", "perception_grasp", "place_top_down"):
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


def _verify_description(engine, agent) -> str:
    from vector_os_nano.vcli.native_loop import _bind_body

    *_, schemas = _bind_body(engine, agent)
    return next(s for s in schemas if s["name"] == "verify")["description"]


def test_dev_verify_tool_reads_the_oracles_own_docstrings() -> None:
    """The verify tool's predicates come from ONE source, each oracle's signature and
    docstring: the dev world's one-file-per-call guidance reaches the model there."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.worlds.dev import DevWorld

    eng = VectorEngine(backend=None)
    eng.bind(None, DevWorld())
    desc = _verify_description(eng, None)
    assert "\n- file_exists(path): True if the file *path* (relative to cwd, inside the project) exists" in desc
    assert "\n- grep_count(pattern, path): Number of regex *pattern* matches in the ONE regular file" in desc
    assert "(path has no default)" in desc
    assert desc.count("'and' the calls together") == 2  # grep_count and path_contains
    assert "\n- path_contains(path, substr): True if the ONE regular file" in desc
    # Only the first docstring paragraph (the model-facing one), and no robot oracle.
    assert "pseudo-fs" not in desc and "at_position" not in desc


def test_robot_verify_tool_lists_every_oracle_and_nothing_else() -> None:
    """A robot world: one line per evidence oracle, signature (defaults included) +
    docstring, and no kernel sentence about any one predicate. An untagged callable
    in the namespace and a dev predicate in a robot world are never offered."""
    import re

    from vector_os_nano.vcli.native_loop import _native_system_prompt
    from vector_os_nano.vcli.worlds.dev import file_exists

    agent, _ = _make_agent(0.0, 0.0)
    eng = _make_engine(agent, FakeToolScriptBackend.from_tool_script([]))
    names = verify_oracle_names(agent, eng)
    desc = _verify_description(eng, agent)
    assert re.search(r"\n- at_position\(x, y, tol=[0-9.]+\): Base within \*tol\* m \(xy\) of \(x, y\)\.", desc)
    listed = re.findall(r"\n- (\w+)\(", desc)
    assert sorted(listed) == sorted(names) and "at_position" in names

    real = eng._build_verifier_namespace
    eng._build_verifier_namespace = lambda a: {**real(a), "memory_reader": lambda: 1,
                                               "file_exists": file_exists}
    desc = _verify_description(eng, agent)
    assert "memory_reader" not in desc and "file_exists" not in desc
    prompt = "\n".join(b["text"] for b in _native_system_prompt(eng, frozenset(names), None))
    assert "tol=" not in prompt and "metres" not in prompt  # no world-specific kernel sentence


def test_a_mobile_arms_oracles_reach_the_model_with_signature_and_meaning() -> None:
    """The verify tool over the spine's GT oracles (a go2+arm body): every oracle is a
    line with its signature and its own docstring's first paragraph."""
    import re

    agent, _arm = _make_arm_agent()
    eng = _make_engine(agent, FakeToolScriptBackend.from_tool_script([]))
    names = verify_oracle_names(agent, eng)
    desc = _verify_description(eng, agent)
    for n in names:
        assert re.search(rf"\n- {n}\([^\n]*\): \S", desc), n
    for line in (
        "- kept_holding(target): Held now AND its weld never dropped since the turn started.",
        "- near_object(name, r=",
        "- object_on(target, surface): Object resting ON the named support surface",
        "- released(target=None): No longer held",
        "- localized(target, tol=0.15): The actor's SceneGraph has an instance of *target*",
    ):
        assert "\n" + line in desc, line


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
    return SimpleNamespace(_registry=_CodeRegistry())


def test_robot_world_drops_mutating_code_tools() -> None:
    from vector_os_nano.vcli.native_tools import _build_motor_tools

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
    from vector_os_nano.vcli.native_tools import _build_motor_tools

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
        context.app_state["agent"] = agent  # what bind_body does
        self._engine[0].bind(agent, RobotWorld())
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
    eng.bind(None, DevWorld())
    app_state: dict = {"agent": None}
    trace = eng.run_turn_native("start go2 and walk 2 m", session=_session(), app_state=app_state)

    new_agent = app_state["agent"]
    assert new_agent is not None and eng._agent is new_agent
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


def test_native_never_sends_a_tool_use_absent_from_its_tools() -> None:
    """I23: an earlier turn's bash tool_use (a dev-world turn, before a sim started) is
    not a tool of this robot-world turn, so native sends it as text."""
    seen: list[tuple[list, set]] = []

    class _Recorder:
        def call(self, messages, tools, **_kw):
            seen.append((list(messages), {t["name"] for t in tools}))
            return LLMResponse(text="", stop_reason="end_turn")

    session = _session()
    session.append_user("ls")
    session.append_assistant("", [{"type": "tool_use", "id": "x1", "name": "bash", "input": {"command": "ls"}}])
    session.append_tool_results([{"tool_use_id": "x1", "content": "README.md"}])
    agent, _ = _make_agent(0.0, 0.0)
    _make_engine(agent, _Recorder()).run_turn_native("往前走一米", session=session)

    messages, names = seen[0]
    blocks = [b for m in messages if isinstance(m["content"], list) for b in m["content"]]
    assert "bash" not in names
    assert not [b for b in blocks if b.get("type") in ("tool_use", "tool_result")]
    assert any("README.md" in b.get("text", "") for b in blocks)


class _Scripted:
    """Backend that replays scripted responses and records every call's messages."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def call(self, messages, tools, **_kw):
        from vector_os_nano.vcli.backends.types import LLMResponse
        self.seen.append(list(messages))
        return self.turns.pop(0) if self.turns else LLMResponse(text="", stop_reason="end_turn")


def _alternates(messages) -> list[str]:
    from vector_os_nano.vcli.backends.openai_compat import convert_messages
    roles = [m["role"] for m in convert_messages(messages, "")]
    assert all(a != b or a == "tool" for a, b in zip(roles, roles[1:])), roles
    return roles


def test_native_turn_after_a_foreign_tool_then_text_turn_keeps_role_alternation() -> None:
    """I23-D1: an earlier turn (bash tool_use, its result, then final text) whose tool is
    not offered now is bash-as-text followed by the final text: one assistant message,
    not two (DeepSeek via openai_compat does not merge them)."""
    rec = _Scripted([])
    session = _session()
    session.append_user("ls")
    session.append_assistant("", [{"type": "tool_use", "id": "x1", "name": "bash", "input": {"command": "ls"}}])
    session.append_tool_results([{"tool_use_id": "x1", "content": "README.md"}])
    session.append_assistant("there is a README", [])
    agent, _ = _make_agent(0.0, 0.0)
    _make_engine(agent, rec).run_turn_native("往前走一米", session=session)
    assert _alternates(rec.seen[0]) == ["user", "assistant", "user"]


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


_CAN = "pickable_can_red"  # the manifest binds "红色的罐子" / "red can" to it


class _HeldArm:
    """An arm whose one object, the apartment's red can, rises to the EE once grasped."""

    _connected = True

    def __init__(self) -> None:
        self.held = False

    def get_object_positions(self):
        return {_CAN: [0.0, 0.0, 0.3] if self.held else [0.6, 0.6, 0.02]}

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
        return {_CAN: self._arm.held}


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
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _base = _make_agent(0.0, 0.0)
    arm = _HeldArm()
    agent._arm, agent._gripper = arm, _WeldGripper(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper")  # a go2+arm body: the gripper's GT oracles are live
    for make in skills_fn:
        agent._skill_registry.register(make(arm))
    return agent, arm


def _grasp(arm):
    return _Skill("grasp_fake", lambda: setattr(arm, "held", True), verify_hint="holding_object({object})")


def _stop(arm):
    stop = _Skill("stop", lambda: None)
    stop.housekeeping = True  # as the real StopSkill declares (skills/go2/stop.py)
    return stop


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
        tool_turn(("grasp_fake", {"object": "red can"})),
        tool_turn(("verify", {"expr": "holding_object('red can')"})),
        tool_turn(("stop", {})),
        tool_turn(("verify", {"expr": "holding_object('red can')"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp, _stop)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起红色的罐子", session=_session())
    assert trace.contract == (f"holding_object('{_CAN}')",)
    assert (trace.contract_start, trace.contract_end) == ((False,), (True,))
    assert [s.superseded for s in trace.steps] == [True, False]
    assert trace.steps[1].actor_caused is ActorCaused.CAUSED
    assert trace.intents == ("grasp",)  # a dispatched gripper skill is a manipulation
    report = _verdict(agent, eng, trace)
    assert report.verified is True, report.reject_reason
    assert report.contract == (f"holding_object('{_CAN}')",)


def test_auto_post_verify_proves_an_action_the_model_did_not_verify() -> None:
    """Graft (c): the skill's verify_hint runs right after it; the finish is accepted
    without a nudge and the auto-verify is the step's GROUNDED, CAUSED evidence."""
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "red can"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp)
    eng = _make_engine(agent, backend)
    session = _session()
    trace = eng.run_turn_native("拿起红色的罐子", session=session)
    (step,) = trace.steps
    assert trace.goal_tree.sub_goals[0].verify == "holding_object('red can')"
    assert step.channels[0] == "auto-verify" and step.actor_caused is ActorCaused.CAUSED
    assert _verdict(agent, eng, trace).verified is True
    assert any("auto-verify holding_object('red can') -> PASS" in r for r in _tool_results(session))


def test_a_model_verify_takes_the_place_of_the_auto_verify() -> None:
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "red can"})),
        tool_turn(("verify", {"expr": "holding_object('red can')"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起红色的罐子", session=_session())
    (step,) = trace.steps
    assert trace.goal_tree.sub_goals[0].verify == "holding_object('red can')"
    assert step.result_data["auto_verify"] == {"expr": "holding_object('red can')", "result": True}


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
    from vector_os_nano.vcli import native_runner

    real = native_runner.NativeStepRunner.handle_verify

    def leaky(self, expr, goal_id=""):  # noqa: ANN001, ANN202
        self._acted_since_verify = True
        return real(self, expr, goal_id)

    monkeypatch.setattr(native_runner.NativeStepRunner, "handle_verify", leaky)
    agent, _base = _make_agent(0.0, 0.0)
    trace = _make_engine(agent, _probe_script()).run_turn_native("go to (3,0)", session=_session())
    assert len(trace.steps) == 2


def test_a_failed_skill_is_recorded_with_its_message() -> None:
    """producers:native-step-collapses-result: the ToolResult is kept (rule 4)."""
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "red can"})),
        tool_turn(("verify", {"expr": "holding_object('red can')"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(
        lambda arm: _Skill("grasp_fake", lambda: None, "holding_object()", fail="ik_unreachable"))
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起红色的罐子", session=_session())
    # Charter 3.5: no arm counter, so no motion evidence: a failed attempt, closed at once
    # as its own FAILED row; the model's verify after it is a row of its own, and it
    # asserts the hint's predicate (holding the red can is holding an object), so it
    # supersedes the failed row (SP1F1V-01) and is graded: it FAILs, nothing acted, UNCAUSED.
    step, _after = trace.steps
    assert step.failed_attempt is True
    assert step.success is False and "ik_unreachable" in step.error
    assert step.result_data["is_error"] is True and "ik_unreachable" in step.result_data["content"]
    rep = _verdict(agent, eng, trace)
    assert [(s.evidence, s.superseded) for s in rep.per_step] == [("FAILED", True), ("RAN", False)]
    assert rep.per_step[1].actor_caused == "UNCAUSED" and rep.per_step[1].verify_result is False
    assert rep.verified is False and rep.reject_reason == "native_step_1: verify FAIL", rep.reject_reason


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


def test_an_auto_verified_step_keeps_its_own_result_after_a_failed_read() -> None:
    """A read that errors after an auto-verified action does not fail that action's row."""
    def _broken_read(arm):
        read = _Skill("look_fake", lambda: None, fail="camera offline")
        read.__skill_actuates__ = False
        return read

    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("grasp_fake", {"object": "red can"})),
        tool_turn(("look_fake", {})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(_grasp, _broken_read)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起红色的罐子", session=_session())
    (step,) = trace.steps
    assert step.success is True and step.result_data["tool"] == "grasp_fake"
    assert step.strategy == "grasp_fake"
    assert _verdict(agent, eng, trace).verified is True


def test_an_aborted_navigate_is_one_failed_row_without_its_auto_verify() -> None:
    """A stop mid-navigate: the interrupted action is its own FAILED 'aborted' row
    (the REPL shows 'navigate → aborted'), never an auto-verify row."""
    from vector_os_nano.vcli.cognitive.abort import CancelToken
    from vector_os_nano.vcli.native_loop import run_turn_native

    token = CancelToken()

    class _StoppedNav(_NavBase):
        def navigate_to(self, x, y, timeout=None):  # noqa: ANN001, ANN202
            token.cancel()  # the owner's stop lands mid-navigate
            return False

    backend = FakeToolScriptBackend.from_tool_script([tool_turn(("navigate", {"x": 3.0, "y": 0.0}))])
    agent, _base = _make_agent(0.0, 0.0)
    agent._base = _StoppedNav(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = run_turn_native(eng, "走到坐标 (3.0,0.0)", agent=agent, session=_session(), cancel=token)
    (step,) = trace.steps
    assert (step.strategy, step.success, step.error) == ("navigate", False, "aborted")
    assert trace.cancelled and _verdict(agent, eng, trace).evidence == "FAILED"


# ---------------------------------------------------------------------------
# A met contract exempts only housekeeping rows (verifier repros, through the producer)
# ---------------------------------------------------------------------------


def _nav_agent(*skills):
    agent, _base = _make_agent(0.0, 0.0)
    agent._base = _NavBase(0.0, 0.0)
    for sk in skills:
        agent._skill_registry.register(sk)
    return agent


@pytest.mark.parametrize("goal, script, contract", [
    # go2-only: nothing here can prove the grab, so no navigation contract stands in
    ("go to (3,0) and grab the red can", [("navigate", {"x": 3.0, "y": 0.0}), ("stop_fake", {})], ()),
    # the turn the coordinate contract does not cover was never proven
    ("走到坐标 (3,0) 然后左转90度", [("navigate", {"x": 3.0, "y": 0.0}), ("turn_fake", {})],
     ("at_position(3.0, 0.0)",)),
])
def test_an_unproven_actuating_step_is_not_verified(goal, script, contract) -> None:
    agent = _nav_agent(_Skill("stop_fake", lambda: None), _Skill("turn_fake", lambda: None))
    backend = FakeToolScriptBackend.from_tool_script(
        [*(tool_turn(call) for call in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    assert trace.contract == contract
    assert [s.housekeeping for s in trace.steps] == [False, False]
    report = _verdict(agent, eng, trace)
    assert report.verified is False and report.reject_reason.startswith("native_step_1:")


def test_a_declared_housekeeping_stop_does_not_fail_a_met_contract() -> None:
    """Graft (i), narrowly: a trailing stop changes no task state; the contract decides."""
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

    assert SkillWrapperTool(StopSkill(), None).housekeeping is True
    stop = _Skill("stop", lambda: None)
    stop.housekeeping = True
    agent = _nav_agent(stop)
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("navigate", {"x": 3.0, "y": 0.0})),
        tool_turn(("stop", {})),
        tool_turn(("finish", {})),
    ])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("走到坐标 (3.0,0.0)", session=_session())
    assert [(s.strategy, s.housekeeping) for s in trace.steps] == [("navigate", False), ("stop", True)]
    assert _verdict(agent, eng, trace).verified is True


@pytest.mark.parametrize("expr", ["True", "len(get_position()) == 3", "facing(0.0)"])
def test_a_stop_never_launders_a_failed_turn(expr) -> None:
    """Verifier P8 through the real producer: the turn skill errs ('Unknown turn
    direction') before commanding motion, then stop and the same re-verify. Charter 3.5:
    that turn was REFUSED, not an action: the verify right after it is a probe (refused
    unevaluated), and the stop's row carries the refusal; nothing was laundered. The
    failed-after-motion twin still rejects (test_native_failed_attempts)."""
    stop = _Skill("stop", lambda: None)
    stop.housekeeping = True
    agent = _nav_agent(_Skill("turn_fake", lambda: None, fail="Unknown turn direction"), stop)
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("navigate", {"x": 9.0, "y": 3.0})),
        tool_turn(("verify", {"expr": "at_position(9.0, 3.0)"})),
        tool_turn(("turn_fake", {})),
        tool_turn(("verify", {"expr": expr})),
        tool_turn(("stop", {})),
        tool_turn(("verify", {"expr": expr})),
        tool_turn(("finish", {})),
    ])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("走到坐标 (9.0,3.0) 然后左转90度", session=_session())
    assert trace.contract == ("at_position(9.0, 3.0)",)
    assert [(s.strategy, s.success, s.housekeeping, s.superseded) for s in trace.steps] == [
        ("navigate", True, False, False), ("stop", True, True, False)]
    report = _verdict(agent, eng, trace)
    assert report.per_step[1].refused == ("turn_fake: Unknown turn direction",)
    assert report.verified is True and report.goal_check == "proven", report.reject_reason


def test_navigating_twice_to_the_same_target_still_verifies() -> None:
    """Verifier P2: the second auto-verify is an attempt of the first's group, graded
    against the group baseline; 'already true before THIS action' is no false red."""
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("navigate", {"x": 3.0, "y": 0.0})),
        tool_turn(("navigate", {"x": 3.0, "y": 0.0})),
        tool_turn(("finish", {})),
    ])
    agent = _nav_agent()
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("走到坐标 (3.0,0.0)", session=_session())
    assert [s.superseded for s in trace.steps] == [True, False]
    assert trace.steps[1].actor_caused is ActorCaused.CAUSED
    assert _verdict(agent, eng, trace).verified is True


def test_a_failed_auto_verify_is_kept_when_the_model_verifies_something_else() -> None:
    """Verifier P3: navigate(3,0) lands at (1,0); the model may ADD a verify of its
    landing, never replace the FAILED auto-verify, which stays its own graded row."""
    class _ShortNav(_NavBase):
        def navigate_to(self, x, y, timeout=None):  # noqa: ANN001, ANN202
            return super().navigate_to(1.0, 0.0)

    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("navigate", {"x": 3.0, "y": 0.0})),
        tool_turn(("verify", {"expr": "at_position(1.0, 0.0)"})),
        tool_turn(("finish", {})),
    ])
    agent = _nav_agent()
    agent._base = _ShortNav(0.0, 0.0)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("go to the kitchen", session=_session())
    assert [(sg.verify, s.verify_result) for sg, s in zip(trace.goal_tree.sub_goals, trace.steps)] == [
        ("at_position(3.0, 0.0)", False), ("at_position(1.0, 0.0)", True)]
    report = _verdict(agent, eng, trace)
    assert report.verified is False and report.reject_reason == "native_step_0: verify FAIL"


def test_a_dispatched_release_skill_is_a_place_intent() -> None:
    """gripper_open / handover declare no gripper verify_hint, but their spec releases
    what is held (``effects['held_object'] = None``): dispatching one is a place, so
    an unrelated walk verify cannot stand in for the release."""
    from vector_os_nano.skills.gripper import GripperOpenSkill
    from vector_os_nano.skills.handover import HandoverSkill
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

    for cls in (GripperOpenSkill, HandoverSkill):
        assert SkillWrapperTool(cls(), None).effects["held_object"] is None
    release = _Skill("gripper_open_fake", lambda: None)
    release.effects = {"gripper_state": "open", "held_object": None}
    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("gripper_open_fake", {})),
        tool_turn(("walk", {"distance": 1.0, "speed": 0.5})),
        tool_turn(("verify", {"expr": "at_position(1.0, 0.0, 0.5)"})),
        tool_turn(("finish", {})),
    ])
    agent, _arm = _make_arm_agent(lambda arm: release)
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("松开它", session=_session())
    assert trace.intents == ("place",)
    assert _verdict(agent, eng, trace).verified is False


# test_the_contract_start_fails_closed_on_an_unevaluable_predicate moved to
# test_task_contract_binding.py (I4): its fetch contract now binds the manifest object.


def test_user_authored_prompt_files_reach_the_native_turn(tmp_path, monkeypatch) -> None:
    """R1V-3: $VECTOR_HOME personality.md and VECTOR.md (and the project's own VECTOR.md
    in cwd) are user-authored file interfaces (charter §5); every native turn sees them."""
    home, project = tmp_path / "home", tmp_path / "project"
    (home / ".vector").mkdir(parents=True)
    project.mkdir()
    (home / ".vector" / "personality.md").write_text("speak like a pirate\n")
    (home / ".vector" / "VECTOR.md").write_text("user notes: metric units\n")
    (project / "VECTOR.md").write_text("project notes: the arm is left-handed\n")
    monkeypatch.setenv("VECTOR_HOME", str(home / ".vector"))
    monkeypatch.chdir(project)
    text = _prompt_text(None, None)
    for wanted in ("speak like a pirate", "user notes: metric units", "project notes: the arm is left-handed"):
        assert wanted in text, wanted
    assert text.index("Never finish while") < text.index("speak like a pirate")  # below the kernel rules


def test_absent_empty_or_foreign_prompt_files_add_nothing(tmp_path, monkeypatch) -> None:
    """No file, an empty file, a directory named VECTOR.md, or another agent's file
    (AGENTS.md / CLAUDE.md: a repo's developer constitution) adds no block and never raises."""
    home, project = tmp_path / "home", tmp_path / "project"
    (home / ".vector").mkdir(parents=True)
    (project / "VECTOR.md").mkdir(parents=True)
    (home / ".vector" / "personality.md").write_text("  \n")
    (project / "AGENTS.md").write_text("agents constitution\n")
    (project / "CLAUDE.md").write_text("claude constitution\n")
    monkeypatch.setenv("VECTOR_HOME", str(home / ".vector"))
    monkeypatch.chdir(project)
    assert "constitution" not in _prompt_text(None, None)
    from vector_os_nano.vcli.native_loop import _native_system_prompt

    assert len(_native_system_prompt(SimpleNamespace(_world=None), frozenset({"holding_object"}), None)) == 1
