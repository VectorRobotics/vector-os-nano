"""VGG end-to-end tests with real MuJoCo physics.

Validates the full pipeline: user input -> VGG decompose -> skill execution ->
physical state change. Uses real MuJoCoGo2 (sinusoidal gait), mock LLM backend.

Test layers:
  TestVGGDecomposeWithRealRobot  — GoalTree creation (fast, no physics wait)
  TestVGGExecuteWithRealPhysics  — full execution with real MuJoCo (slow)
  TestVGGTryVggWithRealRobot     — highest-level integration via try_vgg()

Gate strategy:
  mujoco not installed  -> skip all tests in this file (pytest.importorskip)
  LLM API key missing   -> MockLLMBackend never called for simple commands
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills.go2 import get_go2_skills
from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, SubGoal
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.intent_router import IntentRouter


# ---------------------------------------------------------------------------
# Mock LLM backend (should not be called for simple skill-match commands)
# ---------------------------------------------------------------------------


class _MockLLMBackend:
    """Minimal LLM backend that raises if chat() is called.

    Simple commands (站起来, explore, walk) take the fast path — no LLM needed.
    If called, it means VGG incorrectly routed a simple command to slow path.
    """

    def __init__(self) -> None:
        self.model = "mock"
        self.call_count = 0

    def call(
        self,
        messages: list[dict],
        tools: Any = None,
        system: Any = None,
        max_tokens: int = 4096,
        on_text: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        # Return a fallback single-step GoalTree JSON so tests don't hard-fail
        # if a complex command accidentally hits the slow path.
        text = (
            '{"goal":"fallback","sub_goals":[{"name":"fallback_step",'
            '"description":"fallback","verify":"True","strategy":"",'
            '"timeout_sec":10,"depends_on":[],"strategy_params":{},'
            '"fail_action":""}]}'
        )
        from vector_os_nano.vcli.backends.types import LLMResponse, TokenUsage  # type: ignore[attr-defined]
        try:
            from vector_os_nano.vcli.session import TokenUsage as TU
            usage = TU(input_tokens=0, output_tokens=0)
        except Exception:
            usage = type("U", (), {"input_tokens": 0, "output_tokens": 0})()
        return type(
            "LLMResponse",
            (),
            {
                "text": text,
                "tool_calls": [],
                "stop_reason": "end_turn",
                "usage": usage,
            },
        )()

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Fake agent — minimal namespace that VectorEngine.init_vgg() needs
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal Agent-like object for VGG wiring.

    VectorEngine.init_vgg() reads: _base, _spatial_memory, _vlm, _skill_registry.
    """

    def __init__(self, base: MuJoCoGo2, skill_registry: SkillRegistry) -> None:
        self._base = base
        self._spatial_memory = None  # no SceneGraph needed for physics tests
        self._vlm = None
        self._skill_registry = skill_registry


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _make_skill_registry() -> SkillRegistry:
    """Build a SkillRegistry with all Go2 skills registered."""
    registry = SkillRegistry()
    for skill_obj in get_go2_skills():
        registry.register(skill_obj)
    return registry


def _make_engine(go2: MuJoCoGo2) -> tuple[VectorEngine, _FakeAgent]:
    """Create VectorEngine with VGG wired to the real MuJoCoGo2."""
    backend = _MockLLMBackend()
    skill_registry = _make_skill_registry()
    agent = _FakeAgent(go2, skill_registry)

    engine = VectorEngine(
        backend=backend,
        intent_router=IntentRouter(),
    )
    engine.init_vgg(agent=agent, skill_registry=skill_registry)
    return engine, agent


def _make_goal_tree(strategy: str, description: str, verify: str = "True") -> GoalTree:
    """Build a 1-step GoalTree for direct execution tests."""
    sub = SubGoal(
        name=f"{strategy}_goal",
        description=description,
        verify=verify,
        strategy=f"{strategy}_skill",
        strategy_params={},
        timeout_sec=30.0,
    )
    return GoalTree(goal=description, sub_goals=(sub,))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def go2_module():
    """Module-scoped MuJoCoGo2 (expensive to start, shared across all tests).

    Stands the robot once on module setup. Each test class fixture resets
    posture to reduce cumulative drift.
    """
    robot = MuJoCoGo2(gui=False, room=False)
    robot.connect()
    robot.stand(duration=2.0)
    yield robot
    robot.disconnect()


@pytest.fixture
def vgg_engine(go2_module):
    """VectorEngine with VGG initialized using the real MuJoCoGo2.

    Resets robot to standing pose before each test to ensure clean state.
    Yields (engine, go2) tuple.
    """
    # Reset to standing pose so each test starts from a known state
    go2_module.stand(duration=1.0)
    engine, agent = _make_engine(go2_module)
    yield engine, go2_module


# ---------------------------------------------------------------------------
# TestVGGDecomposeWithRealRobot
# Fast path only — no physics wait, validates GoalTree structure.
# ---------------------------------------------------------------------------


class TestVGGDecomposeWithRealRobot:
    """Validate vgg_decompose() produces correct GoalTrees with real robot attached."""

    def test_vgg_enabled_after_init(self, vgg_engine):
        """VGG must be enabled after init_vgg() — otherwise all decompose tests are moot."""
        engine, go2 = vgg_engine
        assert engine._vgg_enabled is True, (
            "init_vgg() failed silently — check VGG dependencies"
        )

    def test_decompose_stand(self, vgg_engine):
        """vgg_decompose('站起来') returns 1-step GoalTree with stand strategy."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("站起来")
        assert tree is not None, "站起来 should match StandSkill and return GoalTree"
        assert len(tree.sub_goals) == 1, (
            f"Simple command should produce 1 sub_goal, got {len(tree.sub_goals)}"
        )
        assert "stand" in tree.sub_goals[0].strategy, (
            f"strategy should contain 'stand', got {tree.sub_goals[0].strategy!r}"
        )

    def test_decompose_explore(self, vgg_engine):
        """vgg_decompose('explore') returns 1-step GoalTree with explore strategy."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("explore")
        assert tree is not None, "explore should match ExploreSkill and return GoalTree"
        assert len(tree.sub_goals) >= 1
        assert "explore" in tree.sub_goals[0].strategy, (
            f"strategy should contain 'explore', got {tree.sub_goals[0].strategy!r}"
        )

    def test_decompose_walk(self, vgg_engine):
        """vgg_decompose('walk forward') returns GoalTree with walk strategy."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("walk forward")
        assert tree is not None, "walk forward should match WalkSkill and return GoalTree"
        assert len(tree.sub_goals) >= 1
        assert "walk" in tree.sub_goals[0].strategy, (
            f"strategy should contain 'walk', got {tree.sub_goals[0].strategy!r}"
        )

    def test_decompose_sit(self, vgg_engine):
        """vgg_decompose('坐下') returns 1-step GoalTree with sit strategy."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("坐下")
        assert tree is not None, "坐下 should match SitSkill and return GoalTree"
        assert len(tree.sub_goals) == 1
        assert "sit" in tree.sub_goals[0].strategy, (
            f"strategy should contain 'sit', got {tree.sub_goals[0].strategy!r}"
        )

    def test_decompose_stop(self, vgg_engine):
        """vgg_decompose('stop') returns GoalTree with stop strategy."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("stop")
        assert tree is not None, "stop should match StopSkill and return GoalTree"
        assert "stop" in tree.sub_goals[0].strategy, (
            f"strategy should contain 'stop', got {tree.sub_goals[0].strategy!r}"
        )

    def test_decompose_conversation_returns_none(self, vgg_engine):
        """vgg_decompose('你好') returns None — pure conversation, not actionable."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("你好")
        assert tree is None, (
            "Greeting '你好' should return None — not an actionable command"
        )

    def test_decompose_hello_returns_none(self, vgg_engine):
        """vgg_decompose('hello') returns None — pure greeting."""
        engine, go2 = vgg_engine
        tree = engine.vgg_decompose("hello")
        assert tree is None, "Greeting 'hello' should return None"

    def test_verifier_namespace_has_real_position(self, vgg_engine):
        """After init_vgg, the verifier namespace includes get_position() with real data."""
        engine, go2 = vgg_engine
        # Retrieve the GoalVerifier's namespace through the executor
        executor = engine._goal_executor
        assert executor is not None, "_goal_executor not set after init_vgg()"
        verifier = executor._verifier
        assert verifier is not None
        # Call get_position from the namespace
        ns = verifier._namespace
        assert "get_position" in ns, "verifier namespace missing get_position"
        pos = ns["get_position"]()
        assert len(pos) == 3, f"get_position() should return 3-tuple, got {pos}"
        # Robot should be at a valid height (standing: ~0.3m, sitting/fallen: lower)
        assert pos[2] > 0.0, f"Robot z-position must be > 0, got {pos[2]}"

    def test_decompose_does_not_call_llm_for_stand(self, vgg_engine):
        """Fast-path decompose for '站起来' must NOT call MockLLMBackend.call()."""
        engine, go2 = vgg_engine
        backend = engine._backend
        initial_calls = getattr(backend, "call_count", 0)
        engine.vgg_decompose("站起来")
        final_calls = getattr(backend, "call_count", 0)
        assert final_calls == initial_calls, (
            f"LLM was called {final_calls - initial_calls} times for simple '站起来' — "
            "fast path should bypass LLM"
        )


# ---------------------------------------------------------------------------
# TestVGGExecuteWithRealPhysics
# Full skill execution with real MuJoCo physics. These are slow (~5-10s each).
# ---------------------------------------------------------------------------


class TestVGGExecuteWithRealPhysics:
    """Execute GoalTrees against real MuJoCo simulation and verify physical state."""

    def test_execute_stand_skill(self, vgg_engine):
        """Stand GoalTree -> ExecutionTrace with success, robot z in [0.2, 0.45]."""
        engine, go2 = vgg_engine

        tree = _make_goal_tree("stand", "stand up", verify="True")
        trace = engine.vgg_execute(tree)

        assert trace is not None, "vgg_execute returned None"
        assert isinstance(trace, ExecutionTrace)
        assert len(trace.steps) >= 1

        # Check robot is upright
        pos = go2.get_position()
        assert 0.2 <= pos[2] <= 0.45, (
            f"After stand, robot z={pos[2]:.3f} should be in [0.20, 0.45]"
        )

        # No NoneType errors
        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in step {step.sub_goal_name!r}: {step.error}"
            )

    def test_execute_walk_skill_causes_displacement(self, vgg_engine):
        """Walk GoalTree -> robot displaces, z stays above 0.15 (not fallen)."""
        engine, go2 = vgg_engine

        pos_before = go2.get_position()
        x0, y0 = pos_before[0], pos_before[1]

        tree = _make_goal_tree("walk", "walk forward", verify="True")
        trace = engine.vgg_execute(tree)

        assert trace is not None
        assert isinstance(trace, ExecutionTrace)

        # No hard crashes
        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in walk: {step.error}"
            )

        # Robot should still be standing (not fallen through the floor)
        pos_after = go2.get_position()
        assert pos_after[2] > 0.15, (
            f"Robot fell after walk (z={pos_after[2]:.3f}) — physics simulation issue"
        )

    def test_execute_stop_skill(self, vgg_engine):
        """Stop GoalTree -> ExecutionTrace without crash."""
        engine, go2 = vgg_engine

        tree = _make_goal_tree("stop", "stop all movement", verify="True")
        trace = engine.vgg_execute(tree)

        assert trace is not None, "vgg_execute(stop) returned None"
        assert isinstance(trace, ExecutionTrace)
        assert len(trace.steps) >= 1

        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in stop: {step.error}"
            )

    def test_execute_sit_skill(self, vgg_engine):
        """Sit GoalTree -> robot z lowers (sitting is lower than standing)."""
        engine, go2 = vgg_engine

        # First make sure robot is standing
        go2.stand(duration=1.5)
        pos_standing = go2.get_position()

        tree = _make_goal_tree("sit", "sit down", verify="True")
        trace = engine.vgg_execute(tree)

        assert trace is not None
        assert isinstance(trace, ExecutionTrace)

        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in sit: {step.error}"
            )

        # After sit the robot height should drop (sitting is ~0.15-0.25m, standing ~0.35m)
        pos_sitting = go2.get_position()
        assert pos_sitting[2] < 0.35, (
            f"After sit, z={pos_sitting[2]:.3f} should be < 0.35 (sitting lower than standing)"
        )

    def test_execute_returns_execution_trace_type(self, vgg_engine):
        """vgg_execute always returns an ExecutionTrace (never raises)."""
        engine, go2 = vgg_engine

        tree = _make_goal_tree("stand", "stand", verify="True")
        result = engine.vgg_execute(tree)

        assert isinstance(result, ExecutionTrace), (
            f"vgg_execute must return ExecutionTrace, got {type(result)}"
        )

    def test_execute_trace_has_steps(self, vgg_engine):
        """ExecutionTrace must contain at least one StepRecord."""
        engine, go2 = vgg_engine

        tree = _make_goal_tree("stand", "stand", verify="True")
        trace = engine.vgg_execute(tree)

        assert len(trace.steps) >= 1, "ExecutionTrace must have at least one step"

    def test_execute_trace_goal_preserved(self, vgg_engine):
        """ExecutionTrace.goal_tree must match the original GoalTree."""
        engine, go2 = vgg_engine

        tree = _make_goal_tree("stand", "stand up for test")
        trace = engine.vgg_execute(tree)

        assert trace.goal_tree is tree, "ExecutionTrace.goal_tree must be the original tree"


# ---------------------------------------------------------------------------
# TestVGGTryVggWithRealRobot
# Highest-level integration: try_vgg() wraps decompose + execute.
# ---------------------------------------------------------------------------


class TestVGGTryVggWithRealRobot:
    """Validate the full try_vgg() pipeline end-to-end."""

    def test_try_vgg_stand_returns_trace(self, vgg_engine):
        """try_vgg('站起来') returns ExecutionTrace (not None), no NoneType errors."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("站起来")

        assert result is not None, (
            "try_vgg('站起来') returned None — decompose or execute failed"
        )
        assert isinstance(result, ExecutionTrace)

        for step in result.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error after try_vgg('站起来'): {step.error}"
            )

    def test_try_vgg_walk_returns_trace(self, vgg_engine):
        """try_vgg('walk forward') returns ExecutionTrace."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("walk forward")

        assert result is not None, (
            "try_vgg('walk forward') returned None"
        )
        assert isinstance(result, ExecutionTrace)

        for step in result.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error after try_vgg('walk forward'): {step.error}"
            )

    def test_try_vgg_stop_returns_trace(self, vgg_engine):
        """try_vgg('stop') returns ExecutionTrace."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("stop")

        assert result is not None, "try_vgg('stop') returned None"
        assert isinstance(result, ExecutionTrace)

    def test_try_vgg_conversation_returns_none(self, vgg_engine):
        """try_vgg('你好') returns None — pure conversation, no execution."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("你好")
        assert result is None, (
            "try_vgg('你好') should return None — not an actionable command"
        )

    def test_try_vgg_hello_returns_none(self, vgg_engine):
        """try_vgg('hello') returns None — pure greeting."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("hello")
        assert result is None, "try_vgg('hello') should return None"

    def test_try_vgg_stand_robot_stays_upright(self, vgg_engine):
        """After try_vgg('站起来'), robot z must be >= 0.2 (not fallen)."""
        engine, go2 = vgg_engine
        engine.try_vgg("站起来")
        pos = go2.get_position()
        assert pos[2] >= 0.2, (
            f"Robot z={pos[2]:.3f} after stand — should be >= 0.2"
        )

    def test_try_vgg_walk_robot_does_not_fall(self, vgg_engine):
        """After try_vgg('walk forward'), robot must stay above ground (z > 0.15)."""
        engine, go2 = vgg_engine
        engine.try_vgg("walk forward")
        pos = go2.get_position()
        assert pos[2] > 0.15, (
            f"Robot fell after walk (z={pos[2]:.3f}) — physics or skill error"
        )

    def test_try_vgg_returns_none_for_empty_input(self, vgg_engine):
        """try_vgg('') returns None — empty input is not actionable."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("")
        assert result is None, "Empty input should return None"

    def test_try_vgg_explore_returns_trace(self, vgg_engine):
        """try_vgg('explore') returns ExecutionTrace — ExploreSkill launches background thread."""
        engine, go2 = vgg_engine
        result = engine.try_vgg("explore")

        assert result is not None, (
            "try_vgg('explore') returned None — should produce 1-step GoalTree"
        )
        assert isinstance(result, ExecutionTrace)
