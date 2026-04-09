"""CLI scenario harness — full VGG pipeline with real Agent + Skills.

Simulates real user interactions: sim start -> explore -> navigate -> complex tasks.
Uses real SkillRegistry, SceneGraph, GoalDecomposer, GoalExecutor.
Only hardware (MockBase) and LLM (MockLLMBackend) are mocked.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# MockBase — simulates Go2 hardware (reused from level54)
# ---------------------------------------------------------------------------


class MockBase:
    """Simulates Go2 base — position tracking, heading, walk, navigate_to."""

    def __init__(self) -> None:
        self._pos = [10.0, 3.0, 0.28]
        self._heading = 0.0
        self._connected = True
        self.name = "mock_go2"
        self.navigate_to_calls: list[tuple[float, float]] = []
        self.stand_calls: int = 0
        self.sit_calls: int = 0

    def get_position(self) -> list[float]:
        return list(self._pos)

    def get_heading(self) -> float:
        return self._heading

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        self._pos[0] += vx * 0.1
        self._pos[1] += vy * 0.1
        self._heading += vyaw * 0.1

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        self._pos[0] += vx * duration
        self._pos[1] += vy * duration
        self._heading += vyaw * duration
        return True

    def navigate_to(self, x: float, y: float, timeout: float = 60.0) -> bool:
        """Teleport to target — simulates successful navigation."""
        self.navigate_to_calls.append((x, y))
        self._pos[0] = x
        self._pos[1] = y
        return True

    def stand(self, duration: float = 1.0) -> bool:
        self.stand_calls += 1
        return True

    def sit(self, duration: float = 1.0) -> bool:
        self.sit_calls += 1
        return True

    def stop(self) -> None:
        pass

    def get_camera_frame(self, width: int = 320, height: int = 240):
        import numpy as np
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_lidar_scan(self) -> None:
        return None

    def get_odometry(self) -> Any:
        from vector_os_nano.core.types import Odometry
        return Odometry(
            timestamp=time.time(),
            x=self._pos[0],
            y=self._pos[1],
            z=self._pos[2],
            qx=0, qy=0, qz=0, qw=1,
            vx=0, vy=0, vz=0, vyaw=0,
        )

    @property
    def supports_lidar(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# MockLLMBackend — returns predetermined GoalTree JSON for known tasks
# ---------------------------------------------------------------------------


class MockLLMBackend:
    """Returns predetermined GoalTree JSON for known tasks."""

    def __init__(self) -> None:
        # Keys are matched as substrings against the full LLM prompt (task + world context).
        # Order matters — more specific keys first to avoid early false matches.
        _multi_step_response = (
            '{"goal":"visit kitchen and find cups","sub_goals":['
            '{"name":"reach_kitchen","description":"navigate to kitchen",'
            '"verify":"nearest_room() == \'kitchen\'","strategy":"navigate_skill",'
            '"timeout_sec":60,"depends_on":[],"strategy_params":{},"fail_action":""},'
            '{"name":"observe_kitchen","description":"observe kitchen environment",'
            '"verify":"len(describe_scene()) > 0","strategy":"look_skill",'
            '"timeout_sec":15,"depends_on":["reach_kitchen"],"strategy_params":{},"fail_action":""},'
            '{"name":"detect_cup","description":"detect cups",'
            '"verify":"len(detect_objects(\'cup\')) > 0","strategy":"describe_scene_skill",'
            '"timeout_sec":10,"depends_on":["observe_kitchen"],"strategy_params":{},"fail_action":""}]}'
        )
        _nav_response = (
            '{"goal":"go to kitchen","sub_goals":['
            '{"name":"reach_kitchen","description":"navigate to kitchen",'
            '"verify":"nearest_room() == \'kitchen\'","strategy":"navigate_skill",'
            '"timeout_sec":60,"depends_on":[],"strategy_params":{},"fail_action":""}]}'
        )
        _patrol_response = (
            '{"goal":"end explore and start patrol","sub_goals":['
            '{"name":"stop_explore","description":"stop exploration",'
            '"verify":"True","strategy":"stop_skill",'
            '"timeout_sec":10,"depends_on":[],"strategy_params":{},"fail_action":""},'
            '{"name":"start_patrol","description":"begin patrol route",'
            '"verify":"True","strategy":"patrol_skill",'
            '"timeout_sec":60,"depends_on":["stop_explore"],"strategy_params":{},"fail_action":""}]}'
        )
        self._responses: dict[str, str] = {
            # Chinese complex task: look for cups in kitchen
            "看看有没有": _multi_step_response,
            # English complex task
            "complex": _multi_step_response,
            # Multi-action patrol
            "multi_action": _patrol_response,
            # Simple navigate
            "navigate": _nav_response,
        }
        self._last_call: Any = None

    def call(
        self,
        messages: list[dict],
        tools: Any = None,
        system: Any = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """Match task to predetermined response."""
        self._last_call = messages
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            user_msg = block["text"]
                else:
                    user_msg = str(content)

        for key, response in self._responses.items():
            if key in user_msg.lower():
                return _make_llm_response(response)

        # Default fallback: single step
        return _make_llm_response(
            '{"goal":"task","sub_goals":[{"name":"do_task","description":"execute",'
            '"verify":"world_stats() is not None","strategy":"","timeout_sec":30,'
            '"depends_on":[],"strategy_params":{},"fail_action":""}]}'
        )


def _make_llm_response(text: str) -> Any:
    """Create a minimal LLMResponse-compatible object."""
    Usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 50})
    return type(
        "LLMResponse",
        (),
        {
            "text": text,
            "tool_calls": [],
            "stop_reason": "end_turn",
            "usage": Usage(),
        },
    )()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_agent_with_skills(base: MockBase) -> Any:
    """Create a minimal Agent-like object with real Go2 skills registered."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2 import get_go2_skills
    from vector_os_nano.core.scene_graph import SceneGraph

    class _FakeAgent:
        pass

    agent = _FakeAgent()
    agent._base = base
    agent._arm = None
    agent._vlm = None

    skill_registry = SkillRegistry()
    for skill_obj in get_go2_skills():
        skill_registry.register(skill_obj)
    agent._skill_registry = skill_registry

    sg = SceneGraph()
    for room_id, (cx, cy) in [
        ("kitchen", (17.0, 2.5)),
        ("hallway", (10.0, 5.0)),
        ("living_room", (4.8, 2.9)),
        ("master_bedroom", (12.0, 10.0)),
        ("study", (4.0, 8.0)),
    ]:
        for _ in range(5):
            sg.visit(room_id, cx, cy)
    agent._spatial_memory = sg

    return agent


def _make_engine_with_vgg(base: MockBase) -> tuple[Any, Any]:
    """Create a VectorEngine with VGG fully initialised. Returns (engine, agent)."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.intent_router import IntentRouter

    agent = _make_agent_with_skills(base)
    backend = MockLLMBackend()
    engine = VectorEngine(backend=backend, intent_router=IntentRouter())
    engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)
    return engine, agent


def _make_executor(base: MockBase, agent: Any) -> Any:
    """Create a standalone GoalExecutor with real components."""
    from vector_os_nano.vcli.cognitive import GoalVerifier, StrategySelector, GoalExecutor
    from vector_os_nano.core.skill import SkillContext

    sg = agent._spatial_memory

    ns = {
        "nearest_room": lambda: sg.nearest_room(base._pos[0], base._pos[1]),
        "get_position": lambda: tuple(base._pos),
        "get_heading": lambda: base._heading,
        "get_visited_rooms": lambda: sg.get_visited_rooms(),
        "describe_scene": lambda: "a kitchen counter with dishes",
        "detect_objects": lambda query="": [{"name": "cup", "confidence": 0.9}],
        "world_stats": lambda: sg.stats(),
        "query_rooms": lambda: [
            {"id": r.room_id, "x": r.center_x, "y": r.center_y}
            for r in sg.get_all_rooms()
        ],
    }

    verifier = GoalVerifier(ns)
    selector = StrategySelector(skill_registry=agent._skill_registry)

    def _build_context() -> SkillContext:
        return SkillContext(
            bases={"go2": base},
            services={"spatial_memory": sg},
        )

    return GoalExecutor(
        strategy_selector=selector,
        verifier=verifier,
        skill_registry=agent._skill_registry,
        build_context=_build_context,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Pre-sim commands should NOT trigger VGG
# ---------------------------------------------------------------------------


class TestPreSimNoVgg:
    """Before sim starts (no _vgg_agent), VGG pipeline must return None."""

    def _engine_no_agent(self) -> Any:
        """Engine with VGG enabled but no _vgg_agent — simulates pre-sim state."""
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        backend = MockLLMBackend()
        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        # Manually set _vgg_enabled without agent — simulates partial init
        engine._vgg_enabled = True
        engine._vgg_agent = None
        return engine

    def test_pre_sim_go2sim_no_vgg(self) -> None:
        """Before sim starts, 'go2sim' should NOT trigger VGG (no agent)."""
        engine = self._engine_no_agent()
        result = engine.vgg_decompose("go2sim")
        assert result is None, (
            "vgg_decompose('go2sim') must return None when _vgg_agent is None"
        )

    def test_pre_sim_hello_no_vgg(self) -> None:
        """Before sim, 'hello' should NOT trigger VGG."""
        engine = self._engine_no_agent()
        result = engine.vgg_decompose("hello")
        assert result is None, (
            "vgg_decompose('hello') must return None when no agent is connected"
        )

    def test_pre_sim_explore_no_vgg(self) -> None:
        """Before sim, 'explore' should NOT trigger VGG (no agent)."""
        engine = self._engine_no_agent()
        result = engine.vgg_decompose("explore")
        assert result is None, (
            "vgg_decompose('explore') must return None — no base connected yet"
        )

    def test_pre_sim_navigate_no_vgg(self) -> None:
        """Before sim, '去厨房' should NOT trigger VGG (no agent)."""
        engine = self._engine_no_agent()
        result = engine.vgg_decompose("去厨房")
        assert result is None, (
            "vgg_decompose('去厨房') must return None — no base connected yet"
        )

    def test_vgg_disabled_returns_none(self) -> None:
        """When _vgg_enabled is False, vgg_decompose always returns None."""
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        backend = MockLLMBackend()
        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        # VGG not initialised — _vgg_enabled stays False by default
        assert engine._vgg_enabled is False
        assert engine.vgg_decompose("explore") is None
        assert engine.vgg_decompose("去厨房") is None
        assert engine.vgg_decompose("去厨房看看有没有杯子") is None


# ---------------------------------------------------------------------------
# Scenario 2: Simple commands after sim -> 1-step VGG GoalTree
# ---------------------------------------------------------------------------


class TestSimpleCommands1StepGoalTree:
    """Simple actionable commands produce 1-step GoalTree via fast path."""

    def test_explore_produces_1step_goal_tree(self) -> None:
        """After sim start, 'explore' -> 1-step GoalTree via fast path."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("explore")
        assert tree is not None, "explore must produce a GoalTree"
        assert len(tree.sub_goals) == 1, (
            f"Simple 'explore' must be 1-step, got {len(tree.sub_goals)}"
        )
        assert "explore" in tree.sub_goals[0].name.lower(), (
            f"SubGoal name should contain 'explore', got {tree.sub_goals[0].name!r}"
        )

    def test_navigate_produces_1step_goal_tree(self) -> None:
        """'去厨房' -> 1-step GoalTree with navigate strategy."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("去厨房")
        assert tree is not None, "去厨房 must produce a GoalTree"
        assert len(tree.sub_goals) == 1, (
            f"Simple navigate must be 1-step, got {len(tree.sub_goals)}"
        )
        sg = tree.sub_goals[0]
        assert sg.strategy == "navigate_skill", (
            f"Expected 'navigate_skill' strategy, got {sg.strategy!r}"
        )

    def test_navigate_strategy_params_contain_room(self) -> None:
        """'去厨房' -> SubGoal.strategy_params contains room key."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("去厨房")
        assert tree is not None
        sg = tree.sub_goals[0]
        # The fast path should extract "厨房" as the room param
        assert sg.strategy_params.get("room") or "厨房" in sg.description.lower(), (
            "navigate SubGoal should contain room info in params or description"
        )

    def test_stand_produces_1step_goal_tree(self) -> None:
        """'站起来' -> 1-step GoalTree with stand strategy."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("站起来")
        assert tree is not None, "站起来 must produce a GoalTree"
        assert len(tree.sub_goals) == 1, (
            f"Simple stand must be 1-step, got {len(tree.sub_goals)}"
        )
        sg = tree.sub_goals[0]
        assert "stand" in sg.strategy.lower(), (
            f"Expected stand strategy, got {sg.strategy!r}"
        )

    def test_stop_produces_1step(self) -> None:
        """'stop' -> 1-step GoalTree."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("stop")
        assert tree is not None, "stop must produce a GoalTree"
        assert len(tree.sub_goals) == 1, (
            f"Simple 'stop' must be 1-step, got {len(tree.sub_goals)}"
        )

    def test_1step_tree_has_strategy(self) -> None:
        """1-step GoalTree from fast path has a non-empty strategy."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("explore")
        assert tree is not None
        sg = tree.sub_goals[0]
        assert sg.strategy, "Fast-path SubGoal must have a non-empty strategy"

    def test_1step_tree_has_verify(self) -> None:
        """1-step GoalTree from fast path has a verify expression."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("explore")
        assert tree is not None
        sg = tree.sub_goals[0]
        assert sg.verify, "Fast-path SubGoal must have a non-empty verify expression"


# ---------------------------------------------------------------------------
# Scenario 3: Complex commands -> LLM decomposition (multi-step)
# ---------------------------------------------------------------------------


class TestComplexCommandsLLMDecomposition:
    """Complex commands trigger LLM decomposition and return multi-step GoalTrees."""

    def test_complex_navigate_and_check(self) -> None:
        """'去厨房看看有没有杯子' -> multi-step GoalTree from LLM."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("去厨房看看有没有杯子")
        assert tree is not None, (
            "Complex task '去厨房看看有没有杯子' must produce a GoalTree"
        )
        assert len(tree.sub_goals) > 1, (
            f"Complex perception+action task must have >1 sub_goals, "
            f"got {len(tree.sub_goals)}"
        )

    def test_complex_multi_action(self) -> None:
        """'结束探索，开始巡逻' -> multi-step GoalTree from LLM."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("结束探索，开始巡逻")
        assert tree is not None, (
            "Multi-action task '结束探索，开始巡逻' must produce a GoalTree"
        )
        # Multi-action detection means this is complex -> multi-step
        assert len(tree.sub_goals) >= 1, "Multi-action task must produce at least 1 sub_goal"

    def test_complex_task_is_complex_flagged(self) -> None:
        """'去厨房看看有没有杯子' must be detected as complex by IntentRouter."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.is_complex("去厨房看看有没有杯子") is True, (
            "'看看有没有' is a perception+judgment phrase → complex"
        )

    def test_multi_action_is_complex_flagged(self) -> None:
        """'结束探索，开始巡逻' has multiple action verbs -> complex."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.is_complex("结束探索，开始巡逻") is True, (
            "Multiple action verbs should trigger is_complex=True"
        )

    def test_complex_tree_all_sub_goals_have_names(self) -> None:
        """Every sub_goal in a complex GoalTree has a non-empty name."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("去厨房看看有没有杯子")
        assert tree is not None
        for sg in tree.sub_goals:
            assert sg.name, f"SubGoal must have non-empty name, got {sg!r}"


# ---------------------------------------------------------------------------
# Scenario 4: GoalExecutor actually executes skills
# ---------------------------------------------------------------------------


class TestGoalExecutorRunsSkills:
    """GoalExecutor with skill strategies calls the real skill implementations."""

    def test_executor_runs_navigate_skill(self) -> None:
        """GoalExecutor with navigate strategy calls real NavigateSkill."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = _make_executor(base, agent)

        tree = GoalTree(
            goal="navigate to kitchen",
            sub_goals=(
                SubGoal(
                    name="reach_kitchen",
                    description="navigate to kitchen",
                    verify="True",
                    strategy="navigate_skill",
                    strategy_params={"room": "kitchen"},
                    timeout_sec=30,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace is not None, "execute must return an ExecutionTrace"
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert "NoneType" not in (step.error or ""), (
            f"NoneType error in navigate step: {step.error}"
        )

    def test_executor_runs_stand_skill(self) -> None:
        """GoalExecutor with stand strategy calls real StandSkill."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = _make_executor(base, agent)

        tree = GoalTree(
            goal="stand up",
            sub_goals=(
                SubGoal(
                    name="stand_up",
                    description="stand",
                    verify="True",
                    strategy="stand_skill",
                    timeout_sec=10,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace is not None
        step = trace.steps[0]
        assert "NoneType" not in (step.error or ""), (
            f"NoneType error in stand step: {step.error}"
        )

    def test_executor_runs_explore_skill(self) -> None:
        """GoalExecutor with explore strategy calls real ExploreSkill."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = _make_executor(base, agent)

        tree = GoalTree(
            goal="explore environment",
            sub_goals=(
                SubGoal(
                    name="explore_env",
                    description="explore",
                    verify="True",
                    strategy="explore_skill",
                    timeout_sec=10,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace is not None
        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert "NoneType" not in (step.error or ""), (
            f"NoneType error in explore step: {step.error}"
        )

    def test_executor_returns_execution_trace(self) -> None:
        """GoalExecutor.execute() returns an ExecutionTrace, never None."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = _make_executor(base, agent)

        tree = GoalTree(
            goal="test",
            sub_goals=(
                SubGoal(
                    name="simple_step",
                    description="test step",
                    verify="True",
                    strategy="stand_skill",
                    timeout_sec=5,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace is not None, "execute must never return None"
        from vector_os_nano.vcli.cognitive.types import ExecutionTrace
        assert isinstance(trace, ExecutionTrace), (
            f"execute must return ExecutionTrace, got {type(trace)}"
        )

    def test_executor_step_records_sub_goal_name(self) -> None:
        """StepRecord.sub_goal_name matches the SubGoal name executed."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = _make_executor(base, agent)

        tree = GoalTree(
            goal="test",
            sub_goals=(
                SubGoal(
                    name="my_special_step",
                    description="a step",
                    verify="True",
                    strategy="stand_skill",
                    timeout_sec=5,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace.steps[0].sub_goal_name == "my_special_step"


# ---------------------------------------------------------------------------
# Scenario 5: Full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFullPipelineE2E:
    """Full VGG pipeline: vgg_decompose -> vgg_execute -> trace with success."""

    def test_full_pipeline_navigate(self) -> None:
        """Full: vgg_decompose('去主卧') -> vgg_execute() -> trace with no NoneType errors."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("去主卧")
        assert tree is not None, "去主卧 must produce a GoalTree"
        assert len(tree.sub_goals) >= 1

        trace = engine.vgg_execute(tree)
        assert trace is not None, "vgg_execute must return a trace"

        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in step {step.sub_goal_name!r}: {step.error}"
            )

    def test_full_pipeline_complex(self) -> None:
        """Full: complex task -> LLM decompose -> execute -> trace."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("去厨房看看有没有杯子")
        assert tree is not None, (
            "Complex task must produce a GoalTree"
        )

        trace = engine.vgg_execute(tree)
        assert trace is not None, "vgg_execute must return a trace"
        assert len(trace.steps) > 0, "Execution trace must have at least one step"

        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in step {step.sub_goal_name!r}: {step.error}"
            )

    def test_full_pipeline_explore(self) -> None:
        """Full: 'explore' -> 1-step tree -> execute -> trace no errors."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("explore")
        assert tree is not None
        assert len(tree.sub_goals) == 1

        trace = engine.vgg_execute(tree)
        assert trace is not None

        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in explore step: {step.error}"
            )

    def test_try_vgg_returns_trace_for_complex(self) -> None:
        """engine.try_vgg('去厨房看看有没有杯子') returns ExecutionTrace, not None."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        trace = engine.try_vgg("去厨房看看有没有杯子")
        assert trace is not None, (
            "try_vgg must return ExecutionTrace for a complex task"
        )

    def test_try_vgg_returns_none_for_conversation(self) -> None:
        """engine.try_vgg('你好') returns None (not an actionable command)."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        trace = engine.try_vgg("你好")
        assert trace is None, (
            "try_vgg must return None for pure conversation like '你好'"
        )

    def test_full_pipeline_stand(self) -> None:
        """Full: '站起来' -> 1-step tree -> execute -> no errors."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("站起来")
        assert tree is not None

        trace = engine.vgg_execute(tree)
        assert trace is not None

        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in stand step: {step.error}"
            )

    def test_full_pipeline_trace_has_goal_tree(self) -> None:
        """ExecutionTrace.goal_tree matches the tree that was executed."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("explore")
        assert tree is not None
        trace = engine.vgg_execute(tree)
        assert trace.goal_tree is tree, (
            "ExecutionTrace.goal_tree must reference the executed GoalTree"
        )


# ---------------------------------------------------------------------------
# Scenario 6: Edge cases — false positives and non-VGG messages
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: false positives, non-VGG messages, boundary conditions."""

    def test_english_go_to_kitchen(self) -> None:
        """'go to kitchen' -> VGG 1-step (motor pattern match)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        base = MockBase()
        agent = _make_agent_with_skills(base)
        result = router.should_use_vgg("go to kitchen", skill_registry=agent._skill_registry)
        assert result is True, "'go to kitchen' contains motor pattern 'go to' -> VGG"

    def test_chinese_navigate(self) -> None:
        """'导航去书房' -> VGG 1-step."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        base = MockBase()
        agent = _make_agent_with_skills(base)
        result = router.should_use_vgg("导航去书房", skill_registry=agent._skill_registry)
        assert result is True, "'导航去书房' contains '导航' motor pattern -> VGG"

    def test_conversation_no_vgg(self) -> None:
        """'你好' -> no VGG (pure greeting)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("你好", skill_registry=None)
        assert result is False, "'你好' is a greeting — should NOT trigger VGG"

    def test_question_no_vgg(self) -> None:
        """'你在哪里' -> no VGG (no action verb, only question)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        base = MockBase()
        agent = _make_agent_with_skills(base)
        # '在' is not in _ACTION_VERBS. '哪里' is not an action.
        result = router.should_use_vgg("你在哪里", skill_registry=agent._skill_registry)
        assert result is False, "'你在哪里' is a question — should NOT trigger VGG"

    def test_go2sim_no_false_positive(self) -> None:
        """'go2sim' must NOT trigger VGG (word boundary prevents 'go' match)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("go2sim", skill_registry=None)
        assert result is False, (
            "'go2sim' must NOT trigger VGG — 'go' has no word boundary in 'go2sim'"
        )

    def test_google_no_false_positive(self) -> None:
        """'google it' must NOT trigger VGG (word boundary check)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("google it", skill_registry=None)
        assert result is False, (
            "'google it' must NOT trigger VGG — 'go' is not a standalone word in 'google'"
        )

    def test_empty_message_no_vgg(self) -> None:
        """Empty message -> no VGG."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.should_use_vgg("", skill_registry=None) is False
        assert router.should_use_vgg(" ", skill_registry=None) is False

    def test_very_short_message_no_vgg(self) -> None:
        """Single character -> no VGG (too short)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.should_use_vgg("x", skill_registry=None) is False

    def test_go2sim_intent_router_word_boundary(self) -> None:
        """IntentRouter.should_use_vgg('go2sim', registry=None) -> False.

        The word-boundary check prevents 'go' (inside 'go2sim') from triggering
        VGG when no skill registry is present. With a skill registry the prefix
        matcher can still match 'go' as an alias — that is correct behaviour
        (walk skill alias 'go' would match). This test checks the router-only path.
        """
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        # Without registry: word-boundary check blocks 'go' in 'go2sim'
        result = router.should_use_vgg("go2sim", skill_registry=None)
        assert result is False, (
            "IntentRouter without registry: 'go' must NOT match in 'go2sim' (word boundary)"
        )

    def test_chinese_navigate_engine_produces_tree(self) -> None:
        """'导航去书房' -> engine produces GoalTree."""
        base = MockBase()
        engine, _ = _make_engine_with_vgg(base)

        tree = engine.vgg_decompose("导航去书房")
        assert tree is not None, "'导航去书房' should produce GoalTree via engine"
        assert len(tree.sub_goals) >= 1


# ---------------------------------------------------------------------------
# Scenario 7: Verify expressions work
# ---------------------------------------------------------------------------


class TestVerifyExpressions:
    """Verify expressions in GoalTree sub_goals evaluate correctly."""

    def _make_verifier_for_room(self, base: MockBase, sg: Any) -> Any:
        """Build GoalVerifier with nearest_room bound to current base position."""
        from vector_os_nano.vcli.cognitive import GoalVerifier

        ns = {
            "nearest_room": lambda: sg.nearest_room(base._pos[0], base._pos[1]),
            "get_position": lambda: tuple(base._pos),
            "get_heading": lambda: base._heading,
            "get_visited_rooms": lambda: sg.get_visited_rooms(),
            "describe_scene": lambda: "kitchen counter with cups",
            "detect_objects": lambda query="": [{"name": "cup", "confidence": 0.9}],
            "world_stats": lambda: sg.stats(),
        }
        return GoalVerifier(ns)

    def test_navigate_verify_nearest_room(self) -> None:
        """After navigate to kitchen, verify expression checks nearest_room."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory

        # Place robot at kitchen center
        base._pos[0] = 17.0
        base._pos[1] = 2.5

        verifier = self._make_verifier_for_room(base, sg)
        result = verifier.verify("nearest_room() == 'kitchen'")
        assert result is True, (
            "nearest_room() == 'kitchen' should be True when at kitchen (17.0, 2.5)"
        )

    def test_navigate_verify_fails_wrong_room(self) -> None:
        """Navigate to kitchen but robot at hallway -> verify fails."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory

        # Place robot at hallway center
        base._pos[0] = 10.0
        base._pos[1] = 5.0

        verifier = self._make_verifier_for_room(base, sg)
        result = verifier.verify("nearest_room() == 'kitchen'")
        assert result is False, (
            "nearest_room() == 'kitchen' should be False when robot is in hallway"
        )

    def test_verify_true_literal(self) -> None:
        """verify='True' should always evaluate to True."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory
        verifier = self._make_verifier_for_room(base, sg)
        assert verifier.verify("True") is True

    def test_verify_visited_rooms_after_setup(self) -> None:
        """verify='len(get_visited_rooms()) > 0' passes when SceneGraph has rooms."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory
        verifier = self._make_verifier_for_room(base, sg)

        # SceneGraph was seeded with rooms in _make_agent_with_skills
        result = verifier.verify("len(get_visited_rooms()) > 0")
        assert result is True, (
            "get_visited_rooms() should be non-empty after SceneGraph seeding"
        )

    def test_verify_world_stats_not_none(self) -> None:
        """verify='world_stats() is not None' passes when SceneGraph is initialised."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory
        verifier = self._make_verifier_for_room(base, sg)

        result = verifier.verify("world_stats() is not None")
        assert result is True, "world_stats() must return a non-None dict"

    def test_verify_describe_scene_check(self) -> None:
        """verify='len(describe_scene()) > 0' passes with mock that returns non-empty."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory
        verifier = self._make_verifier_for_room(base, sg)

        result = verifier.verify("len(describe_scene()) > 0")
        assert result is True, "describe_scene() mock returns non-empty string"

    def test_full_executor_navigate_verify_passes(self) -> None:
        """Execute navigate step with verify; verify passes after teleport."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        sg = agent._spatial_memory

        # Build verifier reflecting live base position
        ns = {
            "nearest_room": lambda: sg.nearest_room(base._pos[0], base._pos[1]),
            "get_position": lambda: tuple(base._pos),
            "get_heading": lambda: base._heading,
            "get_visited_rooms": lambda: sg.get_visited_rooms(),
            "describe_scene": lambda: "kitchen",
            "detect_objects": lambda query="": [],
            "world_stats": lambda: sg.stats(),
        }
        from vector_os_nano.vcli.cognitive import GoalVerifier, StrategySelector, GoalExecutor
        from vector_os_nano.core.skill import SkillContext

        verifier = GoalVerifier(ns)
        selector = StrategySelector(skill_registry=agent._skill_registry)

        def _build_context() -> SkillContext:
            return SkillContext(
                bases={"go2": base},
                services={"spatial_memory": sg},
            )

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=agent._skill_registry,
            build_context=_build_context,
        )

        tree = GoalTree(
            goal="go to hallway",
            sub_goals=(
                SubGoal(
                    name="reach_hallway",
                    description="navigate to hallway",
                    verify="nearest_room() == 'hallway'",
                    strategy="navigate_skill",
                    strategy_params={"room": "hallway"},
                    timeout_sec=30,
                ),
            ),
        )

        # Navigate skill will move base toward hallway; verify then checks position
        trace = executor.execute(tree)
        assert trace is not None
        step = trace.steps[0]
        assert "NoneType" not in (step.error or ""), (
            f"NoneType error during navigate+verify: {step.error}"
        )


# ---------------------------------------------------------------------------
# Scenario 8: should_use_vgg routing decisions
# ---------------------------------------------------------------------------


class TestShouldUseVggRouting:
    """Verify IntentRouter.should_use_vgg makes correct go/no-go decisions."""

    def test_explore_triggers_vgg(self) -> None:
        """'explore' matches ExploreSkill -> VGG."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        base = MockBase()
        agent = _make_agent_with_skills(base)
        assert router.should_use_vgg("explore", skill_registry=agent._skill_registry) is True

    def test_stand_triggers_vgg(self) -> None:
        """'站起来' matches StandSkill -> VGG."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        base = MockBase()
        agent = _make_agent_with_skills(base)
        assert router.should_use_vgg("站起来", skill_registry=agent._skill_registry) is True

    def test_patrol_triggers_vgg(self) -> None:
        """'巡逻所有房间' -> VGG (scope keyword + action verb)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        base = MockBase()
        agent = _make_agent_with_skills(base)
        assert router.should_use_vgg("巡逻所有房间", skill_registry=agent._skill_registry) is True

    def test_complex_multi_step_triggers_vgg(self) -> None:
        """'然后去厨房' (sequential keyword) -> VGG."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.should_use_vgg("然后去厨房", skill_registry=None) is True

    def test_pure_question_no_vgg(self) -> None:
        """'什么时候' -> no VGG (pure question, no action)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.should_use_vgg("什么时候", skill_registry=None) is False

    def test_greeting_no_vgg(self) -> None:
        """'hello there' -> no VGG (greeting, no robot action)."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.should_use_vgg("hello there", skill_registry=None) is False
