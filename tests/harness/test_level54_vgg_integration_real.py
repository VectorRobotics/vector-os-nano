"""VGG Integration Harness — tests with real Agent + Skills, mocked hardware only.

Tests the full pipeline: should_use_vgg, GoalDecomposer, StrategySelector,
GoalExecutor with real SkillRegistry and SceneGraph — only hardware mocked.
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
# MockBase — simulates Go2 hardware
# ---------------------------------------------------------------------------


class MockBase:
    """Simulates Go2 base — position tracking, heading, walk, navigate_to."""

    def __init__(self) -> None:
        self._pos = [10.0, 3.0, 0.28]
        self._heading = 0.0
        self._connected = True
        self.name = "mock_go2"

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
        self._pos[0] = x
        self._pos[1] = y
        return True

    def stand(self, duration: float = 1.0) -> bool:
        return True

    def sit(self, duration: float = 1.0) -> bool:
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
            qx=0,
            qy=0,
            qz=0,
            qw=1,
            vx=0,
            vy=0,
            vz=0,
            vyaw=0,
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
        self._responses: dict[str, str] = {
            "navigate": (
                '{"goal":"go to kitchen","sub_goals":['
                '{"name":"reach_kitchen","description":"navigate to kitchen",'
                '"verify":"nearest_room() == \'kitchen\'","strategy":"navigate_skill",'
                '"timeout_sec":60,"depends_on":[],"strategy_params":{},"fail_action":""}]}'
            ),
            "complex": (
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
            ),
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

        # Match to response
        for key, response in self._responses.items():
            if key in user_msg.lower():
                return _make_llm_response(response)

        # Default fallback: single step with no strategy
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
# Helpers — build real agent with real skills
# ---------------------------------------------------------------------------


def _make_agent_with_skills(base: MockBase) -> Any:
    """Create a minimal Agent-like object with real Go2 skills registered."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2 import get_go2_skills
    from vector_os_nano.core.scene_graph import SceneGraph

    # Use a plain object instead of Agent to avoid heavy __init__ dependencies
    class _FakeAgent:
        pass

    agent = _FakeAgent()
    agent._base = base  # type: ignore[attr-defined]
    agent._arm = None  # type: ignore[attr-defined]
    agent._vlm = None  # type: ignore[attr-defined]

    skill_registry = SkillRegistry()
    for skill_obj in get_go2_skills():
        skill_registry.register(skill_obj)
    agent._skill_registry = skill_registry  # type: ignore[attr-defined]

    # SceneGraph with rooms seeded to simulate post-exploration state
    sg = SceneGraph()
    for room_id, (cx, cy) in [
        ("kitchen", (17.0, 2.5)),
        ("hallway", (10.0, 5.0)),
        ("living_room", (4.8, 2.9)),
        ("master_bedroom", (12.0, 10.0)),
        ("study", (4.0, 8.0)),
    ]:
        for _ in range(5):  # builds visit_count > 0
            sg.visit(room_id, cx, cy)
    agent._spatial_memory = sg  # type: ignore[attr-defined]

    return agent


def _build_skill_context(base: MockBase, sg: Any) -> Any:
    """Build a SkillContext with the mock base and scene graph."""
    from vector_os_nano.core.skill import SkillContext

    return SkillContext(
        bases={"go2": base},
        services={"spatial_memory": sg},
    )


# ---------------------------------------------------------------------------
# TestVGGDirectSkillBypass
# ---------------------------------------------------------------------------


class TestVGGUnifiedPipeline:
    """ALL actionable commands go through VGG — no bypass."""

    def test_explore_goes_through_vgg(self) -> None:
        """'explore' matches ExploreSkill → VGG (1-step GoalTree)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("explore", skill_registry=agent._skill_registry)
        assert result is True, "explore should go through VGG (unified pipeline)"

    def test_go_to_kitchen_goes_through_vgg(self) -> None:
        """'去厨房' matches NavigateSkill → VGG (1-step GoalTree)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("去厨房", skill_registry=agent._skill_registry)
        assert result is True, "去厨房 should go through VGG (unified pipeline)"

    def test_stand_goes_through_vgg(self) -> None:
        """'站起来' matches StandSkill → VGG (1-step GoalTree)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("站起来", skill_registry=agent._skill_registry)
        assert result is True, "站起来 should go through VGG (unified pipeline)"

    def test_complex_task_triggers_vgg(self) -> None:
        """'去厨房看看有没有杯子' should trigger VGG (multi-step)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg(
            "去厨房看看有没有杯子", skill_registry=agent._skill_registry
        )
        assert result is True, "Complex multi-step task should trigger VGG"

    def test_patrol_triggers_vgg(self) -> None:
        """'巡逻所有房间' should trigger VGG (scope + multi-action)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("巡逻所有房间", skill_registry=agent._skill_registry)
        assert result is True, "巡逻所有房间 should trigger VGG"

    def test_explore_with_no_registry_triggers_vgg(self) -> None:
        """Without skill_registry, 'explore' triggers VGG via motor pattern."""
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        result = router.should_use_vgg("explore", skill_registry=None)
        assert result is True, "Without registry, explore triggers VGG motor pattern"

    def test_explore_keyword_match_exists(self) -> None:
        """ExploreSkill should be matched by 'explore' in the registry."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        match = agent._skill_registry.match("explore")
        assert match is not None, "SkillRegistry should match 'explore' to ExploreSkill"
        assert match.skill_name == "explore"

    def test_navigate_direct_flag(self) -> None:
        """NavigateSkill should have direct=True."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        match = agent._skill_registry.match("navigate")
        assert match is not None
        assert match.direct is True, "NavigateSkill must be direct=True"

    def test_explore_direct_flag_false(self) -> None:
        """ExploreSkill has direct=False — bypass check must use match-any logic."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        match = agent._skill_registry.match("explore")
        assert match is not None
        assert match.direct is False, "ExploreSkill should have direct=False"


# ---------------------------------------------------------------------------
# TestVGGGoalDecomposerIntegration
# ---------------------------------------------------------------------------


class TestVGGGoalDecomposerIntegration:
    """Test GoalDecomposer with real SkillRegistry."""

    def test_known_strategies_built_from_registry(self) -> None:
        """KNOWN_STRATEGIES should be built from real registered skill names."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()
        from vector_os_nano.vcli.cognitive import GoalDecomposer

        decomposer = GoalDecomposer(backend, skill_registry=agent._skill_registry)

        # Core skill-based strategies should exist
        assert "navigate_skill" in decomposer.KNOWN_STRATEGIES, (
            "navigate_skill missing from KNOWN_STRATEGIES"
        )
        assert "explore_skill" in decomposer.KNOWN_STRATEGIES, (
            "explore_skill missing from KNOWN_STRATEGIES"
        )
        assert "look_skill" in decomposer.KNOWN_STRATEGIES, (
            "look_skill missing from KNOWN_STRATEGIES"
        )
        # Primitives should still be present
        assert "walk_forward" in decomposer.KNOWN_STRATEGIES
        assert "turn" in decomposer.KNOWN_STRATEGIES

    def test_no_fake_strategies_in_registry(self) -> None:
        """KNOWN_STRATEGIES should NOT contain strategies absent from the real registry."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()
        from vector_os_nano.vcli.cognitive import GoalDecomposer

        decomposer = GoalDecomposer(backend, skill_registry=agent._skill_registry)
        assert "scan_room_360_then_report" not in decomposer.KNOWN_STRATEGIES
        assert "door_chain_fallback" not in decomposer.KNOWN_STRATEGIES

    def test_decompose_navigate_returns_valid_tree(self) -> None:
        """decompose() returns GoalTree with navigate_skill strategy."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()
        from vector_os_nano.vcli.cognitive import GoalDecomposer

        decomposer = GoalDecomposer(backend, skill_registry=agent._skill_registry)
        tree = decomposer.decompose("navigate to kitchen", "Position: (10.0, 3.0)")

        assert tree is not None
        assert len(tree.sub_goals) > 0, "GoalTree should have at least one sub_goal"
        for sg in tree.sub_goals:
            if sg.strategy:
                assert sg.strategy in decomposer.KNOWN_STRATEGIES, (
                    f"Strategy {sg.strategy!r} not in KNOWN_STRATEGIES"
                )

    def test_decompose_complex_task(self) -> None:
        """decompose() handles multi-step task."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()
        from vector_os_nano.vcli.cognitive import GoalDecomposer

        decomposer = GoalDecomposer(backend, skill_registry=agent._skill_registry)
        tree = decomposer.decompose(
            "visit kitchen and find cups",
            "Position: (10.0, 3.0), Known rooms: kitchen, hallway",
        )

        assert tree is not None
        assert len(tree.sub_goals) >= 1

    def test_list_skills_returns_strings(self) -> None:
        """SkillRegistry.list_skills() returns list of name strings, not objects."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        skill_names = agent._skill_registry.list_skills()
        assert isinstance(skill_names, list)
        for name in skill_names:
            assert isinstance(name, str), (
                f"list_skills() must return str, got {type(name)} for {name!r}"
            )


# ---------------------------------------------------------------------------
# TestVGGStrategySelector
# ---------------------------------------------------------------------------


class TestVGGStrategySelector:
    """Test StrategySelector resolves strategy names to real skills."""

    def test_navigate_skill_resolves(self) -> None:
        """'navigate_skill' strategy should resolve to skill executor_type='skill', name='navigate'."""
        from vector_os_nano.vcli.cognitive import StrategySelector
        from vector_os_nano.vcli.cognitive.types import SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        selector = StrategySelector(skill_registry=agent._skill_registry)

        sg = SubGoal(
            name="reach_kitchen",
            description="navigate to kitchen",
            verify="nearest_room() == 'kitchen'",
            strategy="navigate_skill",
            timeout_sec=60,
        )
        result = selector.select(sg)
        assert result.executor_type == "skill"
        assert result.name == "navigate", (
            f"navigate_skill should strip '_skill' suffix → 'navigate', got {result.name!r}"
        )

    def test_look_skill_resolves(self) -> None:
        """'look_skill' strategy → executor_type='skill', name='look'."""
        from vector_os_nano.vcli.cognitive import StrategySelector
        from vector_os_nano.vcli.cognitive.types import SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        selector = StrategySelector(skill_registry=agent._skill_registry)

        sg = SubGoal(
            name="observe",
            description="observe environment",
            verify="len(describe_scene()) > 0",
            strategy="look_skill",
            timeout_sec=15,
        )
        result = selector.select(sg)
        assert result.executor_type == "skill"
        assert result.name == "look"

    def test_navigate_skill_registered(self) -> None:
        """'navigate' skill must be registered so executor can find it."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        skill = agent._skill_registry.get("navigate")
        assert skill is not None, (
            "navigate skill must be registered in SkillRegistry"
        )

    def test_look_skill_registered(self) -> None:
        """'look' skill must be registered."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        skill = agent._skill_registry.get("look")
        assert skill is not None, "look skill must be registered"

    def test_describe_scene_skill_registered(self) -> None:
        """'describe_scene' skill must be registered."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        skill = agent._skill_registry.get("describe_scene")
        assert skill is not None, "describe_scene skill must be registered"


# ---------------------------------------------------------------------------
# TestVGGGoalExecutorIntegration
# ---------------------------------------------------------------------------


class TestVGGGoalExecutorIntegration:
    """Test GoalExecutor with real Agent + Skills wired via build_context."""

    def _make_executor(self, base: MockBase, agent: Any) -> Any:
        """Create a GoalExecutor with real components and proper build_context."""
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

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=agent._skill_registry,
            build_context=_build_context,
        )
        return executor

    def test_build_context_produces_valid_skill_context(self) -> None:
        """build_context() should return SkillContext with base accessible."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        from vector_os_nano.core.skill import SkillContext

        sg = agent._spatial_memory
        ctx = SkillContext(bases={"go2": base}, services={"spatial_memory": sg})

        assert ctx.base is base, "SkillContext.base should return the mock base"
        assert ctx.services.get("spatial_memory") is sg

    def test_executor_no_nonetype_base_error(self) -> None:
        """GoalExecutor should NOT raise 'NoneType has no attribute base'."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = self._make_executor(base, agent)

        tree = GoalTree(
            goal="navigate to kitchen",
            sub_goals=(
                SubGoal(
                    name="reach_kitchen",
                    description="navigate to kitchen",
                    verify="nearest_room() == 'kitchen'",
                    strategy="navigate_skill",
                    timeout_sec=30,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace is not None
        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in step {step.sub_goal_name}: {step.error}\n"
                "build_context was not wired correctly in GoalExecutor"
            )

    def test_executor_skill_result_is_skill_result_type(self) -> None:
        """GoalExecutor._execute_skill should get a SkillResult from the skill."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = self._make_executor(base, agent)

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
        assert len(trace.steps) == 1
        step = trace.steps[0]
        # StandSkill is direct=True — should either succeed or fail gracefully
        assert "NoneType" not in (step.error or ""), (
            f"NoneType error: {step.error}"
        )

    def test_executor_handles_missing_skill_gracefully(self) -> None:
        """When strategy resolves to a non-existent skill, error is clean (no NoneType crash)."""
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        base = MockBase()
        agent = _make_agent_with_skills(base)
        executor = self._make_executor(base, agent)

        tree = GoalTree(
            goal="do something unknown",
            sub_goals=(
                SubGoal(
                    name="unknown_step",
                    description="unknown action",
                    verify="True",
                    strategy="nonexistent_skill",
                    timeout_sec=5,
                ),
            ),
        )

        trace = executor.execute(tree)
        assert trace is not None
        step = trace.steps[0]
        assert step.success is False
        assert "NoneType" not in (step.error or ""), (
            "Missing skill error should be clean string, not NoneType"
        )


# ---------------------------------------------------------------------------
# TestVGGFullPipelineIntegration
# ---------------------------------------------------------------------------


class TestVGGFullPipelineIntegration:
    """Test the complete VGG pipeline through VectorEngine."""

    def test_engine_vgg_enabled_after_init(self) -> None:
        """engine._vgg_enabled should be True after init_vgg()."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

        assert engine._vgg_enabled is True, "VGG should be enabled after init_vgg()"

    def test_engine_vgg_decompose_complex_task(self) -> None:
        """engine.vgg_decompose() should return GoalTree for complex task.

        Uses a task with perception+judgment phrase (看看有没有) which is a
        confirmed complexity marker in IntentRouter.is_complex().
        """
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

        # 看看有没有 is a confirmed complex phrase → should_use_vgg returns True
        tree = engine.vgg_decompose("去厨房看看有没有杯子")
        assert tree is not None, (
            "Complex task '去厨房看看有没有杯子' should produce GoalTree via VGG"
        )
        assert len(tree.sub_goals) > 0

    def test_engine_vgg_decompose_simple_navigate_returns_1step(self) -> None:
        """engine.vgg_decompose('去厨房') returns 1-step GoalTree (fast path)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

        tree = engine.vgg_decompose("去厨房")
        assert tree is not None, "去厨房 should produce 1-step GoalTree via VGG fast path"
        assert len(tree.sub_goals) == 1, f"Simple navigate should be 1 step, got {len(tree.sub_goals)}"
        assert "navigate" in tree.sub_goals[0].name.lower()

    def test_engine_vgg_execute_no_nonetype_errors(self) -> None:
        """engine.vgg_execute() should not produce NoneType errors in trace."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

        # Use confirmed-complex phrase so vgg_decompose returns a tree
        tree = engine.vgg_decompose("去厨房看看有没有杯子")
        assert tree is not None, (
            "Complex task should produce GoalTree — check should_use_vgg logic"
        )

        trace = engine.vgg_execute(tree)
        assert trace is not None
        for step in trace.steps:
            assert "NoneType" not in (step.error or ""), (
                f"NoneType error in step {step.sub_goal_name}: {step.error}"
            )

    def test_engine_build_context_wired_in_init_vgg(self) -> None:
        """GoalExecutor inside engine must have build_context set (not None)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

        executor = engine._goal_executor
        assert executor is not None
        assert executor._build_context is not None, (
            "GoalExecutor._build_context is None — "
            "engine.init_vgg() must wire build_context"
        )

    def test_engine_vgg_explore_1step(self) -> None:
        """engine.vgg_decompose('explore') returns 1-step GoalTree (fast path)."""
        base = MockBase()
        agent = _make_agent_with_skills(base)
        backend = MockLLMBackend()

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        engine = VectorEngine(backend=backend, intent_router=IntentRouter())
        engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

        tree = engine.vgg_decompose("explore")
        assert tree is not None, "explore should produce 1-step GoalTree via VGG fast path"
        assert len(tree.sub_goals) == 1
        assert "explore" in tree.sub_goals[0].name.lower()
