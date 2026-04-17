"""Test Phase 3 visual verification integration in GoalExecutor."""
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.types import SubGoal, GoalTree, StepRecord
from vector_os_nano.vcli.cognitive.visual_verifier import VisualVerifyResult


class TestVisualVerifyIntegration:
    """GoalExecutor calls visual verifier when primary verify fails."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent._base.get_camera_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        agent._vlm.describe_scene.return_value = MagicMock(summary="kitchen with cup on table")
        # MagicMock(name=...) sets the mock display name, not a .name attribute.
        # Use spec or set attribute explicitly so getattr(o, "name") returns "cup".
        obj_mock = MagicMock()
        obj_mock.name = "cup"
        obj_mock.confidence = 0.9
        agent._vlm.find_objects.return_value = [obj_mock]
        return agent

    @pytest.fixture
    def executor_with_visual(self, mock_agent):
        selector = MagicMock()
        selector.select.return_value = MagicMock(
            executor_type="skill",
            name="look",
            params={},
        )
        verifier = MagicMock()
        skill_registry = MagicMock()
        skill = MagicMock()
        skill.execute.return_value = MagicMock(success=True)
        skill_registry.get.return_value = skill

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=skill_registry,
            build_context=lambda: MagicMock(),
            visual_verifier_agent=mock_agent,
        )
        return executor, verifier

    def test_visual_verify_overrides_failed_verify(self, executor_with_visual):
        """When primary verify fails but VLM confirms, step should succeed."""
        executor, verifier = executor_with_visual
        verifier.verify.return_value = False  # primary verify fails

        sub_goal = SubGoal(
            name="detect_cup",
            description="detect cup on table",
            verify="len(detect_objects('cup')) > 0",
            strategy="look_skill",
        )
        step = executor._execute_sub_goal(sub_goal)
        # Visual verifier should have been triggered and overridden
        assert step.success is True

    def test_visual_verify_not_triggered_when_passed(self, executor_with_visual):
        """When primary verify passes, visual verify should NOT be called."""
        executor, verifier = executor_with_visual
        verifier.verify.return_value = True  # primary passes

        sub_goal = SubGoal(
            name="reach_kitchen",
            description="navigate to kitchen",
            verify="nearest_room() == 'kitchen'",
            strategy="navigate_skill",
        )
        step = executor._execute_sub_goal(sub_goal)
        assert step.success is True

    def test_visual_verify_not_triggered_no_perception(self, executor_with_visual):
        """Non-perception steps should not trigger visual verification."""
        executor, verifier = executor_with_visual
        verifier.verify.return_value = False

        sub_goal = SubGoal(
            name="reach_kitchen",
            description="navigate to kitchen",
            verify="nearest_room() == 'kitchen'",
            strategy="navigate_skill",
        )
        step = executor._execute_sub_goal(sub_goal)
        # should_verify returns False for navigation without perception keywords
        # so visual verify is NOT triggered, step fails normally
        assert step.success is False

    def test_no_visual_verifier_agent(self):
        """GoalExecutor without visual_verifier_agent works normally."""
        selector = MagicMock()
        selector.select.return_value = MagicMock(executor_type="skill", name="look", params={})
        verifier = MagicMock()
        verifier.verify.return_value = False
        skill_registry = MagicMock()
        skill = MagicMock()
        skill.execute.return_value = MagicMock(success=True)
        skill_registry.get.return_value = skill

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=skill_registry,
            build_context=lambda: MagicMock(),
            visual_verifier_agent=None,  # no visual verifier
        )
        sub_goal = SubGoal(
            name="detect_cup",
            description="detect cup",
            verify="len(detect_objects('cup')) > 0",
            strategy="look_skill",
        )
        step = executor._execute_sub_goal(sub_goal)
        # Without visual verifier, failed verify = failed step
        assert step.success is False

    def test_visual_verify_exception_non_blocking(self, executor_with_visual, mock_agent):
        """If visual verifier raises, step continues to fail_action normally."""
        executor, verifier = executor_with_visual
        verifier.verify.return_value = False
        # Make VLM raise
        mock_agent._vlm.find_objects.side_effect = RuntimeError("VLM crash")
        mock_agent._vlm.describe_scene.side_effect = RuntimeError("VLM crash")

        sub_goal = SubGoal(
            name="detect_cup",
            description="observe table to detect cup",
            verify="len(detect_objects('cup')) > 0",
            strategy="look_skill",
        )
        step = executor._execute_sub_goal(sub_goal)
        # VLM failed, so visual verify couldn't help -> step fails
        assert step.success is False

    def test_backward_compatible_no_visual_arg(self):
        """GoalExecutor still works without visual_verifier_agent kwarg."""
        selector = MagicMock()
        verifier = MagicMock()
        # Old-style construction without visual_verifier_agent
        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
        )
        assert executor._visual_verifier_agent is None


class TestGoalExecutorExistingBehavior:
    """Verify existing GoalExecutor tests still pass after modification."""

    def test_execute_success_path_unchanged(self):
        selector = MagicMock()
        selector.select.return_value = MagicMock(executor_type="skill", name="stand", params={})
        verifier = MagicMock()
        verifier.verify.return_value = True
        skill_registry = MagicMock()
        skill = MagicMock()
        skill.execute.return_value = MagicMock(success=True)
        skill_registry.get.return_value = skill

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=skill_registry,
            build_context=lambda: MagicMock(),
        )
        tree = GoalTree(goal="stand up", sub_goals=(
            SubGoal(name="stand", description="stand up", verify="True"),
        ))
        trace = executor.execute(tree)
        assert trace.success is True

    def test_execute_failure_path_unchanged(self):
        selector = MagicMock()
        selector.select.return_value = MagicMock(executor_type="skill", name="nav", params={})
        verifier = MagicMock()
        verifier.verify.return_value = False
        skill_registry = MagicMock()
        skill = MagicMock()
        skill.execute.return_value = MagicMock(success=True)
        skill_registry.get.return_value = skill

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=skill_registry,
            build_context=lambda: MagicMock(),
        )
        tree = GoalTree(goal="go kitchen", sub_goals=(
            SubGoal(name="reach_kitchen", description="go", verify="nearest_room()=='kitchen'", strategy="navigate_skill"),
        ))
        trace = executor.execute(tree)
        assert trace.success is False
