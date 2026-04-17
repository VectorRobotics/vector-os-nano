"""Level 59 — VisualVerifier TDD tests.

VLM-based visual verification fallback for VGG goal steps.
Called when GoalVerifier.verify() returns False to get a second opinion
via camera + VLM.

All tests use mocks — no MuJoCo, no real VLM, no real camera.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vector_os_nano.vcli.cognitive.visual_verifier import (
    VisualVerifyResult,
    _check_description_relevance,
    _extract_query_from_verify,
    should_verify,
    verify_visual,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BLANK_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _make_agent(has_base: bool = True, has_vlm: bool = True) -> MagicMock:
    """Build a minimal mock agent with _base and _vlm."""
    agent = MagicMock()
    if has_base:
        agent._base = MagicMock()
        agent._base.get_camera_frame.return_value = _BLANK_FRAME
    else:
        agent._base = None
    if has_vlm:
        agent._vlm = MagicMock()
    else:
        agent._vlm = None
    return agent


# ===========================================================================
# should_verify tests
# ===========================================================================


class TestShouldVerify:
    """Tests for should_verify() trigger logic."""

    def test_should_verify_false_when_passed(self) -> None:
        """verify_result=True → no second opinion needed → False."""
        result = should_verify(
            sub_goal_name="navigate_to_kitchen",
            sub_goal_description="Go to the kitchen",
            strategy="navigate_skill",
            verify_expr="nearest_room() == 'kitchen'",
            verify_result=True,
        )
        assert result is False

    def test_should_verify_true_perception_keyword_in_strategy(self) -> None:
        """verify_result=False + 'look' in strategy → True."""
        result = should_verify(
            sub_goal_name="look_around",
            sub_goal_description="Scan the room",
            strategy="look_skill",
            verify_expr="nearest_room() == 'kitchen'",
            verify_result=False,
        )
        assert result is True

    def test_should_verify_true_detect_in_verify_expr(self) -> None:
        """verify_result=False + detect_objects in verify expr → True."""
        result = should_verify(
            sub_goal_name="find_cup",
            sub_goal_description="Find the cup",
            strategy="navigate_skill",
            verify_expr="len(detect_objects('cup')) > 0",
            verify_result=False,
        )
        assert result is True

    def test_should_verify_true_describe_scene_in_verify_expr(self) -> None:
        """verify_result=False + describe_scene in verify expr → True."""
        result = should_verify(
            sub_goal_name="check_table",
            sub_goal_description="Check if table is visible",
            strategy="navigate_skill",
            verify_expr="'table' in describe_scene()",
            verify_result=False,
        )
        assert result is True

    def test_should_verify_false_no_perception_context(self) -> None:
        """verify_result=False but no perception keywords or visual functions → False."""
        result = should_verify(
            sub_goal_name="go_kitchen",
            sub_goal_description="Navigate to the kitchen",
            strategy="navigate_skill",
            verify_expr="nearest_room() == 'kitchen'",
            verify_result=False,
        )
        assert result is False

    def test_should_verify_true_chinese_perception_keyword_in_description(self) -> None:
        """verify_result=False + Chinese perception keyword '观察' in description → True."""
        result = should_verify(
            sub_goal_name="observe_kitchen",
            sub_goal_description="观察厨房环境",
            strategy="navigate_skill",
            verify_expr="nearest_room() == 'kitchen'",
            verify_result=False,
        )
        assert result is True

    def test_should_verify_true_chinese_look_keyword(self) -> None:
        """verify_result=False + Chinese '看' in description → True."""
        result = should_verify(
            sub_goal_name="check_room",
            sub_goal_description="看看厨房有没有杯子",
            strategy="navigate_skill",
            verify_expr="nearest_room() == 'kitchen'",
            verify_result=False,
        )
        assert result is True

    def test_should_verify_true_identify_room_in_verify_expr(self) -> None:
        """identify_room is NOT in _VISUAL_VERIFY_FUNCTIONS, but 'scan' keyword in strategy."""
        result = should_verify(
            sub_goal_name="scan_room",
            sub_goal_description="Scan the area",
            strategy="scan_skill",
            verify_expr="identify_room() == 'kitchen'",
            verify_result=False,
        )
        assert result is True  # 'scan' is in _PERCEPTION_KEYWORDS

    def test_should_verify_true_observe_keyword_in_description(self) -> None:
        """'observe' keyword in description triggers visual verify."""
        result = should_verify(
            sub_goal_name="observe_step",
            sub_goal_description="Observe the current scene carefully",
            strategy="navigate_skill",
            verify_expr="nearest_room() == 'living_room'",
            verify_result=False,
        )
        assert result is True


# ===========================================================================
# verify_visual tests
# ===========================================================================


class TestVerifyVisual:
    """Tests for verify_visual() execution logic."""

    def test_verify_visual_no_vlm(self) -> None:
        """No VLM → triggered=False, success=False, error mentions unavailability."""
        agent = _make_agent(has_base=True, has_vlm=False)
        result = verify_visual(agent, "Find the cup", "len(detect_objects('cup')) > 0")
        assert result.triggered is False
        assert result.success is False
        assert "VLM" in result.error or "vlm" in result.error.lower()

    def test_verify_visual_no_base(self) -> None:
        """No camera base → triggered=False, success=False."""
        agent = _make_agent(has_base=False, has_vlm=True)
        result = verify_visual(agent, "Find the cup", "len(detect_objects('cup')) > 0")
        assert result.triggered is False
        assert result.success is False

    def test_verify_visual_detect_objects_found(self) -> None:
        """VLM.find_objects returns non-empty list → triggered=True, success=True."""
        agent = _make_agent()
        mock_obj = MagicMock()
        mock_obj.name = "cup"
        mock_obj.confidence = 0.9
        agent._vlm.find_objects.return_value = [mock_obj]

        result = verify_visual(
            agent, "Find the cup", "len(detect_objects('cup')) > 0"
        )
        assert result.triggered is True
        assert result.success is True
        assert result.duration_sec >= 0.0

    def test_verify_visual_detect_objects_empty(self) -> None:
        """VLM.find_objects returns [] → triggered=True, success=False."""
        agent = _make_agent()
        agent._vlm.find_objects.return_value = []

        result = verify_visual(
            agent, "Find the cup", "len(detect_objects('cup')) > 0"
        )
        assert result.triggered is True
        assert result.success is False

    def test_verify_visual_describe_scene_relevant(self) -> None:
        """VLM.describe_scene returns description containing goal keyword → success=True."""
        agent = _make_agent()
        mock_scene = MagicMock()
        mock_scene.details = "I can see a kitchen table with a cup on it"
        mock_scene.summary = "Kitchen scene with cup"
        agent._vlm.describe_scene.return_value = mock_scene

        result = verify_visual(
            agent,
            "find cup on table",
            "'table' in describe_scene()",
        )
        assert result.triggered is True
        assert result.success is True
        assert len(result.description) > 0

    def test_verify_visual_vlm_exception(self) -> None:
        """VLM raises RuntimeError → triggered=True, success=False, error captured."""
        agent = _make_agent()
        agent._vlm.find_objects.side_effect = RuntimeError("VLM timeout")
        agent._vlm.describe_scene.side_effect = RuntimeError("VLM timeout")

        result = verify_visual(
            agent,
            "Find the cup",
            "len(detect_objects('cup')) > 0",
        )
        assert result.triggered is True
        assert result.success is False
        assert "VLM timeout" in result.error or len(result.error) > 0

    def test_verify_visual_timeout(self) -> None:
        """VLM call that simulates timeout → success=False."""
        agent = _make_agent()

        def slow_find_objects(frame, query):  # noqa: ANN001
            time.sleep(0.05)  # small delay to test timing path
            raise TimeoutError("Request timed out")

        agent._vlm.find_objects.side_effect = slow_find_objects

        result = verify_visual(
            agent,
            "Find the cup",
            "len(detect_objects('cup')) > 0",
        )
        assert result.triggered is True
        assert result.success is False
        assert result.duration_sec >= 0.0

    def test_verify_visual_duration_recorded(self) -> None:
        """duration_sec is set to a non-negative value on success."""
        agent = _make_agent()
        mock_scene = MagicMock()
        mock_scene.details = "I see a cup"
        mock_scene.summary = "cup visible"
        agent._vlm.describe_scene.return_value = mock_scene

        result = verify_visual(agent, "find cup", "describe_scene()")
        assert result.duration_sec >= 0.0


# ===========================================================================
# _extract_query_from_verify tests
# ===========================================================================


class TestExtractQueryFromVerify:
    """Tests for _extract_query_from_verify()."""

    def test_extract_detect_objects_single_quotes(self) -> None:
        """detect_objects('cup') → 'cup'."""
        assert _extract_query_from_verify("len(detect_objects('cup')) > 0") == "cup"

    def test_extract_describe_scene_keyword(self) -> None:
        """'table' in describe_scene() → 'table'."""
        assert _extract_query_from_verify("'table' in describe_scene()") == "table"

    def test_extract_no_match(self) -> None:
        """nearest_room() == 'kitchen' → '' (no visual detection call)."""
        assert _extract_query_from_verify("nearest_room() == 'kitchen'") == ""

    def test_extract_detect_objects_double_quotes(self) -> None:
        """detect_objects("ball") → 'ball'."""
        assert _extract_query_from_verify('detect_objects("ball")') == "ball"

    def test_extract_detect_objects_multi_word(self) -> None:
        """detect_objects('red ball') → 'red ball'."""
        assert _extract_query_from_verify("detect_objects('red ball')") == "red ball"

    def test_extract_empty_expr(self) -> None:
        """Empty string → ''."""
        assert _extract_query_from_verify("") == ""


# ===========================================================================
# _check_description_relevance tests
# ===========================================================================


class TestCheckDescriptionRelevance:
    """Tests for _check_description_relevance()."""

    def test_relevance_match_keyword_in_description(self) -> None:
        """description contains 'cup' and goal says 'find cup' → True."""
        result = _check_description_relevance(
            description="I see a kitchen with a cup on the table",
            sub_goal_description="find cup",
            verify_expr="len(detect_objects('cup')) > 0",
        )
        assert result is True

    def test_relevance_no_match(self) -> None:
        """description has no overlap with goal → False."""
        result = _check_description_relevance(
            description="I see an empty hallway with white walls",
            sub_goal_description="find cup",
            verify_expr="len(detect_objects('cup')) > 0",
        )
        assert result is False

    def test_relevance_empty_description(self) -> None:
        """Empty description → False."""
        result = _check_description_relevance(
            description="",
            sub_goal_description="find cup",
            verify_expr="",
        )
        assert result is False

    def test_relevance_description_matches_verify_expr_keyword(self) -> None:
        """verify_expr mentions 'table'; description has 'table' → True."""
        result = _check_description_relevance(
            description="There is a wooden table in the center",
            sub_goal_description="navigate to room",
            verify_expr="'table' in describe_scene()",
        )
        assert result is True

    def test_relevance_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        result = _check_description_relevance(
            description="I see a CUP on the counter",
            sub_goal_description="Find Cup in Kitchen",
            verify_expr="",
        )
        assert result is True
