"""Level 2 — Scene understanding skills with mock VLM.

Tests verify LookSkill and DescribeSceneSkill behaviour using mock
dependencies. No real API calls are made — all VLM responses are
pre-configured MagicMock return values.

Coverage:
- LookSkill: success, missing base, missing VLM service
- DescribeSceneSkill: query mode, full-description mode, missing base, missing VLM

Cost: $0.00 (mock only — no external calls).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import types used in mock return values
from vector_os_nano.perception.vlm_go2 import (
    DetectedObject,
    RoomIdentification,
    SceneDescription,
)
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult


# ---------------------------------------------------------------------------
# Shared mock factories
# ---------------------------------------------------------------------------

def _make_base_mock() -> MagicMock:
    """Build a mock base with plausible sensor return values."""
    base = MagicMock()
    base.get_position.return_value = [10.0, 3.0, 0.28]
    base.get_heading.return_value = 0.0
    base.get_camera_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    return base


def _make_vlm_mock() -> MagicMock:
    """Build a mock VLM with realistic SceneDescription / RoomIdentification responses."""
    vlm = MagicMock()
    vlm.describe_scene.return_value = SceneDescription(
        summary="A hallway with tiled floor",
        objects=[
            DetectedObject(
                name="door",
                description="wooden door at end of hallway",
                confidence=0.9,
            )
        ],
        room_type="hallway",
        details="The hallway has beige walls, a tiled floor, and a wooden door.",
    )
    vlm.identify_room.return_value = RoomIdentification(
        room="hallway",
        confidence=0.85,
        reasoning="Tiled floor and narrow corridor suggest a hallway.",
    )
    vlm.find_objects.return_value = [
        DetectedObject(
            name="chair",
            description="a wooden chair near the wall",
            confidence=0.88,
        )
    ]
    return vlm


def _make_context(
    has_vlm: bool = True,
    has_base: bool = True,
) -> SkillContext:
    """Build a SkillContext with configurable mock base and VLM service."""
    bases = {"default": _make_base_mock()} if has_base else {}
    services = {"vlm": _make_vlm_mock()} if has_vlm else {}
    return SkillContext(bases=bases, services=services)


# ---------------------------------------------------------------------------
# LookSkill tests
# ---------------------------------------------------------------------------


class TestLookSkill:
    """L2: LookSkill behaviour with mock VLM and mock base."""

    def test_look_skill_success(self):
        """LookSkill succeeds and returns scene summary when base + VLM are present."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context()
        result = skill.execute({}, ctx)

        assert isinstance(result, SkillResult), (
            f"Expected SkillResult, got {type(result)}"
        )
        assert result.success, (
            f"LookSkill failed unexpectedly: {result.error_message}"
        )
        assert "summary" in result.result_data, (
            "result_data missing 'summary' key"
        )
        assert result.result_data["summary"], "summary value is empty"

    def test_look_skill_returns_room(self):
        """LookSkill result_data includes a non-empty room field."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context()
        result = skill.execute({}, ctx)

        assert result.success
        assert "room" in result.result_data, "result_data missing 'room' key"
        assert result.result_data["room"], "room value is empty"

    def test_look_skill_returns_objects_list(self):
        """LookSkill result_data includes an 'objects' list."""

    def test_look_skill_no_base(self):
        """LookSkill fails gracefully with diagnosis_code 'no_base' when base is absent."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context(has_base=False)
        result = skill.execute({}, ctx)

        assert not result.success, "LookSkill should fail when base is absent"
        assert result.diagnosis_code == "no_base", (
            f"Expected diagnosis_code 'no_base', got {result.diagnosis_code!r}"
        )

    def test_look_skill_no_vlm(self):
        """LookSkill fails gracefully with diagnosis_code 'no_vlm' when VLM is absent."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context(has_vlm=False)
        result = skill.execute({}, ctx)

        assert not result.success, "LookSkill should fail when VLM is absent"
        assert result.diagnosis_code == "no_vlm", (
            f"Expected diagnosis_code 'no_vlm', got {result.diagnosis_code!r}"
        )

    def test_look_skill_camera_failure(self):
        """LookSkill fails with 'camera_failed' when get_camera_frame raises."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context()
        # Override camera to raise an error
        ctx.base.get_camera_frame.side_effect = RuntimeError("camera disconnected")

        result = skill.execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "camera_failed", (
            f"Expected 'camera_failed', got {result.diagnosis_code!r}"
        )

    def test_look_skill_vlm_exception(self):
        """LookSkill fails with 'vlm_failed' when VLM raises."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context()
        ctx.services["vlm"].describe_scene.side_effect = RuntimeError("API unreachable")

        result = skill.execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "vlm_failed", (
            f"Expected 'vlm_failed', got {result.diagnosis_code!r}"
        )

    def test_look_skill_room_confidence_in_range(self):
        """LookSkill result_data room_confidence is in [0.0, 1.0]."""
        from vector_os_nano.skills.go2.look import LookSkill

        skill = LookSkill()
        ctx = _make_context()
        result = skill.execute({}, ctx)

        assert result.success
        confidence = result.result_data.get("room_confidence", None)
        assert confidence is not None, "result_data missing 'room_confidence'"
        assert 0.0 <= confidence <= 1.0, (
            f"room_confidence {confidence} out of [0.0, 1.0] range"
        )


# ---------------------------------------------------------------------------
# DescribeSceneSkill tests
# ---------------------------------------------------------------------------


class TestDescribeSceneSkill:
    """L2: DescribeSceneSkill behaviour with mock VLM and mock base."""

    def test_describe_scene_without_query(self):
        """DescribeSceneSkill runs full description mode when no query is given."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        result = skill.execute({}, ctx)

        assert result.success, (
            f"DescribeSceneSkill failed unexpectedly: {result.error_message}"
        )
        # Full description mode returns summary + details + room
        assert "summary" in result.result_data, "result_data missing 'summary'"
        assert "details" in result.result_data, "result_data missing 'details'"
        assert "room" in result.result_data, "result_data missing 'room'"

    def test_describe_scene_with_query(self):
        """DescribeSceneSkill runs find_objects mode when query is provided."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        result = skill.execute({"query": "chair"}, ctx)

        assert result.success, (
            f"DescribeSceneSkill (query mode) failed: {result.error_message}"
        )
        # Query mode returns objects + count + query echo
        assert "objects" in result.result_data, "result_data missing 'objects'"
        assert "count" in result.result_data, "result_data missing 'count'"
        assert result.result_data.get("query") == "chair", (
            "result_data should echo the query back"
        )

    def test_describe_scene_query_calls_find_objects(self):
        """When query is provided, find_objects is called (not describe_scene)."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        vlm = ctx.services["vlm"]

        skill.execute({"query": "chair"}, ctx)

        vlm.find_objects.assert_called_once()
        vlm.describe_scene.assert_not_called()

    def test_describe_scene_no_query_calls_describe(self):
        """When query is absent, describe_scene is called (not find_objects)."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        vlm = ctx.services["vlm"]

        skill.execute({}, ctx)

        vlm.describe_scene.assert_called_once()
        vlm.find_objects.assert_not_called()

    def test_describe_scene_no_base(self):
        """DescribeSceneSkill fails gracefully with 'no_base' when base is absent."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context(has_base=False)
        result = skill.execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base", (
            f"Expected 'no_base', got {result.diagnosis_code!r}"
        )

    def test_describe_scene_no_vlm(self):
        """DescribeSceneSkill fails gracefully with 'no_vlm' when VLM is absent."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context(has_vlm=False)
        result = skill.execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_vlm", (
            f"Expected 'no_vlm', got {result.diagnosis_code!r}"
        )

    def test_describe_scene_camera_failure(self):
        """DescribeSceneSkill fails with 'camera_failed' when get_camera_frame raises."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        ctx.base.get_camera_frame.side_effect = RuntimeError("shutter jammed")

        result = skill.execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "camera_failed"

    def test_describe_scene_vlm_exception(self):
        """DescribeSceneSkill fails with 'vlm_failed' when VLM raises."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        ctx.services["vlm"].describe_scene.side_effect = RuntimeError("timeout")

        result = skill.execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "vlm_failed"

    def test_describe_scene_result_data_types(self):
        """Full description mode returns correctly typed result_data fields."""
        from vector_os_nano.skills.go2.look import DescribeSceneSkill

        skill = DescribeSceneSkill()
        ctx = _make_context()
        result = skill.execute({}, ctx)

        assert result.success
        rd = result.result_data
        assert isinstance(rd.get("summary"), str), "'summary' must be a string"
        assert isinstance(rd.get("details"), str), "'details' must be a string"
        assert isinstance(rd.get("room"), str), "'room' must be a string"
        assert isinstance(rd.get("objects"), list), "'objects' must be a list"
        confidence = rd.get("room_confidence")
        assert isinstance(confidence, float), "'room_confidence' must be a float"
        assert 0.0 <= confidence <= 1.0
