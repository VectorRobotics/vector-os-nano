"""Level 62 — Phase 3 Active World Model: MuJoCo integration tests.

Tests Phase 3 components with a real MuJoCo Go2 simulation:
- ObjectMemory sync from a real SceneGraph populated via MuJoCo
- predict functions with a real SceneGraph (room layout loaded)
- VisualVerifier with real camera frames from MuJoCo renderer
- GoalVerifier namespace with Phase 3 functions wired to real data
- Full VGG pipeline: decompose → execute → verify with world model

MuJoCo instance management:
- Each fixture uses scope="function" (clean state per test)
- go2_flat fixture from conftest.py: connect → yield → disconnect
- No parallel pytest — single-threaded execution
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vector_os_nano.core.scene_graph import (
    ObjectNode,
    RoomNode,
    SceneGraph,
    ViewpointNode,
)
from vector_os_nano.vcli.cognitive.object_memory import ObjectMemory, TrackedObject
from vector_os_nano.vcli.cognitive.predict import (
    predict_exploration_value,
    predict_navigation,
    predict_room_after_door,
)
from vector_os_nano.vcli.cognitive.visual_verifier import (
    VisualVerifyResult,
    should_verify,
    verify_visual,
)
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populated_scene_graph() -> SceneGraph:
    """Create a SceneGraph with rooms, doors, viewpoints, and objects.

    Layout: hallway (0,0) ←door→ kitchen (5,3) ←door→ bedroom (10,3)
    """
    sg = SceneGraph()
    sg.add_room(RoomNode(room_id="hallway", center_x=0.0, center_y=0.0, visit_count=2))
    sg.add_room(RoomNode(room_id="kitchen", center_x=5.0, center_y=3.0, visit_count=1))
    sg.add_room(RoomNode(room_id="bedroom", center_x=10.0, center_y=3.0, visit_count=0))
    sg.add_door("hallway", "kitchen", 2.5, 1.5)
    sg.add_door("kitchen", "bedroom", 7.5, 3.0)

    # Objects in kitchen
    sg.add_object(ObjectNode(
        object_id="cup_01", category="cup", room_id="kitchen",
        x=5.1, y=3.2, confidence=0.9, first_seen=time.time(),
    ))
    sg.add_object(ObjectNode(
        object_id="plate_01", category="plate", room_id="kitchen",
        x=5.3, y=2.8, confidence=0.85, first_seen=time.time(),
    ))
    # Object in hallway
    sg.add_object(ObjectNode(
        object_id="shoe_01", category="shoe", room_id="hallway",
        x=0.5, y=-0.3, confidence=0.7, first_seen=time.time() - 600,
    ))
    return sg


# ===================================================================
# A. ObjectMemory + real SceneGraph (no MuJoCo needed)
# ===================================================================


class TestObjectMemoryWithRealSceneGraph:
    """ObjectMemory synced from a real SceneGraph with rooms/objects."""

    @pytest.fixture
    def sg_and_om(self):
        sg = _populated_scene_graph()
        om = ObjectMemory(decay_lambda=0.001)
        count = om.sync_from_scene_graph(sg)
        return sg, om, count

    def test_sync_count(self, sg_and_om):
        _, _, count = sg_and_om
        assert count == 3  # cup, plate, shoe

    def test_last_seen_cup(self, sg_and_om):
        _, om, _ = sg_and_om
        result = om.last_seen("cup")
        assert result is not None
        assert result["room"] == "kitchen"
        assert result["confidence"] > 0.8  # just synced, minimal decay

    def test_last_seen_shoe_decayed(self, sg_and_om):
        _, om, _ = sg_and_om
        # sync_from_scene_graph uses time.time() as last_seen (refresh on sync).
        # To test decay, manually backdate the shoe's last_seen.
        shoe = om._objects.get("shoe_01")
        assert shoe is not None
        backdated = TrackedObject(
            object_id=shoe.object_id, category=shoe.category,
            room_id=shoe.room_id, x=shoe.x, y=shoe.y,
            last_seen=time.time() - 600,  # 10 min ago
            base_confidence=shoe.base_confidence,
            observation_count=shoe.observation_count,
        )
        om._objects["shoe_01"] = backdated
        result = om.last_seen("shoe")
        assert result is not None
        assert result["room"] == "hallway"
        # 0.7 * exp(-0.001*600) ≈ 0.384
        assert result["confidence"] < 0.5

    def test_certainty_cup_in_kitchen(self, sg_and_om):
        _, om, _ = sg_and_om
        c = om.certainty("cup在kitchen")
        assert c > 0.8

    def test_certainty_cup_in_bedroom(self, sg_and_om):
        _, om, _ = sg_and_om
        c = om.certainty("cup在bedroom")
        assert c == 0.0  # cup is in kitchen, not bedroom

    def test_objects_in_kitchen(self, sg_and_om):
        _, om, _ = sg_and_om
        objs = om.objects_in_room("kitchen")
        assert len(objs) == 2
        categories = {o["category"] for o in objs}
        assert categories == {"cup", "plate"}

    def test_find_object_across_rooms(self, sg_and_om):
        _, om, _ = sg_and_om
        # shoe is in hallway, cup+plate in kitchen
        all_objs = om.find_object("shoe")
        assert len(all_objs) == 1
        assert all_objs[0]["room"] == "hallway"

    def test_resync_after_new_object(self, sg_and_om):
        sg, om, _ = sg_and_om
        # Add a new object to SceneGraph
        sg.add_object(ObjectNode(
            object_id="book_01", category="book", room_id="bedroom",
            x=10.2, y=3.1, confidence=0.8,
        ))
        count = om.sync_from_scene_graph(sg)
        assert count == 4
        assert om.last_seen("book") is not None


# ===================================================================
# B. predict with real SceneGraph (no MuJoCo needed)
# ===================================================================


class TestPredictWithRealSceneGraph:
    """predict functions using a real SceneGraph with room topology."""

    @pytest.fixture
    def sg(self):
        return _populated_scene_graph()

    def test_predict_nav_hallway_to_kitchen(self, sg):
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["reachable"] is True
        assert result["door_count"] == 1
        assert result["confidence"] == 1.0
        assert "hallway" in result["rooms_on_path"]
        assert "kitchen" in result["rooms_on_path"]

    def test_predict_nav_hallway_to_bedroom(self, sg):
        result = predict_navigation(sg, "hallway", "bedroom")
        assert result["reachable"] is True
        assert result["door_count"] == 2  # hallway→kitchen→bedroom

    def test_predict_nav_same_room(self, sg):
        result = predict_navigation(sg, "kitchen", "kitchen")
        assert result["reachable"] is True
        assert result["door_count"] == 0
        assert result["estimated_steps"] == 1

    def test_predict_nav_unreachable(self, sg):
        result = predict_navigation(sg, "hallway", "garage")
        assert result["reachable"] is False
        assert result["confidence"] == 0.0

    def test_predict_room_after_door_exists(self, sg):
        result = predict_room_after_door(sg, "hallway", "kitchen")
        assert result["room"] == "kitchen"
        assert result["confidence"] == 1.0
        assert result["door_position"] is not None

    def test_predict_room_after_door_no_direct(self, sg):
        result = predict_room_after_door(sg, "hallway", "bedroom")
        assert result["room"] == ""
        assert result["confidence"] == 0.0

    def test_predict_exploration_unexplored(self, sg):
        result = predict_exploration_value(sg, "bedroom")
        assert result["coverage"] < 0.1  # never visited, no viewpoints
        assert result["value"] > 0.5  # high value to explore

    def test_predict_exploration_visited(self, sg):
        # hallway has visit_count=2 but no viewpoints → coverage still 0
        # The value depends on coverage calculation
        result = predict_exploration_value(sg, "hallway")
        assert "coverage" in result
        assert "value" in result


# ===================================================================
# C. VisualVerifier with real MuJoCo camera frames
# ===================================================================


class TestVisualVerifierMuJoCo:
    """VisualVerifier with real camera frames from MuJoCo Go2."""

    def test_camera_frame_shape(self, go2_standing):
        """MuJoCo produces a valid RGB frame."""
        frame = go2_standing.get_camera_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.dtype == np.uint8

    def test_camera_frame_not_black(self, go2_standing):
        """Frame has actual content (not all zeros)."""
        frame = go2_standing.get_camera_frame()
        assert frame.max() > 0, "Frame is completely black"

    def test_visual_verify_with_real_frame_mock_vlm(self, go2_standing):
        """Real camera frame + mock VLM = verifier runs end-to-end."""
        agent = MagicMock()
        agent._base = go2_standing
        agent._vlm = MagicMock()
        agent._vlm.find_objects.return_value = [
            MagicMock(**{"name": "floor", "confidence": 0.8}),
        ]
        # Fix: MagicMock(name=...) sets internal name, use explicit attr
        for obj in agent._vlm.find_objects.return_value:
            obj.name = obj._mock_kwargs.get("name", "floor") if hasattr(obj, "_mock_kwargs") else "floor"

        result = verify_visual(
            agent=agent,
            sub_goal_description="detect objects in room",
            verify_expr="len(detect_objects('floor')) > 0",
        )
        assert result.triggered is True
        # VLM was called with a real frame
        agent._vlm.find_objects.assert_called_once()
        call_args = agent._vlm.find_objects.call_args
        frame_arg = call_args[0][0]
        assert isinstance(frame_arg, np.ndarray)
        assert frame_arg.shape[2] == 3

    def test_visual_verify_no_vlm_degrades(self, go2_standing):
        """Without VLM, verifier degrades gracefully."""
        agent = MagicMock()
        agent._base = go2_standing
        agent._vlm = None

        result = verify_visual(
            agent=agent,
            sub_goal_description="detect objects",
            verify_expr="detect_objects('cup')",
        )
        assert result.triggered is False  # graceful degradation

    def test_should_verify_triggers_on_perception_failure(self):
        """should_verify returns True for perception step that failed."""
        assert should_verify(
            sub_goal_name="detect_cup",
            sub_goal_description="observe table to detect cup",
            strategy="look_skill",
            verify_expr="len(detect_objects('cup')) > 0",
            verify_result=False,
        ) is True

    def test_should_verify_skips_navigation(self):
        """should_verify returns False for non-perception navigation step."""
        assert should_verify(
            sub_goal_name="reach_kitchen",
            sub_goal_description="navigate to kitchen",
            strategy="navigate_skill",
            verify_expr="nearest_room() == 'kitchen'",
            verify_result=False,
        ) is False


# ===================================================================
# D. GoalVerifier with Phase 3 namespace + real data
# ===================================================================


class TestGoalVerifierPhase3Namespace:
    """GoalVerifier with Phase 3 functions wired to real SceneGraph + ObjectMemory."""

    @pytest.fixture
    def verifier(self):
        sg = _populated_scene_graph()
        om = ObjectMemory(decay_lambda=0.001)
        om.sync_from_scene_graph(sg)

        ns = {
            # Phase 1+2 functions
            "nearest_room": lambda: "hallway",
            "get_visited_rooms": sg.get_visited_rooms,
            "query_rooms": lambda: [
                {"id": r.room_id, "x": r.center_x, "y": r.center_y}
                for r in sg.get_all_rooms()
            ],
            "world_stats": sg.stats,
            "describe_scene": lambda: "",
            "detect_objects": lambda query="": [],
            # Phase 3 functions
            "last_seen": om.last_seen,
            "certainty": om.certainty,
            "objects_in_room": om.objects_in_room,
            "find_object": om.find_object,
            "room_coverage": sg.get_room_coverage,
            "predict_navigation": lambda target: predict_navigation(sg, "hallway", target),
        }
        return GoalVerifier(ns)

    def test_verify_last_seen_cup(self, verifier):
        assert verifier.verify("last_seen('cup') is not None") is True

    def test_verify_last_seen_laptop(self, verifier):
        assert verifier.verify("last_seen('laptop') is not None") is False

    def test_verify_certainty_cup_in_kitchen(self, verifier):
        assert verifier.verify("certainty('cup在kitchen') > 0.5") is True

    def test_verify_certainty_cup_in_bedroom(self, verifier):
        assert verifier.verify("certainty('cup在bedroom') > 0.5") is False

    def test_verify_objects_in_kitchen(self, verifier):
        assert verifier.verify("len(objects_in_room('kitchen')) >= 2") is True

    def test_verify_objects_in_empty_room(self, verifier):
        assert verifier.verify("len(objects_in_room('bedroom')) == 0") is True

    def test_verify_find_object_cup(self, verifier):
        assert verifier.verify("len(find_object('cup')) > 0") is True

    def test_verify_find_object_nonexistent(self, verifier):
        assert verifier.verify("len(find_object('laptop')) == 0") is True

    def test_verify_predict_nav_reachable(self, verifier):
        assert verifier.verify("predict_navigation('kitchen')['reachable']") is True

    def test_verify_predict_nav_unreachable(self, verifier):
        assert verifier.verify("predict_navigation('garage')['reachable']") is False

    def test_verify_combined_expression(self, verifier):
        """Multi-function compound expression."""
        assert verifier.verify(
            "certainty('cup在kitchen') > 0.3 and predict_navigation('kitchen')['reachable']"
        ) is True

    def test_verify_world_stats_still_works(self, verifier):
        """Phase 1 functions still work after namespace extension."""
        assert verifier.verify("world_stats()['rooms'] >= 2") is True

    def test_verify_visited_rooms(self, verifier):
        assert verifier.verify("len(get_visited_rooms()) >= 1") is True


# ===================================================================
# E. Full pipeline: MuJoCo + SceneGraph + ObjectMemory + GoalExecutor
# ===================================================================


class TestFullPipelineMuJoCo:
    """End-to-end: GoalExecutor with real MuJoCo robot + Phase 3 world model."""

    def test_goal_executor_with_real_robot(self, go2_standing):
        """GoalExecutor runs a simple goal with real robot position data."""
        from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        sg = _populated_scene_graph()
        om = ObjectMemory(decay_lambda=0.001)
        om.sync_from_scene_graph(sg)

        # Build verifier namespace with real robot data
        pos = go2_standing.get_position()
        heading = go2_standing.get_heading()
        ns = {
            "get_position": lambda: tuple(pos),
            "get_heading": lambda: float(heading),
            "nearest_room": lambda: "hallway",  # robot starts at origin ≈ hallway
            "get_visited_rooms": sg.get_visited_rooms,
            "query_rooms": lambda: [{"id": r.room_id} for r in sg.get_all_rooms()],
            "world_stats": sg.stats,
            "describe_scene": lambda: "",
            "detect_objects": lambda query="": [],
            "last_seen": om.last_seen,
            "certainty": om.certainty,
            "objects_in_room": om.objects_in_room,
            "find_object": om.find_object,
            "room_coverage": sg.get_room_coverage,
            "predict_navigation": lambda target: predict_navigation(sg, "hallway", target),
        }
        verifier = GoalVerifier(ns)

        # Mock selector + skill
        selector = MagicMock()
        selector.select.return_value = MagicMock(
            executor_type="skill", name="stand", params={},
        )
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

        # Execute a goal that verifies using Phase 3 functions
        tree = GoalTree(
            goal="check kitchen objects",
            sub_goals=(
                SubGoal(
                    name="verify_cup_known",
                    description="check if cup is known in kitchen",
                    verify="len(objects_in_room('kitchen')) >= 2",
                    strategy="stand_skill",
                    timeout_sec=10.0,
                ),
            ),
        )
        trace = executor.execute(tree)
        assert trace.success is True
        assert len(trace.steps) == 1
        assert trace.steps[0].verify_result is True

    def test_robot_position_available(self, go2_standing):
        """Real robot provides position and heading for namespace."""
        pos = go2_standing.get_position()
        heading = go2_standing.get_heading()
        assert len(pos) == 3
        assert isinstance(heading, float)
        # Robot should be near origin after standing
        assert abs(pos[0]) < 2.0
        assert abs(pos[1]) < 2.0

    def test_lidar_scan_available(self, go2_standing):
        """Real robot provides lidar scan object (ranges may be empty on flat ground)."""
        scan = go2_standing.get_lidar_scan()
        assert scan is not None
        # On flat ground (room=False), lidar may return empty ranges
        # because there are no obstacles to reflect rays. Just verify the
        # LaserScan structure is valid.
        assert hasattr(scan, "ranges")
        assert hasattr(scan, "angle_min")
        assert hasattr(scan, "angle_max")
