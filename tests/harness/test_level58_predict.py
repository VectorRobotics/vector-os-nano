"""Level 58 — predict module TDD tests.

Pure rule-based state prediction using SceneGraph room topology.
All tests use mock SceneGraph — no MuJoCo, no hardware.

Mock topology:
    hallway <-> kitchen   (door at 2.5, 1.5)
    hallway <-> bedroom   (door at -1.0, 2.0)
    kitchen and bedroom are NOT directly connected.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli.cognitive.predict import (
    predict_exploration_value,
    predict_navigation,
    predict_room_after_door,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scene_graph() -> MagicMock:
    """Build a mock SceneGraph with 3-room topology."""
    sg = MagicMock()

    # Door positions
    _doors: dict[tuple[str, str], tuple[float, float]] = {
        ("hallway", "kitchen"): (2.5, 1.5),
        ("kitchen", "hallway"): (2.5, 1.5),
        ("hallway", "bedroom"): (-1.0, 2.0),
        ("bedroom", "hallway"): (-1.0, 2.0),
    }

    def _get_door(a: str, b: str) -> tuple[float, float] | None:
        key = tuple(sorted([a, b]))
        return _doors.get((key[0], key[1]))

    sg.get_door.side_effect = _get_door

    # BFS chain for get_door_chain
    def _get_door_chain(src: str, dst: str) -> list[tuple[float, float, str]]:
        if src == dst:
            centers = {
                "hallway": (0.0, 0.0),
                "kitchen": (3.0, 0.0),
                "bedroom": (-2.0, 0.0),
            }
            if src not in centers:
                return []
            cx, cy = centers[src]
            return [(cx, cy, src)]

        # Direct adjacency
        adjacency: dict[str, list[str]] = {
            "hallway": ["kitchen", "bedroom"],
            "kitchen": ["hallway"],
            "bedroom": ["hallway"],
        }

        if src not in adjacency:
            return []

        # BFS
        from collections import deque
        visited = {src}
        parent: dict[str, str] = {}
        queue: deque[str] = deque([src])
        found = False

        while queue:
            cur = queue.popleft()
            for nb in adjacency.get(cur, []):
                if nb in visited:
                    continue
                visited.add(nb)
                parent[nb] = cur
                if nb == dst:
                    found = True
                    break
                queue.append(nb)
            if found:
                break

        if not found:
            return []

        path: list[str] = []
        node = dst
        while node in parent:
            path.append(node)
            node = parent[node]
        path.append(src)
        path.reverse()

        centers = {
            "hallway": (0.0, 0.0),
            "kitchen": (3.0, 0.0),
            "bedroom": (-2.0, 0.0),
        }
        waypoints: list[tuple[float, float, str]] = []
        for i in range(len(path) - 1):
            door_pos = _get_door(path[i], path[i + 1])
            if door_pos is not None:
                waypoints.append((door_pos[0], door_pos[1], f"{path[i]}_{path[i+1]}_door"))
        if dst in centers:
            cx, cy = centers[dst]
            waypoints.append((cx, cy, dst))
        return waypoints

    sg.get_door_chain.side_effect = _get_door_chain

    # Room coverage
    _coverage: dict[str, float] = {
        "hallway": 0.5,
        "kitchen": 0.0,
        "bedroom": 0.9,
    }

    def _get_room_coverage(room_id: str) -> float:
        return _coverage.get(room_id, 0.0)

    sg.get_room_coverage.side_effect = _get_room_coverage

    # get_room returns a mock RoomNode
    _connected: dict[str, tuple[str, ...]] = {
        "hallway": ("kitchen", "bedroom"),
        "kitchen": ("hallway",),
        "bedroom": ("hallway",),
    }

    def _get_room(room_id: str) -> MagicMock | None:
        if room_id not in _connected:
            return None
        room = MagicMock()
        room.room_id = room_id
        room.connected_rooms = _connected[room_id]
        return room

    sg.get_room.side_effect = _get_room

    return sg


# ---------------------------------------------------------------------------
# predict_navigation
# ---------------------------------------------------------------------------


class TestPredictNavReachable:
    """hallway → kitchen: direct single-door path."""

    def test_reachable_true(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["reachable"] is True

    def test_door_count_one(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["door_count"] == 1

    def test_rooms_on_path(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["rooms_on_path"] == ["hallway", "kitchen"]

    def test_estimated_steps(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "hallway", "kitchen")
        # door_count + 1
        assert result["estimated_steps"] == 2


class TestPredictNavSameRoom:
    """kitchen → kitchen: no doors, already there."""

    def test_reachable(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "kitchen")
        assert result["reachable"] is True

    def test_door_count_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "kitchen")
        assert result["door_count"] == 0

    def test_estimated_steps_one(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "kitchen")
        assert result["estimated_steps"] == 1

    def test_rooms_on_path_contains_room(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "kitchen")
        assert "kitchen" in result["rooms_on_path"]


class TestPredictNavTwoDoors:
    """kitchen → bedroom: goes through hallway, two doors."""

    def test_reachable(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "bedroom")
        assert result["reachable"] is True

    def test_door_count_two(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "bedroom")
        assert result["door_count"] == 2

    def test_rooms_includes_hallway(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "bedroom")
        assert "hallway" in result["rooms_on_path"]

    def test_rooms_on_path_order(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "bedroom")
        path = result["rooms_on_path"]
        assert path[0] == "kitchen"
        assert path[-1] == "bedroom"


class TestPredictNavUnreachable:
    """kitchen → garage (not in graph): unreachable."""

    def test_reachable_false(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "garage")
        assert result["reachable"] is False

    def test_door_count_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "garage")
        assert result["door_count"] == 0

    def test_rooms_on_path_empty(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "garage")
        assert result["rooms_on_path"] == []


class TestPredictNavConfidence:
    """Confidence is 1.0 when path exists, 0.0 when not."""

    def test_confidence_reachable(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["confidence"] == 1.0

    def test_confidence_unreachable(self) -> None:
        sg = _make_scene_graph()
        result = predict_navigation(sg, "kitchen", "garage")
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# predict_room_after_door
# ---------------------------------------------------------------------------


class TestPredictRoomAfterDoorExists:
    """hallway → kitchen: door exists at (2.5, 1.5)."""

    def test_room_name(self) -> None:
        sg = _make_scene_graph()
        result = predict_room_after_door(sg, "hallway", "kitchen")
        assert result["room"] == "kitchen"

    def test_confidence_one(self) -> None:
        sg = _make_scene_graph()
        result = predict_room_after_door(sg, "hallway", "kitchen")
        assert result["confidence"] == 1.0

    def test_door_position(self) -> None:
        sg = _make_scene_graph()
        result = predict_room_after_door(sg, "hallway", "kitchen")
        assert result["door_position"] == (2.5, 1.5)


class TestPredictRoomAfterDoorNone:
    """kitchen → bedroom: no direct door."""

    def test_room_empty(self) -> None:
        sg = _make_scene_graph()
        result = predict_room_after_door(sg, "kitchen", "bedroom")
        assert result["room"] == ""

    def test_confidence_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_room_after_door(sg, "kitchen", "bedroom")
        assert result["confidence"] == 0.0

    def test_door_position_none(self) -> None:
        sg = _make_scene_graph()
        result = predict_room_after_door(sg, "kitchen", "bedroom")
        assert result["door_position"] is None


# ---------------------------------------------------------------------------
# predict_exploration_value
# ---------------------------------------------------------------------------


class TestPredictExplorationValueUnexplored:
    """kitchen: coverage=0.0, 1 connected room (hallway)."""

    def test_coverage_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "kitchen")
        assert result["coverage"] == 0.0

    def test_connected_rooms_count(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "kitchen")
        assert result["connected_rooms"] == 1

    def test_value_positive(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "kitchen")
        assert result["value"] > 0.5  # low coverage → high value

    def test_unexplored_neighbors_count(self) -> None:
        # kitchen's only neighbor is hallway (coverage=0.5, not < 0.1)
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "kitchen")
        assert result["unexplored_neighbors"] == 0


class TestPredictExplorationValueHighlyCovered:
    """bedroom: coverage=0.9 → low exploration value."""

    def test_coverage_high(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "bedroom")
        assert result["coverage"] == pytest.approx(0.9)

    def test_value_low(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "bedroom")
        assert result["value"] < 0.2  # well-explored → low value


class TestPredictExplorationValueUnknownRoom:
    """garage: not in graph → defaults."""

    def test_coverage_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "garage")
        assert result["coverage"] == 0.0

    def test_connected_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "garage")
        assert result["connected_rooms"] == 0

    def test_unexplored_zero(self) -> None:
        sg = _make_scene_graph()
        result = predict_exploration_value(sg, "garage")
        assert result["unexplored_neighbors"] == 0


# ---------------------------------------------------------------------------
# predict_navigation with empty graph
# ---------------------------------------------------------------------------


class TestPredictNavEmptyGraph:
    """Empty SceneGraph: all rooms unreachable."""

    def _make_empty_sg(self) -> MagicMock:
        sg = MagicMock()
        sg.get_door_chain.return_value = []
        sg.get_door.return_value = None
        sg.get_room_coverage.return_value = 0.0
        sg.get_room.return_value = None
        return sg

    def test_unreachable(self) -> None:
        sg = self._make_empty_sg()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["reachable"] is False

    def test_confidence_zero(self) -> None:
        sg = self._make_empty_sg()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["confidence"] == 0.0

    def test_door_count_zero(self) -> None:
        sg = self._make_empty_sg()
        result = predict_navigation(sg, "hallway", "kitchen")
        assert result["door_count"] == 0
