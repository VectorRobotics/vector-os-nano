"""T2 — Primitives API tests (AC-13 to AC-20 + additional).

TDD: write tests first, run RED, then implement GREEN.

Tests cover:
- AC-13: all primitive modules importable, functions exist with correct signatures
- AC-14: all functions have type hints and docstrings
- AC-15: no hardware (ctx.base=None) → locomotion.get_position() raises RuntimeError
- AC-16: locomotion.get_position() returns tuple matching mock
- AC-17: locomotion.get_heading() returns float matching mock
- AC-18: perception.describe_scene() returns non-empty string
- AC-19: navigation.nearest_room() returns "kitchen"
- AC-20: world.query_rooms() returns list with correct count
- Additional: world.get_visited_rooms() returns room list
- Additional: navigation.get_door_chain() wraps scene_graph correctly
- Additional: perception.detect_objects("cup") returns list of dicts
- Additional: world.world_stats() keys match expected
- Additional: init_primitives() sets context in all modules
- Additional: perception.scan_360() returns list of (angle, dist) tuples
- Additional: locomotion.stop() calls set_velocity(0,0,0)
- Additional: locomotion.stand() delegates to base.stand()
- Additional: locomotion.sit() delegates to base.sit()
- Additional: navigation.wait_until_near() returns True when already near
- Additional: world.query_objects() returns list
- Additional: world.path_between() returns list of tuples
- Additional: perception.capture_image() returns array from base
- Additional: perception.identify_room() returns (room_name, confidence) tuple
- Additional: perception.measure_distance() returns float
"""
from __future__ import annotations

import importlib
import inspect
import math
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock helpers — simulate hardware data shapes without real numpy
# ---------------------------------------------------------------------------


class MockLaserScan:
    """Minimal LaserScan mock matching BaseProtocol.get_lidar_scan() return shape."""

    def __init__(self, ranges: list[float], angle_min: float = -math.pi, angle_max: float = math.pi) -> None:
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_max = angle_max
        n = len(ranges)
        self.angle_increment = (angle_max - angle_min) / max(n - 1, 1)


class MockSceneDesc:
    def __init__(self, summary: str) -> None:
        self.summary = summary


class MockObj:
    def __init__(self, name: str, confidence: float) -> None:
        self.name = name
        self.confidence = confidence


class MockRoomId:
    def __init__(self, room: str, confidence: float) -> None:
        self.room = room
        self.confidence = confidence


class MockRoom:
    def __init__(self, room_id: str, x: float, y: float) -> None:
        self.room_id = room_id
        self.center_x = x
        self.center_y = y
        self.visit_count = 1


class FakeNdarray:
    """Fake numpy ndarray so tests don't need numpy installed."""

    def __init__(self, shape: tuple) -> None:
        self.shape = shape
        self.dtype = "uint8"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_base, mock_sg, mock_vlm) pre-configured."""
    mock_base = MagicMock()
    mock_base.get_position.return_value = [10.0, 5.0, 0.3]
    mock_base.get_heading.return_value = 1.57
    scan = MockLaserScan(ranges=[1.0] * 360)
    mock_base.get_lidar_scan.return_value = scan
    mock_base.get_camera_frame.return_value = FakeNdarray((240, 320, 3))
    mock_base.stand.return_value = True
    mock_base.sit.return_value = True

    mock_sg = MagicMock()
    mock_sg.nearest_room.return_value = "kitchen"
    mock_sg.get_all_rooms.return_value = [
        MockRoom("kitchen", 10.0, 5.0),
        MockRoom("hallway", 5.0, 5.0),
    ]
    mock_sg.get_door_chain.return_value = [(7.5, 5.0, "kitchen_door")]
    mock_sg.get_visited_rooms.return_value = ["kitchen", "hallway"]
    mock_sg.get_all_doors.return_value = {("kitchen", "hallway"): (7.5, 5.0)}
    mock_sg.find_objects_in_room.return_value = []
    mock_sg.stats.return_value = {"rooms": 2, "objects": 0, "visited_rooms": 2, "viewpoints": 0}

    mock_vlm = MagicMock()
    mock_vlm.describe_scene.return_value = MockSceneDesc(summary="A kitchen with countertops")
    mock_vlm.find_objects.return_value = [MockObj(name="cup", confidence=0.9)]
    mock_vlm.identify_room.return_value = MockRoomId(room="kitchen", confidence=0.95)

    return mock_base, mock_sg, mock_vlm


def _init_ctx(base: Any = None, sg: Any = None, vlm: Any = None) -> None:
    """Re-initialize primitives with fresh context."""
    from vector_os_nano.vcli.primitives import PrimitiveContext, init_primitives
    ctx = PrimitiveContext(base=base, scene_graph=sg, vlm=vlm)
    init_primitives(ctx)


# ---------------------------------------------------------------------------
# AC-13: importability and function existence
# ---------------------------------------------------------------------------


class TestAC13Importability:
    """AC-13: All primitive modules importable, functions exist."""

    def test_primitives_package_importable(self) -> None:
        import vector_os_nano.vcli.primitives as pkg
        assert pkg is not None

    def test_locomotion_module_importable(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        assert locomotion is not None

    def test_navigation_module_importable(self) -> None:
        from vector_os_nano.vcli.primitives import navigation
        assert navigation is not None

    def test_perception_module_importable(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        assert perception is not None

    def test_world_module_importable(self) -> None:
        from vector_os_nano.vcli.primitives import world
        assert world is not None

    def test_locomotion_functions_exist(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        for fn_name in [
            "get_position", "get_heading", "set_velocity", "stop",
            "walk_forward", "turn", "stand", "sit",
        ]:
            assert hasattr(locomotion, fn_name), f"locomotion.{fn_name} missing"
            assert callable(getattr(locomotion, fn_name))

    def test_navigation_functions_exist(self) -> None:
        from vector_os_nano.vcli.primitives import navigation
        for fn_name in [
            "nearest_room", "publish_goal", "wait_until_near",
            "get_door_chain", "navigate_to_room",
        ]:
            assert hasattr(navigation, fn_name), f"navigation.{fn_name} missing"
            assert callable(getattr(navigation, fn_name))

    def test_perception_functions_exist(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        for fn_name in [
            "capture_image", "describe_scene", "detect_objects",
            "identify_room", "measure_distance", "scan_360",
        ]:
            assert hasattr(perception, fn_name), f"perception.{fn_name} missing"
            assert callable(getattr(perception, fn_name))

    def test_world_functions_exist(self) -> None:
        from vector_os_nano.vcli.primitives import world
        for fn_name in [
            "query_rooms", "query_doors", "query_objects",
            "get_visited_rooms", "path_between", "world_stats",
        ]:
            assert hasattr(world, fn_name), f"world.{fn_name} missing"
            assert callable(getattr(world, fn_name))


# ---------------------------------------------------------------------------
# AC-14: type hints and docstrings
# ---------------------------------------------------------------------------


class TestAC14TypeHintsAndDocstrings:
    """AC-14: All functions have type hints and docstrings."""

    _MODULES_FUNCS = {
        "locomotion": [
            "get_position", "get_heading", "set_velocity", "stop",
            "walk_forward", "turn", "stand", "sit",
        ],
        "navigation": [
            "nearest_room", "publish_goal", "wait_until_near",
            "get_door_chain", "navigate_to_room",
        ],
        "perception": [
            "capture_image", "describe_scene", "detect_objects",
            "identify_room", "measure_distance", "scan_360",
        ],
        "world": [
            "query_rooms", "query_doors", "query_objects",
            "get_visited_rooms", "path_between", "world_stats",
        ],
    }

    def test_all_functions_have_docstrings(self) -> None:
        base_pkg = "vector_os_nano.vcli.primitives"
        for mod_name, fn_names in self._MODULES_FUNCS.items():
            mod = importlib.import_module(f"{base_pkg}.{mod_name}")
            for fn_name in fn_names:
                fn = getattr(mod, fn_name)
                assert fn.__doc__ and fn.__doc__.strip(), (
                    f"{mod_name}.{fn_name} has no docstring"
                )

    def test_all_functions_have_return_annotations(self) -> None:
        base_pkg = "vector_os_nano.vcli.primitives"
        for mod_name, fn_names in self._MODULES_FUNCS.items():
            mod = importlib.import_module(f"{base_pkg}.{mod_name}")
            for fn_name in fn_names:
                fn = getattr(mod, fn_name)
                sig = inspect.signature(fn)
                assert sig.return_annotation is not inspect.Parameter.empty, (
                    f"{mod_name}.{fn_name} missing return annotation"
                )


# ---------------------------------------------------------------------------
# AC-15: no hardware raises RuntimeError
# ---------------------------------------------------------------------------


class TestAC15NoHardwareError:
    """AC-15: ctx.base=None → locomotion functions raise RuntimeError."""

    def setup_method(self) -> None:
        _init_ctx(base=None)

    def test_get_position_raises_without_base(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        with pytest.raises(RuntimeError, match="[Nn]o hardware|[Nn]ot connected|[Nn]o base"):
            locomotion.get_position()

    def test_get_heading_raises_without_base(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        with pytest.raises(RuntimeError):
            locomotion.get_heading()

    def test_set_velocity_raises_without_base(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        with pytest.raises(RuntimeError):
            locomotion.set_velocity(0.1, 0.0, 0.0)

    def test_stop_raises_without_base(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        with pytest.raises(RuntimeError):
            locomotion.stop()


# ---------------------------------------------------------------------------
# AC-16: get_position returns tuple
# ---------------------------------------------------------------------------


class TestAC16GetPosition:
    """AC-16: locomotion.get_position() returns tuple matching mock."""

    def setup_method(self) -> None:
        mock_base, mock_sg, mock_vlm = _make_mocks()
        _init_ctx(base=mock_base, sg=mock_sg, vlm=mock_vlm)

    def test_get_position_returns_tuple(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        pos = locomotion.get_position()
        assert isinstance(pos, tuple)

    def test_get_position_three_elements(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        pos = locomotion.get_position()
        assert len(pos) == 3

    def test_get_position_matches_mock(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        pos = locomotion.get_position()
        assert pos == (10.0, 5.0, 0.3)


# ---------------------------------------------------------------------------
# AC-17: get_heading returns float
# ---------------------------------------------------------------------------


class TestAC17GetHeading:
    """AC-17: locomotion.get_heading() returns float matching mock."""

    def setup_method(self) -> None:
        mock_base, mock_sg, mock_vlm = _make_mocks()
        _init_ctx(base=mock_base, sg=mock_sg, vlm=mock_vlm)

    def test_get_heading_returns_float(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        h = locomotion.get_heading()
        assert isinstance(h, float)

    def test_get_heading_matches_mock(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        h = locomotion.get_heading()
        assert abs(h - 1.57) < 1e-6


# ---------------------------------------------------------------------------
# AC-18: describe_scene returns non-empty string
# ---------------------------------------------------------------------------


class TestAC18DescribeScene:
    """AC-18: perception.describe_scene() returns non-empty string."""

    def setup_method(self) -> None:
        mock_base, mock_sg, mock_vlm = _make_mocks()
        _init_ctx(base=mock_base, sg=mock_sg, vlm=mock_vlm)

    def test_describe_scene_returns_string(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        result = perception.describe_scene()
        assert isinstance(result, str)

    def test_describe_scene_non_empty(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        result = perception.describe_scene()
        assert len(result) > 0

    def test_describe_scene_matches_mock_summary(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        result = perception.describe_scene()
        assert "kitchen" in result.lower()


# ---------------------------------------------------------------------------
# AC-19: nearest_room returns "kitchen"
# ---------------------------------------------------------------------------


class TestAC19NearestRoom:
    """AC-19: navigation.nearest_room() returns "kitchen" matching scene_graph mock."""

    def setup_method(self) -> None:
        mock_base, mock_sg, mock_vlm = _make_mocks()
        _init_ctx(base=mock_base, sg=mock_sg, vlm=mock_vlm)

    def test_nearest_room_returns_string(self) -> None:
        from vector_os_nano.vcli.primitives import navigation
        result = navigation.nearest_room()
        assert isinstance(result, str) or result is None

    def test_nearest_room_returns_kitchen(self) -> None:
        from vector_os_nano.vcli.primitives import navigation
        result = navigation.nearest_room()
        assert result == "kitchen"


# ---------------------------------------------------------------------------
# AC-20: query_rooms returns list with correct count
# ---------------------------------------------------------------------------


class TestAC20QueryRooms:
    """AC-20: world.query_rooms() returns list with correct count."""

    def setup_method(self) -> None:
        mock_base, mock_sg, mock_vlm = _make_mocks()
        _init_ctx(base=mock_base, sg=mock_sg, vlm=mock_vlm)

    def test_query_rooms_returns_list(self) -> None:
        from vector_os_nano.vcli.primitives import world
        rooms = world.query_rooms()
        assert isinstance(rooms, list)

    def test_query_rooms_count(self) -> None:
        from vector_os_nano.vcli.primitives import world
        rooms = world.query_rooms()
        assert len(rooms) == 2

    def test_query_rooms_dict_shape(self) -> None:
        from vector_os_nano.vcli.primitives import world
        rooms = world.query_rooms()
        for r in rooms:
            assert "id" in r
            assert "x" in r
            assert "y" in r


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


class TestAdditionalPrimitives:
    """Additional coverage for all four modules."""

    def setup_method(self) -> None:
        mock_base, mock_sg, mock_vlm = _make_mocks()
        self.mock_base = mock_base
        self.mock_sg = mock_sg
        self.mock_vlm = mock_vlm
        _init_ctx(base=mock_base, sg=mock_sg, vlm=mock_vlm)

    def test_world_get_visited_rooms(self) -> None:
        from vector_os_nano.vcli.primitives import world
        visited = world.get_visited_rooms()
        assert isinstance(visited, list)
        assert "kitchen" in visited
        assert "hallway" in visited

    def test_navigation_get_door_chain(self) -> None:
        from vector_os_nano.vcli.primitives import navigation
        chain = navigation.get_door_chain("kitchen", "hallway")
        assert isinstance(chain, list)
        assert len(chain) > 0
        x, y, label = chain[0]
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(label, str)

    def test_perception_detect_objects_returns_list_of_dicts(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        results = perception.detect_objects("cup")
        assert isinstance(results, list)
        assert len(results) > 0
        obj = results[0]
        assert "name" in obj
        assert "confidence" in obj
        assert obj["name"] == "cup"
        assert isinstance(obj["confidence"], float)

    def test_world_stats_keys(self) -> None:
        from vector_os_nano.vcli.primitives import world
        stats = world.world_stats()
        assert isinstance(stats, dict)
        assert "rooms" in stats
        assert "objects" in stats
        assert "visited" in stats

    def test_init_primitives_sets_context_in_all_modules(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion, navigation, perception, world
        # After setup_method calls _init_ctx, all modules should have _ctx set
        assert locomotion._ctx is not None
        assert navigation._ctx is not None
        assert perception._ctx is not None
        assert world._ctx is not None

    def test_perception_scan_360_returns_list_of_tuples(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        scan = perception.scan_360()
        assert isinstance(scan, list)
        assert len(scan) > 0
        angle, dist = scan[0]
        assert isinstance(angle, float)
        assert isinstance(dist, float)

    def test_locomotion_stop_calls_set_velocity_zero(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        locomotion.stop()
        self.mock_base.set_velocity.assert_called_once_with(0.0, 0.0, 0.0)

    def test_locomotion_stand_delegates_to_base(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        result = locomotion.stand()
        self.mock_base.stand.assert_called_once()
        assert result is True

    def test_locomotion_sit_delegates_to_base(self) -> None:
        from vector_os_nano.vcli.primitives import locomotion
        result = locomotion.sit()
        self.mock_base.sit.assert_called_once()
        assert result is True

    def test_navigation_wait_until_near_already_near(self) -> None:
        from vector_os_nano.vcli.primitives import navigation
        # Robot at (10, 5), target at (10, 5) → distance = 0 → immediate True
        result = navigation.wait_until_near(10.0, 5.0, tolerance=1.0, timeout=5.0)
        assert result is True

    def test_world_query_objects_returns_list(self) -> None:
        from vector_os_nano.vcli.primitives import world
        objs = world.query_objects()
        assert isinstance(objs, list)

    def test_world_path_between_returns_list_of_tuples(self) -> None:
        from vector_os_nano.vcli.primitives import world
        path = world.path_between("kitchen", "hallway")
        assert isinstance(path, list)
        for item in path:
            assert len(item) == 2

    def test_perception_capture_image_returns_array(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        frame = perception.capture_image()
        # Should return the FakeNdarray with shape (240, 320, 3)
        assert hasattr(frame, "shape")
        assert frame.shape == (240, 320, 3)

    def test_perception_identify_room_returns_tuple(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        result = perception.identify_room()
        assert isinstance(result, tuple)
        room_name, confidence = result
        assert isinstance(room_name, str)
        assert isinstance(confidence, float)
        assert room_name == "kitchen"
        assert abs(confidence - 0.95) < 1e-6

    def test_perception_measure_distance_returns_float(self) -> None:
        from vector_os_nano.vcli.primitives import perception
        dist = perception.measure_distance(0.0)
        assert isinstance(dist, float)
        assert dist == 1.0  # mock returns [1.0]*360
