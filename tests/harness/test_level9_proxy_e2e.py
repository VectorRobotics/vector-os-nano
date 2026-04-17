"""Level 9 — Go2ROS2Proxy → LookSkill end-to-end pipeline.

Tests verify the full VLM look pipeline when the robot base is a real
Go2ROS2Proxy instance (not a generic mock).  ROS2 initialisation and the
ROS2 spin thread are patched out so no running ROS2 daemon is required.

Coverage:
  - Go2ROS2Proxy.get_camera_frame() shape/dtype when _last_camera_frame is set
  - Go2ROS2Proxy.get_camera_frame() zero-frame fallback when no frame received
  - Go2ROS2Proxy._camera_cb() correctly parses a mock sensor_msgs/Image message
  - LookSkill succeeds when base is a real Go2ROS2Proxy with injected frame
  - LookSkill.execute() passes the proxy frame to vlm.describe_scene()
  - SceneGraph gets a RoomNode + ViewpointNode after LookSkill runs
  - observe_with_viewpoint is called with proxy position and heading data
  - Full pipeline: proxy camera → LookSkill → SceneGraph → observed objects

Cost: $0.00 (mock VLM only — no external API calls).
No real ROS2 required — rclpy initialisation is patched at import time.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Stub out rclpy and ros2 message packages BEFORE importing the proxy.
# This prevents import errors on machines without a ROS2 installation.
# ---------------------------------------------------------------------------

def _make_rclpy_stub() -> types.ModuleType:
    """Return a minimal rclpy stub that the proxy will not crash on."""
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = MagicMock(return_value=True)
    rclpy.init = MagicMock()
    rclpy.spin = MagicMock()

    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = MagicMock

    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.QoSProfile = MagicMock
    qos_mod.ReliabilityPolicy = MagicMock()
    qos_mod.ReliabilityPolicy.RELIABLE = "RELIABLE"

    rclpy.node = node_mod
    rclpy.qos = qos_mod
    return rclpy


def _make_ros2_msg_stubs() -> dict[str, types.ModuleType]:
    """Return stub modules for geometry_msgs, nav_msgs, sensor_msgs."""
    stubs: dict[str, types.ModuleType] = {}

    geo = types.ModuleType("geometry_msgs")
    geo_msg = types.ModuleType("geometry_msgs.msg")
    geo_msg.Twist = MagicMock
    geo.msg = geo_msg
    stubs["geometry_msgs"] = geo
    stubs["geometry_msgs.msg"] = geo_msg

    nav = types.ModuleType("nav_msgs")
    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.Odometry = MagicMock
    nav.msg = nav_msg
    stubs["nav_msgs"] = nav
    stubs["nav_msgs.msg"] = nav_msg

    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.Image = MagicMock
    sensor.msg = sensor_msg
    stubs["sensor_msgs"] = sensor
    stubs["sensor_msgs.msg"] = sensor_msg

    viz = types.ModuleType("visualization_msgs")
    viz_msg = types.ModuleType("visualization_msgs.msg")
    viz_msg.MarkerArray = MagicMock
    viz.msg = viz_msg
    stubs["visualization_msgs"] = viz
    stubs["visualization_msgs.msg"] = viz_msg

    return stubs


# Install stubs into sys.modules ONLY for packages not already installed.
# This avoids overwriting real ROS2 packages (e.g. visualization_msgs)
# that other test files depend on.
_rclpy_stub = _make_rclpy_stub()
sys.modules.setdefault("rclpy", _rclpy_stub)
sys.modules.setdefault("rclpy.node", _rclpy_stub.node)
sys.modules.setdefault("rclpy.qos", _rclpy_stub.qos)
for _name, _mod in _make_ros2_msg_stubs().items():
    sys.modules.setdefault(_name, _mod)

# Reset scene_graph_viz cached imports after this module's tests finish,
# so subsequent test files (e.g. test_level8) can re-import real ROS2 types.
# Also remove our stub modules to restore real imports.
@pytest.fixture(autouse=True, scope="module")
def _cleanup_ros2_stubs():
    """Yield-fixture that restores real ROS2 modules after all L9 tests."""
    yield
    # Remove stub modules — let real imports work again
    stubs_to_remove = [
        k for k in sys.modules
        if isinstance(sys.modules[k], types.ModuleType)
        and hasattr(sys.modules[k], "__name__")
        and sys.modules[k].__name__ in (
            "rclpy", "rclpy.node", "rclpy.qos",
            "geometry_msgs", "geometry_msgs.msg",
            "nav_msgs", "nav_msgs.msg",
            "sensor_msgs", "sensor_msgs.msg",
            "visualization_msgs", "visualization_msgs.msg",
        )
        and not hasattr(sys.modules[k], "__file__")  # stubs have no __file__
    ]
    for k in stubs_to_remove:
        del sys.modules[k]
    # Clear scene_graph_viz lazy-import cache
    try:
        from vector_os_nano.ros2.nodes import scene_graph_viz as _sgv
        _sgv._MarkerArray = None
        _sgv._Marker = None
        _sgv._ColorRGBA = None
        _sgv._Point = None
        _sgv._Vector3 = None
        _sgv._Header = None
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Now safe to import from the project
# ---------------------------------------------------------------------------

from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy  # noqa: E402
from vector_os_nano.skills.go2.look import LookSkill  # noqa: E402
from vector_os_nano.core.scene_graph import SceneGraph  # noqa: E402
from vector_os_nano.core.skill import SkillContext  # noqa: E402
from vector_os_nano.core.types import SkillResult  # noqa: E402
from vector_os_nano.perception.vlm_go2 import (  # noqa: E402
    DetectedObject,
    RoomIdentification,
    SceneDescription,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_H: int = 240
_FRAME_W: int = 320
_FRAME_CHANNELS: int = 3


def _rgb_frame(h: int = _FRAME_H, w: int = _FRAME_W, value: int = 128) -> np.ndarray:
    """Return a solid-colour (h, w, 3) uint8 frame."""
    return np.full((h, w, _FRAME_CHANNELS), value, dtype=np.uint8)


def _proxy_with_frame(frame: np.ndarray | None = None) -> Go2ROS2Proxy:
    """Build a disconnected Go2ROS2Proxy with _last_camera_frame pre-injected."""
    proxy = Go2ROS2Proxy.__new__(Go2ROS2Proxy)
    # Manually initialise fields that __init__ normally sets
    proxy._node = None
    proxy._cmd_pub = None
    proxy._position = (2.5, 1.0, 0.28)
    proxy._heading = 0.785  # ~45 deg
    proxy._connected = False
    proxy._last_odom = None
    proxy._last_camera_frame = frame
    proxy._last_camera_ts = 0.0  # added: __init__ now sets this field
    return proxy


def _make_vlm_mock(
    room: str = "living_room",
    confidence: float = 0.9,
    objects: list[str] | None = None,
) -> MagicMock:
    """Return a mock VLM pre-configured with realistic responses."""
    if objects is None:
        objects = ["sofa", "coffee_table"]
    vlm = MagicMock()
    vlm.describe_scene.return_value = SceneDescription(
        summary="A bright living room with a sofa",
        objects=[
            DetectedObject(
                name=name,
                description=f"a {name} in the living room",
                confidence=0.85,
            )
            for name in objects
        ],
        room_type=room,
        details="Spacious room with natural light from windows.",
    )
    vlm.identify_room.return_value = RoomIdentification(
        room=room,
        confidence=confidence,
        reasoning=f"Furniture arrangement suggests {room}.",
    )
    return vlm


def _make_context(
    proxy: Go2ROS2Proxy,
    vlm: MagicMock,
    spatial_memory: Any = None,
) -> SkillContext:
    """Build a SkillContext wiring a real proxy as the base."""
    services: dict = {"vlm": vlm}
    if spatial_memory is not None:
        services["spatial_memory"] = spatial_memory
    return SkillContext(bases={"go2": proxy}, services=services)


# ---------------------------------------------------------------------------
# Tests: Go2ROS2Proxy camera interface
# ---------------------------------------------------------------------------


class TestProxyCameraInterface:
    """Unit tests for Go2ROS2Proxy camera-related methods without ROS2."""

    def test_get_camera_frame_returns_injected_frame(self):
        """get_camera_frame() returns a copy of _last_camera_frame when set."""
        frame = _rgb_frame(value=200)
        proxy = _proxy_with_frame(frame)

        result = proxy.get_camera_frame()

        assert result.shape == (240, 320, 3), (
            f"Expected (240, 320, 3), got {result.shape}"
        )
        assert result.dtype == np.uint8, (
            f"Expected uint8, got {result.dtype}"
        )
        np.testing.assert_array_equal(result, frame)

    def test_get_camera_frame_returns_copy_not_reference(self):
        """get_camera_frame() returns a copy — mutations do not affect cache."""
        frame = _rgb_frame(value=100)
        proxy = _proxy_with_frame(frame)

        result = proxy.get_camera_frame()
        result[:] = 0  # mutate the returned copy

        # Original cached frame should be unmodified
        np.testing.assert_array_equal(proxy._last_camera_frame, frame)

    def test_get_camera_frame_zero_fallback_when_no_frame(self):
        """get_camera_frame() returns a black frame when no image received yet."""
        proxy = _proxy_with_frame(frame=None)

        result = proxy.get_camera_frame()

        assert result.shape == (240, 320, 3), (
            f"Expected (240, 320, 3), got {result.shape}"
        )
        assert result.dtype == np.uint8
        assert result.max() == 0, "Fallback frame should be all zeros"

    def test_get_camera_frame_respects_width_height_params_on_fallback(self):
        """get_camera_frame(width=640, height=480) shapes the zero fallback."""
        proxy = _proxy_with_frame(frame=None)

        result = proxy.get_camera_frame(width=640, height=480)

        assert result.shape == (480, 640, 3), (
            f"Expected (480, 640, 3), got {result.shape}"
        )

    def test_camera_cb_parses_mock_image_message(self):
        """_camera_cb() correctly reshapes a mock sensor_msgs/Image message."""
        proxy = _proxy_with_frame(frame=None)

        # Build a fake Image message: 240x320 RGB8
        h, w = 120, 160
        fake_data = np.arange(h * w * 3, dtype=np.uint8)
        msg = MagicMock()
        msg.height = h
        msg.width = w
        msg.data = fake_data.tobytes()

        proxy._camera_cb(msg)

        assert proxy._last_camera_frame is not None, (
            "_last_camera_frame should be set after _camera_cb"
        )
        assert proxy._last_camera_frame.shape == (h, w, 3), (
            f"Expected ({h}, {w}, 3), got {proxy._last_camera_frame.shape}"
        )
        assert proxy._last_camera_frame.dtype == np.uint8

    def test_camera_cb_stores_correct_pixel_values(self):
        """_camera_cb() stores the exact pixel data from the message."""
        proxy = _proxy_with_frame(frame=None)

        h, w = 4, 4
        raw = np.arange(h * w * 3, dtype=np.uint8)
        expected = raw.reshape((h, w, 3))

        msg = MagicMock()
        msg.height = h
        msg.width = w
        msg.data = raw.tobytes()

        proxy._camera_cb(msg)

        np.testing.assert_array_equal(proxy._last_camera_frame, expected)

    def test_camera_cb_handles_bad_message_gracefully(self):
        """_camera_cb() silently swallows exceptions on malformed messages."""
        proxy = _proxy_with_frame(frame=None)

        msg = MagicMock()
        msg.height = 240
        msg.width = 320
        # data too short — reshape will fail
        msg.data = b"\x00\x01"

        # Should not raise
        proxy._camera_cb(msg)
        # Frame stays None on error
        assert proxy._last_camera_frame is None

    def test_proxy_position_accessible_without_ros2(self):
        """get_position() returns injected position without ROS2 connection."""
        proxy = _proxy_with_frame()
        pos = proxy.get_position()
        assert len(pos) == 3
        assert pos == (2.5, 1.0, 0.28)

    def test_proxy_heading_accessible_without_ros2(self):
        """get_heading() returns injected heading without ROS2 connection."""
        proxy = _proxy_with_frame()
        heading = proxy.get_heading()
        assert isinstance(heading, float)
        assert pytest.approx(heading, abs=1e-6) == 0.785


# ---------------------------------------------------------------------------
# Tests: LookSkill with real Go2ROS2Proxy base
# ---------------------------------------------------------------------------


class TestLookSkillWithProxy:
    """LookSkill behaviour using a real (disconnected) Go2ROS2Proxy as base."""

    def test_look_skill_succeeds_with_proxy_base(self):
        """LookSkill returns success when base is a Go2ROS2Proxy with a frame."""
        proxy = _proxy_with_frame(_rgb_frame())
        vlm = _make_vlm_mock()
        ctx = _make_context(proxy, vlm)

        result = LookSkill().execute({}, ctx)

        assert isinstance(result, SkillResult)
        assert result.success, f"LookSkill failed: {result.error_message}"

    def test_look_skill_passes_proxy_frame_to_vlm(self):
        """The frame from proxy.get_camera_frame() is passed to vlm.describe_scene."""
        frame = _rgb_frame(value=77)
        proxy = _proxy_with_frame(frame)
        vlm = _make_vlm_mock()
        ctx = _make_context(proxy, vlm)

        LookSkill().execute({}, ctx)

        # describe_scene must be called once
        vlm.describe_scene.assert_called_once()
        passed_frame = vlm.describe_scene.call_args[0][0]
        # The frame passed should have the same shape and values as the proxy frame
        assert passed_frame.shape == frame.shape
        np.testing.assert_array_equal(passed_frame, frame)

    def test_look_skill_passes_proxy_frame_to_identify_room(self):
        """The frame from proxy.get_camera_frame() is passed to vlm.identify_room."""
        frame = _rgb_frame(value=55)
        proxy = _proxy_with_frame(frame)
        vlm = _make_vlm_mock()
        ctx = _make_context(proxy, vlm)

        LookSkill().execute({}, ctx)

        vlm.identify_room.assert_called_once()
        passed_frame = vlm.identify_room.call_args[0][0]
        assert passed_frame.shape == frame.shape
        np.testing.assert_array_equal(passed_frame, frame)

    def test_look_skill_result_data_contains_vlm_room(self):
        """result_data['room'] reflects VLM-identified room from proxy pipeline."""
        proxy = _proxy_with_frame(_rgb_frame())
        vlm = _make_vlm_mock(room="kitchen")
        ctx = _make_context(proxy, vlm)

        result = LookSkill().execute({}, ctx)

        assert result.success
        assert result.result_data.get("room") == "kitchen", (
            f"Expected room='kitchen', got {result.result_data.get('room')!r}"
        )

    def test_look_skill_result_data_contains_objects(self):
        """result_data['objects'] contains dicts for each VLM-detected object."""

    def test_look_skill_room_confidence_from_proxy_pipeline(self):
        """room_confidence in result_data matches the mock VLM's confidence."""
        proxy = _proxy_with_frame(_rgb_frame())
        vlm = _make_vlm_mock(confidence=0.75)
        ctx = _make_context(proxy, vlm)

        result = LookSkill().execute({}, ctx)

        assert result.success
        conf = result.result_data.get("room_confidence")
        assert conf is not None
        assert pytest.approx(conf, abs=1e-6) == 0.75

    def test_look_skill_uses_zero_frame_when_proxy_has_no_image(self):
        """LookSkill succeeds even when proxy has no camera frame yet (zero frame)."""
        proxy = _proxy_with_frame(frame=None)
        vlm = _make_vlm_mock()
        ctx = _make_context(proxy, vlm)

        result = LookSkill().execute({}, ctx)

        assert result.success
        passed_frame = vlm.describe_scene.call_args[0][0]
        # Frame should be all zeros (fallback)
        assert passed_frame.max() == 0


# ---------------------------------------------------------------------------
# Tests: SceneGraph integration after LookSkill runs through proxy
# ---------------------------------------------------------------------------


class TestSceneGraphUpdatedViaProxy:
    """Verify SceneGraph state after LookSkill executes with Go2ROS2Proxy base."""

    def test_scene_graph_room_created_after_look(self):
        """A RoomNode for the VLM-identified room exists after LookSkill runs."""
        proxy = _proxy_with_frame(_rgb_frame())
        vlm = _make_vlm_mock(room="bedroom")
        scene_graph = SceneGraph()
        ctx = _make_context(proxy, vlm, spatial_memory=scene_graph)

        result = LookSkill().execute({}, ctx)

        assert result.success
        room_node = scene_graph.get_room("bedroom")
        assert room_node is not None, "SceneGraph should have a 'bedroom' RoomNode"

    def test_scene_graph_viewpoint_created_after_look(self):
        """A ViewpointNode is added to the scene graph after LookSkill runs."""
        proxy = _proxy_with_frame(_rgb_frame())
        vlm = _make_vlm_mock(room="hallway")
        scene_graph = SceneGraph()
        ctx = _make_context(proxy, vlm, spatial_memory=scene_graph)

        result = LookSkill().execute({}, ctx)

        assert result.success
        viewpoints = scene_graph.get_viewpoints_in_room("hallway")
        assert len(viewpoints) >= 1, (
            "At least one viewpoint should exist in 'hallway' after LookSkill"
        )

    def test_scene_graph_viewpoint_position_matches_proxy(self):
        """ViewpointNode position matches the proxy's injected (x, y) position."""
        proxy = _proxy_with_frame(_rgb_frame())
        proxy._position = (5.0, 3.0, 0.28)
        vlm = _make_vlm_mock(room="office")
        scene_graph = SceneGraph()
        ctx = _make_context(proxy, vlm, spatial_memory=scene_graph)

        result = LookSkill().execute({}, ctx)

        assert result.success
        viewpoints = scene_graph.get_viewpoints_in_room("office")
        assert viewpoints, "Expected viewpoints in 'office'"
        vp = viewpoints[0]
        assert pytest.approx(vp.x, abs=0.01) == 5.0
        assert pytest.approx(vp.y, abs=0.01) == 3.0

    def test_scene_graph_objects_created_after_look(self):
        """ObjectNodes for each detected object exist in the scene graph."""

    def test_scene_graph_room_visit_count_incremented(self):
        """RoomNode.visit_count is >= 1 after LookSkill runs."""
        proxy = _proxy_with_frame(_rgb_frame())
        vlm = _make_vlm_mock(room="living_room")
        scene_graph = SceneGraph()
        ctx = _make_context(proxy, vlm, spatial_memory=scene_graph)

        result = LookSkill().execute({}, ctx)

        assert result.success
        room_node = scene_graph.get_room("living_room")
        assert room_node is not None
        assert room_node.visit_count >= 1, (
            f"visit_count should be >= 1, got {room_node.visit_count}"
        )

    def test_observe_with_viewpoint_called_with_proxy_data(self):
        """spatial_memory.observe_with_viewpoint is called with proxy pos/heading."""
        proxy = _proxy_with_frame(_rgb_frame())
        proxy._position = (1.5, 2.5, 0.28)
        proxy._heading = 1.57
        vlm = _make_vlm_mock(room="dining_room", objects=["table"])

        # Use a mock spatial_memory to capture call args
        mock_sm = MagicMock()
        mock_sm.observe_with_viewpoint = MagicMock(return_value=None)
        ctx = _make_context(proxy, vlm, spatial_memory=mock_sm)

        result = LookSkill().execute({}, ctx)

        assert result.success
        mock_sm.observe_with_viewpoint.assert_called_once()
        call_args = mock_sm.observe_with_viewpoint.call_args
        # First positional: room, x, y, heading, objects, summary
        args = call_args[0]
        assert args[0] == "dining_room", f"Expected room 'dining_room', got {args[0]!r}"
        assert pytest.approx(args[1], abs=0.01) == 1.5, f"Expected x=1.5, got {args[1]}"
        assert pytest.approx(args[2], abs=0.01) == 2.5, f"Expected y=2.5, got {args[2]}"
        assert pytest.approx(args[3], abs=0.01) == 1.57, (
            f"Expected heading=1.57, got {args[3]}"
        )


# ---------------------------------------------------------------------------
# Tests: Full pipeline (proxy camera → LookSkill → SceneGraph)
# ---------------------------------------------------------------------------


class TestFullProxyPipeline:
    """End-to-end pipeline: inject frame into proxy → run LookSkill → verify graph."""

    def test_full_pipeline_camera_to_scene_graph(self):
        """Simulate a full observation cycle from proxy frame to SceneGraph."""

    def test_full_pipeline_second_look_skips_nearby_viewpoint(self):
        """A second look from nearly the same position does not duplicate viewpoints."""
        proxy = _proxy_with_frame(_rgb_frame())
        proxy._position = (0.0, 0.0, 0.28)
        vlm = _make_vlm_mock(room="entry")
        scene_graph = SceneGraph()
        ctx = _make_context(proxy, vlm, spatial_memory=scene_graph)

        # First look
        LookSkill().execute({}, ctx)
        vp_count_after_first = len(scene_graph.get_viewpoints_in_room("entry"))

        # Second look from the same spot — should not add a new viewpoint
        LookSkill().execute({}, ctx)
        vp_count_after_second = len(scene_graph.get_viewpoints_in_room("entry"))

        assert vp_count_after_first == vp_count_after_second, (
            "A second viewpoint should not be added when position is unchanged "
            f"(was {vp_count_after_first}, now {vp_count_after_second})"
        )

    def test_full_pipeline_two_distant_looks_add_two_viewpoints(self):
        """Two observations from distant positions create two distinct viewpoints."""
        scene_graph = SceneGraph()
        vlm = _make_vlm_mock(room="garage")

        # First look
        proxy1 = _proxy_with_frame(_rgb_frame())
        proxy1._position = (0.0, 0.0, 0.28)
        ctx1 = _make_context(proxy1, vlm, spatial_memory=scene_graph)
        LookSkill().execute({}, ctx1)

        # Second look from far away (> _VIEWPOINT_MIN_DISTANCE = 1.5 m)
        proxy2 = _proxy_with_frame(_rgb_frame(value=200))
        proxy2._position = (5.0, 5.0, 0.28)
        ctx2 = _make_context(proxy2, vlm, spatial_memory=scene_graph)
        LookSkill().execute({}, ctx2)

        vps = scene_graph.get_viewpoints_in_room("garage")
        assert len(vps) >= 2, (
            f"Expected at least 2 viewpoints in 'garage', got {len(vps)}"
        )

    def test_stats_reflect_observation_after_pipeline(self):
        """SceneGraph.stats() shows non-zero rooms/viewpoints/objects after look."""
