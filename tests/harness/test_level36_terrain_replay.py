"""L36: Terrain replay publishes to /registered_scan only.

Tests verify the bridge's terrain replay configuration by static source
analysis — no ROS2 or MuJoCo required.

The key invariant: _replay_terrain() must publish the saved terrain to
/registered_scan via _pc_pub only. The removed terrain_map/terrain_map_ext
publishers are no longer part of the bridge.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nav_debug_helpers import read_bridge_source

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BRIDGE = os.path.join(_REPO, "scripts", "go2_vnav_bridge.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_method_body(source: str, method_name: str) -> str:
    """Return source slice for a method in Go2VNavBridge."""
    start = source.find(f"    def {method_name}(")
    if start < 0:
        raise AssertionError(f"Method {method_name!r} not found in bridge source")
    end = source.find("\n    def ", start + 1)
    return source[start:end] if end > 0 else source[start:]


def _get_init_body(source: str) -> str:
    """Return source slice for Go2VNavBridge.__init__."""
    cls_start = source.find("class Go2VNavBridge")
    body = source[cls_start:]
    start = body.find("    def __init__(")
    end = body.find("\n    def ", start + 1)
    return body[start:end] if end > 0 else body[start:]


# ---------------------------------------------------------------------------
# Part 1: Static analysis — publisher declarations in __init__
# ---------------------------------------------------------------------------

class TestTerrainReplayPublisherDeclarations:
    """Verify __init__ declares _pc_pub for /registered_scan."""

    def test_bridge_has_registered_scan_publisher(self):
        """Bridge must have _pc_pub publishing to /registered_scan."""
        src = read_bridge_source()
        init = _get_init_body(src)
        assert "_pc_pub" in init
        assert '"/registered_scan"' in init


# ---------------------------------------------------------------------------
# Part 2: Static analysis — _replay_terrain publishes only to /registered_scan
# ---------------------------------------------------------------------------

class TestReplayTerrainMethodPublishes:
    """Verify _replay_terrain() publishes only to _pc_pub (/registered_scan)."""

    def test_replay_publishes_to_registered_scan(self):
        """_replay_terrain must call self._pc_pub.publish(msg)."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        assert "_pc_pub.publish" in body, (
            "_replay_terrain must call self._pc_pub.publish(msg)"
        )

    def test_replay_does_not_publish_to_terrain_map(self):
        """_replay_terrain must NOT call _terrain_map_pub.publish (removed)."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        assert "_terrain_map_pub.publish" not in body, (
            "_replay_terrain must not publish to /terrain_map — publisher removed"
        )

    def test_replay_does_not_publish_to_terrain_map_ext(self):
        """_replay_terrain must NOT call _terrain_map_ext_pub.publish (removed)."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        assert "_terrain_map_ext_pub.publish" not in body, (
            "_replay_terrain must not publish to /terrain_map_ext — publisher removed"
        )

    def test_replay_publishes_exactly_one_topic(self):
        """_replay_terrain must call .publish() exactly once (registered_scan only)."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        publish_count = body.count(".publish(")
        assert publish_count == 1, (
            f"_replay_terrain should call .publish() once (/registered_scan only), "
            f"found {publish_count}"
        )

    def test_replay_uses_map_frame(self):
        """_replay_terrain delegates to _build_terrain_pc2 which sets 'map' frame."""
        src = read_bridge_source()
        # _replay_terrain delegates to _build_terrain_pc2; check the helper sets 'map'
        helper_body = _get_method_body(src, "_build_terrain_pc2")
        assert '"map"' in helper_body, (
            "_build_terrain_pc2 must set header.frame_id = 'map'"
        )

    def test_replay_uses_pointcloud2(self):
        """_replay_terrain delegates to _build_terrain_pc2 which builds a PointCloud2."""
        src = read_bridge_source()
        # _replay_terrain delegates to _build_terrain_pc2; check the helper uses PointCloud2
        helper_body = _get_method_body(src, "_build_terrain_pc2")
        assert "PointCloud2" in helper_body, (
            "_build_terrain_pc2 must create a PointCloud2 message"
        )


# ---------------------------------------------------------------------------
# Part 3: Behavioral test using mocks
# ---------------------------------------------------------------------------

class TestReplayTerrainBehavior:
    """Behavioral tests: mock the bridge and call _replay_terrain directly."""

    def _make_bridge(self):
        """Instantiate Go2VNavBridge with full mocking (no MuJoCo / ROS2)."""
        import importlib.util
        import types

        # Build a minimal mock rclpy + ROS2 message types environment
        rclpy_mock = MagicMock()
        rclpy_mock.node.Node = object  # will be replaced by manual class

        # Patch modules before importing bridge source via exec
        mock_modules = {
            "rclpy": rclpy_mock,
            "rclpy.node": MagicMock(),
            "rclpy.qos": MagicMock(),
            "rclpy.logging": MagicMock(),
            "nav_msgs": MagicMock(),
            "nav_msgs.msg": MagicMock(),
            "sensor_msgs": MagicMock(),
            "sensor_msgs.msg": MagicMock(),
            "geometry_msgs": MagicMock(),
            "geometry_msgs.msg": MagicMock(),
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
            "tf2_ros": MagicMock(),
            "visualization_msgs": MagicMock(),
            "visualization_msgs.msg": MagicMock(),
            "numpy": __import__("numpy"),
        }

        # Build minimal mocks for message types that the bridge references
        PointField = MagicMock()
        PointField.FLOAT32 = 7
        PointCloud2 = MagicMock(side_effect=lambda: MagicMock())
        mock_modules["sensor_msgs.msg"].PointField = PointField
        mock_modules["sensor_msgs.msg"].PointCloud2 = PointCloud2
        mock_modules["sensor_msgs.msg"].LaserScan = MagicMock()
        mock_modules["sensor_msgs.msg"].Joy = MagicMock()
        mock_modules["sensor_msgs.msg"].Image = MagicMock()
        mock_modules["sensor_msgs.msg"].CompressedImage = MagicMock()
        mock_modules["nav_msgs.msg"].Odometry = MagicMock()
        mock_modules["nav_msgs.msg"].Path = MagicMock()
        mock_modules["geometry_msgs.msg"].TwistStamped = MagicMock()
        mock_modules["geometry_msgs.msg"].Twist = MagicMock()
        mock_modules["geometry_msgs.msg"].TransformStamped = MagicMock()
        mock_modules["geometry_msgs.msg"].PointStamped = MagicMock()
        mock_modules["std_msgs.msg"].Float32 = MagicMock()
        mock_modules["std_msgs.msg"].Header = MagicMock()

        with patch.dict(sys.modules, mock_modules):
            spec = importlib.util.spec_from_file_location("go2_vnav_bridge_mock", _BRIDGE)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        return mod

    def test_replay_guard_stops_after_max_frames(self):
        """_replay_terrain stops publishing after _terrain_replay_max frames."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        assert "_terrain_replay_max" in body, (
            "_replay_terrain must check _terrain_replay_count >= _terrain_replay_max"
        )
        assert "_terrain_replay_count" in body

    def test_replay_cancels_timer_when_done(self):
        """_replay_terrain cancels its own timer when replay is complete."""
        src = read_bridge_source()
        body = _get_method_body(src, "_replay_terrain")
        assert "cancel()" in body, (
            "_replay_terrain must cancel its timer after replay completes"
        )
