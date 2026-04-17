"""Level 53: FAR V-Graph ceiling point filter harness.

Tests verify:
  - _CEILING_FILTER_HEIGHT module-level constant exists and is reasonable
  - _publish_pointcloud() filters points with intensity > ceiling threshold
  - Ceiling filter applies BEFORE struct.pack (no wasted bytes)
  - msg.width uses filtered count, not len(points)
  - msg.row_step uses filtered count
  - TerrainAccumulator z_max is <= ceiling filter height
  - _build_terrain_pc2 does NOT reference ceiling filter (different purpose)
  - intensity = z - ground_z in _publish_pointcloud
  - /registered_scan is the topic for filtered cloud
  - Wall-height points (0.3-1.5m intensity) are preserved
  - Ceiling-height points (>2.0m intensity) are excluded
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BRIDGE = _REPO / "scripts" / "go2_vnav_bridge.py"


def _read_bridge() -> str:
    return _BRIDGE.read_text()


def _get_publish_pointcloud_src(src: str) -> str:
    """Extract the _publish_pointcloud method body from source."""
    # Find method start
    start = src.find("def _publish_pointcloud(self)")
    assert start != -1, "_publish_pointcloud not found"
    # Find next def at same or lower indentation
    # The method body starts at start; find end by looking for next `    def ` at class level
    after = src[start:]
    # Find next method (4-space indent + def at same level)
    next_method = re.search(r"\n    def ", after[1:])
    if next_method:
        end = next_method.start() + 1
        return after[:end]
    return after


def _get_build_terrain_pc2_src(src: str) -> str:
    """Extract the _build_terrain_pc2 method body from source."""
    start = src.find("def _build_terrain_pc2(self")
    assert start != -1, "_build_terrain_pc2 not found"
    after = src[start:]
    next_method = re.search(r"\n    def ", after[1:])
    if next_method:
        end = next_method.start() + 1
        return after[:end]
    return after


# ===================================================================
# Test 1: _CEILING_FILTER_HEIGHT constant exists and is reasonable
# ===================================================================

def _get_ceiling_filter_height() -> float:
    """Return the runtime value of _CEILING_FILTER_HEIGHT from the bridge module.

    After the nav.yaml refactor, _CEILING_FILTER_HEIGHT is assigned via
    _nav("ceiling_filter_height", 1.8) rather than a numeric literal.
    We import the module to get the actual resolved value.
    """
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("_bridge_tmp", str(_BRIDGE))
    # The bridge imports ROS2 packages — mock them so we can import the module.
    _ros_mocks = [
        "rclpy", "rclpy.node", "rclpy.qos", "rclpy.executors",
        "geometry_msgs", "geometry_msgs.msg",
        "nav_msgs", "nav_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "std_msgs", "std_msgs.msg",
        "tf2_ros",
        "visualization_msgs", "visualization_msgs.msg",
        "unitree_go", "unitree_go.msg",
        "unitree_go2_ros2_interfaces", "unitree_go2_ros2_interfaces.msg",
        "sensor_msgs.msg",
        "std_srvs", "std_srvs.srv",
        "point_cloud2", "sensor_msgs.point_cloud2",
    ]
    import types
    for name in _ros_mocks:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # Read the constant via source regex since the bridge has complex ROS2 deps
    src = _read_bridge()
    # Accept both: literal assignment AND _nav() call
    # Pattern 1 (legacy): _CEILING_FILTER_HEIGHT: float = 1.8
    m_literal = re.search(
        r"^_CEILING_FILTER_HEIGHT\s*(?::\s*float\s*)?=\s*([0-9]+(?:\.[0-9]*)?)",
        src, re.MULTILINE,
    )
    if m_literal:
        return float(m_literal.group(1))
    # Pattern 2 (current): _CEILING_FILTER_HEIGHT: float = _nav("ceiling_filter_height", 1.8)
    m_nav = re.search(
        r'^_CEILING_FILTER_HEIGHT\s*(?::\s*float\s*)?=\s*_nav\s*\([^,]+,\s*([0-9]+(?:\.[0-9]*)?)\)',
        src, re.MULTILINE,
    )
    if m_nav:
        return float(m_nav.group(1))
    raise AssertionError(
        "_CEILING_FILTER_HEIGHT not found as literal or _nav() call at module level"
    )


class TestCeilingFilterConstant:

    def test_ceiling_filter_height_constant_exists(self):
        """_CEILING_FILTER_HEIGHT must be defined at module level."""
        src = _read_bridge()
        assert "_CEILING_FILTER_HEIGHT" in src, (
            "_CEILING_FILTER_HEIGHT constant not found in go2_vnav_bridge.py"
        )

    def test_ceiling_filter_height_is_float_literal(self):
        """_CEILING_FILTER_HEIGHT must be assigned at module level (literal or _nav() call)."""
        src = _read_bridge()
        # Accept both: numeric literal OR _nav() pattern (after nav.yaml refactor)
        m = re.search(
            r"^_CEILING_FILTER_HEIGHT\s*(?::\s*float\s*)?=\s*"
            r"(?:[0-9]+(?:\.[0-9]*)?|_nav\s*\()",
            src,
            re.MULTILINE,
        )
        assert m is not None, (
            "_CEILING_FILTER_HEIGHT must be assigned at module level "
            "(numeric literal or _nav() call)"
        )

    def test_ceiling_filter_height_value_reasonable(self):
        """_CEILING_FILTER_HEIGHT must be between 1.0 and 2.5 for indoor use."""
        value = _get_ceiling_filter_height()
        assert 1.0 <= value <= 2.5, (
            f"_CEILING_FILTER_HEIGHT={value} is outside reasonable indoor range [1.0, 2.5]"
        )


# ===================================================================
# Test 2: _publish_pointcloud filters ceiling points
# ===================================================================

class TestPublishPointcloudCeilingFilter:

    def test_publish_pointcloud_references_ceiling_filter(self):
        """_publish_pointcloud must reference _CEILING_FILTER_HEIGHT."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        assert "_CEILING_FILTER_HEIGHT" in body, (
            "_CEILING_FILTER_HEIGHT not referenced inside _publish_pointcloud"
        )

    def test_ceiling_filter_uses_intensity(self):
        """Ceiling filter must compare intensity, not raw z."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        # Filter must be on intensity variable (not raw z)
        assert re.search(r"intensity\s*>\s*_CEILING_FILTER_HEIGHT", body), (
            "Filter condition must be: intensity > _CEILING_FILTER_HEIGHT"
        )

    def test_ceiling_filter_before_pack(self):
        """The ceiling filter (continue/skip) must appear before struct.pack."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        filter_pos = body.find("_CEILING_FILTER_HEIGHT")
        pack_pos = body.find("struct.pack")
        assert filter_pos != -1, "_CEILING_FILTER_HEIGHT not in _publish_pointcloud"
        assert pack_pos != -1, "struct.pack not in _publish_pointcloud"
        assert filter_pos < pack_pos, (
            "Ceiling filter must come BEFORE struct.pack — filter early, pack only valid points"
        )

    def test_ceiling_filter_has_continue_or_skip(self):
        """Filter must skip ceiling points (continue statement in loop).

        The _publish_pointcloud docstring also contains _CEILING_FILTER_HEIGHT,
        so we search for the SECOND occurrence (the actual comparison) and look
        for 'continue' within 200 chars of that.
        """
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        # Find the second occurrence: docstring has it first, comparison is second
        first = body.find("_CEILING_FILTER_HEIGHT")
        assert first != -1, "_CEILING_FILTER_HEIGHT not in _publish_pointcloud"
        second = body.find("_CEILING_FILTER_HEIGHT", first + 1)
        if second == -1:
            # Only one occurrence — use it (future-proof if docstring is removed)
            filter_pos = first
        else:
            filter_pos = second
        snippet = body[filter_pos: filter_pos + 200]
        assert "continue" in snippet, (
            "Ceiling filter must use 'continue' to skip ceiling points"
        )


# ===================================================================
# Test 3: msg.width and msg.row_step use filtered count
# ===================================================================

class TestFilteredCountInMessage:

    def test_width_not_len_points(self):
        """msg.width must NOT be set to len(points) — must use filtered count."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        # msg.width = len(points) would be wrong
        assert not re.search(r"msg\.width\s*=\s*len\s*\(\s*points\s*\)", body), (
            "msg.width must use filtered count variable, not len(points)"
        )

    def test_width_uses_filtered_variable(self):
        """msg.width must be assigned a variable (not hardcoded or len(points))."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        # msg.width = <something other than len(points)>
        m = re.search(r"msg\.width\s*=\s*(.+)", body)
        assert m is not None, "msg.width assignment not found in _publish_pointcloud"
        rhs = m.group(1).strip()
        # Must not be len(points)
        assert "len(points)" not in rhs, (
            f"msg.width must not be len(points), got: msg.width = {rhs}"
        )

    def test_row_step_uses_filtered_count(self):
        """msg.row_step must not use len(points) directly."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        m = re.search(r"msg\.row_step\s*=\s*(.+)", body)
        assert m is not None, "msg.row_step assignment not found in _publish_pointcloud"
        rhs = m.group(1).strip()
        assert "len(points)" not in rhs, (
            f"msg.row_step must not use len(points), got: msg.row_step = {rhs}"
        )

    def test_filtered_count_variable_exists(self):
        """A filtered count variable must be tracked and used for msg.width."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        # There must be some variable that counts filtered points
        # Common patterns: filtered_count, n_points, count, etc.
        has_counter = (
            "filtered_count" in body
            or re.search(r"\bcount\b", body)
            or "n_pts" in body
            or "num_points" in body
        )
        assert has_counter, (
            "No filtered point counter variable found in _publish_pointcloud"
        )


# ===================================================================
# Test 4: TerrainAccumulator z_max is bounded
# ===================================================================

class TestTerrainAccumulatorBound:

    def test_terrain_accumulator_has_z_max(self):
        """TerrainAccumulator.__init__ must have a z_max parameter."""
        src = _read_bridge()
        assert "z_max" in src, "TerrainAccumulator must have z_max parameter"

    def test_terrain_accumulator_z_max_reasonable(self):
        """TerrainAccumulator default z_max must be <= 2.5m."""
        src = _read_bridge()
        m = re.search(
            r"def __init__.*?z_max\s*(?::\s*float\s*)?=\s*([0-9]+(?:\.[0-9]*)?)",
            src,
        )
        assert m is not None, "TerrainAccumulator z_max default not found"
        value = float(m.group(1))
        assert value <= 2.5, (
            f"TerrainAccumulator z_max={value} is too high; ceiling is typically 2.4m"
        )

    def test_terrain_accumulator_z_max_filters_ceiling(self):
        """TerrainAccumulator.add() must filter points with z > z_max."""
        src = _read_bridge()
        # Find the add method
        start = src.find("def add(self, points")
        assert start != -1
        after = src[start:]
        next_method = re.search(r"\n    def ", after[1:])
        body = after[: next_method.start() + 1] if next_method else after
        assert "z_max" in body or "_z_max" in body, (
            "TerrainAccumulator.add() must check z against z_max"
        )


# ===================================================================
# Test 5: _build_terrain_pc2 is unaffected by ceiling filter
# ===================================================================

class TestBuildTerrainPc2NotAffected:

    def test_build_terrain_pc2_no_ceiling_filter(self):
        """_build_terrain_pc2 must NOT reference _CEILING_FILTER_HEIGHT."""
        src = _read_bridge()
        body = _get_build_terrain_pc2_src(src)
        assert "_CEILING_FILTER_HEIGHT" not in body, (
            "_build_terrain_pc2 must not filter ceiling — it handles terrain replay, "
            "which has its own z_max via TerrainAccumulator"
        )


# ===================================================================
# Test 6: Conceptual range checks
# ===================================================================

class TestConceptualRanges:

    def test_ceiling_filter_above_max_wall_height(self):
        """Filter threshold must be above typical max wall feature height (1.5m)."""
        value = _get_ceiling_filter_height()
        assert value > 1.5, (
            f"_CEILING_FILTER_HEIGHT={value} would filter wall features up to 1.5m — "
            "threshold must be above 1.5m"
        )

    def test_ceiling_filter_below_typical_ceiling(self):
        """Filter threshold must be below typical indoor ceiling height (2.4m)."""
        value = _get_ceiling_filter_height()
        assert value < 2.4, (
            f"_CEILING_FILTER_HEIGHT={value} is >= typical ceiling height (2.4m); "
            "ceiling points would not be filtered"
        )

    def test_ceiling_filter_preserves_door_tops(self):
        """Filter threshold must be >= 1.5m to preserve navigation features.

        Per spec: 1.8m is the chosen threshold (walls < 1.8m for Go2 navigation).
        Tall doors are 2.0m but Go2 cannot traverse them anyway.
        """
        value = _get_ceiling_filter_height()
        # 1.8m is acceptable — tall doors are 2.0m but Go2 cannot traverse them anyway
        assert value >= 1.5, (
            f"_CEILING_FILTER_HEIGHT={value} too low — would filter important navigation features"
        )


# ===================================================================
# Test 7: Source structural checks
# ===================================================================

class TestSourceStructure:

    def test_intensity_computed_as_z_minus_ground_z(self):
        """intensity must be computed as z - ground_z in _publish_pointcloud."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        assert re.search(r"intensity\s*=\s*z\s*-\s*ground_z", body), (
            "intensity = z - ground_z must be computed in _publish_pointcloud"
        )

    def test_registered_scan_is_published(self):
        """_pc_pub (registered_scan) must be published in _publish_pointcloud."""
        src = _read_bridge()
        body = _get_publish_pointcloud_src(src)
        assert "_pc_pub.publish" in body, (
            "_publish_pointcloud must call _pc_pub.publish (the /registered_scan publisher)"
        )

    def test_registered_scan_topic_in_bridge(self):
        """Bridge must publish /registered_scan topic."""
        src = _read_bridge()
        assert "registered_scan" in src, (
            "/registered_scan topic must be present in go2_vnav_bridge.py"
        )
