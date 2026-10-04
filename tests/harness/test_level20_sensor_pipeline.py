# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""
Level 20 — Sensor Pipeline: terrainAnalysis parameter passing.

Source inspection tests: verify that launch_vnav.sh passes Go2-specific
ROS2 parameters to terrainAnalysis and terrainAnalysisExt.  These tests
do NOT launch any processes; they read the shell script and assert that
the correct --ros-args flags are present.
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
LAUNCH_SCRIPT = SCRIPTS_DIR / "launch_vnav.sh"


def _launch_content() -> str:
    """Return the full text of launch_vnav.sh."""
    return LAUNCH_SCRIPT.read_text()


# ---------------------------------------------------------------------------
# terrainAnalysis parameter tests
# ---------------------------------------------------------------------------

def test_terrain_analysis_has_ros_args() -> None:
    """terrainAnalysis command must include --ros-args."""
    content = _launch_content()
    # The block must have terrainAnalysis followed (on same or next logical
    # line) by --ros-args.
    assert "terrainAnalysis --ros-args" in content or (
        "terrainAnalysis" in content and "--ros-args" in content
    ), "launch_vnav.sh: terrainAnalysis missing --ros-args"

    # More strict: the --ros-args must appear in the terrainAnalysis invocation
    # (before the next '&' that terminates that command).
    lines = content.splitlines()
    in_terrain_block = False
    found_ros_args = False
    for line in lines:
        stripped = line.strip()
        if "terrainAnalysis --ros-args" in stripped:
            found_ros_args = True
            break
        if stripped.startswith("ros2 run terrain_analysis terrainAnalysis"):
            in_terrain_block = True
        if in_terrain_block and "--ros-args" in stripped:
            found_ros_args = True
            break
        if in_terrain_block and stripped.endswith("&") and "--ros-args" not in stripped:
            in_terrain_block = False  # command ended without --ros-args

    assert found_ros_args, (
        "launch_vnav.sh: terrainAnalysis invocation does not contain --ros-args"
    )


def test_terrain_clearDyObs_true() -> None:
    """clearDyObs:=true must be passed to terrainAnalysis."""
    content = _launch_content()
    assert "clearDyObs:=true" in content, (
        "launch_vnav.sh: missing '-p clearDyObs:=true' for terrainAnalysis"
    )


def test_terrain_obstacleHeightThre() -> None:
    """obstacleHeightThre:=0.15 must be passed to terrainAnalysis."""
    content = _launch_content()
    assert "obstacleHeightThre:=0.15" in content, (
        "launch_vnav.sh: missing '-p obstacleHeightThre:=0.15' for terrainAnalysis"
    )


def test_terrain_limitGroundLift() -> None:
    """limitGroundLift:=true must be passed to terrainAnalysis."""
    content = _launch_content()
    assert "limitGroundLift:=true" in content, (
        "launch_vnav.sh: missing '-p limitGroundLift:=true' for terrainAnalysis"
    )


def test_terrain_maxRelZ() -> None:
    """maxRelZ:=0.3 must be passed to terrainAnalysis."""
    content = _launch_content()
    assert "maxRelZ:=0.3" in content, (
        "launch_vnav.sh: missing '-p maxRelZ:=0.3' for terrainAnalysis"
    )


# ---------------------------------------------------------------------------
# terrainAnalysisExt parameter tests (bonus coverage)
# ---------------------------------------------------------------------------

def test_terrain_ext_has_ros_args() -> None:
    """terrainAnalysisExt command must include --ros-args."""
    content = _launch_content()
    lines = content.splitlines()
    in_ext_block = False
    found_ros_args = False
    for line in lines:
        stripped = line.strip()
        if "terrainAnalysisExt --ros-args" in stripped:
            found_ros_args = True
            break
        if stripped.startswith("ros2 run terrain_analysis_ext terrainAnalysisExt"):
            in_ext_block = True
        if in_ext_block and "--ros-args" in stripped:
            found_ros_args = True
            break
        if in_ext_block and stripped.endswith("&") and "--ros-args" not in stripped:
            in_ext_block = False

    assert found_ros_args, (
        "launch_vnav.sh: terrainAnalysisExt invocation does not contain --ros-args"
    )


def test_terrain_ext_obstacleHeightThre() -> None:
    """terrainAnalysisExt must also receive obstacleHeightThre:=0.15."""
    content = _launch_content()
    # The param appears in both invocations; count occurrences >= 2
    count = content.count("obstacleHeightThre:=0.15")
    assert count >= 2, (
        f"launch_vnav.sh: obstacleHeightThre:=0.15 appears {count} time(s); "
        "expected at least 2 (terrainAnalysis + terrainAnalysisExt)"
    )


def test_terrain_ext_maxRelZ() -> None:
    """terrainAnalysisExt must also receive maxRelZ:=0.3."""
    content = _launch_content()
    count = content.count("maxRelZ:=0.3")
    assert count >= 2, (
        f"launch_vnav.sh: maxRelZ:=0.3 appears {count} time(s); "
        "expected at least 2 (terrainAnalysis + terrainAnalysisExt)"
    )


# ===========================================================================
# Sensor pipeline rate tests (bridge ↔ MuJoCo lidar synchronisation)
# ===========================================================================
#
# These code-inspection tests do NOT require ROS2 to be running.
# They validate that go2_vnav_bridge.py publishes at 5 Hz to match the
# MuJoCo lidar update interval (200 physics steps @ 1 kHz = 5 Hz).
# Publishing at 10 Hz causes 50% duplicate pointclouds, confusing TARE's
# keypose creation logic.
# ===========================================================================

import re

_BRIDGE_PATH = Path(__file__).parent.parent.parent / "scripts" / "go2_vnav_bridge.py"
_MUJOCO_PATH = (
    Path(__file__).parent.parent.parent
    / "vector_os_nano" / "hardware" / "sim" / "mujoco_go2.py"
)


def _bridge_source() -> str:
    return _BRIDGE_PATH.read_text(encoding="utf-8")


def _mujoco_source() -> str:
    return _MUJOCO_PATH.read_text(encoding="utf-8")


def _find_timer_interval(source: str, callback: str) -> float | None:
    """Return the numeric interval (seconds) for create_timer(..., <callback>).

    Handles both literal floats and simple expressions like ``1.0 / 5.0``.
    """
    pattern = rf"self\.create_timer\(([^,]+),\s*self\.{re.escape(callback)}\)"
    match = re.search(pattern, source)
    if match is None:
        return None
    expr = match.group(1).strip()
    try:
        return float(eval(expr))  # noqa: S307 — evaluates only simple arithmetic
    except Exception:
        return None


def test_scan_publish_rate_matches_lidar() -> None:
    """Bridge timer interval for _publish_pointcloud must be 0.2 s (5 Hz)."""
    source = _bridge_source()
    interval = _find_timer_interval(source, "_publish_pointcloud")
    assert interval is not None, "_publish_pointcloud timer not found in bridge source"
    assert abs(interval - 0.2) < 1e-9, (
        f"_publish_pointcloud timer is {interval:.4f}s — expected 0.2s (5 Hz). "
        "MuJoCo lidar runs at 5 Hz (_LIDAR_UPDATE_INTERVAL=200 @ 1kHz); "
        "publishing at 10 Hz sends 50% duplicate clouds."
    )


def test_scan_freshness_no_duplicates() -> None:
    """MuJoCo lidar period must equal bridge publish period for both scan callbacks."""
    mujoco_src = _mujoco_source()
    bridge_src = _bridge_source()

    m = re.search(r"_LIDAR_PERIOD_S\s*:\s*float\s*=\s*([0-9.]+)", mujoco_src)
    assert m is not None, "_LIDAR_PERIOD_S constant not found in mujoco_go2.py"
    expected_interval = float(m.group(1))  # sim seconds between scans: 0.2 s
    lidar_hz = 1.0 / expected_interval     # expected 5.0 Hz

    pc_interval = _find_timer_interval(bridge_src, "_publish_pointcloud")
    scan_interval = _find_timer_interval(bridge_src, "_publish_scan")

    assert pc_interval is not None, "_publish_pointcloud timer not found in bridge source"
    assert scan_interval is not None, "_publish_scan timer not found in bridge source"

    assert abs(pc_interval - expected_interval) < 1e-9, (
        f"_publish_pointcloud interval ({pc_interval:.4f}s) != "
        f"MuJoCo lidar period ({expected_interval:.4f}s = {lidar_hz:.1f} Hz)"
    )
    assert abs(scan_interval - expected_interval) < 1e-9, (
        f"_publish_scan interval ({scan_interval:.4f}s) != "
        f"MuJoCo lidar period ({expected_interval:.4f}s = {lidar_hz:.1f} Hz)"
    )


def test_pointcloud_has_required_fields() -> None:
    """PointCloud2 must declare x, y, z, intensity with correct byte offsets."""
    source = _bridge_source()

    field_pattern = re.compile(
        r'PointField\s*\(\s*name\s*=\s*"(\w+)"\s*,\s*offset\s*=\s*(\d+)'
    )
    fields: dict[str, int] = {
        name: int(offset)
        for name, offset in field_pattern.findall(source)
    }

    required = {"x": 0, "y": 4, "z": 8, "intensity": 12}
    for name, expected_offset in required.items():
        assert name in fields, (
            f"PointField '{name}' missing from PointCloud2 field declarations"
        )
        assert fields[name] == expected_offset, (
            f"PointField '{name}' offset={fields[name]}, expected {expected_offset}"
        )


def test_odom_publish_rate() -> None:
    """_publish_odom timer must be 200 Hz (interval = 1/200 = 0.005 s)."""
    source = _bridge_source()
    interval = _find_timer_interval(source, "_publish_odom")
    assert interval is not None, "_publish_odom timer not found in bridge source"
    expected = 1.0 / 200.0
    assert abs(interval - expected) < 1e-9, (
        f"_publish_odom timer is {interval:.6f}s — expected {expected:.6f}s (200 Hz)"
    )


def test_scan_and_odom_qos_reliable() -> None:
    """/registered_scan and /state_estimation publishers must use RELIABLE QoS.

    sensorScanGeneration expects RELIABLE on both topics; mismatched QoS
    causes silent message drops and TARE data starvation.
    """
    source = _bridge_source()

    assert "ReliabilityPolicy.RELIABLE" in source, (
        "ReliabilityPolicy.RELIABLE not defined in bridge source"
    )

    # /registered_scan — create_publisher may span multiple lines
    pc_match = re.search(
        r'self\._pc_pub\s*=\s*self\.create_publisher\(\s*\w+\s*,\s*'
        r'"[^"]*registered_scan[^"]*"\s*,\s*(\w+)\s*\)',
        source,
        re.DOTALL,
    )
    assert pc_match is not None, "/registered_scan publisher declaration not found"
    pc_qos = pc_match.group(1)
    assert pc_qos == "reliable_qos", (
        f"/registered_scan uses QoS variable '{pc_qos}', expected 'reliable_qos'"
    )

    # /state_estimation — create_publisher may span multiple lines
    odom_match = re.search(
        r'self\._odom_pub\s*=\s*self\.create_publisher\(\s*\w+\s*,\s*'
        r'"[^"]*state_estimation[^"]*"\s*,\s*(\w+)\s*\)',
        source,
        re.DOTALL,
    )
    assert odom_match is not None, "/state_estimation publisher declaration not found"
    odom_qos = odom_match.group(1)
    assert odom_qos == "reliable_qos", (
        f"/state_estimation uses QoS variable '{odom_qos}', expected 'reliable_qos'"
    )
