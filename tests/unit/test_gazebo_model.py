"""Tests for the Go2 SDF model for Gazebo Harmonic.

Validates that gazebo/models/go2/model.sdf and model.config exist, are valid,
and contain the required links, joints, and sensors.

Level: Unit — pure file-parsing, no Gazebo runtime required.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "gazebo" / "models" / "go2"
_MODEL_SDF = _MODEL_DIR / "model.sdf"
_MODEL_CONFIG = _MODEL_DIR / "model.config"

# Expected joint names matching Go2 12-DOF configuration.
# Note: SDF uses underscore-style; xacro uses _joint suffix but SDF drops it.
_EXPECTED_JOINT_NAMES = {
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sdf_tree() -> ET.ElementTree:
    """Parse model.sdf once for the whole module."""
    return ET.parse(str(_MODEL_SDF))


@pytest.fixture(scope="module")
def sdf_root(sdf_tree: ET.ElementTree) -> ET.Element:
    """Return the root <sdf> element."""
    return sdf_tree.getroot()


@pytest.fixture(scope="module")
def model_elem(sdf_root: ET.Element) -> ET.Element:
    """Return the first <model> element inside the SDF."""
    elem = sdf_root.find("model")
    assert elem is not None, "<model> element not found in SDF"
    return elem


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_go2_model_dir_exists() -> None:
    """The gazebo/models/go2/ directory must exist."""
    assert _MODEL_DIR.is_dir(), f"Directory not found: {_MODEL_DIR}"


def test_go2_has_model_sdf() -> None:
    """model.sdf must exist inside the model directory."""
    assert _MODEL_SDF.is_file(), f"model.sdf not found: {_MODEL_SDF}"


def test_go2_has_model_config() -> None:
    """model.config must exist inside the model directory."""
    assert _MODEL_CONFIG.is_file(), f"model.config not found: {_MODEL_CONFIG}"


def test_go2_sdf_valid_xml() -> None:
    """model.sdf must be parseable as valid XML."""
    tree = ET.parse(str(_MODEL_SDF))
    assert tree.getroot() is not None


def test_go2_sdf_version(sdf_root: ET.Element) -> None:
    """Root <sdf> element must declare version 1.9 or higher."""
    version = sdf_root.get("version", "")
    assert version, "SDF version attribute missing"
    major, minor = (int(x) for x in version.split(".")[:2])
    assert (major, minor) >= (1, 9), f"SDF version too old: {version} (need >= 1.9)"


# ---------------------------------------------------------------------------
# Link tests
# ---------------------------------------------------------------------------


def test_go2_has_base_link(model_elem: ET.Element) -> None:
    """Model must have a link named 'base_link' or 'base'."""
    link_names = {link.get("name") for link in model_elem.findall("link")}
    assert link_names & {"base_link", "base"}, (
        f"No 'base_link' or 'base' found. Links present: {link_names}"
    )


# ---------------------------------------------------------------------------
# Joint tests
# ---------------------------------------------------------------------------


def test_go2_has_12_revolute_joints(model_elem: ET.Element) -> None:
    """Model must have exactly 12 revolute joints (4 legs x 3 DOF)."""
    revolute_joints = [
        j for j in model_elem.findall("joint") if j.get("type") == "revolute"
    ]
    assert len(revolute_joints) == 12, (
        f"Expected 12 revolute joints, found {len(revolute_joints)}"
    )


def test_go2_joint_names(model_elem: ET.Element) -> None:
    """All 12 expected joint names must be present in the model."""
    actual_names = {j.get("name") for j in model_elem.findall("joint")}
    missing = _EXPECTED_JOINT_NAMES - actual_names
    assert not missing, f"Missing joint names: {missing}"


# ---------------------------------------------------------------------------
# Sensor tests
# ---------------------------------------------------------------------------


def test_go2_has_gpu_lidar(sdf_root: ET.Element) -> None:
    """Model must contain a sensor of type 'gpu_lidar' (Livox MID-360)."""
    sensors = sdf_root.findall(".//sensor[@type='gpu_lidar']")
    assert sensors, "No <sensor type='gpu_lidar'> found in SDF"


def test_go2_lidar_range_max_12(sdf_root: ET.Element) -> None:
    """Lidar max range must be 12.0 metres."""
    lidar_sensor = sdf_root.find(".//sensor[@type='gpu_lidar']")
    assert lidar_sensor is not None, "gpu_lidar sensor not found"
    max_elem = lidar_sensor.find(".//range/max")
    assert max_elem is not None, "<range><max> not found inside gpu_lidar sensor"
    assert float(max_elem.text) == pytest.approx(12.0), (
        f"Lidar range max expected 12.0, got {max_elem.text}"
    )


def test_go2_has_camera(sdf_root: ET.Element) -> None:
    """Model must contain a sensor of type 'camera' (D435 RGB)."""
    sensors = sdf_root.findall(".//sensor[@type='camera']")
    assert sensors, "No <sensor type='camera'> found in SDF"


def test_go2_camera_width_640(sdf_root: ET.Element) -> None:
    """D435 RGB camera must have width=640."""
    cam_sensor = sdf_root.find(".//sensor[@type='camera']")
    assert cam_sensor is not None, "camera sensor not found"
    width_elem = cam_sensor.find(".//image/width")
    assert width_elem is not None, "<image><width> not found inside camera sensor"
    assert int(width_elem.text) == 640, (
        f"Camera width expected 640, got {width_elem.text}"
    )


def test_go2_has_depth_camera(sdf_root: ET.Element) -> None:
    """Model must contain a sensor of type 'depth_camera' (D435 Depth)."""
    sensors = sdf_root.findall(".//sensor[@type='depth_camera']")
    assert sensors, "No <sensor type='depth_camera'> found in SDF"


def test_go2_depth_camera_width_640(sdf_root: ET.Element) -> None:
    """D435 depth camera must have width=640."""
    depth_sensor = sdf_root.find(".//sensor[@type='depth_camera']")
    assert depth_sensor is not None, "depth_camera sensor not found"
    width_elem = depth_sensor.find(".//image/width")
    assert width_elem is not None, "<image><width> not found inside depth_camera sensor"
    assert int(width_elem.text) == 640, (
        f"Depth camera width expected 640, got {width_elem.text}"
    )


def test_go2_has_imu(sdf_root: ET.Element) -> None:
    """Model must contain a sensor of type 'imu'."""
    sensors = sdf_root.findall(".//sensor[@type='imu']")
    assert sensors, "No <sensor type='imu'> found in SDF"
