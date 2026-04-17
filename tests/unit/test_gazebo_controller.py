"""Tests for unitree_guide_controller ros2_control integration.

Validates that gazebo/models/go2/ros2_control.yaml exists, is valid YAML,
and contains the correct controller_manager + joint configuration aligned
with our model.sdf joint names.

Level: Unit — pure file-parsing, no Gazebo or ROS2 runtime required.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "gazebo" / "models" / "go2"
_ROS2_CONTROL_YAML = _MODEL_DIR / "ros2_control.yaml"
_MODEL_SDF = _MODEL_DIR / "model.sdf"

# The 12 joint names that must appear in both the SDF and the YAML config.
# Order: FR/FL/RR/RL — matches unitree_guide_controller expectation.
_EXPECTED_JOINT_NAMES = {
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
}

_EXPECTED_COMMAND_INTERFACES = {"position", "velocity", "effort", "kp", "kd"}
_EXPECTED_STATE_INTERFACES = {"position", "velocity", "effort"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ros2_control_config() -> dict:
    """Load ros2_control.yaml once for the whole module."""
    with open(str(_ROS2_CONTROL_YAML)) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sdf_root() -> ET.Element:
    """Parse model.sdf and return its root element."""
    tree = ET.parse(str(_MODEL_SDF))
    return tree.getroot()


@pytest.fixture(scope="module")
def sdf_joint_names(sdf_root: ET.Element) -> set[str]:
    """Extract revolute joint names from the SDF model."""
    joints = sdf_root.findall(".//joint[@type='revolute']")
    return {j.get("name") for j in joints if j.get("name")}


# ---------------------------------------------------------------------------
# File existence tests
# ---------------------------------------------------------------------------


def test_ros2_control_yaml_exists() -> None:
    """gazebo/models/go2/ros2_control.yaml must exist."""
    assert _ROS2_CONTROL_YAML.is_file(), (
        f"ros2_control.yaml not found: {_ROS2_CONTROL_YAML}"
    )


# ---------------------------------------------------------------------------
# YAML validity tests
# ---------------------------------------------------------------------------


def test_ros2_control_yaml_valid() -> None:
    """ros2_control.yaml must be parseable as valid YAML."""
    with open(str(_ROS2_CONTROL_YAML)) as f:
        config = yaml.safe_load(f)
    assert config is not None, "ros2_control.yaml parsed as None — empty file?"
    assert isinstance(config, dict), (
        f"ros2_control.yaml root must be a dict, got {type(config)}"
    )


# ---------------------------------------------------------------------------
# controller_manager section tests
# ---------------------------------------------------------------------------


def test_ros2_control_has_controller_manager(ros2_control_config: dict) -> None:
    """YAML must have a top-level 'controller_manager' key."""
    assert "controller_manager" in ros2_control_config, (
        f"Missing 'controller_manager' key. Top-level keys: "
        f"{list(ros2_control_config.keys())}"
    )


def test_ros2_control_controller_manager_has_ros_params(
    ros2_control_config: dict,
) -> None:
    """controller_manager must have ros__parameters with update_rate."""
    cm = ros2_control_config["controller_manager"]
    assert "ros__parameters" in cm, (
        "controller_manager missing ros__parameters"
    )
    params = cm["ros__parameters"]
    assert "update_rate" in params, (
        "controller_manager.ros__parameters missing update_rate"
    )
    assert isinstance(params["update_rate"], int), (
        f"update_rate must be int, got {type(params['update_rate'])}"
    )


def test_ros2_control_has_unitree_guide_controller(
    ros2_control_config: dict,
) -> None:
    """controller_manager must declare unitree_guide_controller."""
    cm_params = ros2_control_config["controller_manager"]["ros__parameters"]
    assert "unitree_guide_controller" in cm_params, (
        "controller_manager.ros__parameters missing unitree_guide_controller entry"
    )
    ctrl = cm_params["unitree_guide_controller"]
    assert "type" in ctrl, "unitree_guide_controller entry missing 'type' field"
    assert ctrl["type"] == "unitree_guide_controller/UnitreeGuideController", (
        f"Wrong controller type: {ctrl['type']}"
    )


def test_ros2_control_has_joint_state_broadcaster(
    ros2_control_config: dict,
) -> None:
    """controller_manager must declare joint_state_broadcaster."""
    cm_params = ros2_control_config["controller_manager"]["ros__parameters"]
    assert "joint_state_broadcaster" in cm_params, (
        "controller_manager.ros__parameters missing joint_state_broadcaster"
    )


# ---------------------------------------------------------------------------
# Joint configuration tests
# ---------------------------------------------------------------------------


def test_ros2_control_has_12_joints(ros2_control_config: dict) -> None:
    """unitree_guide_controller must configure exactly 12 joints."""
    ctrl = ros2_control_config.get("unitree_guide_controller", {})
    params = ctrl.get("ros__parameters", {})
    joints = params.get("joints", [])
    assert len(joints) == 12, (
        f"Expected 12 joints, got {len(joints)}: {joints}"
    )


def test_ros2_control_joint_names_correct(ros2_control_config: dict) -> None:
    """unitree_guide_controller joints must match the expected 12 Go2 joint names."""
    ctrl = ros2_control_config.get("unitree_guide_controller", {})
    params = ctrl.get("ros__parameters", {})
    actual = set(params.get("joints", []))
    missing = _EXPECTED_JOINT_NAMES - actual
    extra = actual - _EXPECTED_JOINT_NAMES
    assert not missing, f"Missing joints in config: {missing}"
    assert not extra, f"Unexpected joints in config: {extra}"


def test_ros2_control_joint_names_match_sdf(
    ros2_control_config: dict,
    sdf_joint_names: set[str],
) -> None:
    """Joint names in ros2_control.yaml must match revolute joints in model.sdf."""
    ctrl = ros2_control_config.get("unitree_guide_controller", {})
    params = ctrl.get("ros__parameters", {})
    yaml_joints = set(params.get("joints", []))
    # All YAML joints must exist in SDF
    missing_from_sdf = yaml_joints - sdf_joint_names
    assert not missing_from_sdf, (
        f"Joints in YAML not found in SDF: {missing_from_sdf}"
    )


# ---------------------------------------------------------------------------
# Interface configuration tests
# ---------------------------------------------------------------------------


def test_ros2_control_has_command_interfaces(ros2_control_config: dict) -> None:
    """unitree_guide_controller must declare position, velocity, effort, kp, kd command interfaces."""
    ctrl = ros2_control_config.get("unitree_guide_controller", {})
    params = ctrl.get("ros__parameters", {})
    cmd_interfaces = set(params.get("command_interfaces", []))
    missing = _EXPECTED_COMMAND_INTERFACES - cmd_interfaces
    assert not missing, (
        f"Missing command_interfaces: {missing}. "
        f"Present: {cmd_interfaces}"
    )


def test_ros2_control_has_state_interfaces(ros2_control_config: dict) -> None:
    """unitree_guide_controller must declare position, velocity, effort state interfaces."""
    ctrl = ros2_control_config.get("unitree_guide_controller", {})
    params = ctrl.get("ros__parameters", {})
    state_interfaces = set(params.get("state_interfaces", []))
    missing = _EXPECTED_STATE_INTERFACES - state_interfaces
    assert not missing, (
        f"Missing state_interfaces: {missing}. "
        f"Present: {state_interfaces}"
    )


# ---------------------------------------------------------------------------
# model.sdf plugin tests
# ---------------------------------------------------------------------------


def test_sdf_has_gz_quadruped_plugin(sdf_root: ET.Element) -> None:
    """model.sdf must reference the gz_quadruped_hardware::GazeboSimQuadrupedPlugin."""
    plugins = sdf_root.findall(".//plugin")
    plugin_names = [p.get("name", "") for p in plugins]
    assert any("GazeboSimQuadrupedPlugin" in n for n in plugin_names), (
        f"gz_quadruped_hardware::GazeboSimQuadrupedPlugin not found in SDF plugins: "
        f"{plugin_names}"
    )


def test_sdf_plugin_references_ros2_control_yaml(sdf_root: ET.Element) -> None:
    """The gz_quadruped plugin element must reference ros2_control.yaml."""
    plugins = sdf_root.findall(".//plugin")
    for plugin in plugins:
        if "GazeboSimQuadrupedPlugin" in plugin.get("name", ""):
            params_elem = plugin.find("parameters")
            assert params_elem is not None, (
                "GazeboSimQuadrupedPlugin has no <parameters> child element"
            )
            assert params_elem.text is not None, (
                "<parameters> element is empty"
            )
            assert "ros2_control.yaml" in params_elem.text, (
                f"<parameters> does not reference ros2_control.yaml: "
                f"{params_elem.text}"
            )
            return
    pytest.fail("GazeboSimQuadrupedPlugin not found in SDF")
