"""Tests for ros_gz_bridge YAML configuration.

Validates that gazebo/config/bridge.yaml exists, is valid YAML,
and documents all required topic mappings between Gazebo and ROS2.

The YAML uses ros_topic_name / gz_topic_name fields to support
topic renaming (e.g. /scan/points → /registered_scan).
The actual bridge runs via command-line args in the launch file;
this YAML serves as the canonical reference for topic mapping.

Level: Unit — pure file-parsing, no ROS2 or Gazebo runtime needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_BRIDGE_YAML = _REPO_ROOT / "gazebo" / "config" / "bridge.yaml"

# Topics that must be documented in bridge.yaml (ros_topic_name)
_REQUIRED_ROS_TOPICS = {
    "/registered_scan",
    "/camera/image",
    "/camera/depth",
    "/cmd_vel_nav",
    "/imu/data",
}

_VALID_DIRECTIONS = {"GZ_TO_ROS", "ROS_TO_GZ"}
_REQUIRED_FIELDS = {"ros_topic_name", "ros_type_name", "gz_type_name", "direction"}


def _load_bridge() -> list[dict]:
    return yaml.safe_load(_BRIDGE_YAML.read_text(encoding="utf-8"))


def _find_entry(entries: list[dict], ros_topic: str) -> dict | None:
    for entry in entries:
        if entry.get("ros_topic_name") == ros_topic:
            return entry
    return None


def test_bridge_yaml_exists() -> None:
    assert _BRIDGE_YAML.exists()


def test_bridge_yaml_valid_syntax() -> None:
    parsed = yaml.safe_load(_BRIDGE_YAML.read_text(encoding="utf-8"))
    assert parsed is not None


def test_bridge_is_list() -> None:
    entries = _load_bridge()
    assert isinstance(entries, list)


def test_bridge_has_registered_scan() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/registered_scan")
    assert entry is not None
    assert entry.get("gz_topic_name") == "/scan/points"


def test_bridge_has_camera_image() -> None:
    entries = _load_bridge()
    assert _find_entry(entries, "/camera/image") is not None


def test_bridge_has_camera_depth() -> None:
    entries = _load_bridge()
    assert _find_entry(entries, "/camera/depth") is not None


def test_bridge_has_cmd_vel_nav() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/cmd_vel_nav")
    assert entry is not None
    assert entry.get("direction") == "ROS_TO_GZ"


def test_bridge_has_imu() -> None:
    entries = _load_bridge()
    entry = _find_entry(entries, "/imu/data")
    assert entry is not None
    assert entry.get("gz_topic_name") == "/imu"


def test_bridge_all_directions_valid() -> None:
    entries = _load_bridge()
    for entry in entries:
        direction = entry.get("direction")
        assert direction in _VALID_DIRECTIONS, (
            f"{entry.get('ros_topic_name')}: invalid direction {direction}"
        )


def test_bridge_all_entries_have_required_fields() -> None:
    entries = _load_bridge()
    for entry in entries:
        missing = _REQUIRED_FIELDS - set(entry.keys())
        assert not missing, (
            f"{entry.get('ros_topic_name', '?')}: missing fields {missing}"
        )


def test_bridge_covers_required_topics() -> None:
    entries = _load_bridge()
    documented = {e.get("ros_topic_name") for e in entries}
    missing = _REQUIRED_ROS_TOPICS - documented
    assert not missing, f"bridge.yaml missing topics: {missing}"
