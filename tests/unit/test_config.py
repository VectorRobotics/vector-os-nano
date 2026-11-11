# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for vector_os_nano.core.config — TDD RED phase.

Tests cover:
- Default config loading (no user overrides)
- User config override (dict)
- User config override (YAML path)
- Missing required fields raise ConfigError
- Type validation of values
- Unknown keys are silently ignored (permissive merge)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_temp_yaml(data: dict) -> str:
    """Write a dict to a temporary YAML file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# ConfigError import
# ---------------------------------------------------------------------------


class TestConfigErrorExists:
    def test_config_error_importable(self):
        from vector_os_nano.core.config import ConfigError  # noqa: F401


# ---------------------------------------------------------------------------
# Default config loading
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_load_defaults_no_error(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_default_agent_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "agent" in cfg
        assert cfg["agent"]["max_planning_retries"] == 3
        assert cfg["agent"]["max_execution_retries"] == 2
        assert cfg["agent"]["planning_timeout_sec"] == 10.0

    def test_default_llm_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "llm" in cfg
        assert cfg["llm"]["provider"] == "claude"
        assert cfg["llm"]["temperature"] == 0.0
        assert cfg["llm"]["max_tokens"] == 2048

    def test_default_arm_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "arm" in cfg
        assert cfg["arm"]["type"] == "so101"
        assert cfg["arm"]["baudrate"] == 1000000

    def test_default_camera_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "camera" in cfg
        assert cfg["camera"]["type"] == "realsense"
        assert cfg["camera"]["fps"] == 30

    def test_default_perception_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "perception" in cfg
        assert cfg["perception"]["vlm_provider"] == "moondream"
        assert cfg["perception"]["tracker"] == "edgetam"

    def test_default_calibration_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "calibration" in cfg
        assert cfg["calibration"]["method"] == "affine"
        assert cfg["calibration"]["num_points"] == 25

    def test_default_skills_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "skills" in cfg
        assert "pick" in cfg["skills"]
        assert cfg["skills"]["pick"]["z_offset"] == 0.1
        assert cfg["skills"]["pick"]["pre_grasp_height"] == 0.06
        assert cfg["skills"]["pick"]["max_retries"] == 2

    def test_default_ros2_section(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        assert "ros2" in cfg
        assert cfg["ros2"]["enabled"] is False

    def test_default_home_joint_values(self):
        from vector_os_nano.core.config import load_config

        cfg = load_config()
        home_joints = cfg["skills"]["home"]["joint_values"]
        assert isinstance(home_joints, list)
        assert len(home_joints) == 5


# ---------------------------------------------------------------------------
# User override — dict
# ---------------------------------------------------------------------------


class TestUserOverrideDict:
    def test_override_scalar(self):
        from vector_os_nano.core.config import load_config

        override = {"llm": {"provider": "openai", "temperature": 0.5}}
        cfg = load_config(user_config=override)
        assert cfg["llm"]["provider"] == "openai"
        assert cfg["llm"]["temperature"] == 0.5

    def test_override_preserves_non_overridden_keys(self):
        from vector_os_nano.core.config import load_config

        override = {"llm": {"provider": "openai"}}
        cfg = load_config(user_config=override)
        # max_tokens must still be at default
        assert cfg["llm"]["max_tokens"] == 2048

    def test_override_nested_section(self):
        from vector_os_nano.core.config import load_config

        override = {"arm": {"port": "/dev/ttyUSB0"}}
        cfg = load_config(user_config=override)
        assert cfg["arm"]["port"] == "/dev/ttyUSB0"
        # Other arm fields preserved
        assert cfg["arm"]["type"] == "so101"

    def test_override_add_unknown_key(self):
        """Unknown keys in override are silently accepted."""
        from vector_os_nano.core.config import load_config

        override = {"custom_section": {"key": "value"}}
        cfg = load_config(user_config=override)
        assert cfg["custom_section"]["key"] == "value"

    def test_empty_override_gives_defaults(self):
        from vector_os_nano.core.config import load_config

        cfg1 = load_config()
        cfg2 = load_config(user_config={})
        assert cfg1 == cfg2


# ---------------------------------------------------------------------------
# User override — YAML path
# ---------------------------------------------------------------------------


class TestUserOverrideYAMLPath:
    def test_load_from_yaml_path(self):
        from vector_os_nano.core.config import load_config

        override_data = {"llm": {"provider": "local", "model": "llama3"}}
        path = write_temp_yaml(override_data)
        try:
            cfg = load_config(user_config=path)
            assert cfg["llm"]["provider"] == "local"
            assert cfg["llm"]["model"] == "llama3"
        finally:
            os.unlink(path)

    def test_load_from_yaml_preserves_defaults(self):
        from vector_os_nano.core.config import load_config

        override_data = {"arm": {"port": "/dev/ttyUSB1"}}
        path = write_temp_yaml(override_data)
        try:
            cfg = load_config(user_config=path)
            assert cfg["arm"]["port"] == "/dev/ttyUSB1"
            assert cfg["arm"]["baudrate"] == 1000000  # default preserved
        finally:
            os.unlink(path)

    def test_missing_yaml_file_raises(self):
        from vector_os_nano.core.config import ConfigError, load_config

        with pytest.raises((ConfigError, FileNotFoundError)):
            load_config(user_config="/nonexistent/path/config.yaml")

    def test_path_object_accepted(self):
        from vector_os_nano.core.config import load_config

        override_data = {"camera": {"fps": 60}}
        path = write_temp_yaml(override_data)
        try:
            cfg = load_config(user_config=Path(path))
            assert cfg["camera"]["fps"] == 60
        finally:
            os.unlink(path)
