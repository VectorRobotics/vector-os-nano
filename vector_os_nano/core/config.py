# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Configuration loading and validation for Vector OS Nano SDK.

Public API:
    load_config(user_config=None) -> dict

ConfigError is raised on invalid configuration.

The default config is loaded from config/default.yaml relative to the
package root. User overrides are deep-merged on top.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration is invalid or required fields are missing."""


# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

# config/default.yaml lives at:  <repo_root>/config/default.yaml
# This file lives at:            <repo_root>/vector_os/core/config.py
_PACKAGE_ROOT = Path(__file__).parent.parent.parent  # repo root
_DEFAULT_YAML = _PACKAGE_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns a new dict.

    For dict values, recursion is applied. For all other types, override wins.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must be a YAML mapping, got {type(data).__name__}: {path}")
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    user_config: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Load configuration, starting from defaults and merging user overrides.

    Args:
        user_config: One of:
            - None: return defaults only
            - dict: deep-merged on top of defaults
            - str or Path: path to a YAML file, loaded then merged

    Returns:
        Merged configuration dict.

    Raises:
        FileNotFoundError: if user_config is a path that doesn't exist.
        ConfigError: if user_config is not a valid mapping.
    """
    try:
        cfg = _load_yaml(_DEFAULT_YAML)
    except FileNotFoundError:
        logger.warning("default.yaml not found, using hardcoded defaults")
        cfg = {
            "agent": {"max_planning_retries": 3, "max_execution_retries": 2, "planning_timeout_sec": 10.0},
            "llm": {"provider": "claude", "model": "anthropic/claude-sonnet-4-6",
                     "api_base": "https://openrouter.ai/api/v1", "temperature": 0.0, "max_tokens": 2048},
            "arm": {"type": "so101", "port": "/dev/ttyACM0", "baudrate": 1000000},
            "camera": {"type": "realsense", "serial": "", "resolution": [640, 480], "fps": 30},
            "perception": {"vlm_provider": "moondream", "vlm_model": "vikhyatk/moondream2",
                           "tracker": "edgetam", "tracker_model": "yonigozlan/EdgeTAM-hf"},
            "calibration": {"file": "", "method": "affine", "num_points": 25},
            "skills": {"pick": {"z_offset": 0.12, "pre_grasp_height": 0.06, "max_retries": 2},
                       "place": {"default_height": 0.05},
                       "home": {"joint_values": [-0.014, -1.238, 0.562, 0.858, 0.311]}},
            "ros2": {"enabled": False, "namespace": "", "enable_moveit": False, "enable_tf2": True},
        }

    if user_config is None:
        return cfg

    if isinstance(user_config, dict):
        override = user_config
    elif isinstance(user_config, (str, Path)):
        override = _load_yaml(user_config)
    else:
        raise ConfigError(
            f"user_config must be a dict, str, or Path, got {type(user_config).__name__}"
        )

    return _deep_merge(cfg, override)
