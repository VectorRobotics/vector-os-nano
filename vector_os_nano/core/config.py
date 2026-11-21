# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Configuration loading and validation for Vector OS Nano SDK.

Public API:
    load_config(user_config=None) -> dict

ConfigError is raised on invalid configuration.

The default config is bundled beside this module. User overrides are deep-merged on top.
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

_DEFAULT_YAML = Path(__file__).with_name('default.yaml')


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
    cfg = _load_yaml(_DEFAULT_YAML)

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
