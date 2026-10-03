# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Packaging smoke: the CLI kernel imports without pulling robot/heavy deps.

Phase A success criterion SC-1: a base install can import and start the CLI
with zero robot dependencies (no mujoco/torch/pybullet/rclpy/etc.).
"""

from __future__ import annotations

import sys

# Heavy / robot-only modules that must NOT be imported by the kernel path.
_ROBOT_DEPS = (
    "mujoco",
    "pybullet",
    "torch",
    "transformers",
    "pyrealsense2",
    "open3d",
    "rclpy",
)


def test_cli_imports_without_robot_deps(monkeypatch) -> None:
    """Importing the CLI module must not import any robot/heavy dependency."""
    # Hide any pre-imported robot deps so we observe the CLI import's own effect.
    # monkeypatch puts them back: a re-imported torch re-registers its TORCH_LIBRARY
    # namespaces and raises in every later torch user.
    for mod in _ROBOT_DEPS:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    import vector_os_nano.vcli.cli  # noqa: F401

    leaked = [m for m in _ROBOT_DEPS if m in sys.modules]
    assert leaked == [], f"CLI import leaked robot deps: {leaked}"


def test_cli_main_is_callable() -> None:
    """The console-script entry point exists and is callable."""
    from vector_os_nano.vcli.cli import main

    assert callable(main)
