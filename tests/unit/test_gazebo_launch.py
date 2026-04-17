"""Tests for gazebo/launch/go2_sim.launch.py structure.

Validates the ROS2 launch file for Gazebo Harmonic Go2 simulation:
- File exists and is valid Python
- Declares required launch arguments (world, gui, use_rviz)
- Contains required node executables and packages
- Returns a LaunchDescription from generate_launch_description()

Level: Unit — pure file-parsing + import, no Gazebo or ROS2 runtime needed.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_LAUNCH_FILE = _REPO_ROOT / "gazebo" / "launch" / "go2_sim.launch.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_source() -> str:
    """Return the launch file source text."""
    return _LAUNCH_FILE.read_text(encoding="utf-8")


def _load_module():
    """Import and return the launch file module."""
    spec = importlib.util.spec_from_file_location("go2_sim_launch", str(_LAUNCH_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Tests — file existence and basic syntax
# ---------------------------------------------------------------------------


def test_launch_file_exists() -> None:
    """Launch file must be present at gazebo/launch/go2_sim.launch.py."""
    assert _LAUNCH_FILE.exists(), (
        f"Launch file not found at {_LAUNCH_FILE}\n"
        "Create gazebo/launch/go2_sim.launch.py"
    )


def test_launch_is_valid_python() -> None:
    """Launch file must be parseable as valid Python (AST check)."""
    source = _read_source()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"go2_sim.launch.py has syntax error: {exc}")


def test_launch_can_be_imported() -> None:
    """Launch file must import without crashing (no side-effects at import time)."""
    # Guard: skip if ROS2 environment unavailable
    try:
        import launch  # noqa: F401
    except ImportError:
        pytest.skip("launch package not available in this environment")

    module = _load_module()
    assert module is not None


# ---------------------------------------------------------------------------
# Tests — generate_launch_description function
# ---------------------------------------------------------------------------


def test_launch_has_generate_launch_description() -> None:
    """generate_launch_description function must be defined."""
    source = _read_source()
    assert "def generate_launch_description" in source, (
        "go2_sim.launch.py must define generate_launch_description()"
    )


def test_launch_returns_launch_description() -> None:
    """generate_launch_description() must return a LaunchDescription instance."""
    try:
        from launch import LaunchDescription  # noqa: F401
    except ImportError:
        pytest.skip("launch package not available in this environment")

    from launch import LaunchDescription

    module = _load_module()
    result = module.generate_launch_description()
    assert isinstance(result, LaunchDescription), (
        f"generate_launch_description() returned {type(result).__name__}, "
        "expected LaunchDescription"
    )


# ---------------------------------------------------------------------------
# Tests — launch arguments declared
# ---------------------------------------------------------------------------


def test_launch_declares_world_arg() -> None:
    """Launch file must declare a 'world' argument."""
    source = _read_source()
    assert "'world'" in source or '"world"' in source, (
        "go2_sim.launch.py must declare a 'world' launch argument"
    )
    assert "DeclareLaunchArgument" in source, (
        "go2_sim.launch.py must use DeclareLaunchArgument"
    )


def test_launch_declares_gui_arg() -> None:
    """Launch file must declare a 'gui' argument."""
    source = _read_source()
    assert "'gui'" in source or '"gui"' in source, (
        "go2_sim.launch.py must declare a 'gui' launch argument"
    )


def test_launch_declares_use_rviz_arg() -> None:
    """Launch file must declare a 'use_rviz' argument."""
    source = _read_source()
    assert "'use_rviz'" in source or '"use_rviz"' in source, (
        "go2_sim.launch.py must declare a 'use_rviz' launch argument"
    )


def test_launch_world_arg_has_default() -> None:
    """The 'world' argument must have a default value of 'apartment'."""
    source = _read_source()
    assert "apartment" in source, (
        "go2_sim.launch.py must have 'apartment' as the default world name"
    )


def test_launch_gui_arg_default_is_true() -> None:
    """The 'gui' argument must default to 'true'."""
    source = _read_source()
    # Should contain the string 'true' near the gui argument declaration
    assert "'true'" in source or '"true"' in source, (
        "go2_sim.launch.py must default 'gui' to 'true'"
    )


# ---------------------------------------------------------------------------
# Tests — required nodes present
# ---------------------------------------------------------------------------


def test_launch_includes_gz_sim() -> None:
    """Launch file must start Gz Sim (gz_sim or gz sim executable)."""
    source = _read_source()
    has_gz_sim_pkg = "ros_gz_sim" in source
    has_gz_sim_cmd = "gz sim" in source or "gz_sim" in source
    assert has_gz_sim_pkg or has_gz_sim_cmd, (
        "go2_sim.launch.py must include Gz Sim via ros_gz_sim package or gz sim command"
    )


def test_launch_includes_spawn_node() -> None:
    """Launch file must spawn the Go2 model using ros_gz_sim create node."""
    source = _read_source()
    assert "ros_gz_sim" in source, (
        "go2_sim.launch.py must use ros_gz_sim package for model spawning"
    )
    assert "create" in source, (
        "go2_sim.launch.py must use 'create' executable to spawn the Go2 model"
    )


def test_launch_includes_bridge_node() -> None:
    """Launch file must start the ros_gz_bridge parameter_bridge node."""
    source = _read_source()
    assert "ros_gz_bridge" in source, (
        "go2_sim.launch.py must include ros_gz_bridge"
    )
    assert "parameter_bridge" in source, (
        "go2_sim.launch.py must use 'parameter_bridge' executable"
    )



def test_launch_references_model_sdf() -> None:
    """Launch file must reference the Go2 model.sdf for spawning."""
    source = _read_source()
    assert "model.sdf" in source or "go2" in source.lower(), (
        "go2_sim.launch.py must reference Go2 model for spawning"
    )


def test_launch_includes_robot_state_publisher() -> None:
    """Launch file must include robot_state_publisher for TF tree."""
    source = _read_source()
    assert "robot_state_publisher" in source, (
        "go2_sim.launch.py must include robot_state_publisher node for TF"
    )
