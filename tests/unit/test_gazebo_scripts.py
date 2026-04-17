"""
Tests for Gazebo launch/stop shell scripts.

Follows TDD: tests are written first (RED), then the scripts are created (GREEN).
No actual Gazebo processes are started — only script structure is validated.
"""

import os
import stat
import subprocess

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
LAUNCH_SCRIPT = os.path.join(SCRIPTS_DIR, "launch_gazebo.sh")
STOP_SCRIPT = os.path.join(SCRIPTS_DIR, "stop_gazebo.sh")


# ---------------------------------------------------------------------------
# launch_gazebo.sh — existence and metadata
# ---------------------------------------------------------------------------


def test_launch_script_exists():
    assert os.path.isfile(LAUNCH_SCRIPT), (
        f"launch_gazebo.sh not found at {LAUNCH_SCRIPT}"
    )


def test_launch_script_executable():
    assert os.access(LAUNCH_SCRIPT, os.X_OK), (
        "launch_gazebo.sh is not executable (run chmod +x)"
    )


def test_launch_script_has_shebang():
    with open(LAUNCH_SCRIPT) as fh:
        first_line = fh.readline().rstrip("\n")
    assert first_line == "#!/bin/bash", (
        f"Expected shebang '#!/bin/bash', got: {first_line!r}"
    )


# ---------------------------------------------------------------------------
# launch_gazebo.sh — preflight checks
# ---------------------------------------------------------------------------


def test_launch_script_has_preflight_gz_version():
    """Script must check that gz sim is available."""
    content = open(LAUNCH_SCRIPT).read()
    assert "gz sim" in content or "gz version" in content or "--version" in content, (
        "launch_gazebo.sh must contain a gz sim version / availability check"
    )


def test_launch_script_has_preflight_ros2():
    """Script must verify ROS2 is sourced (ros2 topic list or ros2 command)."""
    content = open(LAUNCH_SCRIPT).read()
    assert "ros2" in content, (
        "launch_gazebo.sh must reference ros2 for preflight check"
    )


def test_launch_script_has_preflight_world_file():
    """Script must verify the world SDF file exists before launching."""
    content = open(LAUNCH_SCRIPT).read()
    # Checking for file existence test patterns: -f, -e, test -f, [[ -f
    assert "-f " in content or "-e " in content, (
        "launch_gazebo.sh must check that the world SDF file exists (-f or -e)"
    )


# ---------------------------------------------------------------------------
# launch_gazebo.sh — argument handling
# ---------------------------------------------------------------------------


def test_launch_script_accepts_world_arg():
    content = open(LAUNCH_SCRIPT).read()
    assert "--world" in content, (
        "launch_gazebo.sh must handle --world argument"
    )


def test_launch_script_accepts_headless_arg():
    content = open(LAUNCH_SCRIPT).read()
    assert "--headless" in content, (
        "launch_gazebo.sh must handle --headless argument"
    )


def test_launch_script_accepts_controller_arg():
    content = open(LAUNCH_SCRIPT).read()
    assert "--controller" in content, (
        "launch_gazebo.sh must handle --controller argument"
    )


def test_launch_script_has_help():
    """--help flag must work and print usage without error."""
    result = subprocess.run(
        ["bash", LAUNCH_SCRIPT, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"launch_gazebo.sh --help returned non-zero exit code {result.returncode}.\n"
        f"stderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert combined.strip(), "--help produced no output"


# ---------------------------------------------------------------------------
# stop_gazebo.sh — existence and metadata
# ---------------------------------------------------------------------------


def test_stop_script_exists():
    assert os.path.isfile(STOP_SCRIPT), (
        f"stop_gazebo.sh not found at {STOP_SCRIPT}"
    )


def test_stop_script_executable():
    assert os.access(STOP_SCRIPT, os.X_OK), (
        "stop_gazebo.sh is not executable (run chmod +x)"
    )


def test_stop_script_has_shebang():
    with open(STOP_SCRIPT) as fh:
        first_line = fh.readline().rstrip("\n")
    assert first_line == "#!/bin/bash", (
        f"Expected shebang '#!/bin/bash', got: {first_line!r}"
    )


def test_stop_script_kills_gz():
    """stop_gazebo.sh must kill gz sim processes."""
    content = open(STOP_SCRIPT).read()
    has_kill = "pkill" in content or "killall" in content or "kill " in content
    targets_gz = "gz" in content
    assert has_kill and targets_gz, (
        "stop_gazebo.sh must use pkill/kill targeting gz processes"
    )
