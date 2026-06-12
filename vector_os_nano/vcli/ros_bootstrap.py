# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Self-bootstrap the ROS environment — NL start without terminal rituals.

``LD_LIBRARY_PATH`` is read by the dynamic loader at process START, so an
unsourced vector-cli can never import ``rclpy`` by patching ``sys.path``
mid-flight. Instead, when the ROS / SysNav overlays exist on disk but the
process env lacks them, :func:`maybe_reexec_with_ros` sources them in a bash
child, captures the resulting env, and re-execs the CLI exactly once
(``VECTOR_ROS_BOOTSTRAPPED`` is the loop guard; ``VECTOR_NO_ROS_BOOTSTRAP=1``
opts out — the repo conftest sets it so pytest is never replaced by execve).

Boxes without ROS, fully sourced shells, and compose failures are all silent
no-ops: the SysNav tools keep their own fail-loud paths as the backstop.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ROS_ROOT = Path("/opt/ros")


def overlay_scripts(
    environ: dict[str, str] | os._Environ = os.environ,
    ros_root: Path = _DEFAULT_ROS_ROOT,
) -> list[Path]:
    """The setup.bash overlays to source, base ROS first.

    No ROS on disk means NO scripts at all — a workspace overlay without the
    underlay would only produce a broken half-env.
    """
    distros = sorted(ros_root.glob("*/setup.bash")) if ros_root.is_dir() else []
    if not distros:
        return []
    scripts = [distros[-1]]  # newest distro (lexically: jazzy > humble)
    ws = Path(environ.get("VECTOR_SYSNAV_WS", str(Path.home() / "Desktop/SysNav")))
    ws_setup = ws / "install" / "setup.bash"
    if ws_setup.is_file():
        scripts.append(ws_setup)
    return scripts


def needs_bootstrap(
    environ: dict[str, str] | os._Environ, scripts: list[Path]
) -> bool:
    """True when any overlay on disk is absent from ``AMENT_PREFIX_PATH``.

    A colcon workspace setup.bash registers per-package prefixes under its
    install dir, so a substring check on the script's parent dir (``/opt/ros/
    jazzy`` / ``.../SysNav/install``) discriminates sourced from unsourced.
    """
    if not scripts:
        return False
    ament = environ.get("AMENT_PREFIX_PATH", "")
    return any(str(s.parent) not in ament for s in scripts)


def compose_sourced_env(
    scripts: list[Path], base_env: dict[str, str] | os._Environ
) -> dict[str, str] | None:
    """Source ``scripts`` in a bash child and capture the resulting env.

    ``env -0`` keeps multiline values intact. Returns None on any failure —
    the caller continues unsourced (fail-open; the tools fail loud later).
    """
    cmd = " && ".join(f"source {_quote(s)}" for s in scripts) + " && env -0"
    try:
        out = subprocess.run(
            ["bash", "-c", cmd],
            env=dict(base_env),
            capture_output=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ROS env bootstrap failed (continuing unsourced): %s", exc)
        return None
    env: dict[str, str] = {}
    for entry in out.split(b"\0"):
        if b"=" in entry:
            key, _, val = entry.partition(b"=")
            env[key.decode()] = val.decode()
    return env or None


def maybe_reexec_with_ros(
    environ: os._Environ | dict[str, str] = os.environ,
    ros_root: Path = _DEFAULT_ROS_ROOT,
) -> None:
    """Re-exec the CLI once with the ROS overlays sourced, when needed.

    Does not return when it re-execs. Every other path is a cheap no-op
    (string checks only — no subprocess, no rclpy import).
    """
    if environ.get("VECTOR_ROS_BOOTSTRAPPED"):
        return
    if environ.get("VECTOR_NO_ROS_BOOTSTRAP") == "1":
        return
    scripts = overlay_scripts(environ, ros_root=ros_root)
    if not needs_bootstrap(environ, scripts):
        return
    env = compose_sourced_env(scripts, base_env=environ)
    if env is None:
        return
    # Loop guard; carries the overlay list so the re-exec'd process can show
    # the owner what was auto-sourced.
    env["VECTOR_ROS_BOOTSTRAPPED"] = ":".join(str(s) for s in scripts)
    logger.info("re-exec with ROS overlays sourced: %s", ", ".join(map(str, scripts)))
    os.execve(
        sys.executable,
        [sys.executable, "-m", "vector_os_nano.vcli.cli", *sys.argv[1:]],
        env,
    )


def _quote(p: Path) -> str:
    return "'" + str(p).replace("'", "'\\''") + "'"
