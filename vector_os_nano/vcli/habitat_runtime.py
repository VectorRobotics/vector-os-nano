# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared habitat-world runtime: boot, SysNav wiring, teardown.

Single source for BOTH entry paths (rule 3 — no split-brain):

* ``cli._maybe_init_habitat_agent`` — the ``--scenario apartment`` launch flag;
* ``start_simulation(sim_type="habitat")`` — the NL path ("启动habitat模拟").

The kernel never imports a simulator from config: callers hand in the RESOLVED
playground world (M1 seam carries ``sim_backend``/``scene_ref``); every habitat
import below is lazy and function-local. ``on_status`` is an optional callback
for human-readable progress lines (the CLI prints them dim; tools collect them).
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SYSNAV_LOG_PATH = "/tmp/vector_sysnav.log"


def _emit(on_status: Callable[[str], None] | None, line: str) -> None:
    if on_status is not None:
        try:
            on_status(line)
        except Exception:  # noqa: BLE001 — status display must never break boot
            pass


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def resolve_habitat_gui(requested: bool | None = None) -> bool:
    """Decide whether the habitat server opens its live viewer window.

    Precedence: explicit ``VECTOR_HABITAT_GUI`` env (operator override: "0"
    forces headless, "1" forces the window) > the caller's request (the
    start_simulation ``gui`` param) > desktop default (window when DISPLAY
    is present — the owner wants to SEE the sim; CI/headless boxes get none).
    """
    env = os.environ.get("VECTOR_HABITAT_GUI")
    if env in ("0", "1"):
        return env == "1"
    if requested is not None:
        return bool(requested)
    return bool(os.environ.get("DISPLAY"))


def boot_habitat_agent(
    world: Any,
    on_status: Callable[[str], None] | None = None,
    gui: bool | None = None,
) -> Any:
    """Boot the kinematic habitat base for ``world`` and return a ready Agent.

    Raises on any failure — callers decide how loud to surface it (the CLI
    keeps the REPL alive with fail-safe predicates; the tool returns an error).
    """
    from vector_os_nano.core.agent import Agent  # type: ignore[import]
    from vector_os_nano.playground.habitat import HabitatBase
    from vector_os_nano.playground.habitat.scenes import resolve_scene_ref

    scenario = world.scenario
    scene = resolve_scene_ref(scenario.scene_ref)
    show_gui = resolve_habitat_gui(gui)
    _emit(
        on_status,
        f"Starting habitat scene '{scenario.id}' ({scene}) "
        f"[viewer window: {'on' if show_gui else 'off'}] ...",
    )
    base = HabitatBase(scene=scene, gui=show_gui)
    base.connect()
    agent = Agent(base=base)

    # Base-only world: the registry-derived decompose vocab must teach ONLY
    # base-capable skills (rule 3 — single-source, no split-brain; the arm
    # defaults would put pick/place in the planner's mouth with no arm
    # attached). Rebuild the registry with the mobile set.
    # TODO: promote to a public Agent API (skills_replace=...) later.
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.skills.go2.turn import TurnSkill
    from vector_os_nano.skills.go2.walk import WalkSkill
    from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

    registry = SkillRegistry()
    for s in (WalkSkill(), TurnSkill(), StopSkill(), NavigateToPointSkill()):
        registry.register(s)
    agent._skill_registry = registry
    return agent


# ---------------------------------------------------------------------------
# SysNav semantic perception
# ---------------------------------------------------------------------------


def wire_sysnav_feed(agent: Any, on_status: Callable[[str], None] | None = None) -> None:
    """Wire the in-process SysNav feed + consumer into ``agent`` (idempotent).

    Feed: HabitatSysnavBridge publishes /camera/image /registered_scan
    /state_estimation off the agent's base (thread-safe bridge). Consumer:
    LiveSysnavBridge subscribes /object_nodes_list into the agent's world
    model. The heavy perception NODES run outside the REPL — see
    :func:`launch_sysnav_nodes`.

    Raises ``RuntimeError`` loudly when a dependency is missing (rclpy, or
    tare_planner msgs from the unsourced SysNav workspace) — never degrades
    to a silent no-op consumer.
    """
    if getattr(agent, "_sysnav_feed", None) is not None:
        return  # already wired

    import threading

    try:
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as exc:  # pragma: no cover — env-specific
        raise RuntimeError(f"rclpy unavailable — source ROS2 first: {exc}") from exc

    from vector_os_nano.integrations.sysnav_bridge.live_bridge import LiveSysnavBridge
    from vector_os_nano.playground.habitat.sysnav_bridge import HabitatSysnavBridge

    consumer = LiveSysnavBridge(world_model=agent._world_model)
    if not consumer.start():
        # start() returning False means tare_planner.msg (or rclpy) is missing
        # — a no-op consumer would silently eat every detection. Fail loud.
        raise RuntimeError(
            "SysNav consumer could not subscribe (tare_planner msgs not "
            "importable). Launch vector-cli from a shell with BOTH sourced: "
            "source /opt/ros/jazzy/setup.bash && "
            "source ~/Desktop/SysNav/install/setup.bash"
        )

    feed = HabitatSysnavBridge(agent._base, hz=2.0)
    executor = MultiThreadedExecutor()
    executor.add_node(feed.node)
    threading.Thread(
        target=executor.spin, daemon=True, name="habitat-sysnav-feed"
    ).start()

    agent._sysnav_feed = feed          # keep refs alive with the agent
    agent._sysnav_consumer = consumer
    agent._sysnav_executor = executor
    _emit(
        on_status,
        "SysNav feed up (/camera/image /registered_scan /state_estimation); "
        "objects flow into the world model",
    )


def launch_sysnav_nodes(
    on_status: Callable[[str], None] | None = None,
) -> tuple[subprocess.Popen, Any]:
    """Launch the SysNav perception pair (detection + semantic mapping).

    Runs ``scripts/launch_sysnav_nodes.sh`` in its own process group (heavy
    GPU processes, ~1 min of model loading), logging to ``SYSNAV_LOG_PATH``.
    Returns ``(proc, log_fh)``. Preflights the sibling workspace and fails
    loud with the exact fix — never a half-started silent pair.
    """
    sysnav_ws = Path(os.environ.get("VECTOR_SYSNAV_WS", str(Path.home() / "Desktop/SysNav")))
    venv_python = sysnav_ws / ".venv-sysnav/bin/python"
    if not venv_python.is_file():
        raise RuntimeError(
            f"SysNav workspace not provisioned: {venv_python} missing "
            "(expected the sibling workspace at ~/Desktop/SysNav with its "
            ".venv-sysnav; set VECTOR_SYSNAV_WS to override)"
        )

    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "launch_sysnav_nodes.sh"
    if not script.is_file():
        raise RuntimeError(f"SysNav launcher missing: {script}")

    log_fh = open(SYSNAV_LOG_PATH, "w")  # noqa: SIM115 — handle stored for stop
    proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        cwd=str(repo),
    )
    _emit(
        on_status,
        f"SysNav nodes launching (pid {proc.pid}, log {SYSNAV_LOG_PATH}) — "
        "model loading takes ~1 min before detections appear",
    )
    return proc, log_fh


def shutdown_sysnav(agent: Any) -> list[str]:
    """Tear down everything :func:`wire_sysnav_feed`/:func:`launch_sysnav_nodes`
    attached to ``agent``. Idempotent; returns human-readable summary lines."""
    import signal

    lines: list[str] = []

    proc = getattr(agent, "_sysnav_proc", None)
    if proc is not None:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
                lines.append("SysNav nodes stopped")
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    lines.append("SysNav nodes force-killed")
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"SysNav node kill failed: {exc}")
        agent._sysnav_proc = None
    log_fh = getattr(agent, "_sysnav_log_fh", None)
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:  # noqa: BLE001
            pass
        agent._sysnav_log_fh = None

    consumer = getattr(agent, "_sysnav_consumer", None)
    if consumer is not None:
        try:
            consumer.stop()
            lines.append("SysNav consumer stopped")
        except Exception:  # noqa: BLE001
            pass
        agent._sysnav_consumer = None

    feed = getattr(agent, "_sysnav_feed", None)
    if feed is not None:
        executor = getattr(agent, "_sysnav_executor", None)
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception:  # noqa: BLE001
                pass
            agent._sysnav_executor = None
        try:
            feed.destroy()
            lines.append("SysNav feed stopped")
        except Exception:  # noqa: BLE001
            pass
        agent._sysnav_feed = None

    return lines


def sysnav_status_lines(agent: Any) -> list[str]:
    """Human-readable SysNav component status for tools/status surfaces."""
    lines: list[str] = []
    feed = getattr(agent, "_sysnav_feed", None)
    lines.append("SysNav feed: up" if feed is not None else "SysNav feed: not wired")
    proc = getattr(agent, "_sysnav_proc", None)
    if proc is not None and proc.poll() is None:
        lines.append(f"SysNav nodes: running (pid {proc.pid})")
    elif proc is not None:
        lines.append(f"SysNav nodes: exited (code {proc.poll()}, log {SYSNAV_LOG_PATH})")
    else:
        lines.append("SysNav nodes: not started")
    wm = getattr(agent, "_world_model", None)
    if wm is not None:
        try:
            lines.append(f"Live objects: {len(wm.get_objects())}")
        except Exception:  # noqa: BLE001
            pass
    return lines
