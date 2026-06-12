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


def resolve_habitat_viewer(requested: str | None = None) -> str:
    """Viewer camera mode: ``VECTOR_HABITAT_VIEWER`` env (first|chase) >
    caller request > default 'chase' (N3 — the owner wants to SEE the robot;
    'first' restores the pre-N3 eye view)."""
    env = os.environ.get("VECTOR_HABITAT_VIEWER", "")
    if env in ("first", "chase"):
        return env
    if requested in ("first", "chase"):
        return requested
    return "chase"


def resolve_habitat_viewer_size(requested: int | None = None) -> int:
    """Viewer window size in px: ``VECTOR_HABITAT_VIEWER_SIZE`` env > caller
    request > default 800 (owner finding 2026-06-12 round 2 — the 512 window
    was too small). The server clamps to [256, 1600]."""
    env = os.environ.get("VECTOR_HABITAT_VIEWER_SIZE", "")
    try:
        if env and int(env) > 0:
            return int(env)
    except ValueError:
        pass
    if requested and requested > 0:
        return int(requested)
    return 800


def resolve_robot_glb() -> str:
    """The composed rigid robot-body asset (N3), or '' when not built.

    External runtime data (never vendored): built once by
    ``scripts/build_g1_glb.py`` into the habitat data root.
    """
    from vector_os_nano.playground.habitat.scenes import habitat_data_root

    p = habitat_data_root() / "robots" / "unitree_g1" / "g1_rigid.glb"
    return str(p) if p.exists() else ""


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
    from vector_os_nano.playground.habitat.scenes import (
        dataset_navmesh_path,
        preflight_scene_instance,
        resolve_scene_dataset_config,
        resolve_scene_ref,
    )

    scenario = world.scenario
    if scenario.scene_dataset_config:
        # Composed dataset scene (N0): scene_ref is an instance NAME the
        # dataset config resolves; the authored navmesh rides along because
        # bare habitat_sim ignores the dataset's navmesh_instances mapping.
        dataset_config = resolve_scene_dataset_config(scenario.scene_dataset_config)
        scene = preflight_scene_instance(dataset_config, scenario.scene_ref)
        navmesh = dataset_navmesh_path(dataset_config, scene)
    else:
        dataset_config = ""
        navmesh = ""
        scene = resolve_scene_ref(scenario.scene_ref)
    show_gui = resolve_habitat_gui(gui)
    robot_glb = resolve_robot_glb()
    viewer_mode = resolve_habitat_viewer() if robot_glb else "first"
    if not robot_glb:
        _emit(
            on_status,
            "robot body asset missing (scripts/build_g1_glb.py builds it) — "
            "first-person view only",
        )
    _emit(
        on_status,
        f"Starting habitat scene '{scenario.id}' ({scene}) "
        f"[viewer window: {'on' if show_gui else 'off'}"
        f"{', ' + viewer_mode + ' view' if show_gui else ''}] ...",
    )
    base = HabitatBase(
        scene=scene, gui=show_gui, dataset_config=dataset_config, navmesh=navmesh,
        robot_glb=robot_glb, viewer_mode=viewer_mode,
        viewer_size=resolve_habitat_viewer_size(),
    )
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

    # Owner finding 2026-06-12 round 2: the habitat agent has no SceneGraph,
    # so without this the planner cannot know any room exists ('走到门口'
    # decomposed to a paramless navigate). The scenario's authored rooms
    # become world-model landmarks: listed WITH coordinates in the planner's
    # 'Objects (live)' context line and resolvable via navigate_to(label=...).
    seed_room_landmarks(getattr(agent, "_world_model", None), world.scenario)

    # DQ-6: the habitat agent gets the Qwen3-VL backbone (OpenRouter) for
    # visual verification — fail-soft: no key / import error leaves _vlm
    # unset and the visual verifier simply skips (never blocks boot).
    if getattr(agent, "_vlm", None) is None:
        try:
            import os as _os

            if _os.environ.get("OPENROUTER_API_KEY"):
                from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

                agent._vlm = Go2VLMPerception()
                _emit(on_status, "VLM: Qwen3-VL via OpenRouter (visual verify)")
        except Exception as exc:  # noqa: BLE001
            logger.debug("VLM wiring skipped: %s", exc)
    return agent


# Display/grounding aliases for the authored room names — the planner reads
# these in the objects line and copies the EXACT label into navigate params.
_ROOM_ALIASES: dict[str, str] = {
    "entryway": "门口/玄关",
    "kitchen": "厨房",
    "living_room": "客厅",
    "dining": "餐厅",
    "tv_corner": "电视角",
    "bedroom": "卧室",
    "bathroom": "浴室",
}


def markers_from_world_model(world_model: Any) -> "list[dict]":
    """The viewer-overlay marker set: every non-room world-model object.

    Room landmarks are navigation vocabulary, not detections — drawing five
    permanent room labels would bury the live SysNav objects."""
    out: list[dict] = []
    for o in world_model.get_objects():
        if (getattr(o, "properties", None) or {}).get("type") == "room":
            continue
        out.append({"label": o.label, "x": o.x, "y": o.y, "z": o.z})
    return out


def seed_room_landmarks(world_model: Any, scenario: Any) -> int:
    """Seed ``scenario.rooms`` into ``world_model`` as type=room landmarks.

    Each room rect ``(x0, y0, x1, y1)`` lands at its center with a stable
    ``room_<name>`` id (idempotent — re-seeding overwrites, never duplicates;
    sysnav's ``sysnav_<id>`` ids can never clash). Returns the count seeded.
    """
    rooms = dict(getattr(scenario, "rooms", None) or {})
    if world_model is None or not rooms:
        return 0
    from vector_os_nano.core.world_model import ObjectState

    for name, rect in rooms.items():
        x0, y0, x1, y1 = (float(v) for v in rect)
        # rect rides along: a room is a REGION (batch 2 #3) — navigate's
        # room branch sizes its arrival tolerance from these half-dims and
        # the honest predicate is visited('<name>') (inside-rect).
        props = {"type": "room", "rect": [x0, y0, x1, y1]}
        alias = _ROOM_ALIASES.get(name)
        if alias:
            props["alias"] = alias
        world_model.add_object(
            ObjectState(
                object_id=f"room_{name}",
                label=str(name),
                x=(x0 + x1) / 2.0,
                y=(y0 + y1) / 2.0,
                z=0.0,
                confidence=1.0,
                state="unknown",
                properties=props,
            )
        )
    return len(rooms)


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

    try:
        import rclpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover — env-specific
        raise RuntimeError(f"rclpy unavailable — source ROS2 first: {exc}") from exc

    from vector_os_nano.integrations.sysnav_bridge.live_bridge import LiveSysnavBridge
    from vector_os_nano.playground.habitat.sysnav_bridge import HabitatSysnavBridge

    # Detections made VISIBLE (owner finding 2026-06-12 round 2): every node
    # batch pushes the current non-room object set to the viewer overlay,
    # debounced so a chatty mapper cannot flood the stream channel.
    _last_push = [0.0]

    def _push_markers() -> None:
        import time as _time

        now = _time.monotonic()
        if now - _last_push[0] < 0.5:
            return
        _last_push[0] = now
        base = getattr(agent, "_base", None)
        if base is None or not callable(getattr(base, "set_markers", None)):
            return
        base.set_markers(markers_from_world_model(agent._world_model))

    consumer = LiveSysnavBridge(
        world_model=agent._world_model, on_batch=_push_markers
    )
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
    feed.spin_in_background()  # per-node executors: 50 Hz odom + 2 Hz pano

    agent._sysnav_feed = feed          # keep refs alive with the agent
    agent._sysnav_consumer = consumer
    # N4: the navigate skill discovers the nav-stack transport via the base
    # (SkillContext carries the base, not the agent).
    agent._base._nav_feed = feed
    _emit(
        on_status,
        "SysNav feed up (/camera/image /registered_scan /state_estimation 50Hz "
        "+ /cmd_vel in); objects flow into the world model",
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
        base = getattr(agent, "_base", None)
        if base is not None and getattr(base, "_nav_feed", None) is not None:
            base._nav_feed = None

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
