"""ExploreSkill -- autonomous exploration using TARE planner.

When called, automatically launches the ROS2 bridge on the existing MuJoCoGo2
instance and starts the Vector Nav Stack + TARE planner. RViz opens for
visualization. On subsequent calls the stack is already running.

Fallback: when ROS2 is unavailable, visits rooms via dead-reckoning.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import (
    _ROOM_CENTERS,
    _detect_current_room,
    _distance,
    _navigate_to_waypoint,
)

logger = logging.getLogger(__name__)

_DEFAULT_DURATION: float = 60.0
_POSITION_SAMPLE_INTERVAL: float = 2.0
_VISIT_RADIUS: float = 2.5

# Module-level singleton: once the nav stack is launched, reuse it.
_nav_stack_proc: subprocess.Popen | None = None
_bridge_thread: threading.Thread | None = None


def _start_bridge_on_go2(go2: Any) -> bool:
    """Start Go2VNavBridge as a ROS2 node in a background thread.

    Reuses the existing MuJoCoGo2 instance (no second MuJoCo window).
    Returns True if bridge started, False on failure.
    """
    global _bridge_thread
    if _bridge_thread is not None and _bridge_thread.is_alive():
        return True

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        if not rclpy.ok():
            rclpy.init()

        # Import bridge class (heavy imports happen here)
        import sys
        import types
        import importlib.util
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        bridge_path = os.path.join(_repo, "scripts", "go2_vnav_bridge.py")
        spec = importlib.util.spec_from_file_location("_vnav_bridge", bridge_path)
        mod = importlib.util.module_from_spec(spec)
        # Prevent argparse from running
        sys.modules["_vnav_bridge"] = mod
        spec.loader.exec_module(mod)
        Go2VNavBridge = mod.Go2VNavBridge

        node = Go2VNavBridge(go2)

        def _spin() -> None:
            try:
                rclpy.spin(node)
            except Exception:
                pass

        _bridge_thread = threading.Thread(target=_spin, daemon=True)
        _bridge_thread.start()
        logger.info("[EXPLORE] ROS2 bridge started on existing Go2 instance")
        return True

    except Exception as exc:
        logger.warning("[EXPLORE] Failed to start bridge: %s", exc)
        return False


def _launch_nav_stack() -> bool:
    """Launch nav stack nodes (no bridge) via launch_nav_only.sh."""
    global _nav_stack_proc
    if _nav_stack_proc is not None and _nav_stack_proc.poll() is None:
        return True  # already running

    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    script = os.path.join(_repo, "scripts", "launch_nav_only.sh")
    if not os.path.isfile(script):
        logger.error("[EXPLORE] launch_nav_only.sh not found at %s", script)
        return False

    try:
        log_fh = open("/tmp/vector_nav_only.log", "w")
        _nav_stack_proc = subprocess.Popen(
            [script],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        import atexit

        def _cleanup() -> None:
            try:
                os.killpg(os.getpgid(_nav_stack_proc.pid), signal.SIGTERM)
                _nav_stack_proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(_nav_stack_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            log_fh.close()

        atexit.register(_cleanup)
        logger.info("[EXPLORE] Nav stack launching (log: /tmp/vector_nav_only.log)")
        return True

    except Exception as exc:
        logger.error("[EXPLORE] Failed to launch nav stack: %s", exc)
        return False


@skill(
    aliases=[
        "explore",
        "探索",
        "自主探索",
        "explore the house",
        "look around",
        "四处看看",
    ],
    direct=False,
)
class ExploreSkill:
    """Autonomous exploration — auto-launches nav stack + TARE + RViz."""

    name: str = "explore"
    description: str = (
        "Start autonomous exploration of the house. "
        "Launches navigation stack and RViz automatically."
    )
    parameters: dict = {
        "duration": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_DURATION,
            "description": "Exploration duration in seconds (default 60).",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"explored": True}
    failure_modes: list[str] = ["no_base", "exploration_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        duration: float = max(5.0, float(params.get("duration", _DEFAULT_DURATION)))
        base = context.base

        # Auto-launch bridge + nav stack if not already running
        bridge_ok = _start_bridge_on_go2(base)
        if bridge_ok:
            nav_ok = _launch_nav_stack()
            if nav_ok:
                logger.info("[EXPLORE] Waiting 20s for nav stack to initialize...")
                time.sleep(20)

                # Seed FAR planner with initial movement
                logger.info("[EXPLORE] Seeding planners with initial movement...")
                for _ in range(4):
                    base.set_velocity(0.2, 0.0, 0.0)
                    time.sleep(1.0)
                base.set_velocity(0.0, 0.0, 0.0)
                time.sleep(2.0)

                return self._monitor_exploration(base, duration)

        # Fallback: dead-reckoning
        logger.info("[EXPLORE] Nav stack unavailable, using dead-reckoning")
        return self._dead_reckoning_exploration(base, duration)

    def _monitor_exploration(self, base: Any, duration: float) -> SkillResult:
        """Monitor TARE exploration, track rooms visited."""
        visited: set[str] = set()
        deadline = time.time() + duration

        while time.time() < deadline:
            try:
                pos = base.get_position()
                room = _detect_current_room(float(pos[0]), float(pos[1]))
                if room not in visited:
                    visited.add(room)
                    logger.info("[EXPLORE] Entered room: %s", room)
            except Exception as exc:
                logger.warning("[EXPLORE] Position read error: %s", exc)
            time.sleep(_POSITION_SAMPLE_INTERVAL)

        return _build_result(visited, duration, mode="tare")

    def _dead_reckoning_exploration(
        self, base: Any, duration: float,
    ) -> SkillResult:
        """Visit rooms via turn+walk dead-reckoning."""
        visited: set[str] = set()
        deadline = time.time() + duration

        try:
            pos = base.get_position()
            start_room = _detect_current_room(float(pos[0]), float(pos[1]))
            visited.add(start_room)
        except Exception:
            pass

        visit_order = ["hallway"] + [
            r for r in _ROOM_CENTERS if r != "hallway"
        ]

        for room in visit_order:
            if time.time() >= deadline:
                break
            target = _ROOM_CENTERS[room]
            try:
                pos = base.get_position()
                if _distance(pos[0], pos[1], target[0], target[1]) < _VISIT_RADIUS:
                    visited.add(room)
                    continue
            except Exception:
                pass

            ok = _navigate_to_waypoint(base, target[0], target[1], room)
            if ok:
                visited.add(room)
            else:
                break  # robot fell

        return _build_result(visited, duration, mode="dead_reckoning")


def _build_result(visited: set[str], duration: float, mode: str) -> SkillResult:
    rooms_list = sorted(visited)
    total_rooms = len(_ROOM_CENTERS)
    coverage = round(len(visited) / total_rooms * 100.0, 1) if total_rooms else 0.0

    return SkillResult(
        success=True,
        result_data={
            "rooms_visited": rooms_list,
            "rooms_visited_count": len(visited),
            "total_rooms": total_rooms,
            "coverage_percent": coverage,
            "duration_s": duration,
            "mode": mode,
        },
    )
