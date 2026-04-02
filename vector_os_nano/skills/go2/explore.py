"""ExploreSkill -- non-blocking autonomous exploration.

Launches exploration in a background thread and returns IMMEDIATELY so the
CLI remains responsive.  The user can issue new commands (stop, navigate,
look) while exploration is running.  Any new movement command or stop()
cancels the exploration thread.

Architecture (like Claude Code background tasks):
    explore() → starts _explore_thread → returns instantly
    stop()    → sets _explore_cancel event → thread exits
    navigate()→ sets _explore_cancel event → then navigates

The exploration thread:
    1. Ensures bridge + nav stack are running
    2. Seeds FAR planner with initial movement
    3. Monitors position indefinitely, tracking rooms visited
    4. Checks _explore_cancel every 2 seconds
"""
from __future__ import annotations

import logging
import os
import shutil
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
)

logger = logging.getLogger(__name__)

_POSITION_SAMPLE_INTERVAL: float = 2.0

# Module-level singletons
_nav_stack_proc: subprocess.Popen | None = None
_bridge_thread: threading.Thread | None = None

def _start_tare() -> bool:
    """Start TARE autonomous exploration planner as a subprocess."""
    global _tare_proc
    if _tare_proc is not None and _tare_proc.poll() is None:
        return True  # already running

    # Check if TARE is already running
    if shutil.which("pgrep"):
        result = subprocess.run(["pgrep", "-f", "tare_planner"], capture_output=True)
        if result.returncode == 0:
            logger.info("[EXPLORE] TARE already running")
            return True

    try:
        # TARE needs ROS2 sourced — launch via bash
        cmd = "source /opt/ros/jazzy/setup.bash && "
        nav_stack = os.path.expanduser("~/Desktop/vector_navigation_stack")
        cmd += f"source {nav_stack}/install/setup.bash && "
        cmd += "ros2 launch tare_planner explore.launch scenario:=indoor_small"

        log_fh = open("/tmp/vector_tare.log", "w")
        _tare_proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        import atexit

        def _cleanup_tare():
            try:
                os.killpg(os.getpgid(_tare_proc.pid), signal.SIGTERM)
                _tare_proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(_tare_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            log_fh.close()

        atexit.register(_cleanup_tare)
        logger.info("[EXPLORE] TARE planner started")
        return True
    except Exception as exc:
        logger.error("[EXPLORE] TARE start failed: %s", exc)
        return False


def _stop_tare() -> None:
    """Stop TARE planner."""
    global _tare_proc
    if _tare_proc is not None and _tare_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_tare_proc.pid), signal.SIGTERM)
            _tare_proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(_tare_proc.pid), signal.SIGKILL)
            except Exception:
                pass
        _tare_proc = None
        logger.info("[EXPLORE] TARE planner stopped")


# Background exploration state (shared across skills)
_explore_thread: threading.Thread | None = None
_explore_cancel: threading.Event = threading.Event()
_explore_visited: set[str] = set()
_explore_running: bool = False
_tare_proc: subprocess.Popen | None = None
_on_event: Any = None  # callback(event_type: str, data: dict) — set by CLI
_auto_look: Any = None  # callback(room: str) -> dict | None — VLM look on new room


# ---------------------------------------------------------------------------
# Public API for other skills to check/cancel exploration
# ---------------------------------------------------------------------------

def is_exploring() -> bool:
    """Check if autonomous exploration is currently running."""
    return _explore_running


def cancel_exploration() -> None:
    """Request cancellation of the background exploration thread and TARE."""
    global _explore_running
    if _explore_running:
        _explore_cancel.set()
        _stop_tare()
        _emit("stopped", {"reason": "cancelled", "rooms": sorted(_explore_visited)})


def get_explored_rooms() -> list[str]:
    """Return rooms discovered during current/last exploration."""
    return sorted(_explore_visited)


def set_event_callback(callback: Any) -> None:
    """Set a callback for exploration events. Called from CLI.

    callback(event_type: str, data: dict) where event_type is one of:
        "started", "room_entered", "progress", "stopped", "completed"
    """
    global _on_event
    _on_event = callback


def set_auto_look(callback: Any) -> None:
    """Set auto-look callback, invoked when exploration enters a new room.

    callback(room: str) -> dict | None
        Returns VLM observation data dict or None on failure.
        Called from the background exploration thread — must be thread-safe.
    """
    global _auto_look
    _auto_look = callback


def _emit(event_type: str, data: dict | None = None) -> None:
    """Emit an exploration event to the callback (if set)."""
    if _on_event is not None:
        try:
            _on_event(event_type, data or {})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bridge / nav stack launchers (same dedup logic as before)
# ---------------------------------------------------------------------------

def _start_bridge_on_go2(go2: Any) -> bool:
    global _bridge_thread
    if _bridge_thread is not None and _bridge_thread.is_alive():
        return True

    try:
        import rclpy

        if not rclpy.ok():
            rclpy.init()

        # Check if bridge already running
        temp_node = rclpy.create_node("_bridge_check")
        try:
            topics = temp_node.get_topic_names_and_types()
            has_odom = any(name == "/state_estimation" for name, _ in topics)
        finally:
            temp_node.destroy_node()

        if has_odom:
            logger.info("[EXPLORE] Bridge already running")
            _bridge_thread = threading.Thread(target=lambda: None, daemon=True)
            _bridge_thread.start()
            return True

        import sys
        import importlib.util
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        bridge_path = os.path.join(_repo, "scripts", "go2_vnav_bridge.py")
        spec = importlib.util.spec_from_file_location("_vnav_bridge", bridge_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_vnav_bridge"] = mod
        spec.loader.exec_module(mod)

        node = mod.Go2VNavBridge(go2, quiet=True)
        _bridge_thread = threading.Thread(
            target=lambda: rclpy.spin(node), daemon=True,
        )
        _bridge_thread.start()
        return True

    except Exception as exc:
        logger.warning("[EXPLORE] Bridge start failed: %s", exc)
        return False


def _launch_nav_stack() -> bool:
    global _nav_stack_proc
    if _nav_stack_proc is not None and _nav_stack_proc.poll() is None:
        return True

    if shutil.which("pgrep"):
        result = subprocess.run(["pgrep", "-f", "localPlanner"], capture_output=True)
        if result.returncode == 0:
            logger.info("[EXPLORE] Nav stack already running")
            return True

    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    script = os.path.join(_repo, "scripts", "launch_nav_only.sh")
    if not os.path.isfile(script):
        return False

    try:
        log_fh = open("/tmp/vector_nav_only.log", "w")
        _nav_stack_proc = subprocess.Popen(
            [script], stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        import atexit

        def _cleanup():
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
        return True
    except Exception as exc:
        logger.error("[EXPLORE] Nav stack launch failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Background exploration loop
# ---------------------------------------------------------------------------

def _exploration_loop(base: Any, has_bridge: bool = True) -> None:
    """Background thread: start TARE, seed planner, then monitor rooms.

    Everything runs here — execute() returns immediately.
    Runs indefinitely until _explore_cancel is set.
    """
    global _explore_running, _explore_visited

    _explore_visited.clear()
    _explore_running = True
    _explore_cancel.clear()

    _emit("started", {"total_rooms": len(_ROOM_CENTERS)})

    # Start TARE planner (in this background thread, not blocking CLI)
    if has_bridge:
        tare_ok = _start_tare()
        if not tare_ok:
            logger.warning("[EXPLORE] TARE failed, exploration may be limited")

        # Seed planners: walk forward slowly to generate scan data for TARE/FAR.
        # Keep a small velocity even after seed — TARE needs continuous scans.
        logger.info("[EXPLORE] Seeding planners...")
        for _ in range(5):
            if _explore_cancel.is_set():
                _explore_running = False
                return
            base.set_velocity(0.15, 0.0, 0.0)
            time.sleep(1.0)
        # Don't zero velocity — keep creeping so TARE gets fresh scans
        # while it builds its initial visibility graph. Nav stack will
        # override this once it has a real path to follow.
        base.set_velocity(0.1, 0.0, 0.05)  # slow forward + gentle turn
        time.sleep(3.0)  # let TARE process scans

    # TARE needs continuous scan data to generate viewpoints. The bridge's
    # path follower only moves the robot when it has a /path from localPlanner,
    # which only has a path when TARE or FAR publishes /way_point. This creates
    # a deadlock: no movement → no scans → no viewpoints → no waypoints.
    #
    # Fix: keep sending a slow wander velocity every cycle. The bridge's
    # _cmd_vel_cb sets teleop_until=+0.5s, so this overrides the path
    # follower. Once TARE starts working, we stop sending wander commands
    # and let the nav stack drive.
    _last_wander = 0.0
    _wander_heading = 0.08  # gentle turn rate, alternates direction

    try:
        while not _explore_cancel.is_set():
            try:
                pos = base.get_position()
                if pos[2] < 0.12:
                    _emit("stopped", {"reason": "robot_fell", "rooms": sorted(_explore_visited)})
                    break

                # Wander: keep the robot moving so TARE gets scan data.
                # Send velocity every 2s. The /cmd_vel_nav topic gives
                # the bridge ~0.5s of teleop override, then path follower
                # tries to take over. By re-sending every 2s we ensure
                # continuous movement while still allowing nav stack paths
                # to briefly execute between our wander commands.
                if has_bridge:
                    now = time.time()
                    if now - _last_wander > 2.0:
                        base.set_velocity(0.15, 0.0, _wander_heading)
                        _last_wander = now
                        # Reverse turn direction occasionally to avoid circles
                        if len(_explore_visited) % 2 == 0:
                            _wander_heading = 0.08
                        else:
                            _wander_heading = -0.08

                room = _detect_current_room(float(pos[0]), float(pos[1]))
                if room not in _explore_visited:
                    _explore_visited.add(room)
                    _emit("room_entered", {
                        "room": room,
                        "visited": len(_explore_visited),
                        "total": len(_ROOM_CENTERS),
                        "all_rooms": sorted(_explore_visited),
                    })

                    # Auto-look: VLM scene capture at each new room
                    if _auto_look is not None:
                        try:
                            obs = _auto_look(room)
                            if obs:
                                _emit("room_observed", {
                                    "room": room,
                                    "summary": obs.get("summary", ""),
                                    "objects": obs.get("objects", []),
                                })
                                logger.info(
                                    "[EXPLORE] Auto-look %s: %s",
                                    room, obs.get("summary", "")[:80],
                                )
                        except Exception as exc:
                            logger.warning(
                                "[EXPLORE] Auto-look failed for %s: %s",
                                room, exc,
                            )

            except Exception:
                pass

            _explore_cancel.wait(timeout=_POSITION_SAMPLE_INTERVAL)

        if len(_explore_visited) >= len(_ROOM_CENTERS):
            _emit("completed", {"rooms": sorted(_explore_visited)})
        elif not _explore_cancel.is_set():
            _emit("stopped", {"reason": "finished", "rooms": sorted(_explore_visited)})

    finally:
        _explore_running = False


# ---------------------------------------------------------------------------
# ExploreSkill (non-blocking)
# ---------------------------------------------------------------------------

@skill(
    aliases=[
        "explore", "探索", "自主探索",
        "explore the house", "look around", "四处看看",
    ],
    direct=False,
)
class ExploreSkill:
    """Non-blocking autonomous exploration.

    Starts exploration in a background thread and returns immediately.
    The CLI remains responsive — user can stop, navigate, or look at any time.
    """

    name: str = "explore"
    description: str = (
        "Start autonomous exploration of the house. "
        "Runs in the BACKGROUND — you can give other commands while exploring. "
        "Use stop() to halt exploration."
    )
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"explored": True}
    failure_modes: list[str] = ["no_base", "exploration_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        global _explore_thread

        if context.base is None:
            return SkillResult(
                success=False, error_message="No base connected",
                diagnosis_code="no_base",
            )

        # If already exploring, report status
        if _explore_running:
            return SkillResult(
                success=True,
                result_data={
                    "status": "already_exploring",
                    "rooms_visited": sorted(_explore_visited),
                    "rooms_count": len(_explore_visited),
                },
            )

        base = context.base

        # Wire auto-look if VLM + camera are available
        vlm = context.services.get("vlm")
        spatial_memory = context.services.get("spatial_memory")
        if vlm is not None:
            def _do_auto_look(room: str) -> dict | None:
                """Capture frame, run VLM, record to scene graph."""
                try:
                    frame = base.get_camera_frame()
                    scene = vlm.describe_scene(frame)
                    room_id = vlm.identify_room(frame)
                    detected_room = room_id.room if room_id.room != "unknown" else room

                    obj_names = [o.name for o in scene.objects]
                    if spatial_memory is not None:
                        pos = base.get_position()
                        heading = base.get_heading()
                        if hasattr(spatial_memory, "observe_with_viewpoint"):
                            spatial_memory.observe_with_viewpoint(
                                detected_room, float(pos[0]), float(pos[1]),
                                float(heading), obj_names, scene.summary,
                            )
                        else:
                            spatial_memory.visit(detected_room, float(pos[0]), float(pos[1]))
                            spatial_memory.observe(detected_room, obj_names, scene.summary)

                    return {
                        "room": detected_room,
                        "summary": scene.summary,
                        "objects": [{"name": o.name, "confidence": o.confidence}
                                    for o in scene.objects],
                        "room_confidence": room_id.confidence,
                    }
                except Exception as exc:
                    logger.warning("[EXPLORE] auto-look VLM error: %s", exc)
                    return None

            set_auto_look(_do_auto_look)

        # Ensure bridge + nav stack are running
        bridge_ok = _start_bridge_on_go2(base)
        if bridge_ok:
            nav_ok = _launch_nav_stack()
            if not nav_ok:
                return SkillResult(
                    success=False, error_message="Nav stack failed to start",
                    diagnosis_code="exploration_failed",
                )

        # Start background thread that handles TARE launch + seeding + monitoring
        # Everything runs in background — execute() returns IMMEDIATELY
        _explore_thread = threading.Thread(
            target=_exploration_loop, args=(base, bridge_ok), daemon=True,
        )
        _explore_thread.start()

        return SkillResult(
            success=True,
            result_data={
                "status": "exploration_started",
                "note": "Running in background. Use stop() to halt. "
                        "You can navigate or look while exploring.",
            },
        )


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
