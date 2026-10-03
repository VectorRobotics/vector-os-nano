# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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
from vector_os_nano.perception.object_localizer import localize_objects_3d
from vector_os_nano.core.run_dir import run_path

logger = logging.getLogger(__name__)

_POSITION_SAMPLE_INTERVAL: float = 2.0

# Module-level singletons
_nav_stack_proc: subprocess.Popen | None = None
_bridge_thread: threading.Thread | None = None

def _start_tare() -> bool:
    """Check that TARE autonomous exploration planner is already running.

    TARE must be launched by _launch_ros2_stack() (via launch_explore.sh) before
    exploration starts.  This function only detects a running TARE process — it
    does NOT start one.  Launching TARE here would create a second process group
    with broken DDS connectivity.

    """

    # Check if TARE is already running (launched by _launch_ros2_stack / launch_explore.sh)
    if shutil.which("pgrep"):
        result = subprocess.run(["pgrep", "-f", "tare_planner_node"], capture_output=True)
        if result.returncode == 0:
            logger.info("[EXPLORE] TARE already running — reusing")
            return True

    logger.warning(
        "[EXPLORE] TARE not running. Launch the nav stack first via "
        "sim/hardware stack startup (launch_explore.sh). Exploration cannot start."
    )
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
_spatial_memory: Any = None  # SceneGraph — set by ExploreSkill.execute()
_perception: Any = None  # Go2GraspPerception-like — set by ExploreSkill.execute()
_vlm: Any = None  # VLM service — set by ExploreSkill.execute()


# ---------------------------------------------------------------------------
# Public API for other skills to check/cancel exploration
# ---------------------------------------------------------------------------

def is_exploring() -> bool:
    """Check if autonomous exploration is currently running."""
    return _explore_running


def get_explore_status() -> dict:
    """Get current exploration progress.

    Returns dict with:
        running: bool — is exploration active
        rooms_found: list[str] — rooms discovered this session
        rooms_found_count: int
        total_expected: int — target room count (from SceneGraph)
    """
    total = 0
    if _spatial_memory is not None:
        try:
            total = len(_spatial_memory.get_all_rooms())
        except Exception:
            pass
    return {
        "running": _explore_running,
        "rooms_found": sorted(_explore_visited),
        "rooms_found_count": len(_explore_visited),
        "total_expected": total,
    }


def _own_descendants_named(name: str, root: int | None = None) -> list[int]:
    """PIDs anywhere below *root* (default: this process) whose command line contains *name*.

    TARE runs inside the nav-stack launch (launch_explore.sh) this CLI started, a
    descendant in another process group, so its group cannot be killed without
    FAR + localPlanner. Matching by name only among OUR descendants never touches
    another project's planner on a shared machine (a bare pkill -f did).
    """
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    found, todo = [], [os.getpid() if root is None else root]
    while todo:
        for pid in children.get(todo.pop(), []):
            todo.append(pid)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    if name.encode() in fh.read():
                        found.append(pid)
            except OSError:
                pass
    return found


def stop_tare_only() -> None:
    """Stop TARE planner while keeping FAR + localPlanner alive.

    SIGTERMs only the tare_planner_node processes this process launched (its
    descendants). Also removes the nav flag file so the bridge path follower
    stops following TARE-generated paths.
    """
    stopped = 0
    for pid in _own_descendants_named("tare_planner_node"):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped += 1
        except OSError:
            pass

    # Remove nav flag — bridge path follower stops acting on TARE paths
    try:
        os.remove(run_path("vector_nav_active"))
    except FileNotFoundError:
        pass

    logger.info("[EXPLORE] TARE stopped (%d own process(es)); FAR + localPlanner remain running", stopped)


def is_nav_stack_running() -> bool:
    """Check if localPlanner is still alive (FAR + localPlanner nav stack).

    Returns True if the localPlanner process is found via pgrep, False otherwise.
    Does not require the nav stack to have been launched by this module.
    """
    if not shutil.which("pgrep"):
        return False
    result = subprocess.run(
        ["pgrep", "-f", "localPlanner"],
        capture_output=True, timeout=5,
    )
    return result.returncode == 0


def cancel_exploration() -> None:
    """Request cancellation of the background exploration thread.

    Stops TARE only — FAR + localPlanner remain running for point-to-point nav.
    Does NOT kill the nav stack process group.
    """
    global _explore_running
    if _explore_running:
        _explore_cancel.set()
        stop_tare_only()
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
        log_fh = open(run_path("vector_nav_only.log"), "w")
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


_nav_explore_proc: subprocess.Popen | None = None


def _launch_nav_explore() -> None:
    """Start the full nav stack + TARE as one process group."""
    global _nav_explore_proc

    if _nav_explore_proc is not None and _nav_explore_proc.poll() is None:
        logger.info("[EXPLORE] Nav stack already running")
        return

    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    script = os.path.join(_repo, "scripts", "launch_nav_explore.sh")
    if not os.path.isfile(script):
        logger.error("[EXPLORE] launch_nav_explore.sh not found: %s", script)
        return

    log_fh = open(run_path("vector_nav_explore.log"), "w")
    _nav_explore_proc = subprocess.Popen(
        ["bash", script],
        stdout=log_fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    import atexit

    def _cleanup_nav():
        try:
            os.killpg(os.getpgid(_nav_explore_proc.pid), signal.SIGTERM)
            _nav_explore_proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(_nav_explore_proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log_fh.close()

    atexit.register(_cleanup_nav)
    logger.info("[EXPLORE] Nav stack + TARE launched (PID=%d)", _nav_explore_proc.pid)


def _verify_nav_stack() -> None:
    """Check that critical nav stack topics are active before starting TARE.

    Logs warnings for missing topics. Does NOT block — exploration proceeds
    regardless, but the log helps diagnose issues.
    """
    try:
        import rclpy
        node = rclpy.create_node("_nav_verify_tmp")
        topics = node.get_topic_names_and_types()
        topic_names = {name for name, _ in topics}

        required = [
            "/state_estimation",
            "/registered_scan",
            "/terrain_map",
            "/state_estimation_at_scan",
            "/path",
        ]
        for topic in required:
            status = "OK" if topic in topic_names else "MISSING"
            logger.info("[EXPLORE] Topic check: %-30s %s", topic, status)

        node.destroy_node()
    except Exception as exc:
        logger.warning("[EXPLORE] Topic verification failed: %s", exc)


# ---------------------------------------------------------------------------
# Background exploration loop
# ---------------------------------------------------------------------------

def _exploration_loop(base: Any, has_bridge: bool = True) -> None:
    """Background thread: verify TARE is running, seed planner, then monitor rooms.

    Everything runs here — execute() returns immediately.
    Runs indefinitely until _explore_cancel is set.

    TARE must already be running (started by _launch_ros2_stack via launch_explore.sh).
    If TARE is not detected, emits an error event and exits immediately.
    """
    global _explore_running, _explore_visited

    # Guard: TARE must be pre-launched by the nav stack startup.
    if not _start_tare():
        _emit("error", {
            "reason": "tare_not_running",
            "message": (
                "TARE planner is not running. Start the nav stack first "
                "(launch_explore.sh or sim stack startup)."
            ),
        })
        return

    _explore_visited.clear()
    _explore_running = True
    _explore_cancel.clear()

    _total = len(_spatial_memory.get_all_rooms()) if _spatial_memory else 0
    _emit("started", {"total_rooms": _total})

    # Seed walk: give TARE initial scan data by moving the robot forward briefly.
    # TARE requires 5 scans per keypose at 10 Hz (0.5 s minimum) before it can
    # generate candidate viewpoints. A stationary robot accumulates no new scans,
    # so TARE prints "Cannot get candidate viewpoints" for 10-20 rounds until it
    # happens to drift into a new position. A short forward walk guarantees the
    # first keypose is created before we hand off to autonomous nav.
    if has_bridge:
        try:
            logger.info("[EXPLORE] Seed walk: 0.3 m/s forward for 2s")
            base.walk(0.3, 0.0, 0.0, 2.0)
            # Extra 1s for scan buffer: 10 Hz lidar x 1s = 10 scans, well above
            # the 5-scan threshold TARE needs per keypose.
            time.sleep(1.0)
            logger.info("[EXPLORE] Seed walk complete")
        except Exception as exc:
            logger.warning("[EXPLORE] Seed walk failed (non-fatal): %s", exc)

    # Start TARE exploration (kAutoStart=false, waits for this signal)
    try:
        import subprocess as _sp
        _sp.run([
            "ros2", "topic", "pub", "--once",
            "/start_exploration", "std_msgs/msg/Bool", "{data: true}",
        ], capture_output=True, timeout=5)
        logger.info("[EXPLORE] Sent /start_exploration to TARE")
    except Exception as exc:
        logger.warning("[EXPLORE] Failed to send /start_exploration: %s", exc)

    # Enable bridge path follower via nav flag
    if has_bridge:
        try:
            with open(run_path("vector_nav_active"), "w") as f:
                f.write("1")
            logger.info("[EXPLORE] Navigation enabled (flag file created)")
        except Exception as exc:
            logger.warning("[EXPLORE] Failed to create nav flag: %s", exc)

    # NO WANDER. TARE + FAR + localPlanner handle all movement autonomously.
    # The initial seed above gives TARE enough scan data to start planning.
    # Any /cmd_vel_nav we send would CLEAR the bridge's _current_path and
    # override the nav stack's path follower — causing the dog to circle.
    #
    # Reference: launch_explore.sh does the same: seed once, then hands off
    # to TARE entirely. The nav stack drives at up to 0.8 m/s on its own.

    _prev_room: str | None = None
    _start_time = time.monotonic()
    _last_status_time = _start_time
    _last_progress_time = _start_time

    try:
        while not _explore_cancel.is_set():
            try:
                pos = base.get_position()
                if pos[2] < 0.12:
                    _emit("stopped", {"reason": "robot_fell", "rooms": sorted(_explore_visited)})
                    break

                x, y = float(pos[0]), float(pos[1])
                elapsed = time.monotonic() - _start_time
                room = _spatial_memory.nearest_room(x, y) if _spatial_memory else None

                # Periodic status every 30s
                if time.monotonic() - _last_status_time >= 30.0:
                    _last_status_time = time.monotonic()
                    _emit("status", {
                        "elapsed_sec": round(elapsed),
                        "rooms_found": len(_explore_visited),
                        "total": _total,
                        "position": [round(x, 1), round(y, 1)],
                        "current_room": room or "?",
                    })

                # Periodic progress every 5s (unconditional — fires even with no new rooms)
                if time.monotonic() - _last_progress_time >= 5.0:
                    _last_progress_time = time.monotonic()
                    _emit("progress", {
                        "rooms_found": len(_explore_visited),
                        "total": _total,
                        "position": [round(x, 1), round(y, 1)],
                        "elapsed": round(elapsed),
                    })

                # Record position in SceneGraph.
                # Room counting is informational only — TARE decides when to stop.
                if room is not None and _spatial_memory is not None:
                    try:
                        _spatial_memory.visit(room, x, y)
                    except Exception:
                        pass

                    # Auto-observe hook: capture VLM scene description at new viewpoints.
                    # Only triggers when VLM is available and position is a novel viewpoint.
                    # VLM failures are non-blocking — exploration continues regardless.
                    if _vlm is not None:
                        try:
                            if _spatial_memory.should_add_viewpoint(room, x, y):
                                _base_hook = getattr(base, "_base", base)
                                frame = None
                                if hasattr(_base_hook, "get_camera_frame"):
                                    frame = _base_hook.get_camera_frame()
                                if frame is not None:
                                    desc_result = _vlm.describe_scene(frame)
                                    obj_result = _vlm.find_objects(frame)
                                    scene_summary = str(
                                        getattr(desc_result, "summary", "")
                                    )
                                    object_names = [
                                        str(getattr(o, "name", ""))
                                        for o in (obj_result or [])
                                        if str(getattr(o, "name", ""))
                                    ]
                                    # Depth-localize the VLM-named objects to real
                                    # world (x, y) -- same pipeline LookSkill uses.
                                    # Falls back gracefully to [] when perception is
                                    # unavailable / no usable depth points.
                                    world_positions: dict[
                                        str, tuple[float, float, float]
                                    ] = {}
                                    if object_names:
                                        try:
                                            loc_results = localize_objects_3d(
                                                _perception, object_names
                                            )
                                            world_positions = {
                                                label: (lx, ly, lz)
                                                for label, lx, ly, lz in loc_results
                                            }
                                        except Exception as loc_exc:
                                            logger.debug(
                                                "[EXPLORE] localize failed: %s",
                                                loc_exc,
                                            )
                                    # Build (category, x, y) tuples ONLY for objects
                                    # that localized. NEVER store un-localized
                                    # objects at (0, 0) -- skip them so the graph is
                                    # not polluted with fake zeros. When none
                                    # localize, pass detected_objects=None so
                                    # observe_with_viewpoint uses the names-only path.
                                    detected_objects: (
                                        list[tuple[str, float, float]] | None
                                    ) = None
                                    if world_positions:
                                        detected_objects = [
                                            (
                                                name,
                                                world_positions[name][0],
                                                world_positions[name][1],
                                            )
                                            for name in object_names
                                            if name in world_positions
                                        ]
                                    heading = float(pos[3]) if len(pos) > 3 else 0.0
                                    _spatial_memory.observe_with_viewpoint(
                                        room=room,
                                        x=x,
                                        y=y,
                                        heading=heading,
                                        objects=object_names,
                                        description=scene_summary,
                                        detected_objects=detected_objects,
                                    )
                                    logger.debug(
                                        "[EXPLORE] Auto-observe: %d/%d objects "
                                        "localized in %s",
                                        len(detected_objects or []),
                                        len(object_names),
                                        room,
                                    )
                        except Exception as exc:
                            logger.debug(
                                "[EXPLORE] Auto-observe VLM failed (non-blocking): %s",
                                exc,
                            )

                # Door learning: detect room transitions (uses nearest_room, not _in_room,
                # because door detection should work at the boundary between rooms).
                if _prev_room is not None and room is not None and room != _prev_room:
                    if _spatial_memory is not None:
                        _spatial_memory.add_door(_prev_room, room, x, y)
                        logger.info(
                            "[EXPLORE] Door learned: %s <-> %s at (%.1f, %.1f)",
                            _prev_room, room, x, y,
                        )
                _prev_room = room

                if room is not None and room not in _explore_visited:
                    _explore_visited.add(room)
                    _emit("room_entered", {
                        "room": room,
                        "visited": len(_explore_visited),
                        "total": _total,
                        "elapsed_sec": round(elapsed),
                        "all_rooms": sorted(_explore_visited),
                    })

                    # Auto-look: VLM scene capture in a SEPARATE thread.
                    if _auto_look is not None:
                        def _run_auto_look(r: str = room) -> None:
                            try:
                                obs = _auto_look(r)
                                if obs:
                                    _emit("room_observed", {
                                        "room": r,
                                        "summary": obs.get("summary", ""),
                                        "objects": obs.get("objects", []),
                                    })
                                    logger.info(
                                        "[EXPLORE] Auto-look %s: %s",
                                        r, obs.get("summary", "")[:80],
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "[EXPLORE] Auto-look failed for %s: %s",
                                    r, exc,
                                )

                        threading.Thread(
                            target=_run_auto_look, daemon=True,
                        ).start()

            except Exception:
                pass

            # Check if TARE declared exploration complete (all frontiers covered)
            if os.path.exists(run_path("vector_explore_finished")):
                try:
                    os.remove(run_path("vector_explore_finished"))
                except OSError:
                    pass
                logger.info("[EXPLORE] TARE exploration complete — all frontiers covered")
                _emit("complete", {
                    "reason": "tare_finished",
                    "rooms": sorted(_explore_visited),
                    "total": _total,
                    "elapsed_sec": round(time.monotonic() - _start_time),
                })
                break

            _explore_cancel.wait(timeout=_POSITION_SAMPLE_INTERVAL)

        # Normal exit: user stopped or TARE finished
        if not _explore_cancel.is_set():
            _emit("stopped", {"reason": "finished", "rooms": sorted(_explore_visited)})

    finally:
        # Terrain replay fires on ALL exit paths (finish, cancel, crash)
        # so FAR always gets accumulated map data for V-Graph building.
        try:
            with open(run_path("vector_terrain_replay"), "w") as f:
                f.write("1")
            logger.info("[EXPLORE] Triggered terrain replay for FAR")
        except OSError:
            pass
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
    # Typical REAL-TIME startup duration (non-blocking launch + initial movement):
    # the skill returns immediately (exploration is a background thread), but the
    # executor waits via _wait_for_async_skill which polls until is_exploring()
    # returns False; 150s is a conservative floor for open-ended exploration.
    # GoalExecutor floors the step timeout at this value (R2-2).
    typical_duration_sec: float = 150.0
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

        # Wire spatial memory for position recording during exploration
        global _spatial_memory, _perception, _vlm
        _spatial_memory = context.services.get("spatial_memory")
        # Capture perception + VLM from the context here (execute() is the only
        # scope with the correct accessors -- the exploration thread receives
        # only `base`, the locomotion driver, which carries neither). Mirrors
        # LookSkill, which depth-localizes via context.perception.
        _perception = context.perception
        _vlm = context.services.get("vlm")

        # Ensure room layout is loaded (handles /clear_memory wiping the data)
        if _spatial_memory is not None and hasattr(_spatial_memory, 'load_layout'):
            if not _spatial_memory.get_all_rooms():
                _layout = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    ))),
                    "config", "room_layout.yaml",
                )
                if os.path.isfile(_layout):
                    n = _spatial_memory.load_layout(_layout)
                    if n > 0:
                        logger.info("[EXPLORE] Re-loaded room layout: %d rooms", n)

        # VLM auto-look DISABLED for sim — room detection is config-based.
        # Re-enable for real-world deployment where VLM scene descriptions
        # provide spatial understanding.
        # set_auto_look(None) — explicitly not wiring VLM

        # Bridge + nav stack are already running (started by launch_explore.sh
        # via sim_tool). Do NOT call _start_bridge_on_go2 — it creates a
        # conflicting rclpy context that crashes the bridge process.
        bridge_ok = True

        # Start background thread that handles seeding + monitoring
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
