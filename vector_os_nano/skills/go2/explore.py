# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ExploreSkill -- explore the house with a real backend, or fail loud.

Two backends:

- ``coverage`` (the default when no nav stack runs): the in-process room tour
  (sim/nav/coverage.py). It BLOCKS until the tour ends and reports the rooms the
  base's ground-truth position actually entered; a stop, a navigate or Ctrl-C
  cancels it between and inside legs.
- ``tare``: the CMU TARE planner this session launched (VECTOR_NAV_STACK). The
  stack drives; a background monitor tracks rooms and returns at once.

No backend (no planner on the base, no TARE of ours) is a failure, never a
started-and-did-nothing success. ``is_exploring()`` is True from before the run
starts until it has ended, so a waiter can never miss a run that is starting.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any

from vector_os_nano.core.run_dir import run_path
from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.transport import prepare

logger = logging.getLogger(__name__)

_POSITION_SAMPLE_INTERVAL: float = 2.0
BACKENDS: tuple[str, ...] = ("coverage", "tare")

# Exploration state (one exploration per process)
_explore_thread: threading.Thread | None = None
_explore_cancel: threading.Event = threading.Event()
_explore_visited: set[str] = set()
_explore_running: bool = False
_explore_backend: str = ""
_on_event: Any = None  # callback(event_type: str, data: dict) — set by CLI
_spatial_memory: Any = None  # SceneGraph — set by ExploreSkill.execute()
_perception: Any = None  # CameraPerception-like — set by ExploreSkill.execute()
_vlm: Any = None  # VLM service — set by ExploreSkill.execute()
_vocab: list[str] = []  # the world's object names (discovery without a VLM)
_look_failures: list[str] = []  # why looks on the way (this run) could not check every name


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
        **_perception_note(),
    }


def _perception_note() -> dict[str, str]:
    """``{"perception_failed": ...}`` when a look on the way could not check every name."""
    if not _look_failures:
        return {}
    return {"perception_failed": f"{len(_look_failures)} look(s) on the way could not check every "
                                 f"name: {'; '.join(dict.fromkeys(_look_failures))}"}


def _own_descendants_named(name: str, root: int | None = None) -> list[int]:
    """PIDs anywhere below *root* (default: this process) whose command line contains *name*.

    TARE runs inside the nav-stack launch this CLI started, a descendant in another
    process group. Matching by name only among OUR descendants never touches (or
    mistakes for ours) another project's planner on a shared machine. That stack is
    the opt-in external path: without VECTOR_NAV_STACK nothing is scanned, and
    without a Linux /proc nothing can be found.
    """
    if not os.environ.get("VECTOR_NAV_STACK"):
        return []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    children: dict[int, list[int]] = {}
    for entry in entries:
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


def _tare_running() -> bool:
    """True when a TARE planner WE launched is alive (never another project's)."""
    return bool(_own_descendants_named("tare_planner_node"))


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
    """True when a localPlanner WE launched is alive (periphery:missed2: a bare
    ``pgrep -f`` took another project's nav nodes on the shared stack for ours)."""
    return bool(_own_descendants_named("localPlanner"))


def cancel_exploration() -> None:
    """Cancel the running exploration (coverage: between/inside legs; TARE: stop TARE).

    FAR + localPlanner remain running for point-to-point nav.
    """
    if _explore_running:
        _explore_cancel.set()
        if _explore_backend == "tare" and os.environ.get('VECTOR_NAV_CONTROL') != '1':
            stop_tare_only()
            _emit("stopped", {"reason": "cancelled", "rooms": sorted(_explore_visited)})


def get_explored_rooms() -> list[str]:
    """Return rooms discovered during current/last exploration."""
    return sorted(_explore_visited)


def set_event_callback(callback: Any) -> None:
    """Set a callback for exploration events. Called from CLI.

    callback(event_type: str, data: dict) where event_type is one of:
        "started", "room_entered", "progress", "status", "stopped", "completed"
    """
    global _on_event
    _on_event = callback


def _emit(event_type: str, data: dict | None = None) -> None:
    """Emit an exploration event to the callback (if set)."""
    if _on_event is not None:
        try:
            _on_event(event_type, data or {})
        except Exception:
            pass


class _Cancelled:
    """Coverage cancel signal: cancel_exploration() or the run's stop."""

    @staticmethod
    def is_set() -> bool:
        from vector_os_nano.vcli.cognitive.abort import is_abort_requested

        return _explore_cancel.is_set() or is_abort_requested()


def _discover(base: Any) -> None:
    """One look on the way: localize the world's objects (and the VLM's) and map
    them (look.observe_scene). Never raises into the exploration; a look that could
    not check every name is kept, with its cause, for the explore result."""
    if _spatial_memory is None or (_vlm is None and not (_vocab and _perception is not None)):
        return
    from vector_os_nano.perception.base import failure_cause
    from vector_os_nano.skills.go2.look import ObserveFailed, _causes, observe_scene

    try:
        data = observe_scene(getattr(base, "_base", base), _perception, _spatial_memory, _vlm, _vocab)
    except Exception as exc:  # noqa: BLE001 — the exploration goes on; its result says why
        logger.warning("[EXPLORE] a look on the way failed: %s: %s", type(exc).__name__, exc)
        _look_failures.append(str(exc) if isinstance(exc, ObserveFailed) else failure_cause(exc))
        return
    if data["failed"]:
        _look_failures.append(_causes(data["failed"]))
    if data["objects"]:
        _emit("objects_seen", {"room": data["room"], "objects": [o["name"] for o in data["objects"]]})


# ---------------------------------------------------------------------------
# TARE background monitor
# ---------------------------------------------------------------------------

def _exploration_loop(base: Any) -> None:
    """Background thread: seed TARE, start it, then monitor rooms until cancelled.

    Started by ExploreSkill.execute() only after it checked that our TARE runs and
    set the running flag (so no waiter sees a not-yet-started run as finished).
    """
    global _explore_running

    _total = len(_spatial_memory.get_all_rooms()) if _spatial_memory else 0
    _emit("started", {"total_rooms": _total})

    # Seed walk: give TARE initial scan data by moving the robot forward briefly.
    # TARE requires 5 scans per keypose at 10 Hz (0.5 s minimum) before it can
    # generate candidate viewpoints. A stationary robot accumulates no new scans,
    # so TARE prints "Cannot get candidate viewpoints" for 10-20 rounds until it
    # happens to drift into a new position. A short forward walk guarantees the
    # first keypose is created before we hand off to autonomous nav.
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

                    # Discovery at each new viewpoint: the detector over the world
                    # vocabulary (+ the VLM's names) on one snapshot; only what
                    # localizes is stored. Failures never stop the exploration.
                    if _spatial_memory.should_add_viewpoint(room, x, y):
                        _discover(base)

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
# ExploreSkill
# ---------------------------------------------------------------------------

@skill(requires=('base', 'lidar', 'camera'),
    aliases=[
        "explore", "探索", "自主探索",
        "explore the house", "look around", "四处看看",
    ],
)
class ExploreSkill:
    """Explore the house (coverage tour by default; TARE when our stack runs)."""

    name: str = "explore"
    description: str = (
        "Explore the house: visit every room and look around. Without a nav stack "
        "this is a room tour that returns the rooms actually visited (it blocks "
        "until done; stop() cancels it). With the TARE stack it runs in the "
        "background until stop()."
    )
    parameters: dict = {
        "backend": {
            "type": "string",
            "required": False,
            "enum": list(BACKENDS),
            "description": (
                "coverage (in-process room tour) or tare (the CMU TARE planner). "
                "Default: tare when this session runs it, else coverage."
            ),
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"explored": True}
    failure_modes: list[str] = ["no_base", "exploration_failed", "no_backend", "aborted"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        global _explore_thread, _explore_running, _explore_backend
        global _spatial_memory, _perception, _vlm, _vocab

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

        live_tare = context.services.get('explore_tare')
        backend = str(params.get("backend") or ("tare" if live_tare or _tare_running() else "coverage"))
        if backend not in BACKENDS:
            return SkillResult(
                success=False, diagnosis_code="no_backend",
                error_message=f"Unknown explore backend {backend!r}; valid: {', '.join(BACKENDS)}",
            )
        base = context.base
        error = prepare(context)
        if error:
            return error

        # Wire spatial memory / perception / VLM / vocabulary (discovery and the
        # TARE monitor read them). The world's rooms are seeded at sim start.
        from vector_os_nano.skills.go2.look import _world_vocabulary

        _spatial_memory = context.services.get("spatial_memory")
        _perception = context.perception
        _vlm = context.services.get("vlm")
        _vocab = _world_vocabulary(context)

        if backend == "tare":
            if live_tare:
                _explore_cancel.clear()
                _explore_visited.clear()
                _explore_backend, _explore_running = 'tare', True
                try:
                    out = live_tare(cancel=_Cancelled())
                finally:
                    _explore_running = False
                return SkillResult(bool(out), dict(out),
                    None if out else f"TARE exploration ended: {out['reason']}",
                    None if out else 'exploration_failed')
            if not _tare_running():
                return SkillResult(
                    success=False, diagnosis_code="no_backend",
                    error_message=(
                        "TARE is not running in this session (it needs VECTOR_NAV_STACK and a "
                        "Linux /proc to find it); explore with backend='coverage' instead."
                    ),
                )
            _explore_cancel.clear()
            _explore_visited.clear()
            _look_failures.clear()
            _explore_backend, _explore_running = "tare", True  # before the thread starts
            _explore_thread = threading.Thread(target=_exploration_loop, args=(base,), daemon=True)
            _explore_thread.start()
            return SkillResult(
                success=True,
                result_data={
                    "status": "exploration_started", "backend": "tare",
                    "note": "TARE runs in the background. Use stop() to halt.",
                },
            )

        if not callable(getattr(base, "navigate_to", None)):
            return SkillResult(
                success=False, diagnosis_code="no_backend",
                error_message=(
                    "No exploration backend: the base has no planner (navigate_to) "
                    "and TARE is not running."
                ),
            )
        layout = context.services.get("layout")
        if not layout or not layout.get("rooms"):
            return SkillResult(False, {"diagnosis": "no_layout"},
                               "The session world declares no rooms for coverage exploration")
        return self._coverage(base, layout)

    @staticmethod
    def _coverage(base: Any, layout: dict) -> SkillResult:
        global _explore_running, _explore_backend
        from vector_os_nano.sim.nav.coverage import CoverageExplorer

        def on_event(kind: str, data: dict) -> None:
            if kind == "room_entered":
                _explore_visited.add(data["room"])
                if _spatial_memory is not None:
                    pos = base.get_position()
                    _spatial_memory.visit(data["room"], float(pos[0]), float(pos[1]))
            _emit(kind, data)

        _explore_cancel.clear()
        _explore_visited.clear()
        _look_failures.clear()
        _explore_backend, _explore_running = "coverage", True
        try:
            out = CoverageExplorer(base, layout, cancel=_Cancelled(), on_event=on_event,
                                   observe=lambda: _discover(base)).run()
        finally:
            _explore_running = False
        visited, total = out["visited"], out["total"]
        data = {
            "backend": "coverage",
            "rooms_visited": visited,
            "rooms_failed": out["failed"],
            "coverage": f"{len(visited)}/{total} rooms",
            "elapsed_s": out["elapsed_s"],
            **_perception_note(),
        }
        if out["cancelled"]:
            return SkillResult(
                success=False, diagnosis_code="aborted", result_data=data,
                error_message=f"Exploration stopped after {len(visited)}/{total} rooms",
            )
        if not visited:
            return SkillResult(
                success=False, diagnosis_code="exploration_failed", result_data=data,
                error_message=f"Exploration reached no room ({out['failed']})",
            )
        return SkillResult(success=True, result_data=data)
