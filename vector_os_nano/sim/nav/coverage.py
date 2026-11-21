# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Coverage exploration without TARE: a room tour over the world's layout rooms.

``CoverageExplorer(base, layout).run()`` visits every layout room in
nearest-first order: it navigates to the room's free point (the planner snaps it
out of any furniture) and takes a 360-degree look in ``look_turns`` turns. A room
counts as visited only when the base's GROUND-TRUTH position at the end of that
leg lies inside the room's floor box — never on the follower's own word. Every
leg is appended to a JSONL log (``run_path('explore_coverage.jsonl')``) with the
GT pose, so a grader can re-check the tour from outside.

The base is any in-process base with ``navigate_to``/``walk``/``get_position``/
``get_heading``; ``cancel`` (``is_set()``) stops the tour between and inside legs.
``observe`` (optional) is called after every look turn, the base standing still:
explore's object discovery.
"""
from __future__ import annotations

import functools
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from vector_os_nano.schemas.world import WORLDS_DIR, load_world_file

from vector_os_nano.core.run_dir import run_path
from vector_os_nano.sim.nav.follow import _AbortFlag, nav_config, turn_by

_LAYOUT_YAML = WORLDS_DIR / "apartment" / "world.yaml"
LOG_NAME = "explore_coverage.jsonl"


@functools.lru_cache(maxsize=4)
def room_layout(path: str = str(_LAYOUT_YAML)) -> dict[str, Any]:
    """``{"rooms": {name: (x, y)}, "boxes": {name: (x0, y0, x1, y1)}}`` of a layout file."""
    world = load_world_file(path)
    return {"rooms": world.rooms, "boxes": world.boxes}


def room_at(x: float, y: float, boxes: dict[str, tuple]) -> str | None:
    """The room whose floor box contains (x, y) (the smallest box on a shared edge)."""
    hits = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes.values()]
    inside = [(area, name) for (name, b), area in zip(boxes.items(), hits)
              if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    return min(inside)[1] if inside else None


class CoverageExplorer:
    """Tour every layout room once; report what ground truth says was entered."""

    def __init__(
        self,
        base: Any,
        layout: dict[str, Any] | None = None,
        *,
        cancel: Any = None,
        on_event: Callable[[str, dict], None] | None = None,
        log_path: str | None = None,
        observe: Callable[[], None] | None = None,
    ) -> None:
        self.base = base
        self.layout = layout or room_layout()
        self.cancel = cancel if cancel is not None else _AbortFlag()
        self.on_event = on_event or (lambda _e, _d: None)
        self.log_path = log_path or run_path(LOG_NAME)
        self.observe = observe
        self.visited: list[str] = []
        self.failed: dict[str, str] = {}

    def _gt_xy(self) -> tuple[float, float]:
        p = self.base.get_position()
        return float(p[0]), float(p[1])

    def _order(self) -> list[str]:
        """Nearest-first tour over the room points from the current position."""
        todo, out = dict(self.layout["rooms"]), []
        here = self._gt_xy()
        while todo:
            name = min(todo, key=lambda n: math.dist(here, todo[n]))
            here = todo.pop(name)
            out.append(name)
        return out

    def _look(self) -> bool:
        cfg = nav_config()["exploration"]
        turns = int(cfg["look_turns"])
        ok = True
        for _ in range(turns):
            if self.cancel.is_set():
                return False
            done, _res = turn_by(self.base, 2 * math.pi / turns, vyaw=float(cfg["look_vyaw"]),
                                 tol=math.radians(float(cfg["look_tol_deg"])), cancel=self.cancel)
            ok = ok and done
            if done and self.observe is not None:
                self.observe()
        return ok

    def run(self) -> dict[str, Any]:
        """Run the tour; returns {visited, failed, total, cancelled, elapsed_s}."""
        boxes, rooms = self.layout["boxes"], self.layout["rooms"]
        total = len(rooms)
        t0 = time.monotonic()
        self.on_event("started", {"total_rooms": total})
        with open(self.log_path, "a", encoding="utf-8") as log:
            for name in self._order():
                if self.cancel.is_set():
                    break
                x, y = rooms[name]
                nav = self.base.navigate_to(x, y, cancel=self.cancel)
                reason = getattr(nav, "reason", "arrived" if nav else "failed")
                gx, gy = self._gt_xy()
                in_room = room_at(gx, gy, boxes) == name
                looked = self._look() if in_room else False
                log.write(json.dumps({
                    "room": name, "target": [x, y], "gt_pos": [gx, gy], "nav": reason,
                    "in_room": in_room, "looked": looked,
                    "sim_time": float(self.base.get_sim_time()),
                }) + "\n")
                log.flush()
                if in_room:
                    self.visited.append(name)
                    self.on_event("room_entered", {
                        "room": name, "visited": len(self.visited), "total": total,
                        "elapsed_sec": round(time.monotonic() - t0),
                        "all_rooms": sorted(self.visited),
                    })
                else:
                    self.failed[name] = reason
        cancelled = self.cancel.is_set()
        summary = {
            "visited": list(self.visited), "failed": dict(self.failed), "total": total,
            "cancelled": cancelled, "elapsed_s": round(time.monotonic() - t0, 1),
        }
        self.on_event("stopped" if cancelled else "completed", {
            "reason": "cancelled" if cancelled else "coverage_finished",
            "rooms": sorted(self.visited), "elapsed_sec": round(time.monotonic() - t0),
        })
        return summary
