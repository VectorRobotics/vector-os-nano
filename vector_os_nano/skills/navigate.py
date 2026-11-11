# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""NavigateSkill -- hardware-agnostic room / coordinate navigation.

A room name resolves (aliases, SceneGraph) to its learned centre; a coordinate
goal ({x, y} / target) goes straight to the base's planner. Both drive
``base.navigate_to``, which returns a ``NavResult`` from the ONE global planner
(sim/nav): its ``reason`` is reported.

A room visit is recorded only after a verified arrival (never on failure: the
SceneGraph centres feed the legacy verify). Room positions
come from the SceneGraph; no hardcoded coordinates. No ROS2 imports.
"""
from __future__ import annotations

import logging
import math
import sys
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.run_dir import run_path
from vector_os_nano.sim.nav.follow import NavResult, arrival_radius

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aliases -> canonical room name (Chinese + English + shortcuts)
# ---------------------------------------------------------------------------

_ROOM_ALIASES: dict[str, str] = {
    # English
    "living room":    "living_room",
    "living":         "living_room",
    "lounge":         "living_room",
    "dining room":    "dining_room",
    "dining":         "dining_room",
    "kitchen":        "kitchen",
    "study":          "study",
    "office":         "study",
    "master bedroom": "master_bedroom",
    "master":         "master_bedroom",
    "bedroom":        "master_bedroom",   # default bedroom = master
    "guest bedroom":  "guest_bedroom",
    "guest room":     "guest_bedroom",
    "guest":          "guest_bedroom",
    "bathroom":       "bathroom",
    "bath":           "bathroom",
    "restroom":       "bathroom",
    "toilet":         "bathroom",
    "hallway":        "hallway",
    "hall":           "hallway",
    "corridor":       "hallway",
    "laundry":        "hallway",    # laundry is in hallway area
    # Chinese
    "客厅": "living_room",
    "大厅": "living_room",
    "餐厅": "dining_room",
    "饭厅": "dining_room",
    "厨房": "kitchen",
    "书房": "study",
    "办公室": "study",
    "工作室": "study",
    "主卧": "master_bedroom",
    "卧室": "master_bedroom",
    "主卧室": "master_bedroom",
    "客卧": "guest_bedroom",
    "客房": "guest_bedroom",
    "次卧": "guest_bedroom",
    "卫生间": "bathroom",
    "浴室": "bathroom",
    "洗手间": "bathroom",
    "厕所": "bathroom",
    "走廊": "hallway",
    "过道": "hallway",
    "大厅走廊": "hallway",
    "洗衣房": "hallway",
}

_MIN_VISIT_COUNT: int = 1  # trust SceneGraph position after first visit

# Coordinate-goal navigation (R38). A cross-room FAR leg takes ~30-50 s over the
# lidar terrain map (probe v2: 32 s clean, 41 s through furniture), so the nav
# budget must exceed that or the dog times out mid-drive. The VICINITY radius is
# the step-success threshold (generous: FAR arrival radius 0.8 m + margin) — NOT
# the honest arrival grade, which is the at_position(tol 0.5 m) verify oracle.
_COORD_NAV_TIMEOUT_S: float = 90.0
_COORD_VICINITY_RADIUS_M: float = 1.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_room(name: str, sg: Any = None) -> str | None:
    """Resolve a room name/alias to canonical room key.

    Matching priority:
    1. Exact alias match ("master bedroom" → master_bedroom)
    2. Canonical underscore form ("master_bedroom" → master_bedroom)
    3. Fuzzy: input words match room_id parts ("master room" → master_bedroom)
    4. Fuzzy: input is substring of alias or vice versa

    If sg is provided, verifies the resolved room exists in the SceneGraph.
    Returns None if unknown or not found.
    """
    if not name:
        return None
    key = name.strip().lower().replace("_", " ")
    canonical = key.replace(" ", "_")

    # Priority 1: exact alias
    alias_result = _ROOM_ALIASES.get(key)
    if alias_result:
        canonical = alias_result

    # Check SceneGraph
    if sg is not None and hasattr(sg, "get_room"):
        if sg.get_room(canonical) is not None:
            return canonical
        # Priority 3: fuzzy match against all rooms in SceneGraph
        all_rooms = [r.room_id for r in sg.get_all_rooms()] if hasattr(sg, "get_all_rooms") else []
        fuzzy = _fuzzy_room_match(key, all_rooms)
        if fuzzy is not None:
            return fuzzy
        return None

    # No SceneGraph — alias-only
    if alias_result or canonical in _ROOM_ALIASES.values():
        return canonical
    return None


def _fuzzy_room_match(query: str, room_ids: list[str]) -> str | None:
    """Find best room match using word overlap and substring matching.

    "master room" → "master_bedroom" (word "master" matches)
    "guest" → "guest_bedroom" (alias substring)
    Ignores generic words like "room" to avoid false positives.
    """
    if not query or not room_ids:
        return None
    _STOP_WORDS = {"room", "the", "a", "to", "go", "去", "到"}
    query_words = set(query.split()) - _STOP_WORDS
    if not query_words:
        return None
    best, best_score = None, 0
    for rid in room_ids:
        rid_words = set(rid.replace("_", " ").split()) - _STOP_WORDS
        # Word overlap score (meaningful words only)
        overlap = len(query_words & rid_words)
        # Substring match on the non-stopword query
        query_clean = "".join(sorted(query_words))
        rid_clean = rid.replace("_", "")
        if query_clean in rid_clean or rid_clean in query_clean:
            overlap += 2
        # Check aliases that map to this room
        for alias, target in _ROOM_ALIASES.items():
            if target == rid:
                if alias in query or query in alias:
                    overlap += 1
                    break
        if overlap > best_score:
            best, best_score = rid, overlap
    return best if best_score > 0 else None


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _cancel_exploration() -> None:
    """A navigation takes priority over a running exploration: stop it first."""
    try:
        from vector_os_nano.skills.go2.explore import cancel_exploration, is_exploring
        if is_exploring():
            cancel_exploration()
            logger.info("[NAV] Cancelled background exploration for navigation")
    except Exception:  # noqa: BLE001 — explore is an optional go2 skill
        pass


def _detect_current_room(x: float, y: float, sg: Any = None) -> str:
    """Guess which room the robot is in based on SceneGraph nearest_room.

    If sg is provided and has rooms, delegates to sg.nearest_room(x, y).
    Returns "unknown" if SceneGraph is absent or empty.
    """
    if sg is not None and hasattr(sg, "nearest_room"):
        room = sg.nearest_room(x, y)
        if room is not None:
            return room
    return "unknown"


def _get_room_center_from_memory(
    memory: Any, room_key: str,
) -> tuple[float, float] | None:
    """Look up explored room center from spatial memory (SceneGraph).

    Only trusts positions that have visit_count >= _MIN_VISIT_COUNT
    (not just a doorway drive-by). Returns None if the room is not in memory or
    its position is not trustworthy.
    """
    room_node = memory.get_room(room_key)
    if room_node is not None:
        if (room_node.center_x != 0.0 or room_node.center_y != 0.0) and room_node.visit_count >= _MIN_VISIT_COUNT:
            return (room_node.center_x, room_node.center_y)
    return None


# ---------------------------------------------------------------------------
# NavigateSkill
# ---------------------------------------------------------------------------

@skill(
    aliases=[
        "navigate", "go to", "goto",
        "去", "到", "走到", "去到", "导航",
    ],
)
class NavigateSkill:
    """Navigate the robot to a named room (or a world coordinate).

    The base's own planner (``navigate_to``) drives; a base without one cannot
    navigate. Room coordinates come exclusively from the SceneGraph. Works with ANY
    BaseProtocol implementation (not Go2-specific).
    """

    name: str = "navigate"
    description: str = (
        "Navigate the robot to a named room. "
        "Use this when the user says 'go to X' or '去X'. "
        "Returns an error if the room has not been discovered yet."
    )
    parameters: dict = {
        "room": {
            "type": "string",
            "required": True,
            "description": (
                "Target room name. Examples: kitchen, bedroom, study, "
                "bathroom, living_room, dining_room, guest_bedroom, hallway. "
                "Chinese: 厨房, 卧室, 书房, 卫生间, 客厅, 餐厅, 客房, 走廊."
            ),
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"position": "changed"}
    failure_modes: list[str] = ["no_base", "unknown_room", "navigation_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        # --- Coordinate goal (R38): a navigate step may target a world COORDINATE
        # rather than a named SceneGraph room (e.g. "去桌子那里" -> the table
        # standoff, which has no room entry). When x/y (or target=[x, y]) are
        # present, drive the FAR planner directly via base.navigate_to(x, y),
        # bypassing room resolution. World-agnostic and additive — the named-room
        # path below is unchanged when no coordinates are given. The step's
        # at_position verify (graded RAN per D14) is the source of truth for
        # arrival; we surface the actual position regardless of FAR's verdict.
        coord = self._parse_coordinate(params)
        if coord is not None:
            return self._navigate_to_coordinate(coord, context)

        room_input = str(params.get("room", ""))
        sg = context.services.get("spatial_memory")

        # Resolve room name — SceneGraph is authoritative
        room_key = _resolve_room(room_input, sg=sg)

        if room_key is None:
            # Check if SceneGraph has any rooms at all
            if sg is None or not hasattr(sg, "get_all_rooms") or not sg.get_all_rooms():
                return SkillResult(
                    success=False,
                    error_message="No rooms learned. Run explore first.",
                    diagnosis_code="unknown_room",
                )
            # SceneGraph exists but room not found
            available_rooms = [r.room_id for r in sg.get_all_rooms()]
            available = ", ".join(sorted(available_rooms)) if available_rooms else "none"
            return SkillResult(
                success=False,
                error_message=f"Unknown room: '{room_input}'. Available: {available}",
                diagnosis_code="unknown_room",
            )

        # Get target position from SceneGraph only
        target: tuple[float, float] | None = None
        if sg is not None:
            target = _get_room_center_from_memory(sg, room_key)

        if target is None:
            return SkillResult(
                success=False,
                error_message=f"Room '{room_key}' position unknown. Explore more.",
                diagnosis_code="room_not_explored",
            )

        logger.info("[NAV] Using learned position for %s: (%.1f, %.1f)",
                    room_key, target[0], target[1])

        _cancel_exploration()

        # Ensure nav flag exists so bridge path follower is armed
        try:
            import os
            if not os.path.exists(run_path("vector_nav_active")):
                with open(run_path("vector_nav_active"), "w") as fh:
                    fh.write("1")
        except Exception:
            pass

        if not callable(getattr(context.base, "navigate_to", None)):
            return SkillResult(
                success=False,
                error_message="Room navigation needs a planner (base.navigate_to absent).",
                diagnosis_code="navigation_failed",
            )
        return self._navigate_to_room(room_key, target, context)

    # ------------------------------------------------------------------
    # Navigation modes (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_coordinate(params: dict) -> tuple[float, float] | None:
        """Extract a world (x, y) coordinate goal from params, or None.

        Accepts ``{"x": .., "y": ..}`` or ``{"target": [x, y]}`` /
        ``{"goal": [x, y]}`` (a 2-list/tuple). Returns None when no coordinate is
        present (the named-room path then runs) — back-compatible.
        """
        x = params.get("x")
        y = params.get("y")
        if x is not None and y is not None:
            try:
                return (float(x), float(y))
            except (TypeError, ValueError):
                return None
        for key in ("target", "goal", "coordinate", "coord"):
            val = params.get(key)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    return (float(val[0]), float(val[1]))
                except (TypeError, ValueError):
                    return None
        return None

    def _navigate_to_coordinate(
        self,
        coord: tuple[float, float],
        context: SkillContext,
    ) -> SkillResult:
        """Drive the base to a world coordinate via the FAR planner (navigate_to).

        Used for a coordinate navigate goal (no SceneGraph room). Honest: the
        step's success is the planner's own arrival verdict and the actual landing
        position is reported; the at_position verify oracle (RAN) is what grades the
        step. If the base lacks navigate_to (no nav stack) we fail loud.
        """
        base = context.base
        tx, ty = coord
        navigate_to = getattr(base, "navigate_to", None)
        if not callable(navigate_to):
            return SkillResult(
                success=False,
                error_message="Coordinate navigation needs a nav stack (base.navigate_to absent).",
                diagnosis_code="navigation_failed",
            )

        # A running exploration would otherwise freeze (its nav flag is removed
        # below) while is_exploring() stays True (skills-nav:missed2).
        _cancel_exploration()

        # Ensure the bridge path follower is armed (same as the room path).
        try:
            import os
            if not os.path.exists(run_path("vector_nav_active")):
                with open(run_path("vector_nav_active"), "w") as fh:
                    fh.write("1")
        except Exception:
            pass

        logger.info("[NAV] Coordinate goal -> navigate_to(%.2f, %.2f) via FAR", tx, ty)

        def _progress(dist: float, elapsed: float) -> None:
            print(f"  >> 距目标 {dist:.1f}m, 已走 {int(elapsed)}s",
                  file=sys.stderr, flush=True)

        try:
            nav = navigate_to(tx, ty, timeout=_COORD_NAV_TIMEOUT_S, on_progress=_progress)
        except TypeError:
            # A base whose navigate_to does not accept the extra kwargs.
            nav = navigate_to(tx, ty)
        ok = bool(nav)

        # HOLD at the goal. navigate_to leaves the dog under the planner/TARE,
        # which keeps publishing /way_point and DRIFTS the dog away from the goal
        # (observed: dog wandered to (7.7, 4.5) before the next skill perceived).
        # A downstream manipulation step needs the dog STATIONARY at the table, so
        # disarm the nav flag and stop the base now. (Symmetric with how a stop
        # command clears the flag; the proxy's loops treat the missing flag as
        # "no active nav".)
        try:
            import os
            if os.path.exists(run_path("vector_nav_active")):
                os.remove(run_path("vector_nav_active"))
        except Exception:
            pass
        try:
            if callable(getattr(base, "stop", None)):
                base.stop()
            else:
                base.set_velocity(0.0, 0.0, 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[NAV] coord-nav stop failed: %s", exc)

        pos = base.get_position()
        dist = _distance(pos[0], pos[1], tx, ty)
        # The skill SUCCEEDS when it ran and the dog is in the goal's VICINITY — FAR
        # may time out / not raise its arrival flag yet still leave the dog close
        # (it drives over a lidar terrain map at variable speed). Step success only
        # gates whether DEPENDENTS run; the honest arrival grade is the at_position
        # verify oracle (RAN, tol 0.5 m). So we use a GENEROUS vicinity radius here
        # (the FAR arrival radius 0.8 m + margin) — not a strict re-check — to avoid
        # falsely failing the chain when the dog is geometrically at the table but
        # FAR's bool timed out. Genuinely-far (FAR couldn't route) -> still fail loud.
        arrived = ok or dist <= _COORD_VICINITY_RADIUS_M
        return SkillResult(
            success=arrived,
            error_message="" if arrived else (
                f"navigate_to({tx}, {ty}) did not reach the goal vicinity "
                f"(ended {dist:.1f} m away)"),
            diagnosis_code=None if arrived else "navigation_failed",
            result_data={
                "target": [round(tx, 1), round(ty, 1)],
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "distance_to_target": round(dist, 1),
                "far_confirmed": ok,
                "mode": "planner_coord" if isinstance(nav, NavResult) else "proxy_coord",
                **({"reason": nav.reason, "target_planned": nav["target"]}
                   if isinstance(nav, NavResult) else {}),
            },
        )

    def _navigate_to_room(
        self,
        room_key: str,
        target: tuple[float, float],
        context: SkillContext,
    ) -> SkillResult:
        """Drive to the room centre through the base's own planner (navigate_to);
        its ``NavResult`` is final and its reason is reported."""
        logger.info(
            "[NAV] Planner -> room=%s target=(%.1f, %.1f)",
            room_key, target[0], target[1],
        )

        def _progress(dist: float, elapsed: float) -> None:
            print(
                f"  >> 距目标 {dist:.1f}m, 已走 {int(elapsed)}s",
                file=sys.stderr,
                flush=True,
            )

        nav_result = context.base.navigate_to(target[0], target[1], on_progress=_progress)
        pos = context.base.get_position()
        dist = _distance(pos[0], pos[1], target[0], target[1])
        data = {
            "room": room_key,
            "target": [round(target[0], 1), round(target[1], 1)],
            "position": [round(pos[0], 1), round(pos[1], 1)],
            "distance_to_target": round(dist, 1),
            "reason": nav_result.reason,
        }

        if nav_result:
            # Record the visit only after a verified arrival (skills-nav:
            # navigate-authors-own-oracle): a failed attempt never moves a centre.
            memory = context.services.get("spatial_memory")
            if memory is not None and dist <= arrival_radius():
                memory.visit(room_key, pos[0], pos[1])
            return SkillResult(success=True, result_data=data)
        return SkillResult(
            success=False,
            error_message=(
                f"Navigation to {room_key} failed: {nav_result.reason} "
                f"(ended {dist:.1f} m from the target)"
            ),
            diagnosis_code="navigation_failed",
            result_data=data,
        )
