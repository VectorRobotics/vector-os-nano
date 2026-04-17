"""NavigateSkill -- hardware-agnostic room-to-room navigation.

Supports two navigation modes:
1. NavStackClient (real navigation): when context.services.get("nav") is available
   and nav.is_available is True, publishes a waypoint goal and waits for
   goal_reached feedback from the navigation stack.
2. Dead-reckoning fallback: when no nav stack is present, uses SceneGraph door
   chain to navigate between named rooms via turn+walk sequences.

Room positions and door coordinates come entirely from the SceneGraph
(populated during exploration).  No hardcoded coordinates are used.

This module has no ROS2 imports at the top level.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nav config loader (lazy, module-level cache)
# ---------------------------------------------------------------------------

_NAV_CFG: dict | None = None


def _load_nav_config() -> dict:
    """Load nav.yaml with defaults. Searches relative paths then falls back."""
    import os
    import yaml

    global _NAV_CFG
    if _NAV_CFG is not None:
        return _NAV_CFG

    _search = [
        "config/nav.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "nav.yaml"),
    ]
    for path in _search:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                _NAV_CFG = data
                return _NAV_CFG
            except Exception as exc:
                logger.warning("nav.yaml load failed (%s), using defaults", exc)
    _NAV_CFG = {}
    return _NAV_CFG


def _nav(key: str, default: float) -> float:
    """Look up a navigation parameter by key, return default if absent."""
    cfg = _load_nav_config()
    nav_section = cfg.get("navigation", {})
    return float(nav_section.get(key, default))


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

_WALK_SPEED: float = 0.6     # m/s
_TURN_SPEED: float = 0.8     # rad/s
_ARRIVAL_RADIUS: float = 0.5  # meters -- close enough to target (dead-reckoning helper)
_DOORCHAIN_ARRIVAL_RADIUS: float = 0.8  # meters -- arrival threshold for nav stack door-chain
# Loaded from config/nav.yaml at first use; fallback keeps original behaviour
_DOORCHAIN_WAYPOINT_TIMEOUT: float = _nav("waypoint_timeout", 30.0)

_MIN_VISIT_COUNT: int = 1  # trust SceneGraph position after first visit


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


def _angle_between(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing angle from (x1,y1) to (x2,y2) in radians."""
    return math.atan2(y2 - y1, x2 - x1)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


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
    (not just a doorway drive-by).

    Uses get_room() if available (SceneGraph API), otherwise falls back
    to the older get_location() API (legacy SpatialMemory).

    Returns None if room not in memory or position not trustworthy.
    """
    # SceneGraph direct API — preferred, enforces visit_count threshold
    if hasattr(memory, "get_room"):
        room_node = memory.get_room(room_key)
        if room_node is not None:
            if (room_node.center_x != 0.0 or room_node.center_y != 0.0) and room_node.visit_count >= _MIN_VISIT_COUNT:
                return (room_node.center_x, room_node.center_y)
        # get_room is present but room not found or insufficient visits — do not
        # fall through to get_location(), which would bypass the visit threshold.
        return None

    # Backward-compatible get_location() API (legacy SpatialMemory only)
    if hasattr(memory, "get_location"):
        loc = memory.get_location(room_key)
        if loc is not None:
            x, y = getattr(loc, "x", 0.0), getattr(loc, "y", 0.0)
            if x != 0.0 or y != 0.0:
                return (x, y)

    return None


def _navigate_to_waypoint(
    base: Any,
    target_x: float,
    target_y: float,
    label: str,
) -> bool:
    """Turn toward waypoint and walk to it via dead-reckoning.

    Returns True if arrived upright, False if robot fell.
    """
    pos = base.get_position()
    cx, cy = pos[0], pos[1]
    heading = base.get_heading()

    dist = _distance(cx, cy, target_x, target_y)
    if dist < _ARRIVAL_RADIUS:
        logger.info("[NAV] Already at %s (%.1fm away)", label, dist)
        return True

    # Calculate required heading change
    target_angle = _angle_between(cx, cy, target_x, target_y)
    turn_needed = _normalize_angle(target_angle - heading)

    # Turn in place if heading delta > ~5.7 degrees
    if abs(turn_needed) > 0.1:
        vyaw = _TURN_SPEED if turn_needed > 0 else -_TURN_SPEED
        turn_dur = abs(turn_needed) / _TURN_SPEED
        logger.info("[NAV] Turn %.0f deg toward %s", math.degrees(turn_needed), label)
        base.walk(0.0, 0.0, vyaw, turn_dur)

    # Walk forward to waypoint
    walk_dur = dist / _WALK_SPEED
    logger.info("[NAV] Walk %.1fm to %s", dist, label)
    base.walk(_WALK_SPEED, 0.0, 0.0, walk_dur)

    # Upright check (z < 0.12 means robot has fallen)
    pos = base.get_position()
    if pos[2] < 0.12:
        logger.error("[NAV] Robot fell during navigation to %s", label)
        return False
    return True


# ---------------------------------------------------------------------------
# NavigateSkill
# ---------------------------------------------------------------------------

@skill(
    aliases=[
        "navigate", "go to", "goto",
        "去", "到", "走到", "去到", "导航",
    ],
    direct=True,
)
class NavigateSkill:
    """Navigate the robot to a room discovered during exploration.

    Mode selection (in priority order):
    1. NavStackClient (context.services["nav"]) -- full navigation stack,
       publishes waypoint goal and waits for goal_reached confirmation.
    2. Dead-reckoning -- turns toward waypoints and walks using SceneGraph
       door chain data.

    Room coordinates come exclusively from the SceneGraph populated
    during explore.  No hardcoded room positions are used.

    Works with ANY BaseProtocol implementation (not Go2-specific).
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

        # Cancel background exploration if running (navigate takes priority)
        try:
            from vector_os_nano.skills.go2.explore import cancel_exploration, is_exploring
            if is_exploring():
                cancel_exploration()
                logger.info("[NAV] Cancelled background exploration for navigation")
        except Exception:
            pass

        # Ensure nav flag exists so bridge path follower is armed
        try:
            import os
            if not os.path.exists("/tmp/vector_nav_active"):
                with open("/tmp/vector_nav_active", "w") as fh:
                    fh.write("1")
        except Exception:
            pass

        # --- Mode 0: Direct nav stack via proxy ---
        if hasattr(context.base, "navigate_to"):
            result = self._navigate_with_proxy(room_key, target, context)
            return result

        # --- Mode 1: NavStackClient ---
        nav = context.services.get("nav")
        if nav is not None and nav.is_available:
            result = self._navigate_with_nav_stack(nav, room_key, target, context)
        else:
            # --- Mode 2: Dead-reckoning fallback ---
            result = self._dead_reckoning(room_key, context)

        return result

    # ------------------------------------------------------------------
    # Navigation modes (private)
    # ------------------------------------------------------------------

    def _navigate_with_proxy(
        self,
        room_key: str,
        target: tuple[float, float],
        context: SkillContext,
    ) -> SkillResult:
        """Mode 0: Navigate via Go2ROS2Proxy.navigate_to() — FAR planner path.

        Called when context.base exposes navigate_to() (i.e. the proxy is
        connected to the live nav stack).  Falls back to dead-reckoning if
        the proxy call returns False.
        """
        logger.info(
            "[NAV] Proxy mode -> room=%s target=(%.1f, %.1f)",
            room_key, target[0], target[1],
        )

        def _progress(dist: float, elapsed: float) -> None:
            print(
                f"  >> 距目标 {dist:.1f}m, 已走 {int(elapsed)}s",
                file=sys.stderr,
                flush=True,
            )

        nav_result = context.base.navigate_to(
            target[0], target[1], timeout=45.0, on_progress=_progress
        )

        pos = context.base.get_position()
        dist = _distance(pos[0], pos[1], target[0], target[1])

        # Update spatial memory if available
        memory = context.services.get("spatial_memory")
        if memory is not None:
            memory.visit(room_key, pos[0], pos[1])

        if not nav_result:
            logger.warning(
                "[NAV] Proxy navigate_to timed out; falling back to dead-reckoning"
            )
            return self._dead_reckoning(room_key, context)

        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "target": [round(target[0], 1), round(target[1], 1)],
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "distance_to_target": round(dist, 1),
                "mode": "proxy_nav_stack",
            },
        )

    def _navigate_with_nav_stack(
        self,
        nav: Any,
        room_key: str,
        target: tuple[float, float],
        context: SkillContext,
    ) -> SkillResult:
        """Delegate navigation to NavStackClient.

        Sends /way_point and monitors position. Does not rely on /goal_reached
        since the nav stack doesn't always publish it reliably.
        """
        logger.info("[NAV] Using nav stack -> room=%s target=(%.1f, %.1f)",
                    room_key, target[0], target[1])

        # Send the waypoint and wait for result
        nav_result = nav.navigate_to(target[0], target[1], timeout=30.0)

        # Check actual position after navigation
        pos = context.base.get_position() if context.base else None
        if pos is None:
            odom = nav.get_state_estimation()
            pos = [odom.x, odom.y, odom.z] if odom else [0, 0, 0]

        dist = _distance(pos[0], pos[1], target[0], target[1])

        # Update spatial memory
        memory = context.services.get("spatial_memory")
        if memory is not None:
            memory.visit(room_key, pos[0], pos[1])

        if not nav_result:
            return SkillResult(
                success=False,
                error_message=f"Navigation to {room_key} failed (timeout or rejected)",
                diagnosis_code="navigation_failed",
                result_data={
                    "room": room_key,
                    "target": [round(target[0], 1), round(target[1], 1)],
                    "position": [round(pos[0], 1), round(pos[1], 1)],
                    "distance_to_target": round(dist, 1),
                    "mode": "nav_stack",
                },
            )

        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "target": [round(target[0], 1), round(target[1], 1)],
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "distance_to_target": round(dist, 1),
                "mode": "nav_stack",
            },
        )

    def _dead_reckoning(
        self,
        room_key: str,
        context: SkillContext,
        total_timeout: float = 45.0,
    ) -> SkillResult:
        """Navigate via nav stack door chain using SceneGraph waypoints.

        Publishes each waypoint to /way_point via base.navigate_to() so the
        localPlanner handles obstacle avoidance.  The total_timeout budget is
        divided dynamically across remaining waypoints (min 5s each); arrival
        is confirmed when within _DOORCHAIN_ARRIVAL_RADIUS meters of target.
        """
        base = context.base
        sg = context.services.get("spatial_memory")

        pos = base.get_position()
        cx, cy = pos[0], pos[1]
        src_room = _detect_current_room(cx, cy, sg=sg)

        # Get door chain from SceneGraph
        if sg is None or not hasattr(sg, "get_door_chain"):
            return SkillResult(
                success=False,
                error_message="No door data. Explore first.",
                diagnosis_code="room_not_explored",
            )

        # Check if already at destination
        target_room_node = sg.get_room(room_key) if hasattr(sg, "get_room") else None
        if target_room_node is not None:
            target_cx = target_room_node.center_x
            target_cy = target_room_node.center_y
            if _distance(cx, cy, target_cx, target_cy) < _DOORCHAIN_ARRIVAL_RADIUS:
                return SkillResult(
                    success=True,
                    result_data={
                        "room": room_key,
                        "position": [round(cx, 1), round(cy, 1)],
                        "note": "already here",
                    },
                )

        logger.info("[NAV] Door-chain (nav stack): %s -> %s", src_room, room_key)

        # Get waypoint sequence from SceneGraph door chain
        waypoints = sg.get_door_chain(src_room, room_key)

        if not waypoints:
            return SkillResult(
                success=False,
                error_message=(
                    f"No door data between '{src_room}' and '{room_key}'. "
                    "Explore first."
                ),
                diagnosis_code="room_not_explored",
            )

        # Execute each waypoint via nav stack (obstacle avoidance)
        # Dynamic per-waypoint timeout: divide remaining budget evenly across
        # remaining waypoints, but never less than 5s per waypoint.
        start_time = time.monotonic()

        for i, (wx, wy, label) in enumerate(waypoints):
            # --- Abort check between waypoints ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    return SkillResult(
                        success=False,
                        error_message="Navigation aborted",
                        diagnosis_code="aborted",
                    )
            except ImportError:
                pass

            # Compute remaining budget for this waypoint
            elapsed = time.monotonic() - start_time
            remaining = total_timeout - elapsed
            if remaining <= 0:
                return SkillResult(
                    success=False,
                    error_message="Navigation timeout",
                    diagnosis_code="navigation_failed",
                )
            n_remaining = len(waypoints) - i
            per_wp = max(remaining / n_remaining, 5.0)

            # Check arrival before sending — skip waypoint if already close enough
            cur_pos = base.get_position()
            if _distance(cur_pos[0], cur_pos[1], wx, wy) < _DOORCHAIN_ARRIVAL_RADIUS:
                logger.info("[NAV] Already within %.1fm of %s — skipping", _DOORCHAIN_ARRIVAL_RADIUS, label)
                continue

            logger.info(
                "[NAV] Navigate to waypoint %s (%.1f, %.1f) timeout=%.0fs",
                label, wx, wy, per_wp,
            )
            cur_pos2 = base.get_position()
            seg_dist = _distance(cur_pos2[0], cur_pos2[1], wx, wy)
            print(
                f"  >> 前往 {label} (距离 {seg_dist:.1f}m)",
                file=sys.stderr,
                flush=True,
            )

            def _progress(dist: float, elapsed_s: float) -> None:
                print(
                    f"  >> 距目标 {dist:.1f}m, 已走 {int(elapsed_s)}s",
                    file=sys.stderr,
                    flush=True,
                )

            # Use go_to_waypoint (simple /way_point) to avoid recursive
            # navigate_to → FAR probe → door-chain → navigate_to cascade.
            _go_fn = getattr(base, "go_to_waypoint", None) or base.navigate_to
            ok = _go_fn(
                float(wx), float(wy),
                timeout=per_wp,
                on_progress=_progress,
            )
            if not ok:
                # navigate_to returned False — timed out or rejected by nav stack
                return SkillResult(
                    success=False,
                    error_message=f"Navigation timed out near {label}",
                    diagnosis_code="navigation_failed",
                )

        final_pos = base.get_position()
        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "from_room": src_room,
                "position": [round(final_pos[0], 1), round(final_pos[1], 1)],
                "mode": "dead_reckoning",
            },
        )
