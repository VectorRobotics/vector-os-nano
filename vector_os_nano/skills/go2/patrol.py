# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PatrolSkill -- multi-room patrol for the Go2 quadruped.

Navigates to each target room in sequence, calls LookSkill to observe the scene with VLM, and returns a structured report.

Skill registry lookup order:
1. context.services["skill_registry"].get("look")
2. Direct import of LookSkill

Total patrol timeout: 5 minutes by default, configurable via params or
context.config["patrol_timeout"].

Falls through to the next room on navigation failure; aborts entirely if the
robot falls (base z < 0.12 m) or a stop arrives (checked before each navigate and
each look). Named rooms resolve through the navigate aliases (厨房 -> kitchen); a
failed look is reported, never written into the scene graph as a description.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import _resolve_room
from vector_os_nano.vcli.cognitive.abort import is_abort_requested

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = 300.0   # 5 minutes
_DEFAULT_MAX_ROOMS: int = 8
_FALL_Z_THRESHOLD: float = 0.12   # metres — robot has fallen if z < this


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_skill(registry: Any | None, name: str) -> Any | None:
    """Return a skill from registry, or None if registry absent / skill missing."""
    if registry is None:
        return None
    return registry.get(name)


def _resolve_look_skill(context: SkillContext) -> Any | None:
    """Use the registered observation skill, or the shared camera observer."""
    from vector_os_nano.skills.go2.look import LookSkill

    return _get_skill(context.services.get("skill_registry"), "look") or LookSkill()


def _resolve_navigate_skill(context: SkillContext) -> Any | None:
    """Return NavigateSkill from registry or direct import."""
    registry: Any = context.services.get("skill_registry")
    nav = _get_skill(registry, "navigate")
    if nav is not None:
        return nav

    try:
        from vector_os_nano.skills.navigate import NavigateSkill
        return NavigateSkill()
    except ImportError:
        return None


def _robot_has_fallen(context: SkillContext) -> bool:
    """Return True if base z-position is below the fall threshold."""
    base = context.base
    if base is None:
        return False
    try:
        pos = base.get_position()
        return float(pos[2]) < _FALL_Z_THRESHOLD
    except Exception:
        return False


def _extract_description(look_result: SkillResult) -> str:
    """Pull a human-readable description out of a look/describe SkillResult.

    Key priority (matches actual skill output fields):
    - "summary"     -- LookSkill / DescribeSceneSkill full-description mode
    - "description" -- DescribeSkill caption mode (legacy)
    - "answer"      -- DescribeSkill query mode (legacy)
    """
    if not look_result.success:
        return f"(observation failed: {look_result.error_message})"
    data = look_result.result_data
    for key in ("summary", "description", "answer"):
        value = data.get(key)
        if value:
            return str(value)
    return "(no description returned)"


# ---------------------------------------------------------------------------
# PatrolSkill
# ---------------------------------------------------------------------------

@skill(requires=('base',),
    aliases=[
        "patrol",
        "巡逻",
        "巡视",
        "patrol the house",
        "check all rooms",
        "检查所有房间",
    ],
)
class PatrolSkill:
    """Patrol all rooms in the house, navigating to each and describing what is seen."""

    name: str = "patrol"
    description: str = (
        "Patrol all rooms in the house, navigating to each and describing what is seen."
    )
    parameters: dict = {
        "rooms": {
            "type": "array",
            "items": {"type": "string"},
            "required": False,
            "description": "Specific rooms to patrol, any alias (default: every known room)",
        },
        "max_rooms": {
            "type": "integer",
            "required": False,
            "default": _DEFAULT_MAX_ROOMS,
            "description": "Maximum rooms to visit in one patrol",
        },
        "timeout": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_TIMEOUT,
            "description": "Total patrol timeout in seconds (default 300)",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"patrolled": True}
    failure_modes: list[str] = ["no_base", "navigation_failed", "unknown_room", "aborted"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.base is None:
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        timeout: float = float(
            params.get(
                "timeout",
                context.config.get("patrol_timeout", _DEFAULT_TIMEOUT),
            )
        )
        max_rooms: int = int(params.get("max_rooms", _DEFAULT_MAX_ROOMS))

        # Resolve sub-skills
        navigate_skill = _resolve_navigate_skill(context)
        look_skill = _resolve_look_skill(context)

        if navigate_skill is None:
            return SkillResult(
                success=False,
                error_message="NavigateSkill not available",
                diagnosis_code="no_navigate_skill",
            )

        # Determine patrol target list
        rooms_to_visit, unknown = self._resolve_rooms(params, context, max_rooms)
        if not rooms_to_visit:
            return SkillResult(
                success=False,
                error_message=(
                    f"No patrol rooms: unknown {unknown}" if unknown
                    else "No rooms known. Run explore first."
                ),
                diagnosis_code="unknown_room",
                result_data={"rooms_unknown": unknown},
            )

        # Execute patrol loop
        result = self._run_patrol(
            rooms_to_visit,
            navigate_skill,
            look_skill,
            context,
            timeout,
        )
        if unknown:
            result.result_data["rooms_unknown"] = unknown
        return result

    # ------------------------------------------------------------------
    # Room selection
    # ------------------------------------------------------------------

    def _resolve_rooms(
        self,
        params: dict,
        context: SkillContext,
        max_rooms: int,
    ) -> tuple[list[str], list[str]]:
        """(rooms to patrol in order, explicitly named rooms that resolved to nothing).

        Explicit names resolve through the navigate aliases against the SceneGraph;
        with none given, every known room is patrolled.
        """
        memory: Any = context.services.get("spatial_memory")
        explicit: list[str] | None = params.get("rooms")
        if explicit:
            resolved, unknown = [], []
            for name in explicit:
                key = _resolve_room(str(name), sg=memory)
                if key is None:
                    unknown.append(str(name))
                elif key not in resolved:
                    resolved.append(key)
            return resolved[:max_rooms], unknown
        known = memory.get_visited_rooms() if memory is not None else []
        return list(known)[:max_rooms], []

    # ------------------------------------------------------------------
    # Patrol execution
    # ------------------------------------------------------------------

    def _run_patrol(
        self,
        rooms: list[str],
        navigate_skill: Any,
        look_skill: Any | None,
        context: SkillContext,
        timeout: float,
    ) -> SkillResult:
        """Iterate rooms, navigate + observe each one, collect results."""
        deadline: float = time.monotonic() + timeout

        rooms_visited: list[str] = []
        rooms_failed: list[str] = []
        observations: dict[str, str] = {}

        aborted = False
        for room in rooms:
            # Global timeout / stop / fall guards, before the navigate
            if time.monotonic() >= deadline:
                logger.warning("[PATROL] Timeout reached — stopping patrol early")
                break
            if is_abort_requested():
                aborted = True
                break
            if _robot_has_fallen(context):
                logger.error("[PATROL] Robot fell — aborting patrol")
                break

            # --- Navigate ---
            logger.info("[PATROL] Navigating to %s...", room)
            nav_result: SkillResult = navigate_skill.execute(
                {"room": room}, context
            )

            if not nav_result.success:
                logger.warning(
                    "[PATROL] Navigation to %s failed: %s",
                    room,
                    nav_result.error_message,
                )
                rooms_failed.append(room)
                continue
            rooms_visited.append(room)

            # --- Observe (guards again: the navigate may have taken a while) ---
            if is_abort_requested():
                aborted = True
                break
            if look_skill is None or time.monotonic() >= deadline:
                observations[room] = "(not observed)"
                continue
            logger.info("[PATROL] Observing %s...", room)
            look_result: SkillResult = look_skill.execute({}, context)
            description = _extract_description(look_result)
            observations[room] = description
            if not look_result.success:
                continue  # a failed look is reported, never stored as a description

            memory: Any = context.services.get("spatial_memory")
            if memory is not None:
                try:
                    memory.observe(room, [], description=description)
                except Exception as exc:
                    logger.warning("[PATROL] spatial_memory update failed: %s", exc)

        # --- Build report ---
        total = len(rooms)
        visited_count = len(rooms_visited)

        spatial_summary: str = ""
        memory = context.services.get("spatial_memory")
        if memory is not None:
            try:
                spatial_summary = memory.get_room_summary()
            except Exception as exc:
                logger.warning("[PATROL] get_room_summary failed: %s", exc)

        overall_success: bool = visited_count > 0 and not aborted

        return SkillResult(
            success=overall_success,
            result_data={
                "rooms_visited": rooms_visited,
                "rooms_failed": rooms_failed,
                "observations": observations,
                "coverage": f"{visited_count}/{total} rooms visited",
                "spatial_summary": spatial_summary,
            },
            error_message="" if overall_success else (
                "Patrol stopped" if aborted else "No rooms were successfully patrolled"),
            diagnosis_code="" if overall_success else ("aborted" if aborted else "navigation_failed"),
        )
