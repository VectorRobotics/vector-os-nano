# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Robot state context provider for LLM system prompt.

Collects real-time robot state (position, room, SceneGraph, nav state)
and formats it as an Anthropic system prompt block. Injected into
build_system_prompt() on every LLM turn.
"""
from __future__ import annotations

import math
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


_MAX_KNOWN_OBJECTS = 12


class RobotContextProvider:
    """Collects robot state and formats as Anthropic system block.

    Belief only: the base's own pose, the gripper's own holding flag, the scene
    graph's instances (what perception localized) and the world map — never
    ground truth.
    """

    def __init__(
        self, base: Any = None, scene_graph: Any = None, arm: Any = None,
        gripper: Any = None, world: Any = None,
    ) -> None:
        self._base = base
        self._sg = scene_graph
        self._arm = arm
        self._gripper = gripper
        self._world = world  # WorldManifest (rooms / surfaces), or None

    @classmethod
    def for_agent(cls, agent: Any) -> "RobotContextProvider":
        from vector_os_nano.vcli.worlds.world_manifest import world_of

        try:
            world = world_of(agent)
        except Exception as exc:  # noqa: BLE001 — the block just omits the map
            logger.debug("robot context: no world manifest: %s", exc)
            world = None
        return cls(
            base=getattr(agent, "_base", None), scene_graph=getattr(agent, "_spatial_memory", None),
            arm=getattr(agent, "_arm", None), gripper=getattr(agent, "_gripper", None), world=world,
        )

    def _room(self, x: float, y: float) -> str:
        boxes = getattr(self._world, "boxes", None) or {}
        room = next((r for r, (x0, y0, x1, y1) in boxes.items() if x0 <= x <= x1 and y0 <= y <= y1), None)
        if room is None and self._sg is not None:
            room = self._sg.nearest_room(x, y)
        return room or "unknown"

    def _known_objects(self) -> list[str]:
        """The scene graph's localized instances, newest first: what you believe is where."""
        now = time.time()
        objs = []
        for o in self._sg.find_objects_by_category(""):
            if (o.x, o.y) == (0.0, 0.0):  # never localized
                continue
            seen = float(getattr(o, "last_seen", None) or getattr(o, "first_seen", 0.0))
            objs.append((seen, o))
        objs.sort(key=lambda so: so[0], reverse=True)
        out = []
        for seen, o in objs[:_MAX_KNOWN_OBJECTS]:
            colour = (getattr(o, "attributes", None) or {}).get("color", "")
            label = f"{colour} {o.category}".strip()
            out.append(f"  {label} at ({o.x:.2f}, {o.y:.2f}, {o.z:.2f}), seen {_age(now - seen)} ago")
        return out

    def get_context_block(self) -> dict[str, str]:
        """Return Anthropic system block with current robot state."""
        lines: list[str] = []
        has_hardware = (
            self._base is not None or self._sg is not None or self._arm is not None
        )

        # Position + heading from base
        if self._base is not None:
            try:
                pos = self._base.get_position()
                lines.append(f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.2f})")
            except Exception:
                pass
            try:
                heading = self._base.get_heading()
                deg = math.degrees(heading)
                compass = _heading_to_compass(deg)
                lines.append(f"Heading: {deg:.0f} deg ({compass})")
            except Exception:
                pass
            # Current room: the map's room box, else the SceneGraph's nearest room
            if self._sg is not None or self._world is not None:
                try:
                    pos = self._base.get_position()
                    lines.append(f"Current room: {self._room(pos[0], pos[1])}")
                except Exception:
                    pass

        if self._gripper is not None and callable(getattr(self._gripper, "is_holding", None)):
            try:
                lines.append(f"Gripper: {'holding an object' if self._gripper.is_holding() else 'empty'}")
            except Exception:
                pass

        # SceneGraph stats
        if self._sg is not None:
            try:
                stats = self._sg.stats()
                doors = self._sg.get_all_doors()
                lines.append(
                    f"SceneGraph: {stats['rooms']} rooms "
                    f"({stats['visited_rooms']} visited), "
                    f"{len(doors)} doors, {stats['objects']} objects"
                )
            except Exception:
                pass
            try:
                summary = self._sg.get_room_summary()
                if summary:
                    lines.append(f"Rooms: {summary[:200]}")
            except Exception:
                pass
            try:
                known = self._known_objects()
                lines.append("Known objects (from your perception):" if known
                             else "Known objects: none localized yet")
                lines.extend(known)
            except Exception:
                pass

        # Nav/explore state — only meaningful for a mobile base (Go2), not an
        # arm-only sim, so don't inject quadruped fields into an arm prompt.
        if self._base is not None:
            try:
                from vector_os_nano.skills.go2 import explore
                running = explore.is_exploring()
                lines.append("Exploring: " + (f"yes ({explore._explore_backend})" if running else "no"))
                lines.append("Nav stack: " + ("running (FAR planner, TARE explore)" if explore.is_nav_stack_running()
                                              else "stopped (in-process planner, coverage explore)"))
            except ImportError:
                pass

        # Arm state (an arm-only sim has no base/scene_graph; still real hardware)
        if self._arm is not None:
            name = getattr(self._arm, "name", type(self._arm).__name__)
            dof = getattr(self._arm, "dof", None)
            lines.append(f"Arm: {name}" + (f" ({dof}-DOF)" if dof else "") + " connected")
            try:
                pos = self._arm.get_joint_positions()
                lines.append("Joints: " + ", ".join(f"{v:.2f}" for v in pos))
            except Exception:
                pass

        if not has_hardware:
            return {"type": "text", "text": "[Robot State]\nNo hardware connected."}

        return {"type": "text", "text": "[Robot State]\n" + "\n".join(lines)}


def _age(seconds: float) -> str:
    s = max(0.0, seconds)
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.0f} h"


def _heading_to_compass(degrees: float) -> str:
    """Compass label of a world yaw in degrees (0 = +x = east, CCW positive).

    The same convention as where_am_i (skills-nav:compass-mismatch: this used to
    call yaw 0 "N" while where_am_i called it "east").
    """
    from vector_os_nano.skills.go2.where_am_i import _heading_label

    return _heading_label(math.radians(degrees))
