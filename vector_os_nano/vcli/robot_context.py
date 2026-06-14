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
from typing import Any

logger = logging.getLogger(__name__)


class RobotContextProvider:
    """Collects robot state and formats as Anthropic system block."""

    def __init__(
        self,
        base: Any = None,
        scene_graph: Any = None,
        arm: Any = None,
        world: Any = None,
        world_model: Any = None,
    ) -> None:
        self._base = base
        self._sg = scene_graph
        self._arm = arm
        self._world = world
        self._world_model = world_model

    def _world_backend(self) -> str:
        scenario = getattr(self._world, "scenario", None)
        return getattr(scenario, "sim_backend", "mujoco") if scenario else "mujoco"

    def get_context_block(self) -> dict[str, str]:
        """Return Anthropic system block with current robot state."""
        lines: list[str] = []
        has_hardware = (
            self._base is not None or self._sg is not None or self._arm is not None
        )
        is_habitat = self._world_backend() == "habitat"

        # World/scenario line — a habitat scenario means the simulator is
        # ALREADY live (the base IS the world); say so explicitly, or the LLM
        # reads "Base + nothing else" as "no sim running".
        if is_habitat:
            scenario = self._world.scenario
            lines.append(
                f"World: '{scenario.id}' — habitat photoreal scene "
                f"({scenario.scene_ref}), RUNNING (already started; "
                "do not launch any simulator)"
            )
            n_objects = None
            if self._world_model is not None:
                try:
                    n_objects = len(self._world_model.get_objects())
                except Exception:  # noqa: BLE001 — state line, never raise
                    n_objects = None
            if n_objects is not None:
                if n_objects == 0:
                    lines.append(
                        "Live objects: 0 — start SysNav perception "
                        "(sysnav_perception tool) to populate the world model"
                    )
                else:
                    lines.append(f"Live objects: {n_objects} (semantic world model)")

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
            # Current room from SceneGraph
            if self._sg is not None:
                try:
                    pos = self._base.get_position()
                    room = self._sg.nearest_room(pos[0], pos[1])
                    lines.append(f"Current room: {room or 'unknown'}")
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

        # Nav/explore state — only meaningful for the MuJoCo/ROS2 quadruped
        # stack (pgrep'd FAR/TARE processes). NOT for an arm-only sim, and NOT
        # for the habitat world, where "Nav stack: stopped" reads as "no sim
        # running" despite the live world (owner live-test finding 2026-06-11).
        if self._base is not None and not is_habitat:
            try:
                from vector_os_nano.skills.go2.explore import is_exploring, is_nav_stack_running
                lines.append(f"Exploring: {'yes' if is_exploring() else 'no'}")
                lines.append(f"Nav stack: {'running' if is_nav_stack_running() else 'stopped'}")
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


def _heading_to_compass(degrees: float) -> str:
    """Convert heading in degrees to compass direction."""
    # Normalize to 0-360
    d = degrees % 360
    if d < 0:
        d += 360
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(d / 22.5) % 16
    return directions[idx]
