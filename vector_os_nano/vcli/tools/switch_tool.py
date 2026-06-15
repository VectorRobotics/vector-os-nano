# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SwitchEmbodimentTool — hot-swap the active embodiment in one vector-cli session.

Campaign #11 M1 (ADR-011). Locomotion-only g1<->go2 runtime switch: a single NL
command ('switch to g1' / '切换到 go2') tears down the current sim agent and
brings up the other embodiment WITHOUT restarting the CLI.

Seam (ADR-011, boot-then-swap):
  1. no-op if already on the target embodiment;
  2. boot the TARGET runtime FIRST (so a failed boot never leaves zero
     embodiments — the old agent stays live and usable);
  3. only after the target booted OK: ``SimStartTool._shutdown_agent(old)``
     (kills the Go2 subprocess group + disconnects + joins the physics/gait
     thread + closes the GL/viewer);
  4. ``SimStartTool._rebind_agent(new)`` — the SAME single-source rebind that
     start_simulation uses, which also drops the old embodiment's stale
     robot-category skill tools.

Kernel / BaseProtocol unchanged (rule 2/7). Fails loud (rule 8) on an unknown
target or a boot error, rolling back to a clean single-embodiment state so a
half-switch is never left behind.
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolResult,
    tool,
)
from vector_os_nano.vcli.tools.sim_tool import SimStartTool

logger = logging.getLogger(__name__)

# Target embodiment -> default playground scenario (the furnished locomotion
# room each runtime boots into; both embodiments share the same room so the
# locomotion comparison is apples-to-apples for M1).
_DEFAULT_SCENARIO: dict[str, str] = {
    "g1": "g1_room_vlm",
    "go2": "go2_room_vlm",
}


@tool(
    name="switch_embodiment",
    description=(
        "Hot-swap the active robot embodiment in this session between 'g1' "
        "(Unitree G1 humanoid, RL-policy gait) and 'go2' (Unitree Go2 "
        "quadruped, sinusoidal trot) WITHOUT restarting the CLI. Use when the "
        "user says 'switch to g1' / 'switch to go2' / '切换到 g1' / '换成 go2'. "
        "No-op if already on the target. Boots the target first, then tears "
        "down the old one, so a failed switch never drops to zero robots."
    ),
    read_only=False,
    permission="ask",
)
class SwitchEmbodimentTool:
    """Switch the active locomotion embodiment (g1 <-> go2) at runtime."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["g1", "go2"],
                "description": "Which embodiment to switch to: 'g1' or 'go2'.",
            },
            "scenario": {
                "type": "string",
                "description": (
                    "Optional playground scenario id for the target (default "
                    "'g1_room_vlm' / 'go2_room_vlm'). Its embodiment must match."
                ),
            },
        },
        "required": ["target"],
    }

    @staticmethod
    def _current_embodiment(agent: Any) -> "str | None":
        """'g1' / 'go2' from the connected base's class name, else None."""
        base = getattr(agent, "_base", None)
        if base is None:
            return None
        cls = type(base).__name__.lower()      # MuJoCoG1 / MuJoCoGo2
        if "go2" in cls:
            return "go2"
        if "g1" in cls:
            return "g1"
        return None

    @staticmethod
    def _boot_target(target: str, scenario: str, gui: bool) -> tuple[Any, Any]:
        """Boot the target embodiment's furnished runtime; return (agent, world).

        Mirrors SimStartTool._start_g1: import playground (register scenarios),
        resolve the named world, fail loud on an embodiment mismatch, boot the
        matching runtime. g1 hard-wires prefer_daemon=True (mid-REPL gait must
        run on its own daemon thread, not a returned worker thread)."""
        import vector_os_nano.playground  # noqa: F401  (register scenarios)
        from vector_os_nano.vcli.worlds import resolve_world_named

        world = resolve_world_named(scenario)
        emb = getattr(getattr(world, "scenario", None), "embodiment", "")
        if emb != target:
            raise ValueError(
                f"scenario '{scenario}' is not a {target} scenario "
                f"(embodiment={emb!r})")
        if target == "g1":
            from vector_os_nano.vcli import g1_runtime
            agent = g1_runtime.boot_g1_agent(world, gui=gui, prefer_daemon=True)
        else:
            from vector_os_nano.vcli import go2_runtime
            agent = go2_runtime.boot_go2_agent(world, gui=gui)
        return agent, world

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        target: str = params.get("target", "")
        if target not in _DEFAULT_SCENARIO:
            return ToolResult(  # fail loud (rule 8): name the valid set
                content=(f"Unknown embodiment {target!r}. Valid targets: "
                         f"{sorted(_DEFAULT_SCENARIO)}."),
                is_error=True)

        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available.", is_error=True)
        old_agent = app.get("agent")
        if old_agent is None:
            return ToolResult(
                content=(f"No simulation running. Use start_simulation "
                         f"(sim_type='{target}') first."),
                is_error=True)

        if self._current_embodiment(old_agent) == target:
            return ToolResult(content=f"Already running {target}; no switch needed.")

        scenario = params.get("scenario") or _DEFAULT_SCENARIO[target]
        gui = bool(app.get("sim_gui", True))

        # boot-then-swap: boot the target FIRST; on failure nothing is mutated
        # (old embodiment stays live) — a true rollback (rule 8 fail loud).
        try:
            new_agent, new_world = self._boot_target(target, scenario, gui)
        except Exception as exc:  # noqa: BLE001
            logger.warning("switch_embodiment boot %s failed: %s", target, exc)
            return ToolResult(
                content=(f"Failed to boot '{target}' ({exc}). Kept the current "
                         f"embodiment running — no switch performed."),
                is_error=True)

        # Target is live; now tear down the old one and rebind to the new.
        try:
            teardown = SimStartTool._shutdown_agent(old_agent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("old embodiment teardown error: %s", exc)
            teardown = f"teardown warning: {exc}"

        SimStartTool._rebind_agent(app, context, new_agent, new_world)

        base = getattr(new_agent, "_base", None)
        skill_count = (len(new_agent._skill_registry.list_skills())
                       if hasattr(new_agent, "_skill_registry") else 0)
        return ToolResult(
            content=(f"Switched to {target} ({type(base).__name__}), "
                     f"{skill_count} skills registered. [{teardown}]"))

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext,
    ) -> PermissionResult:
        return PermissionResult(behavior="allow", reason="Embodiment switch")
