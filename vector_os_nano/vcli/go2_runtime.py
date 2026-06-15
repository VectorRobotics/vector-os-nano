# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2 furnished-room runtime: boot the quadruped for VLM recognition (#9 R2).

The go2 sibling of ``g1_runtime.boot_g1_agent`` for the ``--scenario
go2_room_vlm`` world: instantiate ``MuJoCoGo2`` in FURNISHED room mode (the SAME
world-agnostic furnished room g1 uses — collidable chair/sofa/plant targets),
attach an ``Agent``, and register the mobile base set plus ``VlmSeekSkill`` (the
EXACT same recognition→approach skill the g1 furnished room runs — track C
proves the kernel is embodiment-agnostic, rule #2/#7).

This is a LIGHT boot (sim + skills only) — it deliberately does NOT start the
ROS2 bridge / nav stack / Rerun that the full ``--sim-go2`` apartment path spins
up; the furnished VLM room needs none of that.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _emit(on_status: "Callable[[str], None] | None", line: str) -> None:
    if on_status is not None:
        try:
            on_status(line)
        except Exception:  # noqa: BLE001 — status output never breaks boot
            pass
    logger.info("go2_runtime: %s", line)


def boot_go2_agent(
    world: Any,
    on_status: "Callable[[str], None] | None" = None,
    gui: bool = False,
) -> Any:
    """Boot the Go2 quadruped in the furnished VLM room and return a ready Agent.

    Raises on failure — the CLI keeps the REPL alive on a caught exception.
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    # Sinusoidal gait: the convex-MPC backend tracks velocity only under the
    # ROS bridge's continuous path-follower, not the skill's discrete
    # walk()+drive_for; the sinusoidal trot moves reliably from a bare walk()
    # call (measured: 1.06 m in 3×1.5 s), which is what the VLM seek loop issues.
    # Photoreal co-sim (campaign #10 R6): env-gated — RGB from Blender Cycles/OptiX
    # so the VLM grounds a photoreal frame (Go2 VLN + photoreal-driven Piper pick).
    import os
    photoreal = os.environ.get("VECTOR_G1_PHOTOREAL", "") not in ("", "0") or \
        os.environ.get("VECTOR_GO2_PHOTOREAL", "") not in ("", "0")
    _emit(on_status, "booting Go2 quadruped (sinusoidal trot) — furnished room (furniture/VLM)"
          + (" [PHOTOREAL co-sim: Blender/OptiX]" if photoreal else ""))
    base = MuJoCoGo2(gui=gui, room=True, furnished=True, backend="sinusoidal",
                     photoreal=photoreal)
    base.connect()
    base.stand()
    agent = Agent(base=base)

    # Mobile base set + VlmSeekSkill (rule 3 — single-source registry; no arm
    # skills on a quadruped base). RecognizeNavigateSkill is registered by
    # CAPABILITY PROBE (campaign #11 M2): Go2 now has navigate_to (+ geodesic +
    # a lidar return cloud), so the SAME photoreal VLN skill G1 runs becomes
    # available — but only when the base actually exposes navigate_to, never by
    # embodiment name (so a future Go2 mode without it gets no broken option).
    # NavigateToPointSkill stays omitted (recognize_navigate is the M2 target).
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.skills.go2.turn import TurnSkill
    from vector_os_nano.skills.go2.walk import WalkSkill
    from vector_os_nano.skills.vlm_seek import VlmSeekSkill

    registry = SkillRegistry()
    for s in (WalkSkill(), TurnSkill(), StopSkill(), VlmSeekSkill()):
        registry.register(s)
    vln = False
    if callable(getattr(base, "navigate_to", None)):
        from vector_os_nano.skills.recognize_navigate import RecognizeNavigateSkill
        registry.register(RecognizeNavigateSkill())
        vln = True
    agent._skill_registry = registry

    # GT furniture targets → world model: the deterministic at_position verify
    # anchor (semantic labels so verify binds to the object the VLM grounded).
    # The MEANS is VLM recognition (VlmSeekSkill), never a GT teleport (rule 5).
    from vector_os_nano.core.world_model import ObjectState
    from vector_os_nano.hardware.sim import g1_room
    labels = {f.name: f.label for f in g1_room.FURNITURE}
    for name, (tx, ty) in base.list_targets().items():
        agent._world_model.add_object(ObjectState(
            object_id=name, label=labels.get(name, name),
            x=float(tx), y=float(ty),
            confidence=1.0, state="placed",
            properties={"source": "go2_room_ground_truth"}))

    _emit(on_status, "Go2 base connected — walk/turn/stop/vlm_seek"
          + ("/recognize_navigate" if vln else "") + " ready"
          f" — {len(base.list_targets())} known targets")
    return agent
