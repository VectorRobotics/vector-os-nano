# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""G1 humanoid real-gait runtime: boot the policy-driven MuJoCo base.

Campaign #5. The mirror of ``habitat_runtime.boot_habitat_agent`` for the
in-process G1 gait scenario (``--scenario g1_flat``): instantiate
``G1MuJoCoBase`` on the pretrained unitree_rl_gym policy, attach it to an
``Agent``, and rebuild the agent's skill registry with the BASE-ONLY set
(rule 3 — no split-brain: a humanoid base gets walk/turn/stop, never arm
skills). The kernel never imports a simulator from config; the CLI hands in
the RESOLVED playground world and this module does the lazy domain wiring.

Assets are NOT vendored — ``scripts/setup_g1_gait.sh`` fetches the policy +
MJCF into ``assets/g1_gait/``; a missing asset fails LOUD with that hint.
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
    logger.info("g1_runtime: %s", line)


def boot_g1_agent(
    world: Any,
    on_status: "Callable[[str], None] | None" = None,
    gui: bool = False,
) -> Any:
    """Boot the policy-driven G1 base for ``world`` and return a ready Agent.

    ``gui`` opens a live MuJoCo viewer window (campaign #8 R0) so the owner can
    WATCH the gait while driving it from vector-cli — the CLI passes
    ``gui=not args.headless``. Raises on any failure — callers decide how loud
    to surface it (the CLI keeps the REPL alive on a caught exception).
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_g1 import G1MuJoCoBase, g1_assets_ready

    if not g1_assets_ready():
        raise FileNotFoundError(
            "G1 gait assets missing — run scripts/setup_g1_gait.sh first "
            "(downloads the pretrained policy + MJCF into assets/g1_gait/; "
            "heavy assets are never vendored into git)")

    # Room mode (campaign #8 R3): the g1_room scenario builds a closed room
    # with real-collision walls/obstacles + a virtual lidar; g1_flat stays the
    # flat gait scene. Detected from the resolved world's scenario id.
    scenario = getattr(world, "scenario", None)
    scenario_id = getattr(scenario, "id", "")
    # g1_room = colour-box room (campaign #8); g1_room_vlm = furnished room with
    # real furniture meshes for VLM semantic recognition (campaign #9 R1).
    furnished = scenario_id == "g1_room_vlm"
    room = scenario_id in ("g1_room", "g1_room_vlm")
    _emit(on_status, "booting G1 humanoid (unitree_rl_gym policy gait)"
          + (" — furnished room (furniture/VLM)" if furnished
             else " — room (walls/obstacles/lidar)" if room else ""))
    base = G1MuJoCoBase(gui=gui, room=room, furnished=furnished)
    base.connect()
    agent = Agent(base=base)

    # Base-only world: rebuild the registry with the mobile set (rule 3 —
    # single-source, no split-brain; arm defaults would put pick/place in the
    # planner's mouth with no arm attached). Same shape as boot_habitat_agent.
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.skills.go2.turn import TurnSkill
    from vector_os_nano.skills.go2.walk import WalkSkill
    from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

    # NavigateToPointSkill is base-generic (campaign #6): it drives
    # base.navigate_to(x, y, tol) and reads the three-value contract —
    # G1MuJoCoBase implements both navigate_to and geodesic_distance
    # (euclidean on the flat scene), so the skill works UNCHANGED.
    registry = SkillRegistry()
    for s in (WalkSkill(), TurnSkill(), StopSkill(), NavigateToPointSkill()):
        registry.register(s)
    if room:
        # ExploreSkill + VisionSeekSkill need the room's grid/lidar/camera.
        from vector_os_nano.skills.explore import ExploreSkill
        from vector_os_nano.skills.explore_seek import ExploreAndSeekSkill
        from vector_os_nano.skills.vision_seek import VisionSeekSkill
        registry.register(ExploreSkill())
        registry.register(VisionSeekSkill())
        registry.register(ExploreAndSeekSkill())
    if furnished:
        # VlmSeekSkill grounds a SEMANTIC class with a real VLM (Qwen-VL) —
        # the track-A perception path for the furnished room (campaign #9 R1).
        from vector_os_nano.skills.vlm_seek import VlmSeekSkill
        registry.register(VlmSeekSkill())
        # RecognizeNavigateSkill (campaign #9 R5): the RELIABLE arrival path —
        # VLM recognise → DEPTH-AT-BBOX locate (depth at the recognised pixels =
        # the object's distance, skipping intervening obstacles, Case 22) →
        # g1 navigate_to. 3/3 GUI-verified arrivals (vs vlm_seek's flaky servoing,
        # Case 21). Preferred for getting all the way to a furniture object.
        from vector_os_nano.skills.recognize_navigate import RecognizeNavigateSkill
        registry.register(RecognizeNavigateSkill())
    agent._skill_registry = registry

    if room:
        # Register the room's GT-known targets into the world model. They are the
        # deterministic VERIFY anchor (at_position(x, y, tol) with the object's
        # real coordinates) — the honest judge that the robot ACTUALLY reached
        # the thing it recognised. The MEANS is recognition (vision_seek colour
        # / vlm_seek semantic), never a GT teleport — rule 5. For the furnished
        # room the label is the semantic class ('chair'), so verify can bind to
        # the object the VLM grounded.
        from vector_os_nano.core.world_model import ObjectState
        labels = {}
        if furnished:
            from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
            # Combined "en zh" label so visited('chair') OR visited('椅子') both
            # match (get_objects_by_label is case-insensitive substring) — the
            # planner may emit either tongue; the verify must bind regardless (R6).
            _zh = {"chair": "椅子", "sofa": "沙发", "potted plant": "盆栽"}
            labels = {f.name: f"{f.label} {_zh.get(f.label, '')}".strip()
                      for f in g1_room.FURNITURE}
        for name, (tx, ty) in base.list_targets().items():
            agent._world_model.add_object(ObjectState(
                object_id=name, label=labels.get(name, name),
                x=float(tx), y=float(ty),
                confidence=1.0, state="placed",
                properties={"source": "g1_room_ground_truth"}))

    _emit(on_status, "G1 base connected — walk/turn/stop/navigate_to ready"
          + (f" — {len(base.list_targets())} known targets" if room else ""))
    return agent
