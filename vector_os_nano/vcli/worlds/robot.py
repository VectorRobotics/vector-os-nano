# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Robot world: the persona composed from the body's manifests, the sim session's
ground-truth oracles and the learned detector capability. Skill tools come from the
agent's skill registry (``wrap_skills``). It imports nothing robot-specific at
module load.
"""

from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.vcli.prompt import ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS

logger = logging.getLogger(__name__)


def _agent_has_camera(agent: Any) -> bool:
    """True iff *agent* can supply an RGB frame the detector can run on.

    Single-sourced (Rule 11) onto the capability resolver — the ``camera`` flag of the
    one declared ``CapabilityProfile`` — so the detector's camera gate can never drift
    from the navigate/base gate. The resolver's ``camera`` is the SAME world-agnostic
    duck-typed check this used to inline (a bound ``_perception`` exposing
    ``get_color_frame()`` OR a base/arm exposing ``get_camera_frame()``), so the
    behavior is byte-identical; fails safe to False (no-agent / sensorless).
    """
    from vector_os_nano.embodiments.capability_profile import resolve_capability_profile

    return resolve_capability_profile(agent).camera


def task_policy(agent: Any) -> str:
    """How THIS body does physical tasks, composed from its manifests and skills.

    The embodiment manifest decides which lines apply (base, arm + gripper), the
    world manifest supplies the map (rooms, support surfaces, where the user is)
    and a line naming a skill appears only when that skill is registered. Nothing
    here is ground truth: the map is configuration, object positions come from
    perception (the per-turn [Robot State] block).
    """
    from vector_os_nano.embodiments.capability_profile import resolve_capability_profile
    from vector_os_nano.schemas.world import world_of

    if agent is None:
        return ""
    caps = resolve_capability_profile(agent)
    registry = getattr(agent, "_skill_registry", None)
    skills = set(registry.list_skills()) if registry is not None else set()
    world = world_of(agent)
    lines = ["Task policy for this body (each goal: the action, then its verify):"]
    lines.append("Available skills: " + ", ".join(sorted(skills)))
    if caps.has_base:
        lines.append(
            "- Reach a coordinate with navigate(x, y): it plans around obstacles. Use walk only for a "
            "short relative step the user asked for. Verify at_position(x, y)."
        )
        if world is not None and world.rooms:
            lines.append(
                f"- Reach a room with go_to_room(room), then verify in_room('<room>'). Rooms: {', '.join(world.rooms)}."
            )
    grasp = "grasp" if "grasp" in skills else ""
    if caps.has_gripper and grasp:
        if "look" in skills:
            search = "look (it localizes the objects the camera sees)"
            if caps.has_base and "explore" in skills:
                search += "; still unknown: go_to_room to a likely room and look again, or explore"
            lines.append(
                "- Know where an object is before you grasp it. The [Robot State] block lists the objects "
                f"you have localized; a target that is not there is unknown: find it with {search}."
            )
            lines.append(
                "- Perception names things in its own words: a green bottle may be listed as a 'green "
                "cylinder'. Match by colour and shape, and query a detector with the user's words."
            )
        if caps.has_base and "navigate_to_object" in skills:
            lines.append(
                "- A known object more than about 1.5 m away: navigate_to_object first, then grasp."
            )
        lines.append(
            f"- An object in view and within reach: grasp it with {grasp}, naming "
            "it as the user did, then verify holding_object('<object>')."
        )
        if caps.has_base and world is not None and world.user is not None:
            ux, uy = world.user
            lines.append(
                f"- 拿过来 / bring / fetch = grasp it, then carry it to the user: navigate({ux}, {uy}), "
                "then verify near_object('user') and kept_holding('<object>')."
            )
        place = "place_top_down" if "place_top_down" in skills else ""
        if place and world is not None and world.surfaces:
            tops = "; ".join(
                f"{n} centre ({(s.box[0] + s.box[2]) / 2:.2f}, {(s.box[1] + s.box[3]) / 2:.2f}) top {s.height:.2f} m"
                for n, s in world.surfaces.items()
            )
            lines.append(
                f"- 放 / put / place it on a surface: {place} with target_xyz = [centre x, centre y, top], "
                f"then verify object_on('<object>', '<surface>'). Surfaces: {tops}."
            )
        lines.append(
            "- Name an object in a verify the way the user did, e.g. holding_object('绿色瓶子') or "
            "holding_object('green bottle'); 'it' / '它' is the object of the earlier turn."
        )
    if not skills:
        return ""
    lines.append(
        "- A skill that fails twice for the same reason will not succeed a third time: stop and "
        "report why. Do not read or debug code in the middle of a physical task."
    )
    return "\n".join(lines)


class RobotWorld:
    """Adapter for the robot domain (Go2 / SO-101 / Piper)."""

    name = "robot"

    def is_robot(self) -> bool:
        return True

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
        policy = task_policy(agent)
        return ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS + ("\n" + policy if policy else "")

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        """The GT oracles of the live sim session that owns *agent*.

        Every predicate reads the session's ground truth (``vcli/worlds/sim_oracles``),
        never an agent or device attribute. Real hardware / no session contributes
        nothing, so every verify there grades RAN. This is the engine's whole verify
        namespace: the kernel adds no predicate of its own.
        """
        session = self.session(agent)
        if session is None or session.gt is None:
            return {}
        from vector_os_nano.vcli.worlds.sim_oracles import session_namespace

        return session_namespace(session)

    def session(self, agent: Any) -> Any:
        from vector_os_nano.sim.session import session_for

        return session_for(agent)

    def mark_turn(self, agent: Any, phase: str, goal: str = "") -> None:
        from vector_os_nano.sim.session import mark_turn

        mark_turn(agent, phase, goal)

    def sim_id(self, agent: Any) -> str:
        from vector_os_nano.sim.session import sim_id

        return sim_id(agent)

    def has_base(self, agent: Any) -> bool:
        from vector_os_nano.embodiments.capability_profile import resolve_capability_profile

        return resolve_capability_profile(agent).has_base

    def actor_epoch(self) -> Any:
        from vector_os_nano.sim.runtime import ACTOR_EPOCH

        return ACTOR_EPOCH.tool()

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        # Register the learned open-vocabulary DETECTOR (grounding-dino-tiny) as a
        # routable second model family. Read-only — it only perceives; the verify
        # spine remains the sole grader (the capability never self-certifies).
        #
        # Guarded two ways so dev / sensorless / CI without weights stay byte-identical:
        #   1. torch/transformers must be importable (ImportError -> register nothing);
        #   2. the agent must actually have a CAMERA to detect on (a perception with
        #      ``get_color_frame()`` OR a base/arm with ``get_camera_frame()``) — a
        #      no-agent probe or a sensorless agent registers nothing. The detector
        #      needs ONLY a camera, not an arm: a go2+arm grasps with it, a g1 (camera,
        #      NO arm) localizes with it. So the gate is a CAMERA-presence CAPABILITY
        #      check (R4), world-agnostic — NOT an arm/embodiment special-case (rule 7).
        #      The model itself is NOT loaded here — the DetectorCapability /
        #      GroundingDinoDetector loads lazily on first detect.
        if agent is None or not _agent_has_camera(agent):
            return None
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            logger.info(
                "[ROBOT-WORLD] torch/transformers absent — detector capability "
                "not registered (path stays detector-free)"
            )
            return None
        from vector_os_nano.perception.detector_capability import DetectorCapability

        # Bind the agent's live RGB source so a PRODUCER-routed ``detect`` sub-goal
        # can perceive on the capability-dispatch path. The kernel builds a
        # SkillContext for capability invoke (it has no camera frame), so without
        # this the routed detector returns "no RGB frame". World-side wiring only —
        # the kernel/cognitive seam is untouched.
        #
        # COLD-TURN rebind (R37 Task B): register_capabilities runs at engine.bind,
        # which can precede the NL sim-start that actually boots the arm+camera — so
        # ``agent._perception`` may be None RIGHT NOW. Bind the AGENT (not the
        # current snapshot) so the capability pulls the LIVE ``_perception`` lazily at
        # invoke time; a cold product turn (sim started this same turn) then perceives
        # with no pre-boot. We still pass the snapshot perception when present (it
        # wins in _live_perception, keeping the warm path identical).
        perception = getattr(agent, "_perception", None)
        registry.register(DetectorCapability(perception=perception, agent=agent))
        logger.info(
            "[ROBOT-WORLD] registered 'detect' capability (grounding-dino), "
            "perception=%s (agent-bound for cold-turn rebind)",
            type(perception).__name__ if perception else None,
        )
        return None
