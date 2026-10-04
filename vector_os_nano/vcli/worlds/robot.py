# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Robot world shim.

Phase A keeps the robot path working exactly as before: this shim is the
registration *surface* (persona + "use kernel defaults" for vocab), while the
robot verify namespace and skill tools continue to be built by the engine/CLI
as today. It deliberately imports nothing robot-specific at module load.
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


class RobotWorld:
    """Adapter for the robot domain (Go2 / SO-101 / Piper)."""

    name = "robot"

    def is_robot(self) -> bool:
        return True

    def persona_blocks(self) -> tuple[str, str]:
        return ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS

    def register_tools(self, registry: Any, agent: Any) -> None:
        # Robot/diag/sim tools are registered by the CLI from discover_*; skill
        # wrappers are added via wrap_skills(agent) at startup. No-op here in
        # Phase A (the full robot-world migration is Phase C).
        return None

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        """The GT oracles of the live sim session that owns *agent*.

        Every predicate reads the session's ground truth (``vcli/worlds/sim_oracles``),
        never an agent or device attribute. Real hardware / no session contributes
        nothing, so every verify there grades RAN. The engine merges this over its
        own bindings, so the GT ``get_position`` / ``get_heading`` replace its
        agent-read ones in a sim.
        """
        from vector_os_nano.sim.session import session_for

        session = session_for(agent)
        if session is None or session.gt is None:
            return {}
        return session.oracles()

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
        # COLD-TURN rebind (R37 Task B): register_capabilities runs at init_vgg,
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

    def decompose_vocab(self) -> None:
        # None => the engine derives the vocab from the live skill registry
        # (see derive_vocab_from_registry); falls back to GoalDecomposer class
        # defaults if no registry/namespace is available.
        return None

    def derive_vocab_from_registry(self) -> bool:
        # Single-source the decompose vocabulary from the skill registry so the
        # prompt, the validator allowlist, and the params-help can never drift.
        # Serves both go2 (has_base=True) and the arm (has_base=False); the
        # engine inspects the agent to decide has_base.
        return True
