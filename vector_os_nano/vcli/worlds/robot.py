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


# R5 CORRECTION (moat integrity): there is deliberately NO camera-no-arm
# ``detect_objects`` verify oracle here. R4 added ``_make_perceived_detections``,
# which built ``detect_objects()`` over ``agent._last_detection`` — the LEARNED
# detector's OWN stashed boxes (the MEANS' output). Binding that into the verify
# namespace made ``len(detect_objects()) > 0`` a *truth-bearing* oracle over the
# means' own product, so the R1 evidence gate graded the g1 detect step GROUNDED:
# the detector certifying ITSELF (a TAUTOLOGY — the actor's output verifying the
# actor). That was a FALSE GREEN (red-team, D61). A detector running + localizing
# is READ-ONLY perception: there is no actor-caused world change to ground, and
# the verify must never read the means' own output as if it were an independent
# oracle (rule 5: the sandbox only gets stricter; rule 4 #moat: GT anchors stay
# invisible to the means). With NO ``detect_objects`` in the camera-no-arm
# namespace, the native detect tool's ``verify(len(detect_objects()) > 0)`` finds
# no oracle name in the live verify-namespace, so ``classify_verify_expr`` returns
# RAN (no oracle call) — fail-closed to the honest D50 grade. The detector still
# REGISTERS + ROUTES + localizes (the genuine cross-EMBODIMENT x cross-MODEL
# achievement, kept); only the dishonest GROUNDED is removed.
#
# A legitimate FUTURE GROUNDED for detection (possible R6, NOT done) would require
# a GT-backed spatial match — the detector's box back-projects to where the SIM GT
# says the object is — i.e. an oracle INDEPENDENT of the means, not a self-read.


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
        # Ground the verifier on the SIM deterministic oracle(s). The engine
        # builds its dev/robot bindings + empty perception stubs first
        # (engine._build_verifier_namespace), then merges THIS on top — so when a
        # sim embodiment is present these predicates REPLACE the stubs (e.g.
        # detect_objects->[], describe_scene->"") with real ground-truth lookups,
        # and the planner's verify allowlist (derived from this namespace) gains
        # them. Single-sourced from the kernel-side oracles (ADR-008 C1 / kernel
        # rule 3); lazily imported here so the module load stays robot-free.
        #
        # Compose whatever embodiment(s) the agent actually has, so an arm-only,
        # a go2-base-only, OR a go2+arm agent each get a GROUNDED verify namespace
        # (world-agnostic — no embodiment is special-cased). Real hardware / no
        # sim oracle contributes nothing, leaving the engine namespace byte-
        # identical. object_names=() => all scene objects (the plain robot world
        # has no Scenario to declare a known-set).
        ns: dict[str, Any] = {}

        arm = getattr(agent, "_arm", None)
        if arm is not None and hasattr(arm, "get_object_positions"):
            from vector_os_nano.vcli.worlds.arm_sim_oracle import (
                make_arm_at_home,
                make_describe_scene,
                make_detect_objects,
                make_holding_object,
                make_placed_count,
                make_released,
            )
            ns.update({
                "detect_objects": make_detect_objects(agent, ()),
                "describe_scene": make_describe_scene(agent, ()),
                "holding_object": make_holding_object(agent),
                "arm_at_home": make_arm_at_home(agent),
                "placed_count": make_placed_count(agent),
                "released": make_released(agent),
            })

        base = getattr(agent, "_base", None)
        if (
            base is not None
            and hasattr(base, "get_position")
            and hasattr(base, "get_heading")
        ):
            # Ground the base predicates the go2 vocab verifies against. The plain
            # robot world has no Scenario, so ``visited`` (which needs a named-room
            # set) is left to the playground; ``at_position`` / ``facing`` need
            # only the live base and replace the engine stubs.
            from vector_os_nano.vcli.worlds.go2_sim_oracle import (
                make_at_position,
                make_facing,
            )
            ns.update({
                "at_position": make_at_position(agent),
                "facing": make_facing(agent),
            })

        # R7: the HONEST GROUNDED for a camera-only embodiment (g1: head camera, NO
        # arm). R5 (D61) correctly REFUSED to bind a camera-no-arm ``detect_objects``
        # oracle, because the only anchor THEN available was the detector's OWN output
        # (a self-read → the R4 false green). R7 supplies what was missing: INDEPENDENT
        # SIM GROUND TRUTH (a MuJoCo SEGMENTATION render — the renderer's own per-pixel
        # geom-id image, never passed to the detector) and a SPATIAL-MATCH oracle,
        # ``detection_matches_gt``, that returns True IFF the detector's box CENTER
        # lands within tol px of the segmentation centroid of the matching-colour geoms.
        # The TRUTH is the render (the detector cannot author it); the detection is the
        # CLAIM being judged. A wrong box (wall / wrong object) or the object OUT OF VIEW
        # (no matching-colour geom in the segmentation) → no match → False → RAN. This
        # is the GT-backed match the R5 note (above) anticipated as the legitimate
        # future GROUNDED — NOT a ``len(detect_objects()) > 0`` self-read. (R7 used the
        # segmentation render rather than a hand-rolled world→pixel projection because
        # the freejoint pickables are occluded in g1's view and the head-camera
        # pose-vs-frame did not reconcile under a pinhole convention — segmentation is
        # strictly more reliable + more independent; see the oracle module docstring.)
        #
        # Bound ONLY for a camera-bearing, ARM-LESS sim base exposing the live MuJoCo
        # model/data (the g1 shape). The go2+arm path already has its GT ``detect_objects``
        # (above) and grasp weld-causation, so it does not need this; gating on
        # arm-absence keeps the two paths from double-binding a detect oracle. The frozen
        # ``vcli/cognitive/`` spine is UNTOUCHED — the model reaches GROUNDED via the
        # existing state-oracle-vs-constant rule (``detection_matches_gt('red') == True``).
        if (
            arm is None
            and base is not None
            and _agent_has_camera(agent)
            and getattr(base, "_model", None) is not None
            and getattr(base, "_data", None) is not None
        ):
            from vector_os_nano.vcli.worlds.g1_perception_oracle import (
                make_detection_matches_gt,
            )
            ns["detection_matches_gt"] = make_detection_matches_gt(agent)

        return ns

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
