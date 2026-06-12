# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Robot world shim.

Phase A keeps the robot path working exactly as before: this shim is the
registration *surface* (persona + "use kernel defaults" for vocab), while the
robot verify namespace and skill tools continue to be built by the engine/CLI
as today. It deliberately imports nothing robot-specific at module load.
"""

from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.prompt import ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS


class RobotWorld:
    """Adapter for the robot domain (Go2 / SO-101 / Piper)."""

    name = "robot"

    def is_robot(self) -> bool:
        return True

    def evidence_exempt_strategies(self) -> "frozenset[str]":
        # TRANSITIONAL (invariant II, design review #4): the real-hardware
        # world's async/locomotion skills carry no symbolic post-condition
        # yet, so they stay exempt — manipulation (pick/place) must always
        # carry real predicates. Invariant III shrinks this set as skills
        # gain real verify hints; the set is BOUNDED and reviewed, never
        # derived from runtime state.
        # navigate/look/describe_scene graduated in invariant III (they
        # declare real verify templates now); motion skills remain until the
        # batch-2 moved/duration result contract gives them real predicates.
        return frozenset({
            "explore", "patrol", "walk", "turn", "stand", "sit", "stop",
            "where_am_i",
        })

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
            )
            ns.update({
                "detect_objects": make_detect_objects(agent, ()),
                "describe_scene": make_describe_scene(agent, ()),
                "holding_object": make_holding_object(agent),
                "arm_at_home": make_arm_at_home(agent),
                "placed_count": make_placed_count(agent),
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

        return ns

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        # No-op in C.1 — the robot path keeps its skill/primitive routing,
        # byte-identical. C.3 registers detectors / planners / VLA policies here.
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
