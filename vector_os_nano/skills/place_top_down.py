# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Release a held object at a target location using a top-down drop.

Phase C (place counterpart) of the v2.1 manipulation stack.
DEMO-QUALITY — mirrors pick_top_down.py assumptions. Summary:

* Gripper always approaches straight down (world -Z). No angled approach.
* Target location is either explicit world XYZ or resolved from a receptacle's
  perceived coordinates. No perception is involved.
* Dog stands still during the drop. No base-arm coordination.
* No collision checking against dog body, furniture, or other objects.
* The arm must be an ``ArmProtocol`` implementation that exposes an
  ``ik_top_down(xyz)`` method (the generic ``sim/drivers`` Arm).
* After release, the arm lifts back to the pre-place hover pose only.
  It does NOT return to the URDF-zero "home". Moving to home after a
  top-down grasp rotates the wrist 90° and any residual object (e.g.
  liquid in a cup) spills. Callers that want home must invoke a separate
  home skill.
* ``drop_height`` keeps the end-effector slightly above the surface so
  the gripper fingers clear the receptacle rim on approach.
* Pre-place hover height (``_DEFAULT_PRE_PLACE_HEIGHT``) is additive with
  the target z, not the drop_height — the arm hovers above the final drop
  point to allow safe descent.
* IK is solved in two phases: pre-place (hover) first, then place (drop
  point) seeded from pre-place joints to minimise wrist rotation.
* ``gripper.open()`` fires AFTER the descent move (arm is over the target),
  then the arm lifts back. This matches real-world pick-and-place semantics.

Flow:
    1. Hardware preconditions (arm, gripper)
    2. Verify arm exposes ``ik_top_down`` (ArmProtocol extension)
    3. Resolve target XYZ (explicit perceived coordinates)
    4. Compute pre-place hover = (tx, ty, tz + PRE_PLACE_HEIGHT)
    5. Compute place pose = (tx, ty, tz + drop_height)
    6. IK for pre-place; IK for place (seeded from pre-place)
    7. Move to pre-place (approach)
    8. Descend to place pose
    9. Open gripper; sleep for settle
    10. Lift back to pre-place (NOT home)
    11. Report success with ``placed_at`` coordinates

No ROS2 imports. No perception imports.
"""
from __future__ import annotations

import logging
import math
import time

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (exposed in config.skills.place_top_down.* for per-deploy override)
# ---------------------------------------------------------------------------

# Default Z offset above the drop target that the gripper opens at (meters).
# Small positive keeps jaws above the receptacle rim; too large drops the
# object from a height that tips it over.
_DEFAULT_DROP_HEIGHT: float = 0.05

# Z offset above the drop target used for the pre-place hover (meters).
# Additive with target z. Enough clearance for the arm to descend without
# crashing into a raised receptacle edge.
_DEFAULT_PRE_PLACE_HEIGHT: float = 0.05

# Motion durations (seconds). Mirror pick_top_down values so the two skills
# feel symmetric to the operator.
_APPROACH_DURATION: float = 4.0
_DESCENT_DURATION: float = 2.0
_OPEN_SETTLE: float = 0.3
_LIFT_DURATION: float = 2.0


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@skill(requires=('arm', 'gripper'),
    aliases=["put", "drop", "放", "放下", "放到", "放置", "put down"],
)
class PlaceTopDownSkill:
    """Release a held object at a target location using a top-down drop.

    Target is either explicit world XYZ (``target_xyz``) or a receptacle
    looked up in the world model (``receptacle_id``). The arm lifts back
    to the pre-place hover pose after opening the gripper — it does NOT
    return home to avoid wrist rotation spilling any residue.

    Assumptions (demo-quality):
    * Gripper approaches straight down, no angled release.
    * Dog stands still; no base-arm coordination during drop.
    * No collision checking against body or furniture.
    * ``context.arm`` must expose ``ik_top_down`` (e.g. the Piper).
    * Pre-place hover height is _DEFAULT_PRE_PLACE_HEIGHT above target z.
    * drop_height keeps end-effector _DEFAULT_DROP_HEIGHT above surface.
    * Gripper opens after descent; arm lifts back to hover, not to home.
    * Object ownership (held_object) is not tracked here; callers that
      verify the final object pose independently.
    """

    name: str = "place_top_down"
    description: str = (
        "Release a held object at a target location using a top-down drop. "
        "Target is either explicit xyz or a receptacle in the world model. "
        "Arm lifts back to pre-place pose (does NOT return home to avoid "
        "wrist rotation spilling any residue)."
    )
    # Grounded success predicate for the PLACE step: the object rests (unwelded, no
    # robot contact) within 0.3 m of the drop XY. The producer substitutes the real
    # object and coordinates, e.g. object_at('pickable_bottle_green', 10.6, 2.7).
    # Consumed by vcli/cognitive/vocab_from_registry.py:_verify_hint via
    # schema["verify_hint"].
    verify_hint: str = "object_at('<object>', x, y)"
    parameters: dict = {
        "target_xyz": {
            "type": "list",
            "required": False,
            "description": "Explicit world XYZ (3 floats) to drop at.",
            "source": "explicit",
        },
        "receptacle_id": {
            "type": "string",
            "required": False,
            "description": (
                "ID of a receptacle object in the world model (fallback to target_xyz)."
            ),
        },
        "drop_height": {
            "type": "number",
            "required": False,
            "default": 0.05,
            "description": "Z offset above target at which the gripper releases (m).",
            "source": "static",
        },
    }
    preconditions: list[str] = ["gripper_holding_any"]
    postconditions: list[str] = []
    effects: dict = {"gripper_state": "open", "held_object": None}
    failure_modes: list[str] = [
        "no_arm",
        "no_gripper",
        "arm_unsupported",
        "ik_unreachable",
        "move_failed",
        "receptacle_not_found",
        "invalid_target_xyz",
        "missing_target",
    ]

    # ------------------------------------------------------------------

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        arm = context.arm
        gripper = context.gripper
        cfg = context.config.get("skills", {}).get("place_top_down", {})

        # Hardware preconditions
        if arm is None:
            return SkillResult(
                success=False,
                error_message="No arm connected",
                result_data={"diagnosis": "no_arm"},
            )
        if gripper is None:
            return SkillResult(
                success=False,
                error_message="No gripper connected",
                result_data={"diagnosis": "no_gripper"},
            )
        if not hasattr(arm, "ik_top_down"):
            return SkillResult(
                success=False,
                error_message=(
                    f"Arm {getattr(arm, 'name', type(arm).__name__)!r} does not "
                    "implement ik_top_down; PlaceTopDownSkill requires a 6-DoF arm "
                    "with orientation-constrained IK (e.g. the Piper)."
                ),
                result_data={"diagnosis": "arm_unsupported"},
            )

        # Resolve target XYZ
        target_xyz = self._resolve_target(params)
        if isinstance(target_xyz, SkillResult):
            # _resolve_target returns a SkillResult on failure
            return target_xyz
        tx, ty, tz = target_xyz

        # Read tunables from config (allow per-deploy override)
        drop_h = float(params.get("drop_height", cfg.get("drop_height", _DEFAULT_DROP_HEIGHT)))
        pre_h = float(cfg.get("pre_place_height", _DEFAULT_PRE_PLACE_HEIGHT))

        pre_place = (tx, ty, tz + pre_h)
        place = (tx, ty, tz + drop_h)
        logger.info("[PLACE-TD] target=(%.3f, %.3f, %.3f)  pre_place=%s  place=%s",
                    tx, ty, tz, pre_place, place)

        # IK — solve pre-place first, seed place from it for minimal joint change
        logger.info("[PLACE-TD] IK pre_place=%s", pre_place)
        q_pre = arm.ik_top_down(pre_place)
        if q_pre is None:
            return SkillResult(
                success=False,
                error_message=f"IK unreachable for pre-place {pre_place}",
                result_data={
                    "diagnosis": "ik_unreachable",
                    "phase": "pre_place",
                    "target": list(pre_place),
                },
            )

        logger.info("[PLACE-TD] IK place=%s (seeded from pre_place joints)", place)
        q_place = arm.ik_top_down(place, current_joints=q_pre)
        if q_place is None:
            return SkillResult(
                success=False,
                error_message=f"IK unreachable for place {place}",
                result_data={
                    "diagnosis": "ik_unreachable",
                    "phase": "place",
                    "target": list(place),
                },
            )

        # --- Execute motion sequence ------------------------------------
        logger.info("[PLACE-TD] approach pre_place (%.1fs)", _APPROACH_DURATION)
        if not arm.move_joints(q_pre, duration=_APPROACH_DURATION):
            return SkillResult(
                success=False,
                error_message="Approach move returned False",
                result_data={"diagnosis": "move_failed", "phase": "approach"},
            )

        logger.info("[PLACE-TD] descent to place pose (%.1fs)", _DESCENT_DURATION)
        if not arm.move_joints(q_place, duration=_DESCENT_DURATION):
            return SkillResult(
                success=False,
                error_message="Descent move returned False",
                result_data={"diagnosis": "move_failed", "phase": "descent"},
            )

        logger.info("[PLACE-TD] opening gripper")
        gripper.open()
        time.sleep(_OPEN_SETTLE)

        logger.info("[PLACE-TD] lift back to pre_place (%.1fs)", _LIFT_DURATION)
        # Do NOT return to URDF-zero home — moving from top-down to zero joints
        # rotates the wrist and spills any residue. Hold at pre_place pose.
        arm.move_joints(q_pre, duration=_LIFT_DURATION)

        placed_at = [tx, ty, tz + drop_h]
        logger.info("[PLACE-TD] placed at %s", placed_at)
        return SkillResult(
            success=True,
            result_data={
                "placed_at": placed_at,
                "diagnosis": "ok",
            },
        )

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _resolve_target(
        self,
        params: dict,
    ) -> tuple[float, float, float] | SkillResult:
        """Return (tx, ty, tz) or a failure SkillResult.

        Priority:
        1. ``target_xyz`` — explicit 3-float list in params.
        2. ``receptacle_id`` — refused until the destination is localized.
        3. Neither present → missing_target.
        """
        if "target_xyz" in params:
            raw = params["target_xyz"]
            try:
                xyz = tuple(float(v) for v in raw)
            except (TypeError, ValueError):
                xyz = ()
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                return SkillResult(
                    success=False,
                    error_message=(
                        f"target_xyz must be 3 finite floats; got {raw!r}"
                    ),
                    result_data={"diagnosis": "invalid_target_xyz", "target_xyz": list(raw) if isinstance(raw, (list, tuple)) else raw},
                )
            return xyz  # type: ignore[return-value]

        if "receptacle_id" in params:
            rid: str = params["receptacle_id"]
            return SkillResult(False, {"diagnosis": "receptacle_not_found", "receptacle_id": rid},
                               "Named receptacles need a perceived target_xyz; use look to localize the destination.")

        return SkillResult(
            success=False,
            error_message=(
                "Neither target_xyz nor receptacle_id provided; "
                "cannot determine place location"
            ),
            result_data={"diagnosis": "missing_target"},
        )
