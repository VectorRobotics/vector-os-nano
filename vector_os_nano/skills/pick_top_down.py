"""Pick up an object with the Piper arm using a top-down grasp.

Phase C of the v2.1 manipulation stack. DEMO-QUALITY — this skill assumes
a LOT; see docs/pick_top_down_spec.md for the full assumption list. Summary:

* Gripper always approaches straight down (world -Z). No angled approach.
* Object world pose is known a priori from ``context.world_model`` (populated
  from MJCF body names prefixed with ``pickable_``). No perception.
* Dog stands still during grasp. No base-arm coordination.
* No collision checking against dog body, furniture, or other objects.
* Single-object only, no clutter reasoning.
* The arm must be an ``ArmProtocol`` implementation that exposes an
  ``ik_top_down(xyz)`` method (currently just ``MuJoCoPiper``).

Flow:
    1. Resolve target object (by id, label, or explicit xyz)
    2. IK top-down for pre-grasp (PRE_GRASP_HEIGHT above object)
    3. IK top-down for grasp (object z + GRASP_Z_ABOVE)
    4. Open gripper
    5. Move to pre-grasp pose (PREGRASP_DURATION)
    6. Descend to grasp pose (DESCENT_DURATION)
    7. Close gripper; wait SETTLE_AFTER_CLOSE
    8. Lift back to pre-grasp pose (LIFT_DURATION)
    9. Return home (HOME_DURATION)
    10. Report success + grasp heuristic (gripper.is_holding())

No ROS2 imports. No perception imports.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.world_model import ObjectState, WorldModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (exposed in config.skills.pick_top_down.* for per-deploy override)
# ---------------------------------------------------------------------------

# Z offset above object centre used for pre-grasp hover (meters, world frame).
# Piper's reach envelope is largely below its base (z ≈ 0.34 m when Go2 stands).
# At top-down orientation, target z > Piper_base − 2 cm forces the arm into
# near-fully-folded configs that IK can't solve. 5 cm keeps pre-grasp below
# the base for 8-cm-tall cylinders (center z ≈ 0.25 → pre-grasp 0.30).
# Larger than the object's half-height plus ~1 cm so the fingers clear the
# object top on approach; short objects may need a smaller override.
_DEFAULT_PRE_GRASP_HEIGHT: float = 0.05

# Extra Z offset applied to the grasp target (object z + this). Small positive
# value keeps the jaws wrapping the upper body of the object rather than
# driving into the table. Tuned empirically for cylinder objects on the
# 20 cm table.
_DEFAULT_GRASP_Z_ABOVE: float = 0.01

# Motion durations (seconds). Piper PD gains are moderate (kp=80 on the big
# joints, kp=10 on wrist) so large motions need multi-second interpolation.
_PREGRASP_DURATION: float = 4.0
_DESCENT_DURATION: float = 2.0
_CLOSE_SETTLE: float = 0.8
_LIFT_DURATION: float = 2.0
_HOME_DURATION: float = 4.0

# Home joints = Piper URDF zero configuration (matches the stow pose set by
# MuJoCoGo2 on connect/stand).
_HOME_JOINTS: list[float] = [0.0] * 6


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@skill(
    aliases=[
        "grab", "grasp", "take", "pick",
        "抓", "拿", "抓起", "抓住", "抓取", "拿起", "取",
    ],
    direct=False,
)
class PickTopDownSkill:
    """Top-down grasp with the Piper arm on Go2. See module docstring."""

    name: str = "pick_top_down"
    description: str = (
        "Pick up an object with the Piper arm using a top-down grasp. "
        "Object world pose is read from the world model by object_id or "
        "object_label. Gripper approaches straight down. Dog must be "
        "standing still; no perception, no collision check."
    )
    parameters: dict = {
        "object_id": {
            "type": "string",
            "required": False,
            "description": "ID of the object in the world model (e.g. 'pickable_bottle_blue').",
            "source": "world_model.objects.object_id",
        },
        "object_label": {
            "type": "string",
            "required": False,
            "description": "Label of the object (e.g. 'blue bottle', 'red can').",
            "source": "world_model.objects.label",
        },
    }
    preconditions: list[str] = ["gripper_empty"]
    postconditions: list[str] = []
    effects: dict = {"gripper_state": "holding"}
    failure_modes: list[str] = [
        "no_arm",
        "no_gripper",
        "no_world_model",
        "arm_unsupported",
        "object_not_found",
        "ik_unreachable",
        "move_failed",
    ]

    # ------------------------------------------------------------------

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        arm = context.arm
        gripper = context.gripper
        wm = context.world_model
        cfg = context.config.get("skills", {}).get("pick_top_down", {})

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
        if wm is None:
            return SkillResult(
                success=False,
                error_message="No world_model available; this skill requires a populated world model",
                result_data={"diagnosis": "no_world_model"},
            )
        if not hasattr(arm, "ik_top_down"):
            return SkillResult(
                success=False,
                error_message=(
                    f"Arm {getattr(arm, 'name', type(arm).__name__)!r} does not "
                    "implement ik_top_down; PickTopDownSkill requires a 6-DoF arm "
                    "with orientation-constrained IK (e.g. MuJoCoPiper)."
                ),
                result_data={"diagnosis": "arm_unsupported"},
            )

        # Resolve target
        target = self._resolve_target(params, wm)
        if target is None:
            return SkillResult(
                success=False,
                error_message="Cannot locate target object",
                result_data={
                    "diagnosis": "object_not_found",
                    "query": params.get("object_label") or params.get("object_id") or "",
                    "known_objects": [o.label for o in wm.get_objects()],
                },
            )
        obj_id, obj_xyz = target
        logger.info("[PICK-TD] target %s at world xyz=%s", obj_id, obj_xyz)

        pre_h = float(cfg.get("pre_grasp_height", _DEFAULT_PRE_GRASP_HEIGHT))
        grasp_dz = float(cfg.get("grasp_z_above", _DEFAULT_GRASP_Z_ABOVE))

        pre_grasp = (obj_xyz[0], obj_xyz[1], obj_xyz[2] + pre_h)
        grasp = (obj_xyz[0], obj_xyz[1], obj_xyz[2] + grasp_dz)
        logger.info("[PICK-TD] pre_grasp=%s  grasp=%s", pre_grasp, grasp)

        # IK — solve pre-grasp first, seed grasp from it for minimal joint change
        q_pre = arm.ik_top_down(pre_grasp)
        if q_pre is None:
            return SkillResult(
                success=False,
                error_message=f"IK unreachable for pre-grasp {pre_grasp}",
                result_data={
                    "diagnosis": "ik_unreachable",
                    "phase": "pre_grasp",
                    "target": list(pre_grasp),
                },
            )
        q_grasp = arm.ik_top_down(grasp, current_joints=q_pre)
        if q_grasp is None:
            return SkillResult(
                success=False,
                error_message=f"IK unreachable for grasp {grasp}",
                result_data={
                    "diagnosis": "ik_unreachable",
                    "phase": "grasp",
                    "target": list(grasp),
                },
            )

        # --- Execute motion sequence ----------------------------------
        logger.info("[PICK-TD] opening gripper")
        gripper.open()
        time.sleep(0.3)

        logger.info("[PICK-TD] pre-grasp move (%.1fs)", _PREGRASP_DURATION)
        if not arm.move_joints(q_pre, duration=_PREGRASP_DURATION):
            return SkillResult(
                success=False,
                error_message="Pre-grasp move returned False",
                result_data={"diagnosis": "move_failed", "phase": "pre_grasp"},
            )

        logger.info("[PICK-TD] descent (%.1fs)", _DESCENT_DURATION)
        if not arm.move_joints(q_grasp, duration=_DESCENT_DURATION):
            return SkillResult(
                success=False,
                error_message="Descent move returned False",
                result_data={"diagnosis": "move_failed", "phase": "descent"},
            )

        logger.info("[PICK-TD] closing gripper")
        gripper.close()
        time.sleep(_CLOSE_SETTLE)

        logger.info("[PICK-TD] lift (%.1fs)", _LIFT_DURATION)
        arm.move_joints(q_pre, duration=_LIFT_DURATION)
        # IMPORTANT: we do NOT return to the URDF-zero "home" pose after a
        # successful grasp. Moving from the top-down grasp pose to zero-joints
        # rotates the wrist 90° and the object tips out of the jaws (Piper
        # has no force feedback, only position control). Holding at the
        # pre-grasp pose keeps the gripper top-down, so the object stays
        # gripped until a follow-up place/drop skill.

        # Grasp heuristic: if gripper reports cmd=closed but jaws held open
        # by an object, we consider it a grasp. No force sensor in sim, so
        # this is position-only.
        grasped = False
        try:
            grasped = bool(gripper.is_holding())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PICK-TD] is_holding() raised: %s", exc)

        result_data = {
            "diagnosis": "ok" if grasped else "possibly_missed",
            "object_id": obj_id,
            "grasp_world": list(obj_xyz),
            "grasped_heuristic": grasped,
        }
        return SkillResult(success=True, result_data=result_data)

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _resolve_target(
        self,
        params: dict,
        wm: WorldModel,
    ) -> Optional[tuple[str, tuple[float, float, float]]]:
        """Return (object_id, world_xyz) or None if not resolvable."""
        # Explicit xyz overrides everything (useful for testing / debug)
        if "target_xyz" in params:
            xyz = tuple(float(v) for v in params["target_xyz"])  # type: ignore[assignment]
            if len(xyz) == 3:
                return (params.get("object_id") or "xyz_target", xyz)  # type: ignore[return-value]

        obj_id = params.get("object_id")
        if obj_id:
            obj = wm.get_object(obj_id)
            if obj is not None:
                return (obj.object_id, _xyz_of(obj))

        label = params.get("object_label")
        if label:
            matches = wm.get_objects_by_label(label)
            if matches:
                obj = matches[0]
                return (obj.object_id, _xyz_of(obj))

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xyz_of(obj: ObjectState) -> tuple[float, float, float]:
    return (float(obj.x), float(obj.y), float(obj.z))
