# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pick up an object with the Piper arm using a top-down grasp.

Phase C of the v2.1 manipulation stack. DEMO-QUALITY — this skill assumes
a LOT; see docs/pick_top_down_spec.md for the full assumption list. Summary:

* Gripper always approaches straight down (world -Z). No angled approach.
* Object world pose is known a priori from ``context.world_model`` (populated
  from MJCF body names prefixed with ``pickable_``). No perception.
* Dog stands still during grasp; every arm segment is re-solved against the
  live base pose right before it runs.
* No collision checking against dog body, furniture, or other objects.
* Single-object only, no clutter reasoning.
* The arm must be an ``ArmProtocol`` implementation that exposes an
  ``ik_top_down(xyz)`` method (currently just ``MuJoCoPiper``).

Flow:
    1. Resolve target object (by id, label, or explicit xyz)
    2. IK top-down for pre-grasp (PRE_GRASP_HEIGHT above object)
    3. IK top-down for grasp (object z + GRASP_Z_ABOVE)
    4. Open gripper
    5. Move to the approach point short of the pre-grasp (toward the base), then
       slide in level to the pre-grasp (no base: straight to the pre-grasp)
    6. Re-solve the grasp IK, descend (DESCENT_DURATION)
    7. Close gripper (settles; welds only an object both fingers grip)
    8. Lift back to the pre-grasp point (LIFT_DURATION)
    9. success iff gripper.is_holding(); a miss is a failure (grasp_missed)

No ROS2 imports. No perception imports.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

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

# With a base, the open jaws first go to a point this far (m) short of the
# pre-grasp, on the line from the base, then slide in level: the long swing from
# stow happens clear of the object, which the fingers would otherwise sweep aside
# (measured: a direct swing pushed the bottle 2 cm and left a finger on its rim).
_DEFAULT_APPROACH_BACK: float = 0.10

# Motion durations (seconds). Piper PD gains are moderate (kp=80 on the big
# joints, kp=10 on wrist) so large motions need multi-second interpolation.
_PREGRASP_DURATION: float = 4.0
_SLIDE_DURATION: float = 1.5
_DESCENT_DURATION: float = 2.0
_LIFT_DURATION: float = 2.0


# ---------------------------------------------------------------------------
# Chinese color-keyword normaliser (used by T7 inside _resolve_target)
# ---------------------------------------------------------------------------

_CN_COLOR_MAP: dict[str, str] = {
    "红": "red",   "红色": "red",
    "绿": "green", "绿色": "green",
    "蓝": "blue",  "蓝色": "blue",
    "黄": "yellow","黄色": "yellow",
    "白": "white", "白色": "white",
    "黑": "black", "黑色": "black",
}

# Keys sorted longest-first so "红色" matches before "红".
_CN_COLOR_KEYS: list[str] = sorted(_CN_COLOR_MAP, key=len, reverse=True)


def _normalise_color_keyword(label: str) -> str | None:
    """Replace Chinese color tokens in *label* with English equivalents.

    Rules:
    - Longest-match first ("红色" before "红") to avoid partial replacements.
    - Replaces ALL occurrences.
    - Returns the modified string, or ``None`` if no Chinese color token was found.

    Examples::

        _normalise_color_keyword("抓前面绿色瓶子")  # "抓前面green瓶子"
        _normalise_color_keyword("bottle")          # None
        _normalise_color_keyword("紫色")            # None  (not in map)
    """
    modified = label
    matched = False
    for cn in _CN_COLOR_KEYS:
        if cn in modified:
            modified = modified.replace(cn, _CN_COLOR_MAP[cn])
            matched = True
    return modified if matched else None


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
        "grasp_missed",
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

        # Resolve target. The explicit target_xyz path (the honest perception path —
        # PerceptionGraspSkill passes the depth-perceived grasp point) needs NO world
        # model; only the object_id/label lookups do. So a None world_model is fatal
        # ONLY when there is no target_xyz to fall back on (refactor direction: the
        # SceneGraph, not a GT world model, is the object source — D89).
        target = self._resolve_target(params, wm)

        # v2.3 hot-fix: perception-driven auto-detect retry on world_model miss.
        # Symmetric with MobilePickSkill — VGG routes "抓 X" to whichever skill
        # it thinks fits, and both must self-heal via the same auto-detect path.
        # Only meaningful when a world model exists to populate/consult.
        if target is None and wm is not None:
            from vector_os_nano.skills.utils import run_autodetect_retry
            if run_autodetect_retry(params, context, log_tag="PICK-TD") > 0:
                target = self._resolve_target(params, wm)

        if target is None:
            if wm is None:
                return SkillResult(
                    success=False,
                    error_message=(
                        "No target_xyz supplied and no world_model available — "
                        "cannot locate the object to pick."
                    ),
                    result_data={"diagnosis": "no_world_model"},
                )
            query = params.get("object_label") or params.get("object_id") or ""
            known_labels = [
                o.label for o in wm.get_objects()
                if o.object_id.startswith("pickable_")
            ]
            return SkillResult(
                success=False,
                error_message=(
                    f"Cannot locate target object (query={query!r}). "
                    f"Known pickable objects in world model: {known_labels}. "
                    f"Retry with one of these labels (e.g. object_label=\"{known_labels[0] if known_labels else '...'}\")."
                ),
                result_data={
                    "diagnosis": "object_not_found",
                    "query": query,
                    "known_objects": known_labels,
                },
            )
        obj_id, obj_xyz = target
        logger.info("[PICK-TD] target %s at world xyz=%s", obj_id, obj_xyz)

        pre_h = float(cfg.get("pre_grasp_height", _DEFAULT_PRE_GRASP_HEIGHT))
        grasp_dz = float(cfg.get("grasp_z_above", _DEFAULT_GRASP_Z_ABOVE))

        pre_grasp = (obj_xyz[0], obj_xyz[1], obj_xyz[2] + pre_h)
        grasp = (obj_xyz[0], obj_xyz[1], obj_xyz[2] + grasp_dz)
        logger.info("[PICK-TD] pre_grasp=%s  grasp=%s", pre_grasp, grasp)

        # IK — both poses up front (never start a motion that cannot finish), the
        # grasp seeded from the pre-grasp for minimal joint change. Every segment is
        # re-solved against the LIVE base right before it runs: the dog sways under
        # the moving arm, and a stale solution lands centimetres off.
        q_pre = arm.ik_top_down(pre_grasp)
        logger.info("[PICK-TD] pre-grasp IK %s", "OK" if q_pre is not None else "FAIL")
        if q_pre is None:
            return _ik_fail("pre_grasp", pre_grasp)
        q_grasp = arm.ik_top_down(grasp, current_joints=q_pre)
        if q_grasp is None:
            return _ik_fail("grasp", grasp)

        # --- Execute motion sequence ----------------------------------
        logger.info("[PICK-TD] opening gripper")
        if not gripper.open():
            return _move_fail("open")

        back = _approach_point(context.base, pre_grasp,
                               float(cfg.get("approach_back", _DEFAULT_APPROACH_BACK)))
        q_back = arm.ik_top_down(back, current_joints=q_pre) if back else None
        if q_back is not None:
            logger.info("[PICK-TD] approach %s (%.1fs), slide in", back, _PREGRASP_DURATION)
            if not arm.move_joints(q_back, duration=_PREGRASP_DURATION):
                return _move_fail("approach")
            q_pre = arm.ik_top_down(pre_grasp, current_joints=arm.get_joint_positions()) or q_pre
            if not arm.move_joints(q_pre, duration=_SLIDE_DURATION):
                return _move_fail("pre_grasp")
        else:
            logger.info("[PICK-TD] pre-grasp move (%.1fs)", _PREGRASP_DURATION)
            if not arm.move_joints(q_pre, duration=_PREGRASP_DURATION):
                return _move_fail("pre_grasp")

        q_grasp = arm.ik_top_down(grasp, current_joints=arm.get_joint_positions())
        if q_grasp is None:
            return _ik_fail("grasp", grasp)
        logger.info("[PICK-TD] descent (%.1fs)", _DESCENT_DURATION)
        if not arm.move_joints(q_grasp, duration=_DESCENT_DURATION):
            return _move_fail("descent")

        # close() settles the jaws and welds only an object both fingers grip.
        logger.info("[PICK-TD] closing gripper at grasp=%s obj_xyz=%s", grasp, obj_xyz)
        gripper.close()

        logger.info("[PICK-TD] lift (%.1fs)", _LIFT_DURATION)
        cur = arm.get_joint_positions()
        arm.move_joints(arm.ik_top_down(pre_grasp, current_joints=cur) or q_pre,
                        duration=_LIFT_DURATION)
        # IMPORTANT: we do NOT return to the URDF-zero "home" pose after a
        # successful grasp. Moving from the top-down grasp pose to zero-joints
        # rotates the wrist 90° and the object tips out of the jaws (Piper
        # has no force feedback, only position control). Holding at the
        # pre-grasp pose keeps the gripper top-down, so the object stays
        # gripped until a follow-up place/drop skill.

        grasped = bool(gripper.is_holding())
        result_data = {
            "diagnosis": "ok" if grasped else "grasp_missed",
            "object_id": obj_id,
            "grasp_world": list(obj_xyz),
            "grasped": grasped,
        }
        if not grasped:
            return SkillResult(
                success=False,
                error_message=f"Grasp missed: the gripper closed on nothing at {list(grasp)}",
                result_data=result_data,
            )
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
        # 1. Explicit xyz overrides everything (useful for testing / debug).
        # Invalid shape / NaN / inf returns None, letting step 2+ try other
        # resolution paths. Caller sees `object_not_found` in the failure mode.
        if "target_xyz" in params:
            try:
                xyz = tuple(float(v) for v in params["target_xyz"])  # type: ignore[assignment]
            except (TypeError, ValueError):
                xyz = ()
            if len(xyz) == 3 and all(math.isfinite(v) for v in xyz):
                return (params.get("object_id") or "xyz_target", xyz)  # type: ignore[return-value]

        # Steps 2+ look the object up in the world model. With no world model
        # (the honest SceneGraph-driven path, where target_xyz is supplied), there
        # is nothing more to resolve — return None rather than dereferencing None.
        if wm is None:
            return None

        # 2. Exact object_id
        obj_id = params.get("object_id")
        if obj_id:
            obj = wm.get_object(obj_id)
            if obj is not None:
                return (obj.object_id, _xyz_of(obj))

        # 3. Exact label match
        label = params.get("object_label")
        if label:
            matches = wm.get_objects_by_label(label)
            if matches:
                obj = matches[0]
                return (obj.object_id, _xyz_of(obj))

        # 4. Chinese color normaliser pass — extract English color tokens and
        # retry lookup against each English color keyword found in the label.
        if label:
            normalised = _normalise_color_keyword(label)
            if normalised is not None:
                # Try each unique English color value that appears in the normalised string.
                for en_color in dict.fromkeys(_CN_COLOR_MAP.values()):
                    if en_color in normalised:
                        norm_matches = wm.get_objects_by_label(en_color)
                        if norm_matches:
                            obj = norm_matches[0]
                            logger.info(
                                "[PICK-TD] resolved %r via colour-normalisation → %r (%s)",
                                label, en_color, obj.object_id,
                            )
                            return (obj.object_id, _xyz_of(obj))

        # No match. Caller sees object_not_found with the known-objects list;
        # the decomposer is expected to retry with a detect/look step to
        # populate the world model via perception (SO-101 pattern), not to
        # silently substitute a nearby pickable.
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approach_point(base, pre_grasp, back: float) -> tuple[float, float, float] | None:
    """*back* metres from *pre_grasp* toward the base, same height; None without a base."""
    try:
        bx, by = (float(v) for v in base.get_position()[:2])
    except Exception:  # noqa: BLE001 — no base (tabletop arm): approach directly
        return None
    dx, dy = pre_grasp[0] - bx, pre_grasp[1] - by
    d = math.hypot(dx, dy)
    if back <= 0.0 or d <= back:
        return None
    return (pre_grasp[0] - back * dx / d, pre_grasp[1] - back * dy / d, pre_grasp[2])


def _ik_fail(phase: str, target: tuple[float, float, float]) -> SkillResult:
    return SkillResult(
        success=False,
        error_message=f"IK unreachable for {phase} {target}",
        result_data={"diagnosis": "ik_unreachable", "phase": phase, "target": list(target)},
    )


def _move_fail(phase: str) -> SkillResult:
    return SkillResult(
        success=False,
        error_message=f"{phase} move did not reach its target (blocked or stopped)",
        result_data={"diagnosis": "move_failed", "phase": phase},
    )


def _xyz_of(obj: ObjectState) -> tuple[float, float, float]:
    return (float(obj.x), float(obj.y), float(obj.z))
