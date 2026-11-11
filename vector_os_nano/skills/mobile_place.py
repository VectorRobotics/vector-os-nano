# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Walk the dog to an approach pose, then delegate to PlaceTopDownSkill.

Phase C mobile variant of the manipulation stack (v2.2 Wave 3, Task T9).
Mirrors MobilePickSkill composition pattern:

  1. Hardware guards (base / arm / gripper).
  2. Resolve target XYZ (explicit or receptacle_id from world_model).
  3. Compute approach pose via compute_approach_pose.
  4. Check already_reachable / skip_navigate flag.
  5. navigate_to approach pose (if needed).
  6. wait_stable so arm IK is computed from a stationary base.
  7. Delegate to PlaceTopDownSkill (force-passing resolved target_xyz).
  8. Enrich result_data with mobile_place metadata.

No ROS2 imports. No perception imports.
"""
from __future__ import annotations

import logging
import math
import time

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.utils.approach_pose import compute_approach_pose

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_DEFAULT_CLEARANCE: float = 0.55  # metres from object to approach position
_APPROACH_XY_TOL: float = 0.10   # metres — within this → already_reachable
_APPROACH_YAW_DEG: float = 20.0  # degrees — yaw tolerance for already_reachable
_NAV_TIMEOUT: float = 20.0       # seconds for navigate_to
_STABLE_MAX_SPEED: float = 0.05  # m/s — dog counts as stable below this
_STABLE_SETTLE: float = 1.0      # seconds to remain stable
_STABLE_TIMEOUT: float = 5.0     # maximum seconds to wait for stability


# ---------------------------------------------------------------------------
# Inline helpers (mirrored from MobilePickSkill to avoid cross-import)
# ---------------------------------------------------------------------------


def _dist_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in XY plane."""
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def _ang_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _wait_stable(
    base: object,
    max_speed: float,
    settle_duration: float,
    timeout: float,
) -> bool:
    """Block until the base reports low speed for settle_duration seconds.

    Returns True if stable within timeout, False on timeout.
    Polls at 10 Hz. Uses time.sleep for test patching compatibility.
    """
    deadline = time.monotonic() + timeout
    stable_since: float | None = None

    while time.monotonic() < deadline:
        pos_a = base.get_position()
        time.sleep(0.1)
        pos_b = base.get_position()
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        speed = math.sqrt(dx * dx + dy * dy) / 0.1
        if speed < max_speed:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= settle_duration:
                return True
        else:
            stable_since = None

    return False


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


@skill(
    aliases=[
        "去放", "送到", "搬到", "拿去放",
        "deliver", "put at", "carry to",
    ],
    direct=False,
)
class MobilePlaceSkill:
    """Walk the dog to a reachable pose near a target location / receptacle,
    then release the held object with a top-down drop. (source: world_model)
    """

    name: str = "mobile_place"
    verify_hint: str = "released()"
    description: str = (
        "Walk the dog to a reachable pose near a target location / receptacle, "
        "then release the held object with a top-down drop. (source: world_model)"
    )
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
            "description": "Receptacle object id in world model.",
            "source": "world_model.objects.object_id",
        },
        "drop_height": {
            "type": "number",
            "required": False,
            "default": 0.05,
            "description": "Z offset above target to release (m).",
            "source": "static",
        },
        "skip_navigate": {
            "type": "boolean",
            "required": False,
            "default": False,
            "description": "Skip the navigation step (debug use).",
            "source": "static",
        },
    }
    preconditions: list[str] = ["gripper_holding_any"]
    postconditions: list[str] = []
    effects: dict = {"gripper_state": "open", "held_object": None}
    failure_modes: list[str] = [
        "no_base", "no_arm", "no_gripper",
        "receptacle_not_found", "invalid_target_xyz", "missing_target",
        "nav_failed", "wait_stable_timeout",
        "ik_unreachable", "move_failed",
    ]

    def __init__(self) -> None:
        from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
        self._place = PlaceTopDownSkill()

    # ------------------------------------------------------------------

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        arm = context.arm
        gripper = context.gripper
        wm = context.world_model
        cfg = context.config.get("skills", {}).get("mobile_place", {})

        # Step 1 — Hardware guards
        if base is None:
            return SkillResult(
                success=False,
                error_message="No base connected",
                result_data={"diagnosis": "no_base"},
            )
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

        # Step 2 — Resolve target XYZ
        resolved = self._resolve_target(params, wm)
        if isinstance(resolved, SkillResult):
            return resolved
        tx, ty, tz = resolved

        # Step 3 — Read dog pose, compute approach pose
        dog_pos = base.get_position()
        dog_heading = base.get_heading()
        dog_pose = (dog_pos[0], dog_pos[1], dog_heading)

        clearance = float(cfg.get("clearance", _DEFAULT_CLEARANCE))
        approach_x, approach_y, approach_yaw = compute_approach_pose(
            (tx, ty, tz), dog_pose, clearance=clearance
        )
        nav_distance = _dist_xy(dog_pos[0], dog_pos[1], approach_x, approach_y)

        # Step 4 — Check already_reachable or skip_navigate
        xy_close = nav_distance < _APPROACH_XY_TOL
        yaw_close = abs(_ang_diff(dog_heading, approach_yaw)) < math.radians(
            _APPROACH_YAW_DEG
        )
        already_reachable = xy_close and yaw_close
        skip_navigate = bool(params.get("skip_navigate", False))

        logger.info(
            "[MOBILE-PLACE] target=(%.3f, %.3f, %.3f) approach=(%.3f, %.3f) "
            "dist=%.3f already_reachable=%s skip=%s",
            tx, ty, tz, approach_x, approach_y,
            nav_distance, already_reachable, skip_navigate,
        )

        # Step 5 — Navigate (if needed)
        if not already_reachable and not skip_navigate:
            logger.info(
                "[MOBILE-PLACE] navigating to approach (%.3f, %.3f) timeout=%.0fs",
                approach_x, approach_y, _NAV_TIMEOUT,
            )
            nav_ok = base.navigate_to(approach_x, approach_y, timeout=_NAV_TIMEOUT)
            if not nav_ok:
                return SkillResult(
                    success=False,
                    error_message="Navigation to approach pose failed",
                    result_data={"diagnosis": "nav_failed"},
                )

        # Step 6 — Wait for stable
        if not _wait_stable(base, _STABLE_MAX_SPEED, _STABLE_SETTLE, _STABLE_TIMEOUT):
            return SkillResult(
                success=False,
                error_message="Base did not stabilise after navigation",
                result_data={"diagnosis": "wait_stable_timeout"},
            )

        # Step 7 — Delegate to PlaceTopDownSkill with resolved target_xyz
        place_params = {**params, "target_xyz": [tx, ty, tz]}
        logger.info("[MOBILE-PLACE] delegating to PlaceTopDownSkill target=%s", [tx, ty, tz])
        place_result = self._place.execute(place_params, context)

        # Step 8 — Return (propagate place failure or enrich success)
        mobile_meta = {
            "approach": [approach_x, approach_y, approach_yaw],
            "nav_distance": nav_distance,
            "skipped_navigate": already_reachable or skip_navigate,
        }
        if not place_result.success:
            # Propagate diagnosis from place; add mobile_place meta
            merged = {**place_result.result_data, "mobile_place": mobile_meta}
            return SkillResult(
                success=False,
                error_message=place_result.error_message,
                result_data=merged,
            )

        merged = {**place_result.result_data, "mobile_place": mobile_meta}
        return SkillResult(success=True, result_data=merged)

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _resolve_target(
        self,
        params: dict,
        wm: object,
    ) -> tuple[float, float, float] | SkillResult:
        """Return (tx, ty, tz) or a failure SkillResult.

        Priority:
        1. target_xyz — explicit 3-float list.
        2. receptacle_id — ID in world model.
        3. Neither → missing_target.
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
            obj = wm.get_object(rid) if wm is not None else None
            if obj is None:
                return SkillResult(
                    success=False,
                    error_message=f"Receptacle {rid!r} not found in world model",
                    result_data={"diagnosis": "receptacle_not_found", "receptacle_id": rid},
                )
            return (float(obj.x), float(obj.y), float(obj.z))

        return SkillResult(
            success=False,
            error_message=(
                "Neither target_xyz nor receptacle_id provided; "
                "cannot determine place location"
            ),
            result_data={"diagnosis": "missing_target"},
        )
