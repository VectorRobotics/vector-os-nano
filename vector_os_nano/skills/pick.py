"""PickSkill — pick up an object from the workspace.

Full port of skill_node_v2._execute_pick() + _single_pick_attempt().
All numeric constants preserved exactly from the original source.

Algorithm:
  1. Locate target position in base frame (from world model or perception)
  2. Apply camera→base calibration transform if coming from perception
  3. Apply z_offset (gripper height) and x_offset (position-dependent)
  4. Apply gripper asymmetry Y compensation (right jaw opens, left fixed)
  5. Check workspace boundary (5–35 cm from origin)
  6. Solve IK for pre-grasp (pre_grasp_height above target)
  7. Solve IK for grasp position (warm-started from pre-grasp)
  8. Open gripper
  9. Move to pre-grasp
  10. Descend to grasp (1s, minimal drift)
  11. Close gripper 3x with 0.2s interval
  12. Lift back to pre-grasp
  13. Return home
  14. On failure: retry up to max_retries times, home between retries

No ROS2 imports.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from vector_os_nano.core.skill import Skill, SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.calibration import camera_to_base, load_calibration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (preserved exactly from skill_node_v2.py)
# ---------------------------------------------------------------------------

# Gripper height above table (z offset added to the raw calibrated position)
_DEFAULT_Z_OFFSET: float = 0.10        # 10 cm — tuned empirically for SO-101 + D405

# Pre-grasp approach height above the target grasp point
_DEFAULT_PRE_GRASP_HEIGHT: float = 0.06  # 6 cm above grasp — enough clearance for approach

# Maximum pick attempts before reporting failure
_DEFAULT_MAX_RETRIES: int = 2

# Position sampling for density-cluster estimation (from _get_target_camera_pos)
_DEFAULT_SAMPLE_COUNT: int = 20
_DEFAULT_SAMPLE_INTERVAL: float = 0.05   # 50 ms

# Density-cluster threshold for mode estimation (1.5 cm)
_DEFAULT_CLUSTER_THRESHOLD: float = 0.015  # meters

# Joint motion durations
_PREGRASP_DURATION: float = 3.0   # seconds, same as TRAJECTORY_DURATION
_DESCENT_DURATION: float = 1.0    # seconds — fast, minimal drift
_LIFT_DURATION: float = 1.0       # seconds
_HOME_DURATION: float = 3.0       # seconds

# Workspace limits (from boundary check in _single_pick_attempt)
_WORKSPACE_MIN_DIST: float = 0.05   # 5 cm
_WORKSPACE_MAX_DIST: float = 0.35   # 35 cm

# Calibrated home joints (DEFAULT_HOME_VALUES in v2)
_DEFAULT_HOME_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]


@skill(
    aliases=["grab", "grasp", "take", "抓", "拿", "抓起", "抓住", "抓取", "拿起", "取"],
    direct=False,
    auto_steps=["scan", "detect", "pick"],
)
class PickSkill:
    """Pick up an object from the workspace.

    The skill can locate the target object in two ways:
    - From the world model by object_id or object_label (preferred).
    - From context.perception by sampling live detections (fallback).

    Parameters:
        object_id (str, optional): ID of the object in the world model.
        object_label (str, optional): Label of the object to pick if no ID.

    When neither is provided, the skill uses context.perception to locate
    an object in the camera frame and applies the calibration transform.
    """

    name: str = "pick"
    description: str = "Pick up an object. Use mode='hold' to keep it, mode='drop' to discard it to the side."
    parameters: dict = {
        "object_id": {
            "type": "string",
            "required": False,
            "description": "ID of the object in the world model",
        },
        "object_label": {
            "type": "string",
            "required": False,
            "description": "Label of the object to pick (e.g. 'mug', 'banana')",
        },
        "mode": {
            "type": "string",
            "required": False,
            "default": "drop",
            "description": "'hold' = grasp and hold at home (for subsequent place), 'drop' = grasp and discard to side",
        },
    }
    preconditions: list[str] = ["gripper_empty"]
    postconditions: list[str] = []  # pick ends with gripper open (object dropped)
    effects: dict = {"gripper_state": "open"}  # pick ends with drop
    failure_modes: list[str] = [
        "no_arm", "object_not_found", "no_detections", "no_3d_samples",
        "out_of_workspace", "ik_unreachable", "move_failed", "track_failed",
        "calibration_error",
    ]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute pick with retry logic.

        On failure between retries the arm returns to home before the next
        attempt.  The world model is NOT updated on success — the executor
        calls world_model.apply_skill_effects() after execute() returns.

        Args:
            params: optional object_id or object_label.
            context: SkillContext with arm, gripper, perception, world_model.

        Returns:
            SkillResult(success=True, result_data={"position_cm": [x, y]}) on success.
            SkillResult(success=False, error_message=...) on failure.
        """
        if context.arm is None:
            return SkillResult(
                success=False,
                error_message="No arm connected",
                result_data={"diagnosis": "no_arm"},
            )

        max_retries: int = (
            context.config.get("skills", {}).get("pick", {}).get("max_retries", _DEFAULT_MAX_RETRIES)
        )
        home_joints: list[float] = (
            context.config.get("skills", {}).get("home", {}).get("joint_values", _DEFAULT_HOME_JOINTS)
        )

        last_error: str = "unknown error"
        last_diagnosis: str = "unknown"
        last_result_data: dict = {}
        for attempt in range(1, max_retries + 1):
            logger.info("[PICK] Attempt %d/%d", attempt, max_retries)
            result = self._single_pick_attempt(params, context)
            if result.success:
                return result
            last_error = result.error_message
            last_result_data = dict(result.result_data)
            last_diagnosis = last_result_data.get("diagnosis", "unknown")
            logger.warning("[PICK] Attempt %d failed: %s", attempt, last_error)

            if attempt < max_retries:
                logger.info("[PICK] Returning home for retry ...")
                context.arm.move_joints(home_joints, duration=_HOME_DURATION)
                time.sleep(1.0)

        # Merge retry metadata into the last attempt's result_data so callers
        # can inspect both the failure diagnosis and retry statistics.
        retry_data = dict(last_result_data)
        retry_data.update({
            "diagnosis": last_diagnosis,
            "attempts": max_retries,
            "hint": "All retry attempts exhausted.",
        })
        return SkillResult(
            success=False,
            error_message=f"Pick failed after {max_retries} attempts: {last_error}",
            result_data=retry_data,
        )

    # ------------------------------------------------------------------
    # Single pick attempt
    # ------------------------------------------------------------------

    def _single_pick_attempt(
        self,
        params: dict,
        context: SkillContext,
    ) -> SkillResult:
        """Execute one pick attempt.  Full port of _single_pick_attempt().

        Returns SkillResult — does NOT retry on its own.
        """
        # Always open gripper first — ensures clean grip regardless of current state
        if context.gripper is not None:
            logger.info("[PICK] Ensuring gripper is open ...")
            context.gripper.open()

        cfg = context.config.get("skills", {}).get("pick", {})
        z_offset: float = cfg.get("z_offset", _DEFAULT_Z_OFFSET)
        x_offset: float = cfg.get("x_offset", 0.0)
        y_scale: float = cfg.get("y_scale", 1.0)
        pre_grasp_h: float = cfg.get("pre_grasp_height", _DEFAULT_PRE_GRASP_HEIGHT)
        home_joints: list[float] = (
            context.config.get("skills", {}).get("home", {}).get("joint_values", _DEFAULT_HOME_JOINTS)
        )

        # Step 1: Get target in base frame
        base_pos_result = self._get_target_base_pos(params, context)
        if base_pos_result is None:
            label = params.get("object_label") or params.get("object_id") or ""
            return SkillResult(
                success=False,
                error_message="Cannot locate target object",
                result_data={
                    "diagnosis": "object_not_found",
                    "query": label,
                    "world_model_objects": [
                        o.label for o in context.world_model.get_objects()
                    ],
                },
            )
        base_pos = base_pos_result.copy()

        # Step 2: Apply z_offset (gripper height above table)
        base_pos[2] += z_offset

        # Step 3: Empirical X/Y offsets (from vector_ws hardware tuning)
        # These offsets absorb tip-to-link offset + servo errors + URDF inaccuracy
        # Skipped when hardware_offsets=false (sim mode — positions already correct)
        if cfg.get("hardware_offsets", True):
            base_pos[0] += x_offset + 0.02  # uniform +2cm forward (empirical)
            # Gripper asymmetry Y compensation — uniform +2cm (half gripper width)
            base_pos[1] += 0.02

        logger.info(
            "[PICK] Raw base: (%.1f, %.1f, %.1f) cm, z_offset=%.0fcm, pre_grasp_h=%.0fcm",
            base_pos_result[0] * 100, base_pos_result[1] * 100, base_pos_result[2] * 100,
            z_offset * 100, pre_grasp_h * 100,
        )
        logger.info(
            "[PICK] Grasp target: (%.1f, %.1f, %.1f) cm | Pre-grasp: %.1f cm",
            base_pos[0] * 100, base_pos[1] * 100, base_pos[2] * 100,
            (base_pos[2] + pre_grasp_h) * 100,
        )

        # Step 5: Workspace boundary check
        dist_xy = float(np.linalg.norm(base_pos[:2]))
        if dist_xy > _WORKSPACE_MAX_DIST or dist_xy < _WORKSPACE_MIN_DIST:
            return SkillResult(
                success=False,
                error_message=(
                    f"Object at ({base_pos[0]*100:.1f}, {base_pos[1]*100:.1f}) cm "
                    f"outside workspace ({_WORKSPACE_MIN_DIST*100:.0f}–{_WORKSPACE_MAX_DIST*100:.0f} cm)"
                ),
                result_data={
                    "diagnosis": "out_of_workspace",
                    "target_base_cm": [
                        round(base_pos[0] * 100, 1),
                        round(base_pos[1] * 100, 1),
                        round(base_pos[2] * 100, 1),
                    ],
                    "distance_cm": round(dist_xy * 100, 1),
                    "workspace_bounds_cm": [
                        int(_WORKSPACE_MIN_DIST * 100),
                        int(_WORKSPACE_MAX_DIST * 100),
                    ],
                },
            )

        # Step 6: Simple IK — solve for gripper_link directly
        # The calibration matrix + z_offset already account for real-world errors.
        # Do NOT try tip compensation — the URDF model doesn't match the real arm
        # accurately enough for model-based tip correction to help.
        current_joints = context.arm.get_joint_positions()

        # Pre-grasp position (higher Z)
        pre_grasp_pos = base_pos.copy()
        pre_grasp_pos[2] += pre_grasp_h

        q_pregrasp_result = context.arm.ik(
            (pre_grasp_pos[0], pre_grasp_pos[1], pre_grasp_pos[2]),
            current_joints,
        )
        if q_pregrasp_result is None:
            return SkillResult(
                success=False,
                error_message="IK failed for pre-grasp",
                result_data={
                    "diagnosis": "ik_unreachable",
                    "target_base_cm": [
                        round(base_pos[0] * 100, 1),
                        round(base_pos[1] * 100, 1),
                        round(base_pos[2] * 100, 1),
                    ],
                    "pre_grasp_cm": [
                        round(pre_grasp_pos[0] * 100, 1),
                        round(pre_grasp_pos[1] * 100, 1),
                        round(pre_grasp_pos[2] * 100, 1),
                    ],
                },
            )
        q_pregrasp = list(q_pregrasp_result)

        # Wrist roll offset (sim mode: +pi/2 to orient gripper fingers for top-down grasp)
        wrist_roll_offset: float = cfg.get("wrist_roll_offset", 0.0)
        if wrist_roll_offset != 0.0 and len(q_pregrasp) == 5:
            q_pregrasp[4] += wrist_roll_offset

        # Grasp position (warm-started from pre-grasp for minimal joint change)
        q_grasp_result = context.arm.ik(
            (base_pos[0], base_pos[1], base_pos[2]),
            q_pregrasp,
        )
        if q_grasp_result is None:
            return SkillResult(
                success=False,
                error_message="IK failed for grasp position",
                result_data={
                    "diagnosis": "ik_unreachable",
                    "target_base_cm": [
                        round(base_pos[0] * 100, 1),
                        round(base_pos[1] * 100, 1),
                        round(base_pos[2] * 100, 1),
                    ],
                },
            )
        q_grasp = list(q_grasp_result)

        if wrist_roll_offset != 0.0 and len(q_grasp) == 5:
            q_grasp[4] += wrist_roll_offset

        logger.info(
            "[PICK] IK solved: pre-grasp Z=%.1fcm, grasp Z=%.1fcm",
            pre_grasp_pos[2] * 100, base_pos[2] * 100,
        )

        # Step 8: Open gripper
        logger.info("[PICK] Opening gripper ...")
        if context.gripper is not None:
            context.gripper.open()

        # Step 9: Move to pre-grasp
        logger.info("[PICK] Moving to pre-grasp ...")
        if not context.arm.move_joints(q_pregrasp, duration=_PREGRASP_DURATION):
            return SkillResult(
                success=False,
                error_message="Pre-grasp move failed",
                result_data={"diagnosis": "move_failed", "phase": "pre-grasp"},
            )

        # Step 10: Descend to grasp
        logger.info("[PICK] Descending to grasp ...")
        if not context.arm.move_joints(q_grasp, duration=_DESCENT_DURATION):
            return SkillResult(
                success=False,
                error_message="Descent to grasp failed",
                result_data={"diagnosis": "move_failed", "phase": "descent"},
            )

        # Step 11: Open → wait → Close gripper sequence
        # Open first to ensure full grip range, then close to grasp
        logger.info("[PICK] Gripper sequence: open → close ...")
        if context.gripper is not None:
            context.gripper.open()
            time.sleep(0.3)
            for _ in range(3):
                context.gripper.close()
                time.sleep(0.2)

        # Step 12: Lift straight back to pre-grasp (one move)
        logger.info("[PICK] Lifting ...")
        context.arm.move_joints(q_pregrasp, duration=_LIFT_DURATION)

        # Step 13: Return home (holding object)
        logger.info("[PICK] Returning home ...")
        if not context.arm.move_joints(home_joints, duration=_HOME_DURATION):
            return SkillResult(
                success=False,
                error_message="Return home after pick failed",
                result_data={"diagnosis": "move_failed", "phase": "home"},
            )

        # Step 14: Mode-dependent behavior
        pick_mode = params.get("mode", cfg.get("default_mode", "drop"))

        if pick_mode == "hold":
            # Hold mode: keep object in gripper, ready for place command
            logger.info("[PICK] Holding object (mode=hold)")
        else:
            # Drop mode: rotate 90deg and drop outside workspace
            drop_joints = list(home_joints)
            drop_joints[0] = drop_joints[0] + 1.57  # shoulder_pan +90deg
            logger.info("[PICK] Rotating to drop position ...")
            context.arm.move_joints(drop_joints, duration=_HOME_DURATION)

            logger.info("[PICK] Dropping object ...")
            if context.gripper is not None:
                context.gripper.open()
                time.sleep(0.5)
                context.gripper.close()

            logger.info("[PICK] Returning home ...")
            context.arm.move_joints(home_joints, duration=_HOME_DURATION)

        # Remove only the picked object from world model
        # (other objects are still valid; clearing all was a bug)
        picked_id = params.get("object_id")
        if not picked_id:
            label = params.get("object_label", "")
            matches = context.world_model.get_objects_by_label(label)
            if matches:
                picked_id = matches[0].object_id
        if picked_id:
            context.world_model.remove_object(picked_id)
            logger.info("[PICK] Removed %s from world model", picked_id)
        else:
            logger.info("[PICK] No specific object to remove from world model")

        logger.info(
            "[PICK] Pick complete! Grasped at (%.1f, %.1f) cm",
            base_pos[0] * 100, base_pos[1] * 100,
        )
        return SkillResult(
            success=True,
            result_data={
                "position_cm": [
                    round(base_pos[0] * 100, 2),
                    round(base_pos[1] * 100, 2),
                ],
                "diagnosis": "ok",
            },
        )

    # ------------------------------------------------------------------
    # Target position resolution
    # ------------------------------------------------------------------

    def _get_target_base_pos(
        self,
        params: dict,
        context: SkillContext,
    ) -> Optional[np.ndarray]:
        """Resolve target object position in base frame.

        Resolution order:
        1. object_id → look up in world_model.get_object()
        2. object_label → look up in world_model.get_objects_by_label()
        3. context.perception → live camera sampling with density cluster

        Returns:
            (3,) numpy array in base frame metres, or None if unresolvable.
        """
        # ALWAYS re-detect with perception if available (object may have moved)
        if context.perception is not None:
            label = params.get("object_label") or params.get("object_id") or "object"
            logger.info("[PICK] Live perception for %r (always re-detect)", label)
            result = self._sample_from_perception(params, context)
            if result is not None:
                return result
            logger.warning("[PICK] Perception failed, falling back to world model")

        # Fallback: world model (only if no perception or perception failed)
        obj_id = params.get("object_id")
        if obj_id:
            obj = context.world_model.get_object(obj_id)
            if obj is not None and (abs(obj.x) > 0.01 or abs(obj.y) > 0.01):
                logger.info("[PICK] Fallback: world_model object_id=%s", obj_id)
                return np.array([obj.x, obj.y, obj.z], dtype=float)

        label = params.get("object_label")
        if label:
            objects = context.world_model.get_objects_by_label(label)
            valid = [o for o in objects if abs(o.x) > 0.01 or abs(o.y) > 0.01]
            if valid:
                closest = min(valid, key=lambda o: o.x ** 2 + o.y ** 2)
                logger.info("[PICK] Fallback: world_model label=%r", label)
                return np.array([closest.x, closest.y, closest.z], dtype=float)

        return None

    def _sample_from_perception(
        self,
        params: dict,
        context: SkillContext,
    ) -> Optional[np.ndarray]:
        """Sample target position from live perception using density clustering.

        Port of skill_node_v2._get_target_camera_pos(), then transforms to
        base frame using the calibration matrix from context.calibration.

        Step 1: detect() to get 2D bboxes, track() to initialise tracker and get
                3D poses from depth projection (TrackedObject.pose).
        Step 2: Collect _DEFAULT_SAMPLE_COUNT 3D position readings by calling
                update() repeatedly at _DEFAULT_SAMPLE_INTERVAL intervals.
        Step 3: Density cluster (1.5cm threshold) to find modal position.
        Step 4: camera_to_base() using calibration transform.
        """
        # Resolve query label: try object_label, then extract from object_id, then "object"
        query = params.get("object_label") or params.get("object") or ""
        if not query:
            obj_id = params.get("object_id", "")
            # Extract label from object_id like "battery_0" → "battery"
            if obj_id and "_" in obj_id:
                query = obj_id.rsplit("_", 1)[0].replace("_", " ")
            elif obj_id:
                query = obj_id
            else:
                query = "object"
        logger.info("[PICK] Sampling perception for %r ...", query)

        cfg = context.config.get("skills", {}).get("pick", {})
        n_samples: int = cfg.get("sample_count", _DEFAULT_SAMPLE_COUNT)
        interval: float = cfg.get("sample_interval", _DEFAULT_SAMPLE_INTERVAL)
        threshold: float = cfg.get("cluster_threshold", _DEFAULT_CLUSTER_THRESHOLD)

        # First: detect to get 2D bbox, then track to get 3D pose from depth
        try:
            detections = context.perception.detect(query)
        except Exception as exc:
            logger.warning("[PICK] Initial detect failed: %s", exc)
            return None

        if not detections:
            logger.warning("[PICK] No detections found for %r", query)
            return None

        # Initialise tracker — this gives us TrackedObject with 3D pose
        try:
            tracked = context.perception.track(detections)
        except Exception as exc:
            logger.warning("[PICK] Track init failed: %s", exc)
            return None

        if not tracked:
            logger.warning("[PICK] Tracker returned no objects")
            return None

        # Collect samples using update() loop (mirrors _get_target_camera_pos sampling)
        samples: list[np.ndarray] = []

        # Add pose from initial track() result
        t0 = tracked[0]
        if t0.pose is not None:
            samples.append(np.array([t0.pose.x, t0.pose.y, t0.pose.z], dtype=float))

        # Collect remaining samples via update()
        has_update = hasattr(context.perception, "update")
        for _ in range(n_samples - len(samples)):
            time.sleep(interval)
            try:
                if has_update:
                    updated = context.perception.update()
                else:
                    updated = context.perception.track(detections)
            except Exception as exc:
                logger.warning("[PICK] Perception update error: %s", exc)
                continue

            if updated:
                obj = updated[0]
                if obj.pose is not None:
                    samples.append(
                        np.array([obj.pose.x, obj.pose.y, obj.pose.z], dtype=float)
                    )

        if not samples:
            logger.warning("[PICK] No valid 3D position samples from perception")
            return None

        logger.info(
            "[PICK] Collected %d/%d valid 3D samples", len(samples), n_samples
        )

        arr = np.array(samples)
        if len(arr) < 3:
            cam_pos = np.median(arr, axis=0)
        else:
            cam_pos = self._density_cluster_mean(arr, threshold)

        logger.info(
            "[PICK] Camera-frame position: (%.3f, %.3f, %.3f)",
            cam_pos[0], cam_pos[1], cam_pos[2],
        )

        # Transform camera→base using calibration
        T = _get_calibration_matrix(context)
        return camera_to_base(cam_pos, T)

    @staticmethod
    def _density_cluster_mean(
        arr: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Find the densest cluster and return its mean.

        Port of the density-cluster logic in skill_node_v2._get_target_camera_pos().

        Args:
            arr: (N, 3) array of position samples.
            threshold: neighbourhood radius in metres.

        Returns:
            (3,) mean position of the densest cluster.
        """
        best_count = 0
        best_idx = 0
        for i in range(len(arr)):
            dists = np.linalg.norm(arr - arr[i], axis=1)
            count = int(np.sum(dists < threshold))
            if count > best_count:
                best_count = count
                best_idx = i

        center = arr[best_idx]
        dists = np.linalg.norm(arr - center, axis=1)
        cluster = arr[dists < threshold]

        logger.debug(
            "[PICK] Cluster: %d/%d samples (threshold=%.1f cm)",
            len(cluster), len(arr), threshold * 100,
        )
        return np.mean(cluster, axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_calibration_matrix(context: SkillContext) -> np.ndarray:
    """Extract calibration matrix from context, loading from file if needed.

    context.calibration can be:
    - a (4,4) numpy ndarray directly
    - a dict with key "transform_matrix"
    - a string path to workspace_calibration.yaml
    - None → load from default path

    Returns:
        (4, 4) numpy float64 homogeneous transform.
    """
    cal = context.calibration
    if cal is None:
        return load_calibration()
    if isinstance(cal, np.ndarray) and cal.shape == (4, 4):
        return cal
    if isinstance(cal, dict) and "transform_matrix" in cal:
        return np.array(cal["transform_matrix"], dtype=np.float64)
    if isinstance(cal, str):
        return load_calibration(cal)
    # Handle Calibration object (from vector_os_nano.perception.calibration)
    if hasattr(cal, '_matrix') and cal._matrix is not None:
        return np.array(cal._matrix, dtype=np.float64)
    if hasattr(cal, 'camera_to_base'):
        # Calibration class — extract matrix or use it directly
        # Store the Calibration object reference for _camera_to_base to use
        logger.info("[PICK] Using Calibration object directly")
        return getattr(cal, '_matrix', np.eye(4))
    # Unknown type — use identity with a warning
    logger.warning("[PICK] Unknown calibration type %s — using identity", type(cal))
    return np.eye(4)
