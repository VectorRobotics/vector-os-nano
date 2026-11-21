# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""One camera-driven grasp: resolve, view, perceive, approach, grip, lift, hold."""
from __future__ import annotations

import math
import threading

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import Pose3D, SkillResult
from vector_os_nano.perception.base import failure_cause
from vector_os_nano.skills.grasp_perception import ALIASES, _fail, _target, perceive
from vector_os_nano.skills.utils.approach_pose import compute_approach_pose


def _cancelled(token):
    return _fail("cancelled", "Grasp cancelled; the gripper keeps its current hold.") if token.is_set() else None


def _move(arm, joints, duration, phase, token):
    if token.is_set():
        return _cancelled(token)
    ok = arm.move_joints(list(joints), duration=duration)
    return _cancelled(token) or (None if ok else _fail("move_failed", f"Arm blocked during {phase}; inspect the clearance before retrying.", phase=phase))


def view(context, cfg, token):
    joints = context.config.get("arm", {}).get("postures", {}).get("view")
    error = _cancelled(token) if joints is None else _move(context.arm, joints, cfg["view_duration"], "view", token)
    if error or 'frontal' not in cfg:
        return error
    error = _group_move(context, cfg['frontal']['prepare'], cfg['view_duration'], 'prepare', token)
    if error:
        return error
    ready = context.config['arm']['postures'][cfg['frontal']['ready_posture']]
    return _move(context.arm, ready, cfg['durations'][0], 'ready', token)


def _group_move(context, phase, duration, name, token):
    group = context.services.get('joint_groups', {}).get(phase['group'])
    joints = context.config.get('joint_groups', {}).get(phase['group'], {}).get('postures', {}).get(phase['posture'])
    if group is None or joints is None:
        return _fail('not_configured', f'Missing joint group/posture for grasp {name}.')
    return _move(group, joints, duration, name, token)


def _poses(point, cfg, base=None):
    if 'frontal' not in cfg:
        return ((point.x, point.y, point.z + cfg['pre_grasp_height']),
                (point.x, point.y, point.z + cfg['grasp_z_above']), None, None)
    yaw = None if base is None else base.get_heading()
    if yaw is None or not math.isfinite(yaw):
        raise ValueError('frontal grasp requires a measured base heading')
    f = cfg['frontal']
    forward, left = (math.cos(yaw), math.sin(yaw), 0.), (-math.sin(yaw), math.cos(yaw), 0.)
    def offset(values):
        return (point.x + values[0]*forward[0] + values[1]*left[0],
                point.y + values[0]*forward[1] + values[1]*left[1], point.z + values[2])
    def rotation(pitch):
        c, s = math.cos(pitch), math.sin(pitch)
        return [[left[i], s*forward[i] + c*(i == 2), c*forward[i] - s*(i == 2)] for i in range(3)]
    return offset(f['pre_offset']), offset(f['contact_offset']), rotation(0.), rotation(f['pitch'])


def _solve(arm, point, seed=None, rotation=None):
    return (arm.ik_top_down(point, current_joints=seed) if rotation is None else
            arm.ik(point, current_joints=seed, target_rotation=rotation))


def _grasp_point(point, meta, cfg):
    if 'frontal' not in cfg:
        return point
    # shortcut: observed box bounds assume an upright aligned package; use oriented fitting for rotated packages.
    bounds = meta.get('visible_bounds')
    if (not isinstance(bounds, list) or len(bounds) != 2 or any(len(row) != 3 for row in bounds)
            or any(not math.isfinite(v) for row in bounds for v in row)
            or any(lo >= hi for lo, hi in zip(*bounds))):
        raise ValueError('frontal grasp needs nondegenerate observed package bounds')
    point = Pose3D(*(sum(pair)/2 for pair in zip(*bounds)))
    meta.update(surface_world=meta['grasp_world'], grasp_world=[point.x, point.y, point.z],
                geometry='observed upright box bounds; occluded extent is unverified')
    return point


def _walk(base, cfg, token, *, vx=0., vy=0., vyaw=0., duration=None):
    if token.is_set():
        return _cancelled(token)
    limits = base.limits
    command = [max(-limit, min(limit, value)) for value, limit in zip((vx, vy, vyaw), limits)]
    if not any(command):
        return _fail("approach_failed", "The base cannot perform this approach; choose another view with look.")
    ok = base.walk(vx=command[0], vy=command[1], vyaw=command[2], duration=duration or cfg["step_duration"])
    return _cancelled(token) or (None if ok else _fail("approach_failed", "Base approach did not complete; inspect the route and look again."))


def _offset(base, point, lateral_offset=0.):
    pos, yaw = base.get_position(), base.get_heading()
    if pos is None or yaw is None:
        raise ValueError("base pose is unknown")
    dx, dy = point.x - pos[0], point.y - pos[1]
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    angle = math.atan2(lateral, forward)
    if lateral_offset:
        angle -= math.asin(max(-1., min(1., lateral_offset/max(math.hypot(dx, dy), 1e-9))))
        angle = math.atan2(math.sin(angle), math.cos(angle))
    return forward, lateral-lateral_offset, angle, tuple(pos[:2]), yaw


def _reachable(arm, point, cfg, base=None):
    if 'frontal' in cfg and 'base' in cfg:
        bc = cfg['base']
        forward, lateral, _, _, _ = _offset(base, point, bc.get('lateral_offset', 0.))
        if abs(forward-bc['standoff']) > bc['lateral_deadband'] or abs(lateral) > bc['lateral_deadband']:
            return False
    pre, target, rp, rg = _poses(point, cfg, base)
    q = _solve(arm, pre, rotation=rp)
    return q is not None and _solve(arm, target, q, rg) is not None


def point(context, query, target, params, cfg, token, expected_bounds=None):
    """Search locally, then use remembered position only to get a fresh observation."""
    base, bc = context.base, cfg.get("base")
    gp, meta, error = perceive(context.perception, query, target, params, cancel=lambda: token.is_set(), expected_bounds=expected_bounds)
    if token.is_set():
        return None, meta, _cancelled(token)
    if base is None or bc is None:
        return gp, meta, error
    for _ in range(bc["scan_steps"]):
        if token.is_set():
            return None, meta, _cancelled(token)
        if gp is not None and math.hypot(gp.x - base.get_position()[0], gp.y - base.get_position()[1]) <= bc["scan_range"]:
            return gp, meta, None
        if error and error.diagnosis_code in {"perception_failed", "no_camera"}:
            return None, meta, error
        rate = min(bc["yaw_speed"], base.limits[2])
        if rate <= 0:
            break
        error = _walk(base, bc, token, vyaw=rate, duration=bc["scan_step"] / rate)
        if error:
            return None, meta, error
        gp, meta, error = perceive(context.perception, query, target, cancel=lambda: token.is_set(), expected_bounds=expected_bounds)
    if gp is not None and math.hypot(gp.x - base.get_position()[0], gp.y - base.get_position()[1]) <= bc["scan_range"]:
        return gp, meta, None
    memory = context.services.get("spatial_memory")
    matches = [] if memory is None else memory.find_objects_by_category(target.prompt or query)
    matches = [o for o in matches if o.x is not None and o.y is not None]
    if len(matches) == 1:
        if token.is_set():
            return None, meta, _cancelled(token)
        o, pos = matches[0], base.get_position()
        x, y, _ = compute_approach_pose((o.x, o.y, o.z or 0.), (*pos[:2], base.get_heading()), clearance=bc["view_standoff"])
        if not base.navigate_to(x, y, cancel=token):
            return None, meta, _cancelled(token) or _fail("approach_failed", "Cannot reach the remembered object's viewing position; clear the path and use look.")
        error = face(base, o, bc, token)
        if error:
            return None, meta, error
        gp, meta, error = perceive(context.perception, query, target, cancel=lambda: token.is_set(), expected_bounds=expected_bounds)
        meta["memory_used_for_navigation"] = True
    if gp is not None and math.hypot(gp.x - base.get_position()[0], gp.y - base.get_position()[1]) > bc["scan_range"]:
        gp = None
    return gp, meta, _cancelled(token) or error or (None if gp is not None else _fail("no_detections", f"Cannot see {query!r}; use look near its last known location, then grasp again."))


def face(base, gp, cfg, token):
    for _ in range(cfg["approach_steps"]):
        _, _, angle, _, _ = _offset(base, gp, cfg.get('lateral_offset', 0.))
        if abs(angle) <= cfg["yaw_deadband"]:
            return _cancelled(token)
        rate = min(cfg["yaw_speed"], base.limits[2])
        if rate <= 0:
            return _fail("approach_failed", "Base cannot turn to face the target; select another approach.")
        error = _walk(base, cfg, token, vyaw=math.copysign(rate, angle), duration=min(cfg["step_duration"], abs(angle) / rate))
        if error:
            return error
    return _fail("approach_failed", "Base could not face the object within its approach budget; look again.")


def reach(context, gp, cfg, token, *, reseat=False):
    """Heading-relative progress; finite seat budget, followed by a fresh observation."""
    arm, base, bc = context.arm, context.base, cfg.get("base")
    if not reseat and _reachable(arm, gp, cfg, base):
        return (True, face(base, gp, bc, token)) if base is not None and bc is not None else (False, None)
    if base is None or bc is None:
        ee, _ = arm.fk(arm.get_joint_positions())
        distance = math.dist(ee, (gp.x, gp.y, gp.z))
        return False, _fail("out_of_reach", f"Target is outside the top-down reach envelope ({cfg['reach_m']:.2f} m; current tip-to-target {distance:.2f} m); reposition the object within reach.", distance=distance, reach_m=cfg["reach_m"])
    error = face(base, gp, bc, token)
    if error:
        return True, error
    stalls, previous, seated = 0, None, 0
    for _ in range(bc["approach_steps"] + bc["seat_steps"]):
        if token.is_set():
            return True, _cancelled(token)
        if not reseat and _reachable(arm, gp, cfg, base):
            return True, face(base, gp, bc, token)
        forward, lateral, angle, pos, yaw = _offset(base, gp, bc.get('lateral_offset', 0.))
        if previous is not None:
            dx, dy = pos[0] - previous[0], pos[1] - previous[1]
            progress = dx * math.cos(previous[2]) + dy * math.sin(previous[2])
            stalls = stalls + 1 if progress < bc["stall_distance"] else 0
        if stalls >= 2:
            seated += 1
            if seated > bc["seat_steps"] or (previous is not None and progress < bc["seat_distance"] and seated >= 2):
                break
        previous = (*pos, yaw)
        # Too close: back out toward the standoff; never creep through the target.
        gap = forward - bc["standoff"]
        vx = math.copysign(bc["step_speed"], gap) if abs(gap) > bc["lateral_deadband"] else 0.
        vy = max(-bc["lateral_speed"], min(bc["lateral_speed"], lateral * bc["lateral_gain"])) if abs(lateral) > bc["lateral_deadband"] else 0.
        vyaw = max(-bc["yaw_speed"], min(bc["yaw_speed"], angle * bc["lateral_gain"])) if abs(angle) > bc["yaw_deadband"] else 0.
        if not any((vx, vy, vyaw)):
            break
        error = _walk(base, bc, token, vx=vx, vy=vy, vyaw=vyaw)
        reseat = False
        if error:
            return True, error
    return True, None  # the caller re-observes and retries IK, never trusts seating alone


def _approach_point(base, arm, q_pre, pre, back):
    if base is None or back <= 0:
        return None
    bx, by = base.get_position()[:2]
    _, rot = arm.fk(q_pre)
    ax, ay = rot[0][0], rot[1][0]
    n = math.hypot(ax, ay)
    if n < 1e-6:
        return None
    if ax * (bx - pre[0]) + ay * (by - pre[1]) < 0:
        ax, ay = -ax, -ay
    return pre[0] + back * ax / n, pre[1] + back * ay / n, pre[2]


def grip(context, gp, cfg, token, identity=None):
    arm, gripper = context.arm, context.gripper
    pre, grasp, r_pre, r_grasp = _poses(gp, cfg, context.base)
    q_pre = _solve(arm, pre, rotation=r_pre)
    if q_pre is None or _solve(arm, grasp, q_pre, r_grasp) is None:
        return _fail("ik_unreachable", "Configured grasp is unreachable at this pose; use look after another approach.")
    if token.is_set():
        return _cancelled(token)
    if not gripper.open():
        return _cancelled(token) or _fail("gripper_failed", "Gripper did not open; inspect it before retrying.")
    t_pre, t_slide, t_down, t_lift = cfg["durations"]
    back = None if 'frontal' in cfg else _approach_point(context.base, arm, q_pre, pre, cfg["approach_back"])
    phases = [("pre_grasp", pre, t_pre, r_pre), ("descent", grasp, t_down, r_grasp)]
    if back is not None:
        phases = [("approach", back, t_pre, None), ("pre_grasp", pre, t_slide, None), ("descent", grasp, t_down, None)]
    for phase, xyz, duration, rotation in phases:
        if token.is_set():
            return _cancelled(token)
        q = _solve(arm, xyz, arm.get_joint_positions(), rotation)
        if q is None:
            return _fail("ik_unreachable", f"IK became unreachable during {phase}; look again before retrying.", phase=phase)
        error = _move(arm, q, duration, phase, token)
        if error:
            return error
    if token.is_set():
        return _cancelled(token)
    if not gripper.close():
        return _cancelled(token) or _fail("gripper_failed", "Gripper did not close; inspect it before retrying.")
    if token.is_set():
        return _cancelled(token)
    if not (gripper.is_holding() if identity is None else gripper.is_holding(identity)):
        return _fail("not_holding", "Gripper closed without a hold; look again and retry grasp.")
    if 'frontal' in cfg:
        error = _group_move(context, cfg['frontal']['lift'], t_lift, 'lift', token)
    else:
        lift = (gp.x, gp.y, gp.z + cfg["lift_height"])
        q = arm.ik_top_down(lift, current_joints=arm.get_joint_positions())
        if q is None:
            q = arm.ik_top_down(lift, current_joints=q_pre)
        if q is None:
            return _fail("ik_unreachable", "Object is held, but lift is unreachable; keep the hold and choose a clear place to release.", phase="lift", holding=True)
        error = _move(arm, q, t_lift, "lift", token)
    if error:
        return error
    hold = context.config.get("arm", {}).get("postures", {}).get("hold")
    error = _move(arm, arm.get_joint_positions() if hold is None else hold,
                  cfg["hold_duration"], "hold", token)
    if error:
        return error
    return _cancelled(token) or (None if (gripper.is_holding() if identity is None else gripper.is_holding(identity)) else _fail("not_holding", "Object slipped during lift; look again before grasping."))


@skill(requires=('arm', 'gripper', 'camera', 'depth'), aliases=ALIASES)
class GraspSkill:
    name = "grasp"
    description = "Find one named object with RGB-D perception, approach if needed, grasp and lift it. Refuses ambiguous names; uses fresh observations after moving."
    verify_hint = "holding_object({query})"
    parameters = {
        "query": {"type": "string", "required": True, "description": "The exact object to grasp."},
        "detections": {"type": "array", "required": False, "description": "Optional detections with label, box, confidence and camera frame provenance."},
        "bbox": {"type": "array", "required": False, "description": "Optional pixel box; re-detected without current camera provenance."},
        "frame": {"type": "object", "required": False, "description": "Box camera provenance: size, cam_pos, cam_mat."},
    }
    preconditions = ["gripper_empty"]
    postconditions = []
    effects = {"gripper_state": "holding"}
    failure_modes = ["unknown_object", "ambiguous_object", "no_arm", "no_gripper", "arm_unsupported",
                     "no_perception", "no_camera", "no_detections", "implausible_target", "perception_failed",
                     "not_configured", "already_holding", "out_of_reach", "approach_failed", "ik_unreachable",
                     "move_failed", "gripper_failed", "not_holding", "cancelled", "device_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        token = context.services.get("cancel") or threading.Event()
        if token.is_set():
            return _cancelled(token)
        if context.arm is None:
            return _fail("no_arm", "Connect an arm before grasping.")
        if context.gripper is None:
            return _fail("no_gripper", "Connect a gripper before grasping.")
        if not callable(getattr(context.arm, "ik_top_down", None)):
            return _fail("arm_unsupported", "This arm has no top-down IK; configure a compatible arm controller.")
        if context.perception is None:
            return _fail("no_perception", "Connect RGB-D perception, then use look.")
        if not all(callable(getattr(context.perception, m, None)) for m in ("capture", "detect_in", "segment")):
            return _fail("no_camera", "Perception does not provide a calibrated RGB-D snapshot; configure the camera.")
        query = str(params.get("query") or "").strip()
        if not query:
            return _fail("unknown_object", "Name the object to grasp, or explicitly ask for the object in front.")
        world = context.services.get("object_world")
        if context.config.get("world") and world is None:
            return _fail("not_configured", "This world has no injected object resolver; bind its manifest before grasping.")
        target = _target(query, world, hasattr(context.perception, "front_object_mask"))
        if isinstance(target, SkillResult):
            return target
        cfg = context.config.get("skills", {}).get("grasp")
        if not cfg:
            return _fail("not_configured", "This deployment has no grasp geometry in its manifest; configure skills.grasp.")
        try:
            if context.gripper.is_holding():
                return _fail("already_holding", "The gripper already holds an object; place it before grasping another.")
            expected_bounds = None
            if 'frontal' in cfg:
                bc = cfg.get('base')
                if bc is None:
                    return _fail('not_configured', 'Frontal grasp requires a preparation standoff and base approach configuration.')
                gp, meta, error = point(context, query, target, params, cfg, token)
                if error:
                    return error
                gp = _grasp_point(gp, meta, cfg)
                expected_bounds = meta['visible_bounds']
                forward, _, _, _, _ = _offset(context.base, gp, bc.get('lateral_offset', 0.))
                if forward < bc['view_standoff']-bc['lateral_deadband']:
                    return _fail('approach_failed', 'Too close to extend the arm safely; back away, observe again, then retry grasp.',
                                 required_forward_clearance_m=bc['view_standoff'], observed_forward_m=forward)
            error = view(context, cfg, token)
            if error:
                return error
            gp, meta, error = point(context, query, target, params, cfg, token, expected_bounds)
            if error:
                return error
            gp = _grasp_point(gp, meta, cfg)
            retries = cfg.get("base", {}).get("retries", 0)
            for attempt in range(retries + 1):
                if token.is_set():
                    return _cancelled(token)
                moved, error = reach(context, gp, cfg, token, reseat=attempt > 0)
                if error:
                    return error
                if moved:
                    gp, meta, error = perceive(context.perception, query, target, cancel=lambda: token.is_set(),
                                               expected_bounds=meta.get('visible_bounds'))
                    if error:
                        return error
                    gp = _grasp_point(gp, meta, cfg)
                error = grip(context, gp, cfg, token, target.label if world is not None and target.prompt else None)
                if error is None:
                    return SkillResult(True, {**meta, "query": query, "object_id": target.label or query, "attempts": attempt + 1, "diagnosis": "ok"})
                if error.diagnosis_code != "ik_unreachable" or context.gripper.is_holding():
                    return error
            return error
        except Exception as exc:
            return _cancelled(token) or _fail("device_failed", f"Grasp device failed: {failure_cause(exc)}; check the device and use look before retrying.")
