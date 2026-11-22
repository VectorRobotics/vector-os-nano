"""Observe a handle, establish a force-limited stop, slide, release and withdraw."""
import math
import threading

import numpy as np

from vector_os_nano.core.skill import skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.base import failure_cause
from vector_os_nano.schemas.robot import CameraSpec, EmbodimentConfigError, _keys, _require, _floats
from vector_os_nano.skills.grasp import _cancelled, _group_move, _move, _solve
from vector_os_nano.skills.grasp_config import _positive
from vector_os_nano.skills.grasp_perception import _fail, _Target, perceive

_POSITIVE = {'max_extent', 'tracking_tolerance', 'correction_limit', 'retreat', 'speed',
             'homing_speed', 'stroke_margin', 'homing_margin', 'endstop_force', 'max_force', 'backoff',
             'timeout', 'gain', 'goal_tolerance', 'dwell'}


def parse(raw, cfg, ctx):
    if (cfg.base is None or cfg.arm is None or cfg.arm.gripper is None
            or not cfg.arm.gripper.force_feedback or cfg.arm.gripper.assist):
        raise EmbodimentConfigError(f'{ctx}: requires base, arm and contact-only force-sensing gripper')
    row = _keys(raw, ctx, _POSITIVE | {'camera', 'view_posture', 'prepare', 'rotation', 'contact_offset', 'durations'})
    result = {k: _positive(_require(row, k, ctx), f'{ctx}.{k}') for k in _POSITIVE-{'homing_margin'}}
    result['homing_margin'] = _positive(row.get('homing_margin', result['stroke_margin']), f'{ctx}.homing_margin')
    for k, choices in (('camera', {s.name for s in cfg.sensors if isinstance(s, CameraSpec) and s.depth}),
                       ('view_posture', cfg.arm.postures)):
        value = _require(row, k, ctx)
        if not isinstance(value, str) or value not in choices:
            raise EmbodimentConfigError(f'{ctx}.{k}: expected one of {sorted(choices)}')
        result[k] = value
    prepare = _keys(_require(row, 'prepare', ctx), f'{ctx}.prepare', {'group', 'posture'})
    group, posture = prepare.get('group'), prepare.get('posture')
    if (not isinstance(group, str) or group not in cfg.joint_groups or
            not isinstance(posture, str) or posture not in cfg.joint_groups[group].postures):
        raise EmbodimentConfigError(f'{ctx}.prepare: expected a configured joint group and posture')
    result['prepare'] = dict(group=group, posture=posture)
    for k, size in (('rotation', 9), ('contact_offset', 3), ('durations', 4)):
        result[k] = _floats(_require(row, k, ctx), size, f'{ctx}.{k}')
    rotation = np.array(result['rotation']).reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or abs(np.linalg.det(rotation)-1)>1e-6:
        raise EmbodimentConfigError(f'{ctx}.rotation: expected a proper orthonormal rotation')
    if (min(result['durations']) <= 0 or result['endstop_force'] >= result['max_force']
            or result['homing_speed'] > result['speed'] or result['speed'] > cfg.base.limits[0]
            or result['tracking_tolerance'] >= result['correction_limit']
            or result['homing_margin'] < result['stroke_margin']
            or result['goal_tolerance'] >= result['backoff']):
        raise EmbodimentConfigError(f'{ctx}: inconsistent motion or force limits')
    return result


def _tip(arm):
    tip = np.asarray(arm.fk(arm.get_joint_positions())[0], dtype=float)
    if tip.shape != (3,) or not np.isfinite(tip).all():
        raise ValueError('end-effector position unavailable')
    return tip


def _reach(context, target, rotation, duration, cfg, token):
    command = np.array(target, dtype=float)
    for attempt in range(4):
        joints = _solve(context.arm, command.tolist(), context.arm.get_joint_positions(), rotation.tolist())
        if joints is None:
            return _fail('ik_unreachable', 'Handle approach is unreachable; choose another base/view pose.')
        error = _move(context.arm, joints, duration if attempt == 0 else 1., 'drawer approach', token)
        if error:
            return error
        residual = target-_tip(context.arm)
        if np.linalg.norm(residual) <= cfg['tracking_tolerance']:
            return None
        if np.linalg.norm(residual) > cfg['correction_limit']:
            break
        command += residual
    return _fail('move_failed', 'Handle approach did not track; inspect clearance before retrying.')


def _drive(context, forward, distance, cfg, token, *, home=False):
    """Bounded own-FK motion; a force stop is an estimate, never an object-state oracle."""
    try:
        if token.is_set():
            return _cancelled(token)
        start = _tip(context.arm)
        previous = -math.inf
        def velocity():
            nonlocal previous
            if token.is_set():
                raise ValueError('cancelled')
            sample = context.gripper.get_finger_load()
            if (sample is None or len(sample.force) != 3 or
                    not all(math.isfinite(v) for v in (*sample.force, sample.sim_time)) or
                    sample.sim_time <= previous or np.linalg.norm(sample.force) > cfg['max_force']):
                raise ValueError('fingertip force missing, stale, nonfinite or over limit')
            previous = sample.sim_time
            if not context.gripper.is_holding():
                raise ValueError('handle grip lost')
            progress = float((_tip(context.arm)-start) @ forward)
            if home:
                if progress > distance:
                    raise ValueError('no endstop within the bounded travel')
                if -np.dot(sample.force, forward) >= cfg['endstop_force']:
                    return None
                return cfg['homing_speed'], 0., 0.
            remaining = distance-progress
            if abs(remaining) <= cfg['goal_tolerance']:
                return None
            return float(np.clip(cfg['gain']*remaining, -cfg['speed'], cfg['speed'])), 0., 0.
        if not context.base.walk(duration=cfg['timeout'], velocity_fn=velocity):
            return _cancelled(token) or _fail('motion_failed', 'Drawer motion stopped or timed out; keep the current grasp.')
    except Exception as exc:
        return _cancelled(token) or _fail('contact_failed', f'Drawer feedback stopped motion: {failure_cause(exc)}; keep the current grasp.')
    finally:
        context.base.stop()
    return _cancelled(token)


def _operate(params, context, *, opening):
    cfg = context.config.get('skills', {}).get('drawer')
    if cfg is None:
        return _fail('not_configured', 'Configure skills.drawer with calibrated view, force and motion limits.')
    world = context.services.get('object_world')
    name = None if world is None else world.resolve_articulation(params.get('query'))
    if name is None:
        return _fail('unknown_articulation', f'Name exactly one declared drawer: {sorted(world.articulations) if world else []}.')
    token = context.services.get('cancel') or threading.Event()
    if token.is_set():
        return _cancelled(token)
    if context.gripper.is_holding():
        return _fail('already_holding', 'Place the held item safely before operating a drawer.')
    spec, observation = world.articulations[name], {}
    try:
        context.base.stop()
        error = _group_move(context, cfg['prepare'], cfg['durations'][0], 'drawer view', token)
        if error:
            return error
        if not context.gripper.open():
            return _fail('gripper_failed', 'Gripper did not open for the handle approach.')
        error = _move(context.arm, context.config['arm']['postures'][cfg['view_posture']], cfg['durations'][1], 'drawer view', token)
        if error:
            return error
        _, observation, error = perceive(context.perception, spec.handle_query,
            _Target(False, None, spec.handle_query, name), camera=cfg['camera'], cancel=token.is_set)
        if error:
            return error
        bounds = np.asarray(observation['visible_bounds'], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]) or np.any(bounds[1]-bounds[0] >= cfg['max_extent']):
            return _fail('implausible_target', 'Handle segmentation includes too much furniture; choose another view.')
        yaw = context.base.get_heading()
        if yaw is None or not math.isfinite(yaw):
            return _fail('no_pose', 'Base heading unavailable; keep stopped.')
        c, s = math.cos(yaw), math.sin(yaw)
        body = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        forward, rotation = body[:, 0], body @ np.array(cfg['rotation']).reshape(3, 3)
        target = bounds.mean(0) + body @ np.array(cfg['contact_offset'])
        error = _reach(context, target, rotation, cfg['durations'][2], cfg, token)
        if error:
            return error
        if token.is_set():
            return _cancelled(token)
        if not context.gripper.close() or not context.gripper.is_holding():
            return _fail('not_holding', 'No bilateral handle grip; observe again before moving.')
        # shortcut: force homing assumes an unobstructed slide; independent joint verification rejects an early obstruction.
        travel = abs(spec.opened-spec.closed)
        stroke = travel + cfg['stroke_margin']
        motions = [(travel + cfg['homing_margin'], True)]
        if opening:
            motions.extend([(-cfg['backoff'], False), (-stroke, False)])
        for distance, home in motions:
            error = _drive(context, forward, distance, cfg, token, home=home)
            if error:
                return error
        error = _move(context.arm, context.arm.get_joint_positions(), cfg['dwell'], 'drawer settle', token)
        if error:
            return error
        if not context.gripper.open():
            return _fail('gripper_failed', 'Handle was not released; keep the base stopped.')
        error = _reach(context, _tip(context.arm)-forward*cfg['retreat'], rotation, cfg['durations'][3], cfg, token)
        if error:
            return error
        error = _move(context.arm, context.arm.get_joint_positions(), cfg['dwell'], 'released drawer settle', token)
        return error or SkillResult(True, {'articulation': name, 'estimated_state': 'open' if opening else 'closed',
            'observation': observation, 'state_verification': 'independent joint predicate required',
            'uncertainty': 'force stop can be an obstruction; hidden handle geometry and sensor calibration unverified'})
    except Exception as exc:
        return _cancelled(token) or _fail('device_failed', f'Drawer operation failed: {failure_cause(exc)}.', observation=observation)
    finally:
        context.base.stop()


@skill(requires=('base', 'arm', 'gripper', 'camera', 'depth', 'joint_groups'), aliases=['open drawer', '打开抽屉'])
class OpenDrawerSkill:
    name = 'open_drawer'
    description = 'Experimental contact-only drawer opening. First face and approach its handle; uses fresh wrist RGB-D, force-limited homing, then pulls, releases and withdraws. Independently verify the resulting drawer state.'
    parameters = {'query': {'type': 'string', 'required': True, 'description': 'One declared drawer name.'}}
    verify_hint = 'articulation_open({query})'
    preconditions, postconditions, effects = [], [], {}
    failure_modes = ['not_configured', 'unknown_articulation', 'already_holding', 'cancelled',
                     'no_camera', 'no_detections', 'implausible_target', 'ik_unreachable',
                     'move_failed', 'not_holding', 'contact_failed', 'motion_failed', 'gripper_failed', 'device_failed']

    def execute(self, params, context):
        return _operate(params, context, opening=True)


@skill(requires=('base', 'arm', 'gripper', 'camera', 'depth', 'joint_groups'), aliases=['close drawer', '关闭抽屉'])
class CloseDrawerSkill(OpenDrawerSkill):
    name = 'close_drawer'
    description = 'Experimental contact-only drawer closing from a fresh wrist observation. Pushes slowly to a bounded force stop, settles, releases and withdraws. An obstruction can mimic closure; independently verify the joint state.'
    verify_hint = 'articulation_closed({query})'

    def execute(self, params, context):
        return _operate(params, context, opening=False)
