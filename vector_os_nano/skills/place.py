# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""One named-surface place: resolve, reach, hover, descend, check release, retreat."""
from __future__ import annotations
import math
import threading
from dataclasses import replace

import numpy as np

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import Pose3D, SkillResult
from vector_os_nano.perception.base import failure_cause
from vector_os_nano.skills.grasp import _cancelled, _move, _solve, face
from vector_os_nano.skills.grasp_perception import _fail, _Target, perceive
from vector_os_nano.skills.utils.approach_pose import compute_approach_pose
from vector_os_nano.skills import place_support


def target(params, context):
    world = context.services.get('object_world')
    name = params.get('surface')
    key = world.resolve_surface(name) if world is not None and isinstance(name, str) else None
    if key is None:
        return None, _fail('unknown_surface', f'Name a known surface; valid: {sorted(world.surfaces) if world else []}.')
    surface = world.surfaces[key]
    names = (surface.name, surface.name.replace('_', ' '), *getattr(surface, 'aliases', ()))
    if name.strip().casefold() not in {word.casefold() for word in names}:
        return None, _fail('unknown_surface', f'Name exactly one surface; valid: {sorted(world.surfaces)}.')
    if surface.place_point is None:
        return None, _fail('not_configured', f'Surface {key!r} has no declared place point.')
    return surface, None


def poses(arm, surface, cfg):
    x, y = surface.place_point
    heights = [surface.height + cfg[k] for k in ('hover_height', 'release_height')]
    heights.append(heights[1] + cfg['retreat_height'])
    rotation = arm.fk(arm.get_joint_positions())[1] if cfg['keep_orientation'] else None
    joints, seed = [], arm.get_joint_positions()
    for z in heights:
        seed = _solve(arm, (x, y, z), seed, rotation)
        if seed is None:
            return None
        joints.append(seed)
    return joints


def localize(surface, context, cfg, token):
    """Resolve a moving support from current pixels and its declared landmark geometry."""
    anchor = getattr(surface, 'local_anchor', None)
    if anchor is None:
        return surface, {}, None
    if context.perception is None:
        return None, {}, _fail('no_perception', 'A moving support requires a fresh RGB-D observation.')
    _, observation, error = perceive(context.perception, anchor.query,
        _Target(False, None, anchor.query, surface.name), camera=cfg.get('camera'), cancel=token.is_set)
    if error:
        return None, observation, error
    bounds = np.asarray(observation['visible_bounds'], dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or
            np.any(bounds[1] <= bounds[0]) or np.any(bounds[1]-bounds[0] >= anchor.max_extent)):
        return None, observation, _fail('implausible_target', 'Support landmark segmentation is not plausible; choose another view.')
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    point = bounds.mean(0) + rotation @ (np.array([*surface.place_point, surface.height])-anchor.point)
    # shortcut: orientation is authored for a braked support; use a pose estimator when furniture can rotate.
    return replace(surface, place_point=tuple(point[:2]), height=float(point[2])), observation, None


def reach(context, surface, cfg, token):
    joints = poses(context.arm, surface, cfg)
    if joints is not None:
        return joints, None
    base, bc = context.base, cfg.get('base')
    if base is None or bc is None:
        tip, _ = context.arm.fk(context.arm.get_joint_positions())
        return None, _fail('out_of_reach', 'No reachable place trajectory; choose a reachable surface.',
                           reach_m=cfg['reach_m'], distance=math.dist(tip, (*surface.place_point, surface.height)))
    carry = context.config['arm']['postures'][bc['carry_posture']]
    error = _move(context.arm, carry, cfg['durations'][0], 'carry', token)
    if error:
        return None, error
    if not context.gripper.is_holding():
        return None, _fail('not_holding', 'Object slipped before navigation; observe it again.')
    # Recheck once after turning: braking can move the mounted arm out of reach.
    for _ in range(2):
        pos, yaw = base.get_position(), base.get_heading()
        if pos is None or yaw is None or not all(math.isfinite(v) for v in (*pos[:2], yaw)):
            return None, _fail('approach_failed', 'Base pose is unknown; localize before placing.')
        x, y, heading = compute_approach_pose((*surface.place_point, surface.height), (*pos[:2], yaw), bc['standoff'])
        x += math.sin(heading)*bc['lateral_offset']
        y -= math.cos(heading)*bc['lateral_offset']
        if not base.navigate_to(x, y, tol=cfg['position_tolerance'], cancel=token):
            return None, _cancelled(token) or _fail('approach_failed', 'Cannot reach the surface standoff; holding the current grasp.')
        if token.is_set():
            return None, _cancelled(token)
        error = face(base, Pose3D(*surface.place_point, surface.height), bc, token)
        if error:
            return None, error
        if not context.gripper.is_holding():
            return None, _fail('not_holding', 'Object slipped during navigation; observe it again.')
        joints = poses(context.arm, surface, cfg)
        if joints is not None:
            return joints, None
    return None, _fail('ik_unreachable', 'Surface remains unreachable from the approached pose; keep holding and reposition.')



@skill(requires=('arm', 'gripper'), aliases=['put', 'drop', 'put down', '放', '放下', '放到', '放置', '放在'])
class PlaceSkill:
    name = 'place'
    description = 'Place the held object on a named surface; observe moving supports afresh, descend, check support, release and retreat.'
    parameters = {'surface': {'type': 'string', 'required': True, 'description': 'Named support surface from the world manifest.'}}
    verify_hint = "object_on('<object>', '<surface>')"
    preconditions = ['gripper_holding_any']
    postconditions = ['gripper_empty']
    effects = {'gripper_state': 'open', 'held_object': None}
    failure_modes = ['no_arm', 'no_gripper', 'unknown_surface', 'not_configured', 'not_holding',
                     'out_of_reach', 'approach_failed', 'ik_unreachable', 'move_failed', 'release_failed', 'support_unconfirmed',
                     'no_perception', 'implausible_target', 'cancelled', 'device_failed']

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        token = context.services.get('cancel') or threading.Event()
        if token.is_set():
            return _cancelled(token)
        if context.arm is None:
            return _fail('no_arm', 'Connect an arm before placing.')
        if context.gripper is None:
            return _fail('no_gripper', 'Connect a gripper before placing.')
        try:
            surface, error = target(params, context)
            if error:
                return error
            cfg = context.config.get('skills', {}).get('place')
            if not cfg:
                return _fail('not_configured', 'Configure skills.place in this embodiment manifest.')
            if not context.gripper.is_holding():
                return _fail('not_holding', 'The gripper holds no object; nothing was placed.')
            source = surface
            surface, observation, error = localize(source, context, cfg, token)
            if error:
                return error
            if getattr(source, 'local_anchor', None) is not None and not context.gripper.is_holding():
                return _fail('not_holding', 'Object slipped while observing the support; keep the base stopped and observe the object.')
            support = cfg.get('support')
            initial = place_support.read(context.gripper, support) if support else None
            joints, error = reach(context, surface, cfg, token)
            if error:
                return error
            if getattr(source, 'local_anchor', None) is not None:
                surface, observation, error = localize(source, context, cfg, token)
                if error:
                    return error
                if not context.gripper.is_holding():
                    return _fail('not_holding', 'Object slipped while observing the support; keep the base stopped and observe the object.')
                joints = poses(context.arm, surface, cfg)
                if joints is None:
                    return _fail('ik_unreachable', 'The freshly observed support is unreachable; keep holding and reposition.')
            rotation = context.arm.fk(context.arm.get_joint_positions())[1] if cfg['keep_orientation'] else None
            for height, duration, phase in zip(('hover_height', 'release_height'), cfg['durations'][:2], ('approach', 'descend')):
                position = (*surface.place_point, surface.height + cfg[height])
                q = _solve(context.arm, position, context.arm.get_joint_positions(), rotation)
                if q is None:
                    return _fail('ik_unreachable', f'Place trajectory became unreachable during {phase}; keep holding.', phase=phase)
                error = _move(context.arm, q, duration, phase, token)
                if error:
                    return error
                if not context.gripper.is_holding():
                    return _fail('not_holding', f'Object slipped during {phase}; observe it again.')
                if support and phase == 'approach':
                    # The extended arm can move its mobile mount while reaching the hover.
                    q = _solve(context.arm, position, context.arm.get_joint_positions(), rotation)
                    if q is None:
                        return _fail('ik_unreachable', 'Hover became unreachable after approach; keep holding.', phase=phase)
                    error = _move(context.arm, q, duration, phase, token)
                    if error:
                        return error
                    error = _move(context.arm, q, support['dwell'], 'load measurement', token)
                    if error:
                        return error
                    baseline = place_support.read(context.gripper, support, initial[1])
                    if baseline[0] < support['minimum_load']:
                        return _fail('support_unconfirmed', 'No measurable suspended load; inspect before releasing.')
                    break  # Measure force throughout descent, including contact above the nominal release height.
            joints[1] = q
            tip, _ = context.arm.fk(context.arm.get_joint_positions())
            if len(tip) != 3 or not all(math.isfinite(v) for v in tip) or math.dist(tip, position) > cfg['position_tolerance']:
                return _fail('move_failed', f'Place position not reached during {phase}; keep holding.', phase=phase)
            if support:
                joints[1], error = place_support.confirm(context, joints[1], position, support,
                    baseline, cfg['position_tolerance'], token,
                    floor=surface.height+cfg['release_height']-support['max_descent'])
                if error:
                    return error
                position = tuple(context.arm.fk(context.arm.get_joint_positions())[0])
            if token.is_set():
                return _cancelled(token)
            if not context.gripper.open():
                return _fail('release_failed', 'Gripper did not complete its open command.')
            error = _move(context.arm, joints[1], cfg['settle_duration'], 'settle', token)
            if error:
                return error
            if context.gripper.is_holding():
                return _fail('release_failed', 'Gripper still reports a held object; do not pull it away.')
            retreat = (*surface.place_point, surface.height + cfg['release_height'] + cfg['retreat_height'])
            q = _solve(context.arm, retreat, context.arm.get_joint_positions(), rotation)
            if q is None:
                return _fail('ik_unreachable', 'Object released, but retreat became unreachable; inspect before moving.', phase='retreat')
            error = _move(context.arm, q, cfg['durations'][2], 'retreat', token)
            if error:
                return error
            if context.gripper.is_holding():
                return _fail('release_failed', 'Gripper reports a load after retreat; inspect the placement.')
            return SkillResult(True, dict(diagnosis='ok', surface=surface.name,
                placed_at=[*surface.place_point, surface.height],
                target_source='fresh RGB-D landmark with authored orientation' if observation else 'authored world manifest',
                observation=observation,
                release_position=list(position), support_confirmed=bool(support)))
        except Exception as exc:
            return _cancelled(token) or _fail('device_failed', f'Place device failed: {failure_cause(exc)}; inspect before retrying.')
