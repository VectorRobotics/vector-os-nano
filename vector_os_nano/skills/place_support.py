"""Confirm load transfer using a declared fingertip sensor, without object truth."""
import math
from vector_os_nano.skills.grasp import _move, _solve
from vector_os_nano.skills.grasp_perception import _fail


def read(gripper, cfg, after=-math.inf):
    sample = gripper.get_finger_load()
    if (sample is None or len(sample.force) != 3 or
            not all(math.isfinite(v) for v in (*sample.force, sample.sim_time)) or sample.sim_time <= after):
        raise ValueError('fingertip force sensor missing, stale or nonfinite')
    if math.sqrt(sum(v*v for v in sample.force)) > cfg['max_force']:
        raise ValueError('fingertip resultant force exceeds the configured limit')
    return -sample.force[2], sample.sim_time


def confirm(context, joints, position, cfg, baseline, tolerance, token, *, floor):
    """Search for unloaded fingertips, backing off compression; require a sustained dwell."""
    arm, grip = context.arm, context.gripper
    weight, previous = baseline
    rotation = arm.fk(arm.get_joint_positions())[1]
    ceiling = position[2]
    z, lower, upper = position[2], None, None
    count = math.ceil((ceiling-floor)/cfg['step'])
    for attempt in range(2*count+1):
        target = (*position[:2], z)
        if attempt:
            joints = _solve(arm, target, joints, rotation)
            if joints is None:
                return None, _fail('support_unconfirmed', 'No reachable support approach; keep holding.')
        error = _move(arm, joints, cfg['duration'], 'support approach', token)
        if error:
            return None, error
        tip, _ = arm.fk(arm.get_joint_positions())
        if not all(math.isfinite(v) for v in tip) or math.dist(tip, target)>tolerance:
            return None, _fail('move_failed', 'Support approach did not track; keep holding.')
        for sample in range(3):
            if sample:
                error = _move(arm, joints, cfg['dwell']/2, 'support dwell', token)
                if error:
                    return None, error
            load, previous = read(grip, cfg, previous)
            if not grip.is_holding():
                return None, _fail('not_holding', 'Grasp lost during support detection; inspect the object.')
            if abs(load)>weight*cfg['unload_fraction']:
                break
        else:
            return joints, None
        if load > 0:
            upper = z
            if lower is not None and lower >= z:
                lower = None  # contact geometry can settle while the load is held
            next_z = max(floor, z-cfg['step']) if lower is None else (lower+upper)/2
        else:
            lower = z
            if upper is not None and upper <= z:
                upper = None
            next_z = min(ceiling, z+cfg['step']) if upper is None else (lower+upper)/2
        if next_z == z:
            break
        z = next_z
    return None, _fail('support_unconfirmed', 'No sustained load transfer within the bounded search; keep the base stopped and lift clear before repositioning.', fingertip_load_N=load)
