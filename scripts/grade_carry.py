"""Independent contact-only transport checks over a declared continuous interval."""
import math
import numpy as np
from scripts.grade_grasp import package_trajectory, contact_support, CRITERIA as GRASP

CRITERIA = dict(minimum_displacement_m=.12, return_error_m=.04, clearance_m=.05)


def carry_window(model, states, contacts, cfg, world, target, start, end):
    t = states['sim_time'][:, 0]
    if (not all(math.isfinite(v) for v in (start, end)) or end-start < .5 or
            start < t[0] or end > t[-1] or cfg.arm.gripper.assist):
        raise ValueError('carry interval must cover recorded contact-only transport')
    mask = (t >= start) & (t <= end)
    sample = {k: v[mask] for k, v in states.items()}
    times = sample['sim_time'][:, 0]
    if len(times) < 2 or times[0]-start > model.opt.timestep*1.01 or end-times[-1] > model.opt.timestep*1.01:
        raise ValueError('carry interval is not fully sampled')
    loads = {k: v[(contacts['sim_time'][:, 0] >= start) &
                  (contacts['sim_time'][:, 0] <= end)] for k, v in contacts.items()} if contacts else {}
    body, address, corners = package_trajectory(model, sample, target)
    fingers = [model.body(n).id for n in cfg.arm.gripper.finger_bodies]
    checks, metrics = contact_support(model, sample, loads, body, fingers, hold_s=times[-1]-times[0])
    root = model.body(cfg.model.root_body).id
    base_q = int(model.jnt_qposadr[model.body_jntadr[root]])
    base = sample['qpos'][:, base_q:base_q+2]
    obj = sample['qpos'][:, address:address+2]
    metrics.update(base_excursion_m=float(np.linalg.norm(base-base[0], axis=1).max()),
        object_excursion_m=float(np.linalg.norm(obj-obj[0], axis=1).max()),
        return_error_m=float(np.linalg.norm(base[-1]-base[0])),
        minimum_clearance_m=float(corners[:, :, 2].min()-world.surfaces[world.pickables[target].surface].height),
        duration_s=float(times[-1]-times[0]))
    checks.update(transported=all(metrics[k] >= CRITERIA['minimum_displacement_m'] for k in
                                 ('base_excursion_m', 'object_excursion_m')),
        returned=metrics['return_error_m'] <= CRITERIA['return_error_m'],
        stayed_clear=bool(np.all(corners[:, :, 2] >=
            world.surfaces[world.pickables[target].surface].height+CRITERIA['clearance_m'])),
        no_other_support=None)
    if loads:
        a, b = loads['body_a'][:, 0], loads['body_b'][:, 0]
        significant = np.linalg.norm(loads['force_on_b'], axis=1) > GRASP['min_compression_N']
        checks['no_other_support'] = not bool(np.any(((a == body) | (b == body)) & significant &
            ~np.isin(np.where(a == body, b, a), fingers)))
    return checks, metrics
