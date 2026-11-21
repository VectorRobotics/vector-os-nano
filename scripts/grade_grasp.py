"""Independent force-only lift acceptance for declared rigid box packages, over every physics step."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from scripts.grade_navigation import grade as grade_safety
from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.backends.mujoco.recording import read_recording


CRITERIA = dict(clearance_m=.03, hold_s=2., max_finger_force_N=7., min_compression_N=.05,
                opposing_cosine=-.5, supported_weight_fraction=.8, initial_support_tolerance_m=.02,
                initially_empty_s=.2, object_state_step_error=1e-6)


def contact_support(model, states, contacts, target, fingers, *, hold_s=2.):
    """Reconstruct loads on the object; missing steps or same-side touches cannot prove a hold."""
    t = states['sim_time'][:, 0]
    keep = t >= t[-1]-hold_s-1e-8
    times = t[keep]
    if not contacts or not len(contacts['sim_time']) or len(times) < 2 or times[-1]-times[0] < hold_s-1e-8:
        return dict(sustained_opposing_loads=False, finger_force_limit=None, load_support=None), {}
    dt = float(model.opt.timestep)
    ticks = np.rint(times/dt).astype(int)
    ct = np.rint(contacts['sim_time'][:, 0]/dt).astype(int)
    if (np.any(contacts['sim_time'][:, 0] < t[0]-1e-8) or
            np.any(contacts['sim_time'][:, 0] > t[-1]+1e-8) or
            np.any(abs(contacts['sim_time'][:, 0]-ct*dt) > 1e-8)):
        raise ValueError('contact times must lie on recorded physics steps')
    unique_ticks, tick_indices = np.unique(ct, return_inverse=True)
    a, b = contacts['body_a'][:, 0].astype(int), contacts['body_b'][:, 0].astype(int)
    normals = contacts['normal_on_b']*np.where((b == target)[:, None], 1, -1)
    forces = contacts['force_on_b']*np.where((b == target)[:, None], 1, -1)
    compression = np.maximum(0., np.sum(forces*normals, axis=1))
    loads, normal_sums, peaks = [], [], {}
    for finger in fingers:
        pair = ((a == target) & (b == finger)) | ((b == target) & (a == finger))
        all_sum = np.zeros(len(unique_ticks))
        np.add.at(all_sum, tick_indices[pair], np.linalg.norm(forces[pair], axis=1))
        peaks[model.body(int(finger)).name] = float(all_sum.max())
        forces_in_window = np.zeros((len(times), 3))
        normals_in_window = np.zeros((len(times), 3))
        mask = pair & (ct >= ticks[0]) & (ct <= ticks[-1]) & (compression > CRITERIA['min_compression_N'])
        indices = np.searchsorted(ticks, ct[mask])
        exact = indices < len(times)
        indices = indices[exact]
        np.add.at(forces_in_window, indices, forces[mask][exact])
        np.add.at(normals_in_window, indices, normals[mask][exact]*compression[mask][exact, None])
        loads.append(forces_in_window)
        normal_sums.append(normals_in_window)
    norms = np.linalg.norm(normal_sums, axis=2)
    bilateral = np.all(norms > CRITERIA['min_compression_N'], axis=0)
    opposed = np.zeros(len(times), dtype=bool)
    for i in range(len(fingers)):
        for j in range(i):
            dot = np.sum(normal_sums[i]*normal_sums[j], axis=1)
            opposed |= dot < CRITERIA['opposing_cosine']*norms[i]*norms[j]
    weight = float(model.body_mass[target]*np.linalg.norm(model.opt.gravity))
    supported = float(np.sum(loads, axis=0)[:, 2].mean())
    return dict(sustained_opposing_loads=bool(np.all(np.diff(ticks)==1) and np.all(bilateral & opposed)),
                finger_force_limit=all(v <= CRITERIA['max_finger_force_N'] for v in peaks.values()),
                load_support=supported >= weight*CRITERIA['supported_weight_fraction']), dict(
                    finger_peak_force_N=peaks, mean_gripper_support_N=supported, target_weight_N=weight)


def grade(directory, cfg, world, audit, goal, target):
    result = grade_safety(directory, cfg, world, audit, goal, grasp_target=target)
    meta, states, contacts = read_recording(directory)
    m = mj.MjModel.from_binary_path(meta['model_path'])
    g = cfg.arm.gripper
    if g.assist:
        raise ValueError('force-only acceptance requires gripper.assist=false')
    body = m.body(target).id
    joint = int(m.body_jntadr[body])
    if joint < 0 or m.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE or m.body_parentid[body] != 0:
        raise ValueError('lift acceptance requires a free package attached to the world')
    geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == body and (m.geom_contype[i] or m.geom_conaffinity[i])]
    if not geoms or any(m.geom_type[i] != mj.mjtGeom.mjGEOM_BOX for i in geoms):
        raise ValueError('this acceptance supports box collision geometry only')
    corners = []
    signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for i in geoms:
        rotation = np.empty(9)
        mj.mju_quat2Mat(rotation, m.geom_quat[i])
        corners.extend(signs*m.geom_size[i] @ rotation.reshape(3, 3).T+m.geom_pos[i])
    q = states['qpos'][:, int(m.jnt_qposadr[joint]):int(m.jnt_qposadr[joint])+7]
    w, x, y, z = q[:, 3:].T
    row_z = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1)
    bottom = q[:, 2] + np.min(row_z @ np.asarray(corners).T, axis=1)
    support = world.surfaces[world.pickables[target].surface].height
    t = states['sim_time'][:, 0]
    window = t >= t[-1]-CRITERIA['hold_s']-1e-8
    fingers = [m.body(n).id for n in g.finger_bodies]
    checks, loads = contact_support(m, states, contacts, body, fingers, hold_s=CRITERIA['hold_s'])
    opening = states['qpos'][window, int(m.jnt_qposadr[m.joint(g.joint).id])]
    checks.update(lift_clearance=bool(np.all(bottom[window]-support >= CRITERIA['clearance_m'])),
                  jaw_held_open=bool(np.all(abs(opening-g.closed) >= g.min_opening)),
                  initially_supported=abs(float(bottom[0])-support) <= CRITERIA['initial_support_tolerance_m'])
    error = 0.
    address = int(m.jnt_qposadr[joint])
    for i in np.flatnonzero(np.diff(t)>0)+1:
        expected = states['qpos'][i-1].copy()
        mj.mj_integratePos(m, expected, states['qvel'][i], float(t[i]-t[i-1]))
        error = max(error, float(np.max(abs(expected[address:address+7]-q[i]))))
    checks['integrated_object_motion'] = error <= CRITERIA['object_state_step_error']
    checks['initially_ungrasped'] = checks['no_other_load_support'] = None
    if contacts:
        a, b = contacts['body_a'][:, 0].astype(int), contacts['body_b'][:, 0].astype(int)
        active = np.linalg.norm(contacts['force_on_b'], axis=1) > CRITERIA['min_compression_N']
        touching = active & ((a == body) | (b == body))
        on_fingers = np.isin(np.where(a == body, b, a), fingers)
        checks['initially_ungrasped'] = not bool(np.any(touching & on_fingers &
            (contacts['sim_time'][:, 0] <= t[0]+CRITERIA['initially_empty_s'])))
        checks['no_other_load_support'] = not bool(np.any(touching & ~on_fingers &
            (contacts['sim_time'][:, 0] >= t[-1]-CRITERIA['hold_s']-1e-8)))
    result['checks'].update(checks)
    result.update(scope=__doc__, target=target, grasp_criteria=CRITERIA, **loads,
                  max_object_state_step_error=error,
                  minimum_held_clearance_m=float((bottom[window]-support).min()))
    result['physical_verdict'] = ('fail' if False in result['checks'].values() else
                                 'unverified' if None in result['checks'].values() else 'pass')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--robot', required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--goal', type=float, nargs=2, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();cfg = registry.load(a.robot)
    result = grade(a.recording, cfg, load_world(cfg.world), json.loads(a.audit.read_text()), a.goal, a.target)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('checks', 'physical_verdict', 'motion_verdict')}))
    raise SystemExit(0 if result['physical_verdict'] == 'pass' else 1)
