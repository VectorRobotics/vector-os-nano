"""Independent contact-only drawer opening, closure and release on a free cart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from scripts.grade_grasp import integration_error
from scripts.grade_navigation import grade as grade_safety
from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.backends.mujoco.recording import read_recording


CRITERIA = dict(open_hold_s=.5, closed_hold_s=2., stopped_speed_m_s=.005,
                cart_drift_m=.005, cart_tilt_deg=5., drawer_speed_m_s=.08,
                max_finger_force_N=7., max_net_hand_force_N=6.,
                min_compression_N=.05, opposing_cosine=-.5,
                handle_padding_m=.001, integration_error_m=1e-6)


def _rotation(quaternion):
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1., atol=1e-6, rtol=0):
        raise ValueError('recorded body quaternions must be normalized')
    w, x, y, z = np.asarray(quaternion).T
    return np.stack((np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), axis=-1),
                     np.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), axis=-1),
                     np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1)), axis=-2)


def drawer_pose(model, qpos, body):
    """Reconstruct the declared slide below a free cart, without moving a simulation."""
    chain, parent = [], body
    while parent and not (model.body_jntnum[parent] == 1 and
            model.jnt_type[model.body_jntadr[parent]] == mj.mjtJoint.mjJNT_FREE):
        chain.append(parent)
        parent = int(model.body_parentid[parent])
    if not parent or model.body_parentid[parent] != 0:
        raise ValueError('drawer acceptance requires a free cart rooted in the world')
    address = int(model.jnt_qposadr[model.body_jntadr[parent]])
    position, rotation = qpos[:, address:address+3].copy(), _rotation(qpos[:, address+3:address+7])
    for item in reversed(chain):
        position += np.einsum('tij,j->ti', rotation, model.body_pos[item])
        rotation = rotation @ _rotation(model.body_quat[item])
        count, joint = int(model.body_jntnum[item]), int(model.body_jntadr[item])
        if count:
            if item != body or count != 1 or model.jnt_type[joint] != mj.mjtJoint.mjJNT_SLIDE:
                raise ValueError('only the declared drawer may articulate below the cart')
            qa = int(model.jnt_qposadr[joint])
            position += np.einsum('tij,j->ti', rotation, model.jnt_axis[joint]) * (qpos[:, qa]-model.qpos0[qa])[:, None]
    return position, rotation, address


def _handle_bounds(model, geom):
    if model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom]); start = int(model.mesh_vertadr[mesh])
        vertices = model.mesh_vert[start:start+int(model.mesh_vertnum[mesh])]
    elif model.geom_type[geom] == mj.mjtGeom.mjGEOM_BOX:
        vertices = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*model.geom_size[geom]
    else:
        raise ValueError('handle acceptance requires mesh or box geometry')
    vertices = vertices @ _rotation(model.geom_quat[geom]).T + model.geom_pos[geom]
    return vertices.min(axis=0), vertices.max(axis=0)


def drawer_checks(model, states, contacts, cfg, spec):
    joint, body = model.joint(spec.joint).id, model.body(spec.body).id
    if (model.jnt_bodyid[joint] != body or model.jnt_type[joint] != mj.mjtJoint.mjJNT_SLIDE
            or not model.jnt_limited[joint] or spec.handle_geom is None):
        raise ValueError('drawer needs its own limited slide and a declared handle')
    geom = model.geom(spec.handle_geom).id
    if model.geom_bodyid[geom] != body or cfg.arm.gripper.assist:
        raise ValueError('drawer acceptance requires an attached handle and contact-only gripper')
    t, q = states['sim_time'][:, 0], states['qpos'][:, model.jnt_qposadr[joint]]
    v = states['qvel'][:, model.jnt_dofadr[joint]]
    if len(t) < 2 or np.any(np.diff(t) < 0) or np.any((np.diff(t) == 0) & (t[1:] != 0)):
        raise ValueError('drawer grade requires one continuous reset episode')
    direction = np.sign(spec.opened-spec.closed)
    opened = (q-spec.closed)*direction >= abs(spec.opened-spec.closed)
    n_open = int(np.ceil(CRITERIA['open_hold_s']/model.opt.timestep))+1
    sustained_open = len(opened) >= n_open and bool(np.any(np.convolve(opened.astype(int), np.ones(n_open, int), 'valid') == n_open))
    final = t >= t[-1]-CRITERIA['closed_hold_s']-1e-8
    stable_close = (t[final][-1]-t[final][0] >= CRITERIA['closed_hold_s']-1e-8 and
                    np.all(abs(q[final]-spec.closed) <= spec.closed_tolerance) and
                    np.all(abs(v[final]) <= CRITERIA['stopped_speed_m_s']))
    position, rotation, cart_address = drawer_pose(model, states['qpos'], body)
    cart = states['qpos'][:, cart_address:cart_address+7]
    drift = float(np.linalg.norm(cart[:, :2]-cart[0, :2], axis=1).max())
    tilt = float(np.degrees(np.arccos(np.clip(_rotation(cart[:, 3:])[:, 2, 2], -1, 1))).max())
    error = float(np.max(abs(np.diff(q)-v[1:]*np.diff(t))))
    checks = dict(initially_closed=abs(float(q[0])-spec.closed) <= spec.closed_tolerance,
                  sustained_open=bool(sustained_open), closed_and_settled=bool(stable_close),
                  passive_drawer=not bool(np.any((model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT) & (model.actuator_trnid[:, 0] == joint))),
                  drawer_limits=bool(np.all(q >= model.jnt_range[joint, 0]-.0001) and np.all(q <= model.jnt_range[joint, 1]+.0001)),
                  drawer_speed=float(abs(v).max()) <= CRITERIA['drawer_speed_m_s'],
                  integrated_drawer_motion=error <= CRITERIA['integration_error_m'],
                  integrated_cart_motion=integration_error(model, states, cart_address) <= CRITERIA['integration_error_m'],
                  cart_stable=drift <= CRITERIA['cart_drift_m'] and tilt <= CRITERIA['cart_tilt_deg'])
    metrics = dict(open_stroke_m=float(((q-spec.closed)*direction).max()),
                   final_close_error_m=float(abs(q[-1]-spec.closed)), cart_drift_m=drift, cart_tilt_deg=tilt,
                   peak_drawer_speed_m_s=float(abs(v).max()), max_drawer_integration_error_m=error)
    checks.update(dict.fromkeys(('handle_only', 'opposed_grip_during_motion', 'finger_force', 'net_hand_force', 'released'), None))
    if not contacts or not len(contacts['sim_time']):
        return checks, metrics
    ct = contacts['sim_time'][:, 0]; indices = np.searchsorted(t, ct)
    if np.any(indices >= len(t)) or np.any(abs(t[indices]-ct) > 1e-8):
        raise ValueError('contact times must match recorded physics steps')
    a, b = contacts['body_a'][:, 0].astype(int), contacts['body_b'][:, 0].astype(int)
    fingers = [model.body(name).id for name in cfg.arm.gripper.finger_bodies]
    pair = ((a == body) & np.isin(b, fingers)) | ((b == body) & np.isin(a, fingers))
    active = np.linalg.norm(contacts['force_on_b'], axis=1) > CRITERIA['min_compression_N']
    selected = pair & active
    local = np.einsum('tji,tj->ti', rotation[indices[selected]], contacts['position'][selected]-position[indices[selected]])
    lo, hi = _handle_bounds(model, geom)
    checks['handle_only'] = bool(np.all(local >= lo-CRITERIA['handle_padding_m']) and np.all(local <= hi+CRITERIA['handle_padding_m']))
    force = contacts['force_on_b']*np.where((b == body)[:, None], 1, -1)
    normal = contacts['normal_on_b']*np.where((b == body)[:, None], 1, -1)
    compression = np.maximum(0, np.sum(force*normal, axis=1))
    normals, peaks = [], []
    for finger in fingers:
        mask = pair & ((a == finger) | (b == finger))
        loads = np.zeros(len(t)); np.add.at(loads, indices[mask], np.linalg.norm(force[mask], axis=1))
        peaks.append(float(loads.max()))
        sums = np.zeros((len(t), 3)); np.add.at(sums, indices[mask], normal[mask]*compression[mask, None])
        normals.append(sums)
    magnitudes = np.linalg.norm(normals, axis=2)
    opposed = np.zeros(len(t), bool)
    for i in range(len(fingers)):
        for j in range(i):
            opposed |= np.sum(normals[i]*normals[j], axis=1) < CRITERIA['opposing_cosine']*magnitudes[i]*magnitudes[j]
    # A released stationary drawer must remain open for loading; opening alone is not a grasp requirement.
    required = abs(v) > CRITERIA['stopped_speed_m_s']
    checks['opposed_grip_during_motion'] = bool(required.any() and np.all((np.all(magnitudes > CRITERIA['min_compression_N'], axis=0) & opposed)[required]))
    touching = np.any(magnitudes > CRITERIA['min_compression_N'], axis=0)
    edges = np.diff(np.r_[False, ~touching, False].astype(int))
    checks['released_open_stable'] = all(
        not np.any(opened[a:b]) or (np.all(opened[a:b]) and np.ptp(q[a:b]) <= spec.closed_tolerance
                                   and np.max(abs(v[a:b])) <= CRITERIA['stopped_speed_m_s'])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    net = np.zeros((len(t), 3))
    sign = np.isin(b, fingers).astype(int)-np.isin(a, fingers).astype(int)
    np.add.at(net, indices, contacts['force_on_b']*sign[:, None])
    net_peak = float(np.linalg.norm(net, axis=1).max())
    checks['finger_force'] = max(peaks) <= CRITERIA['max_finger_force_N']
    checks['net_hand_force'] = net_peak <= CRITERIA['max_net_hand_force_N']
    checks['released'] = not bool(np.any(selected & (ct >= t[final][0])))
    metrics.update(finger_peak_force_N=dict(zip(cfg.arm.gripper.finger_bodies, peaks)), net_hand_peak_force_N=net_peak)
    return checks, metrics


def grade(directory, cfg, world, audit, goal, articulation):
    spec = world.articulations[articulation]
    result = grade_safety(directory, cfg, world, audit, goal, grasp_target=spec.body)
    meta, states, contacts = read_recording(directory)
    model = mj.MjModel.from_binary_path(meta['model_path'])
    checks, metrics = drawer_checks(model, states, contacts, cfg, spec)
    result['checks'].update(checks)
    result.update(scope=__doc__, articulation=articulation, drawer_criteria={**CRITERIA, 'closed': spec.closed,
        'opened': spec.opened, 'closed_tolerance': spec.closed_tolerance}, drawer_metrics=metrics)
    result['physical_verdict'] = ('fail' if False in result['checks'].values() else
                                 'unverified' if None in result['checks'].values() else 'pass')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--robot', required=True); p.add_argument('--world', required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--goal', type=float, nargs=2, required=True)
    p.add_argument('--articulation', required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = grade(a.recording, registry.load(a.robot), load_world(a.world), json.loads(a.audit.read_text()), a.goal, a.articulation)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('checks', 'physical_verdict', 'motion_verdict')}))
    raise SystemExit(0 if result['physical_verdict'] == 'pass' else 1)
