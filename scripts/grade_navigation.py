"""Independent wheel-contact navigation checks over a complete recorded rollout."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco as mj
import numpy as np

from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.backends.mujoco.recording import read_recording


# Fixed before evaluation; motion quality is judged separately by continuous video.
CRITERIA = dict(goal_error_m=.15, stopped_speed_m_s=.03, tilt_deg=5.,
                penetration_m=.0025, joint_limit_slack=.0001, object_drop_m=.02,
                support_height_m=.02, support_normal_z=.94, significant_contact_N=.05,
                integration_error_m=1e-6, supported_weight_fraction=.9, base_speed_slack=.01)


def grade(directory, cfg, world, audit, goal, *, grasp_target=None):
    meta, states, contacts = read_recording(directory)
    if cfg.base is None or cfg.base.policy.ref != 'swerve':
        raise ValueError('this acceptance checks swerve wheel contacts')
    path = meta['model_path']
    model = mj.MjModel.from_binary_path(path) if path.endswith('.mjb') else mj.MjModel.from_xml_path(path)
    root = model.body(cfg.model.root_body).id
    joint = int(model.body_jntadr[root])
    if model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError('navigation requires the recorded free robot root')
    qi, vi = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    q, v, t = states['qpos'], states['qvel'], states['sim_time'][:, 0]
    quat = q[:, qi+3:qi+7]
    tilt = np.degrees(np.arccos(np.clip(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1, 1)))
    error = float(np.linalg.norm(q[-1, qi:qi+2]-goal))
    stopped = float(np.linalg.norm(v[-1, vi:vi+3]))
    checks = dict(arrived=error<=CRITERIA['goal_error_m'], stopped=stopped<=CRITERIA['stopped_speed_m_s'],
                  tilt=float(tilt.max())<=CRITERIA['tilt_deg'],
                  continuous_steps=bool(len(t)>1 and np.all(np.diff(t)>=0) and
                      np.all(np.diff(t)<=meta['timestep']*1.001) and
                      np.all((np.diff(t)>0)|(t[1:]==0))),
                  no_welds=not bool(np.any(states['eq_active'][:, model.eq_type==mj.mjtEq.mjEQ_WELD])))
    residual = states.get('max_contact_residual', states.get('max_penetration'))
    speculative = bool(model.opt.enableflags & int(getattr(mj.mjtEnableBit, 'mjENBL_SPECULATIVE', 0)))
    checks['penetration'] = None if speculative else float(residual.max())<=CRITERIA['penetration_m']
    moving_time = np.diff(t)>0
    integration_error = np.linalg.norm(np.diff(q[:, qi:qi+3], axis=0)-v[1:, vi:vi+3]*np.diff(t)[:, None], axis=1)
    checks['integrated_root_motion'] = bool(np.all(integration_error[moving_time]<=CRITERIA['integration_error_m']))
    checks['no_applied_forces'] = (bool(np.all(states['max_applied_force']==0)) if 'max_applied_force' in states else None)
    w, x, y, z = quat.T
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    vx = np.cos(yaw)*v[:, vi]+np.sin(yaw)*v[:, vi+1]
    vy = -np.sin(yaw)*v[:, vi]+np.cos(yaw)*v[:, vi+1]
    checks['base_speed'] = bool(all(float(np.abs(speed).max())<=limit+CRITERIA['base_speed_slack']
                                    for speed, limit in zip((vx, vy, v[:, vi+5]), cfg.base.limits)))
    joint_peaks, violations = {}, []
    for name, limits in audit['urdf_limits'].items():
        j = model.joint(name).id
        velocity = float(np.abs(v[:, model.jnt_dofadr[j]]).max())
        position = q[:, model.jnt_qposadr[j]]
        lo, hi = model.jnt_range[j]
        joint_peaks[name] = velocity
        if velocity>limits['velocity']+1e-6 or (model.jnt_limited[j] and
                (position.min()<lo-CRITERIA['joint_limit_slack'] or position.max()>hi+CRITERIA['joint_limit_slack'])):
            violations.append(name)
    checks['joint_limits'] = not violations
    force = states['actuator_force']
    checks['effort_limits'] = bool(np.all(force>=model.actuator_forcerange[:, 0]-1e-6) and
                                  np.all(force<=model.actuator_forcerange[:, 1]+1e-6))
    robot = {root}
    for body in range(root+1, model.nbody):
        if int(model.body_parentid[body]) in robot:
            robot.add(body)
    wheels = {int(model.jnt_bodyid[model.joint(n).id]) for n in cfg.base.joints[1::2]}
    bad_contacts = []
    contact_check = 'only_wheel_support_contacts' if grasp_target is None else 'permitted_robot_contacts'
    fingers, target = set(), -1
    if grasp_target is not None:
        declared = grasp_target in world.pickables or any(
            item.body == grasp_target for item in getattr(world, 'articulations', {}).values())
        if not declared or cfg.arm is None or cfg.arm.gripper is None:
            raise ValueError('grasp contact allowance needs a declared target and gripper')
        target = model.body(grasp_target).id
        if target in robot:
            raise ValueError('grasp target cannot be part of the robot')
        fingers = {model.body(n).id for n in cfg.arm.gripper.finger_bodies}
    if not contacts:
        checks[contact_check] = None
        checks['final_wheel_support'] = None
    else:
        a, b = contacts['body_a'][:, 0].astype(int), contacts['body_b'][:, 0].astype(int)
        ra, rb = np.isin(a, list(robot)), np.isin(b, list(robot))
        active = np.linalg.norm(contacts['force_on_b'], axis=1)>CRITERIA['significant_contact_N']
        normal = contacts['normal_on_b']*np.where(rb[:, None], 1, -1)
        ground = ((ra != rb) & np.isin(np.where(rb, b, a), list(wheels)) &
                  (np.abs(contacts['position'][:, 2]-world.spawn[2])<=CRITERIA['support_height_m']) &
                  (normal[:, 2]>=CRITERIA['support_normal_z']))
        hand = (((a == target) & np.isin(b, list(fingers))) |
                ((b == target) & np.isin(a, list(fingers))))
        bad = active & (ra | rb) & ~(ground | hand)
        checks[contact_check] = not bool(bad.any())
        final_support = ground & active & np.isclose(contacts['sim_time'][:, 0], t[-1], rtol=0, atol=1e-8)
        force_robot = contacts['force_on_b']*np.where(rb[:, None], 1, -1)
        weight = float(model.body_mass[list(robot)].sum()*np.linalg.norm(model.opt.gravity))
        checks['final_wheel_support'] = bool(force_robot[final_support, 2].sum()>=weight*CRITERIA['supported_weight_fraction'] and
                len(set(np.where(rb, b, a)[final_support]))>=min(3, len(wheels)))
        bad_contacts = sorted({(model.body(int(x)).name, model.body(int(y)).name) for x,y in zip(a[bad],b[bad])})
    dropped = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mj.mjtJoint.mjJNT_FREE or int(model.jnt_bodyid[j]) in robot:
            continue
        z = q[:, model.jnt_qposadr[j]+2]
        if float(z[0]-z.min())>CRITERIA['object_drop_m']:
            dropped.append(model.body(int(model.jnt_bodyid[j])).name)
    checks['no_dropped_objects'] = not dropped
    verdict = 'fail' if False in checks.values() else 'unverified' if None in checks.values() else 'pass'
    return dict(schema=1, assessed_at=datetime.now(timezone.utc).isoformat(),
                recording=str(Path(directory).resolve()), criteria={**CRITERIA, 'grasp_target': grasp_target}, video_path=None,
                model_sha256=meta['model_sha256'], reset_seed=meta['reset_seed'],
                checks=checks, goal_error_m=error, final_speed_m_s=stopped, max_tilt_deg=float(tilt.max()),
                max_contact_residual_m=float(residual.max()), joint_peak_velocity=joint_peaks,
                joint_violations=violations, forbidden_contact_pairs=bad_contacts, dropped_objects=dropped,
                physical_verdict=verdict, motion_verdict='unverified', deployment_gate='closed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--robot', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--goal', type=float, nargs=2, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = registry.load(args.robot)
    result = grade(args.recording, cfg, load_world(cfg.world), json.loads(args.audit.read_text()), args.goal)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('checks', 'physical_verdict', 'motion_verdict')}))
    raise SystemExit(0 if result['physical_verdict']=='pass' else 1)
