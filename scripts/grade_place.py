"""Independent full-box, sustained-support acceptance after a force-only grasp and release."""
from pathlib import Path
import argparse
import json
import mujoco as mj
import numpy as np
from scripts.grade_grasp import package_trajectory, integration_error, contact_support, CRITERIA as GRASP
from scripts.grade_navigation import grade as grade_safety
from vector_os_nano.sim.backends.mujoco.recording import read_recording

# Fixed before the first household placement evaluation; no centre-only containment.
CRITERIA = dict(stable_s=2., boundary_slack_m=.0001, support_gap_m=.005,
                penetration_m=.0025, linear_speed_m_s=.01, angular_speed_rad_s=.05,
                support_normal_z=.94, minimum_weight_fraction=.9, significant_force_N=.05,
                pre_release_s=.3, pre_release_weight_fraction=.8)


def terminal_support(model, states, contacts, body, surface, *, duration=None, fraction=None):
    t = states['sim_time'][:, 0]
    duration = CRITERIA['stable_s'] if duration is None else duration
    fraction = CRITERIA['minimum_weight_fraction'] if fraction is None else fraction
    window = t >= t[-1]-duration-1e-8
    ticks = np.rint(t[window]/model.opt.timestep).astype(int)
    enough = bool(len(ticks)>1 and t[window][-1]-t[window][0]>=duration-1e-8 and np.all(np.diff(ticks)==1))
    if not enough or not contacts or not len(contacts['sim_time']):
        return dict(sustained_surface_support=False, no_other_terminal_support=None), {}
    a, b = contacts['body_a'][:, 0].astype(int), contacts['body_b'][:, 0].astype(int)
    ct = np.rint(contacts['sim_time'][:, 0]/model.opt.timestep).astype(int)
    if np.any(abs(contacts['sim_time'][:, 0]-ct*model.opt.timestep)>1e-8) or ct.min()<0 or ct.max()>ticks[-1]:
        raise ValueError('contact times must lie on recorded physics steps')
    force = contacts['force_on_b']*np.where((b==body)[:, None], 1, -1)
    normal = contacts['normal_on_b']*np.where((b==body)[:, None], 1, -1)
    active = ((a==body)|(b==body)) & (np.linalg.norm(force, axis=1)>CRITERIA['significant_force_N']) & (ct>=ticks[0])
    other = np.where(a==body, b, a)
    pos = contacts['position']
    x0,y0,x1,y1 = surface.box
    support = ((other==model.body(surface.body).id) & (normal[:, 2]>=CRITERIA['support_normal_z']) &
               (abs(pos[:, 2]-surface.height)<=CRITERIA['support_gap_m']) &
               (pos[:, 0]>=x0) & (pos[:, 0]<=x1) & (pos[:, 1]>=y0) & (pos[:, 1]<=y1))
    load = np.zeros(len(ticks))
    selected = active & support
    np.add.at(load, ct[selected]-ticks[0], force[selected, 2])
    weight = float(model.body_mass[body]*np.linalg.norm(model.opt.gravity))
    return dict(sustained_surface_support=bool(np.all(load>=weight*fraction)),
                no_other_terminal_support=not bool(np.any(active & ~support))), dict(
                    minimum_support_N=float(load.min()), object_weight_N=weight)


def grade(directory, cfg, world, audit, goal, target, destination):
    if cfg.arm.gripper.assist:
        raise ValueError('physical placement acceptance requires gripper.assist=false')
    result = grade_safety(directory, cfg, world, audit, goal, grasp_target=target)
    meta, states, contacts = read_recording(directory)
    model = mj.MjModel.from_binary_path(meta['model_path'])
    body, address, vertices = package_trajectory(model, states, target)
    surface = world.surfaces[destination]
    t = states['sim_time'][:, 0]; window = t>=t[-1]-CRITERIA['stable_s']-1e-8
    points = vertices[window]; bottom = vertices[:, :, 2].min(axis=1)
    x0,y0,x1,y1 = surface.box; slack = CRITERIA['boundary_slack_m']
    checks, loads = terminal_support(model, states, contacts, body, surface)
    checks['full_box_inside_surface'] = bool(np.all(points[:, :, 0]>=x0-slack) and np.all(points[:, :, 0]<=x1+slack)
        and np.all(points[:, :, 1]>=y0-slack) and np.all(points[:, :, 1]<=y1+slack))
    checks['resting_height'] = bool(np.all(bottom[window]>=surface.height-CRITERIA['penetration_m']) and
                                    np.all(bottom[window]<=surface.height+CRITERIA['support_gap_m']))
    vi = int(model.jnt_dofadr[model.body_jntadr[body]])
    velocity = states['qvel'][window, vi:vi+6]
    checks['stable_object'] = bool(np.all(np.linalg.norm(velocity[:, :3], axis=1)<=CRITERIA['linear_speed_m_s']) and
                                   np.all(np.linalg.norm(velocity[:, 3:], axis=1)<=CRITERIA['angular_speed_rad_s']))
    checks['integrated_object_motion'] = integration_error(model, states, address)<=GRASP['object_state_step_error']
    g = cfg.arm.gripper; fingers = [model.body(n).id for n in g.finger_bodies]
    all_loads, peaks = contact_support(model, states, contacts, body, fingers)
    checks['finger_force_limit'] = all_loads['finger_force_limit']
    opening = states['qpos'][window, int(model.jnt_qposadr[model.joint(g.joint).id])]
    checks['released_jaws'] = bool(np.all(abs(opening-g.open)<=g.min_opening))
    jaw = model.joint(g.joint).id
    actuators = [a for a in range(model.nu) if model.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT
                 and model.actuator_trnid[a, 0] == jaw]
    if len(actuators) != 1:
        raise ValueError('placement release acceptance requires one jaw position actuator')
    closed = np.flatnonzero(abs(states['ctrl'][:, actuators[0]]-g.closed)<=1e-8)
    checks['supported_before_release'] = False
    if len(closed) and closed[-1] < len(t)-1:
        end = closed[-1]
        prefix = {k: v[:end+1] for k,v in states.items()}
        loads_before = {k: v[contacts['sim_time'][:, 0]<=t[end]+1e-8] for k,v in contacts.items()}
        supported, _ = terminal_support(model, prefix, loads_before, body, surface,
            duration=CRITERIA['pre_release_s'], fraction=CRITERIA['pre_release_weight_fraction'])
        checks['supported_before_release'] = supported['sustained_surface_support']
    initial_height = world.surfaces[world.pickables[target].surface].height
    checks['initially_supported'] = abs(float(bottom[0])-initial_height)<=GRASP['initial_support_tolerance_m']
    raised = np.flatnonzero((bottom-initial_height>=GRASP['clearance_m']) & (t<t[-1]-CRITERIA['stable_s']))
    checks['prior_force_only_hold'] = False
    checks['initially_ungrasped'] = None
    if contacts:
        a, b = contacts['body_a'][:, 0].astype(int), contacts['body_b'][:, 0].astype(int)
        touching = ((a==body)|(b==body)) & (np.linalg.norm(contacts['force_on_b'], axis=1)>GRASP['min_compression_N'])
        on_fingers = np.isin(np.where(a==body, b, a), fingers)
        checks['initially_ungrasped'] = not bool(np.any(touching & on_fingers &
            (contacts['sim_time'][:, 0]<=t[0]+GRASP['initially_empty_s'])))
    if len(raised) and contacts:
        end = raised[-1]
        prior_states = {k: v[:end+1] for k,v in states.items()}
        prior_contacts = {k: v[contacts['sim_time'][:, 0]<=t[end]+1e-8] for k,v in contacts.items()}
        hold, _ = contact_support(model, prior_states, prior_contacts, body, fingers)
        clear = bottom[(t>=t[end]-GRASP['hold_s']-1e-8)&(t<=t[end])]-initial_height
        other_support = touching & ~on_fingers & (contacts['sim_time'][:, 0]>=t[end]-GRASP['hold_s']-1e-8) & (contacts['sim_time'][:, 0]<=t[end]+1e-8)
        checks['prior_force_only_hold'] = bool(all(hold.values()) and np.all(clear>=GRASP['clearance_m']) and not other_support.any())
    result['checks'].update(checks)
    result.update(scope=__doc__, target=target, destination=destination, place_criteria=CRITERIA,
                  finger_peak_force_N=peaks.get('finger_peak_force_N'), **loads)
    result['physical_verdict'] = ('fail' if False in result['checks'].values() else
                                 'unverified' if None in result['checks'].values() else 'pass')
    return result


if __name__ == '__main__':
    from vector_os_nano.embodiments import registry
    from vector_os_nano.schemas.world import load_world
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('--robot', required=True)
    p.add_argument('--world', required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--goal', type=float, nargs=2, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--surface', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = grade(args.recording, registry.load(args.robot), load_world(args.world),
                   json.loads(args.audit.read_text()), args.goal, args.target, args.surface)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ('checks', 'physical_verdict', 'motion_verdict')}))
    raise SystemExit(0 if result['physical_verdict']=='pass' else 1)
