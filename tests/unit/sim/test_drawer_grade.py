"""A final closed coordinate cannot hide teleportation, lost grip or handle misses."""
from types import SimpleNamespace as NS

import mujoco as mj
import numpy as np
import pytest

from scripts.grade_drawer import drawer_checks, drawer_pose
from vector_os_nano.schemas.world import Articulation

pytestmark = pytest.mark.sim


def trace():
    model = mj.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><option timestep=".01"/>
      <worldbody><body name="cart" pos="1 2 .4" euler="0 0 .7"><freejoint/><geom size=".1" mass="10"/>
      <body pos=".1 .2 .3" euler="0 0 .2"><body name="drawer" pos=".2 0 0" euler="0 .1 0">
      <joint name="slide" type="slide" axis="1 0 0" range="-.3 0"/>
      <geom name="handle" type="box" size=".02 .06 .01" pos="-.1 0 .1" mass="1"/>
      </body></body></body>
      <body name="left" pos="3 0 0"><geom size=".01"/></body>
      <body name="right" pos="4 0 0"><geom size=".01"/></body></worldbody></mujoco>''')
    times = np.arange(1451)*.01
    q = np.tile(model.qpos0, (len(times), 1))
    qa = int(model.jnt_qposadr[model.joint('slide').id])
    q[:, qa] = -.27*np.minimum(np.clip((times-1)/5.4, 0, 1), np.clip((12.4-times)/5.4, 0, 1))
    v = np.zeros((len(times), model.nv)); v[1:, -1] = np.diff(q[:, qa])/.01
    states = dict(sim_time=times[:, None], qpos=q, qvel=v)
    spec = Articulation('drawer', 'slide', 0., -.27, .003, handle_geom='handle')
    cfg = NS(arm=NS(gripper=NS(assist=False, finger_bodies=('left', 'right'))))
    body = model.body('drawer').id
    pos, rot, _ = drawer_pose(model, q, body)
    rows = []
    for k, time in enumerate(times):
        if 1 <= time <= 12.4:
            for finger, sign in (('left', 1), ('right', -1)):
                normal = rot[k] @ np.array([0., sign, 0.])
                point = pos[k] + rot[k] @ np.array([-.1, -.06*sign, .1])
                rows.append((time, model.body(finger).id, body, 2.*normal, normal, point))
    contacts = {key: np.array([row[i] for row in rows]).reshape((-1, width)) for i, (key, width) in enumerate(
        [('sim_time', 1), ('body_a', 1), ('body_b', 1), ('force_on_b', 3), ('normal_on_b', 3), ('position', 3)])}
    return model, states, contacts, cfg, spec


@pytest.mark.parametrize('attack, failed', [('none', None), ('teleport', 'integrated_drawer_motion'),
    ('short_open', 'sustained_open'), ('not_flush', 'closed_and_settled'),
    ('handle_miss', 'handle_only'), ('one_finger', 'opposed_grip_during_motion'),
    ('same_side', 'opposed_grip_during_motion'), ('excess_force', 'finger_force'),
    ('cart_jump', 'integrated_cart_motion'), ('missing_contact', 'opposed_grip_during_motion')])
def test_drawer_complete_trajectory(attack, failed):
    model, states, contacts, cfg, spec = trace()
    if attack == 'teleport': states['qpos'][500:510, -1] += .001
    if attack == 'short_open': states['qpos'][:, -1] *= .9
    if attack == 'not_flush': states['qpos'][-200:, -1] = -.004
    if attack == 'handle_miss': contacts['position'][300] += [0, 0, .1]
    if attack == 'one_finger': contacts['body_a'][::2] = model.body('right').id
    if attack == 'same_side':
        contacts['normal_on_b'][::2] *= -1; contacts['force_on_b'][::2] *= -1
    if attack == 'excess_force': contacts['force_on_b'] *= 4
    if attack == 'cart_jump': states['qpos'][-100:, 0] += .001
    if attack == 'missing_contact':
        contacts = {key: np.delete(value, 500, axis=0) for key, value in contacts.items()}
    checks, _ = drawer_checks(model, states, contacts, cfg, spec)
    if failed: assert checks[failed] is False
    else: assert all(checks.values()), checks


def test_recorded_pose_matches_mujoco_kinematics():
    model, states, _, _, _ = trace()
    body = model.body('drawer').id
    states['qpos'][500, 1] += .02
    states['qpos'][500, 3:7] = [np.cos(.2), 0, 0, np.sin(.2)]
    positions, rotations, _ = drawer_pose(model, states['qpos'], body)
    data = mj.MjData(model)
    for index in (0, 500, 1000):
        data.qpos[:] = states['qpos'][index]; mj.mj_kinematics(model, data)
        np.testing.assert_allclose(positions[index], data.xpos[body], rtol=0, atol=1e-12)
        np.testing.assert_allclose(rotations[index], data.xmat[body].reshape(3, 3), rtol=0, atol=1e-12)


def test_missing_contacts_cannot_verify_and_assisted_grip_is_rejected():
    model, states, contacts, cfg, spec = trace()
    checks, _ = drawer_checks(model, states, {}, cfg, spec)
    assert checks['opposed_grip_during_motion'] is None
    cfg.arm.gripper.assist = True
    with pytest.raises(ValueError, match='contact-only'):
        drawer_checks(model, states, contacts, cfg, spec)


def test_stationary_open_drawer_can_be_released_for_loading_but_cannot_drift():
    model, states, contacts, cfg, spec = trace()
    keep = ~((contacts['sim_time'][:, 0] >= 6.5) & (contacts['sim_time'][:, 0] < 6.9))
    contacts = {key: value[keep] for key, value in contacts.items()}
    assert all(drawer_checks(model, states, contacts, cfg, spec)[0].values())
    states['qpos'][650:690, -1] = -.27-np.arange(40)*.0001
    states['qvel'][650:690, -1] = -.01
    checks, _ = drawer_checks(model, states, contacts, cfg, spec)
    assert not checks['released_open_stable']
    assert not checks['opposed_grip_during_motion']


def test_authored_time_zero_resets_are_allowed_but_midrun_resets_are_not():
    model, states, contacts, cfg, spec = trace()
    states = {key: np.concatenate((value[:1], value), axis=0) for key, value in states.items()}
    assert all(drawer_checks(model, states, contacts, cfg, spec)[0].values())
    states['sim_time'][500] = states['sim_time'][499]
    with pytest.raises(ValueError, match='continuous reset'):
        drawer_checks(model, states, contacts, cfg, spec)
