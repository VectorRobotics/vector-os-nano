"""Adversarial recorded traces cannot turn a terminal-position pass into a physical pass."""
from types import SimpleNamespace as NS

import mujoco as mj
import numpy as np
import pytest

from scripts import grade_navigation as grader

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('attack, verdict', [('none', 'pass'), ('teleport', 'fail'),
    ('external_force', 'fail'), ('body_contact', 'fail'), ('missing_contacts', 'unverified')])
def test_navigation_requires_continuous_supported_motion(tmp_path, monkeypatch, attack, verdict):
    model = mj.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><body name="base" pos="0 0 .5"><freejoint/>
      <geom type="sphere" size=".05" mass="1"/>
      <body><joint name="steer" range="-1 1"/><geom type="sphere" size=".02" mass="1"/>
      <body name="wheel" pos="0 0 -.45"><joint name="wheel" range="-1 1"/>
      <geom type="sphere" size=".05" mass="1"/></body></body></body></worldbody></mujoco>''')
    path = tmp_path/'fixture.mjb'
    mj.mj_saveModel(model, str(path))
    meta = dict(model_path=str(path), model_sha256='fixture', reset_seed=0, timestep=.001)
    states = dict(qpos=np.tile(model.qpos0, (3, 1)), qvel=np.zeros((3, model.nv)),
                  sim_time=np.arange(3)[:, None]*.001, eq_active=np.zeros((3, 0)),
                  actuator_force=np.zeros((3, 0)), max_contact_residual=np.zeros((3, 1)),
                  max_applied_force=np.zeros((3, 1)))
    contacts = dict(body_a=np.zeros((3, 1)), body_b=np.full((3, 1), model.body('wheel').id),
                    position=np.zeros((3, 3)), normal_on_b=np.tile([0, 0, 1.], (3, 1)),
                    force_on_b=np.tile([0, 0, 3*9.81], (3, 1)), sim_time=states['sim_time'])
    if attack == 'teleport': states['qpos'][-1, 0] = .1
    if attack == 'external_force': states['max_applied_force'][1] = 1
    if attack == 'body_contact': contacts['body_b'][-1] = model.body('base').id
    if attack == 'missing_contacts': contacts = {}
    monkeypatch.setattr(grader, 'read_recording', lambda _: (meta, states, contacts))
    cfg = NS(model=NS(root_body='base'), base=NS(policy=NS(ref='swerve'), joints=('steer', 'wheel'), limits=(.25, .18, .3)))
    result = grader.grade(tmp_path, cfg, NS(spawn=[0, 0, 0, 0]),
                           {'urdf_limits': {n: {'velocity': 1} for n in ('steer', 'wheel')}}, [0, 0])
    assert result['checks']['arrived']
    assert result['physical_verdict'] == verdict
    assert result['motion_verdict'] == 'unverified' and result['deployment_gate'] == 'closed'
