"""Synthetic adversarial traces exercise the full independent grader, not task success."""
from types import SimpleNamespace as NS
import mujoco as mj
import numpy as np
import pytest
from scripts import grade_grasp as grader, grade_navigation

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('attack,failed', [('none', None), ('teleport', 'integrated_object_motion'),
    ('rotated_corner', 'lift_clearance'), ('initially_held', 'initially_ungrasped'),
    ('initially_airborne', 'initially_supported'), ('table_support', 'no_other_load_support'),
    ('palm_contact', 'permitted_robot_contacts'), ('other_object', 'permitted_robot_contacts')])
def test_full_grasp_grade(tmp_path, monkeypatch, attack, failed):
    model = mj.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><option timestep=".01"/>
      <worldbody><body name="base" pos="0 0 .5"><freejoint/>
      <geom type="sphere" size=".05" mass="1"/>
      <body><joint name="steer"/><geom type="sphere" size=".02" mass="1"/>
      <body name="wheel" pos="0 0 -.45"><joint name="wheel"/>
      <geom type="sphere" size=".05" mass="1"/></body></body>
      <body name="palm"><geom type="sphere" size=".02" mass=".1"/>
      <body name="left"><joint name="jaw" type="slide" range="0 .04"/>
      <geom type="sphere" size=".01" mass=".1"/></body>
      <body name="right"><geom type="sphere" size=".01" mass=".1"/></body></body></body>
      <body name="target" pos="1 0 .813"><freejoint/><geom type="box" size=".025 .08 .035" mass=".18"/></body>
      <body name="other" pos="2 0 .813"><geom type="box" size=".025 .08 .035" mass=".18"/></body>
      </worldbody></mujoco>''')
    path = tmp_path/'fixture.mjb'; mj.mj_saveModel(model, str(path))
    times = np.arange(331)*.01
    q = np.tile(model.qpos0, (len(times), 1)); q[:, model.joint('jaw').qposadr[0]] = .025
    target, left, right = [model.body(n).id for n in ('target', 'left', 'right')]
    address = int(model.jnt_qposadr[model.body_jntadr[target]])
    q[:, address+2] += np.clip((times-.4)/.8, 0, 1)*.12
    if attack == 'initially_airborne': q[:, address+2] += .1
    if attack == 'rotated_corner':
        q[:, address+2] = .813+np.clip((times-.4)/.8, 0, 1)*.04
        angles = np.clip((times-.4)/.8, 0, 1)*np.pi/2
        q[:, address+3] = np.cos(angles/2); q[:, address+4] = np.sin(angles/2)
    v = np.zeros((len(times), model.nv))
    for i in range(1, len(times)):
        mj.mj_differentiatePos(model, v[i], .01, q[i-1], q[i])
    if attack == 'teleport': q[-100:, address] += .1
    rows = []
    for t in times:
        rows.append((t, 0, model.body('wheel').id, [0, 0, 3.3*9.81], [0, 0, 1]))
        if t >= .4 or attack == 'initially_held':
            rows.extend([(t, left, target, [2., 0, .18*9.81/2], [1, 0, 0]),
                         (t, right, target, [-2., 0, .18*9.81/2], [-1, 0, 0])])
    if attack in ('table_support', 'palm_contact', 'other_object'):
        a, b = {'table_support': (0, target), 'palm_contact': (model.body('palm').id, target),
                'other_object': (left, model.body('other').id)}[attack]
        rows.append((times[-1], a, b, [0, 0, 1], [0, 0, 1]))
    contacts = dict(sim_time=np.array([[r[0]] for r in rows]), body_a=np.array([[r[1]] for r in rows]),
        body_b=np.array([[r[2]] for r in rows]), force_on_b=np.array([r[3] for r in rows]),
        normal_on_b=np.array([r[4] for r in rows]), position=np.zeros((len(rows), 3)))
    states = dict(sim_time=times[:, None], qpos=q, qvel=v, eq_active=np.zeros((len(times), 0)),
        actuator_force=np.zeros((len(times), 0)), max_contact_residual=np.zeros((len(times), 1)),
        max_applied_force=np.zeros((len(times), 1)))
    meta = dict(model_path=str(path), model_sha256='fixture', reset_seed=0, timestep=.01)
    for module in (grader, grade_navigation):
        monkeypatch.setattr(module, 'read_recording', lambda _: (meta, states, contacts))
    cfg = NS(model=NS(root_body='base'), base=NS(policy=NS(ref='swerve'), joints=('steer', 'wheel'), limits=(.25, .18, .3)),
             arm=NS(gripper=NS(assist=False, joint='jaw', closed=0., min_opening=.005, finger_bodies=('left', 'right'))))
    world = NS(spawn=[0, 0, 0, 0], pickables={'target': NS(surface='shelf')}, surfaces={'shelf': NS(height=.778)})
    result = grader.grade(tmp_path, cfg, world, {'urdf_limits': {}}, [0, 0], 'target')
    assert result['physical_verdict'] == ('pass' if failed is None else 'fail'), result
    if failed: assert result['checks'][failed] is False
    assert result['motion_verdict'] == 'unverified' and result['deployment_gate'] == 'closed'
    if attack == 'none':
        # Manipulation's narrow fingertip allowance must never apply to ordinary navigation.
        assert grade_navigation.grade(tmp_path, cfg, world, {'urdf_limits': {}}, [0, 0])['physical_verdict'] == 'fail'
        world.pickables = {}
        with pytest.raises(ValueError, match='declared target'):
            grade_navigation.grade(tmp_path, cfg, world, {'urdf_limits': {}}, [0, 0], grasp_target='target')
        world.articulations = {'fixture': NS(body='target')}
        assert grade_navigation.grade(tmp_path, cfg, world, {'urdf_limits': {}}, [0, 0], grasp_target='target')['physical_verdict'] == 'pass'
