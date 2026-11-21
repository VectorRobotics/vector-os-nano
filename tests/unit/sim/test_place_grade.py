"""Synthetic hostile evidence tests placement acceptance; these are not successful robot episodes."""
from types import SimpleNamespace as NS
import mujoco as mj
import numpy as np
import pytest
from scripts import grade_place as grader

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('attack,failed', [('none', None), ('corner_outside', 'full_box_inside_surface'),
    ('no_support', 'sustained_surface_support'), ('wrong_support', 'sustained_surface_support'),
    ('still_touching', 'no_other_terminal_support'), ('airborne', 'resting_height'),
    ('sliding', 'stable_object'), ('never_held', 'prior_force_only_hold'),
    ('hidden_hold_support', 'prior_force_only_hold'), ('initially_held', 'initially_ungrasped'),
    ('teleport', 'integrated_object_motion'), ('future_contact', 'invalid'),
    ('missing_contacts', 'prior_force_only_hold'), ('empty_contacts', 'prior_force_only_hold'),
    ('late_support', 'supported_before_release')])
def test_complete_box_and_sustained_support_are_required(tmp_path, monkeypatch, attack, failed):
    model = mj.MjModel.from_xml_string('''<mujoco><option timestep=".01"/><worldbody>
      <body name="target" pos="1 0 .05"><freejoint/><geom type="box" size=".15 .10 .05" mass=".18"/></body>
      <body name="left"><joint name="jaw" type="slide" range="0 .04"/><geom size=".01" mass=".1"/></body>
      <body name="right"><geom size=".01" mass=".1"/></body>
      <body name="other"><geom size=".01" mass=".1"/></body>
      </worldbody><actuator><position joint="jaw" kp="1"/></actuator></mujoco>''')
    path = tmp_path/'fixture.mjb'; mj.mj_saveModel(model, str(path))
    t = np.arange(661)*.01
    q = np.tile(model.qpos0, (len(t), 1))
    q[:, 2] += .12*(np.clip((t-.4)/.6, 0, 1)-np.clip((t-3.1)/.8, 0, 1))
    q[:, model.joint('jaw').qposadr[0]] = np.where((t>=.4)&(t<4.3), .02, .04)
    if attack == 'corner_outside': q[:, 0] = 1.2
    if attack == 'airborne': q[:, 2] += .02
    v = np.zeros((len(t), model.nv))
    for i in range(1, len(t)):
        mj.mj_differentiatePos(model, v[i], .01, q[i-1], q[i])
    if attack == 'sliding': v[-201:, 0] = .02
    if attack == 'teleport': q[-100:, 0] += .01
    body, left, right, other = [model.body(n).id for n in ('target','left','right','other')]
    rows = []
    for time in t:
        if .4<=time<4.3 and attack != 'never_held':
            rows += [(time,left,body,[2,0,.8829],[1,0,0]), (time,right,body,[-2,0,.8829],[-1,0,0])]
        if (time<.4 or time>=(4.3 if attack=='late_support' else 3.9)) and attack != 'no_support':
            rows.append((time,other if attack=='wrong_support' else 0,body,[0,0,1.7658],[0,0,1]))
    if attack == 'still_touching': rows.append((t[-1],left,body,[.1,0,0],[1,0,0]))
    if attack == 'hidden_hold_support': rows.append((2.,0,body,[0,0,.1],[0,0,1]))
    if attack == 'initially_held': rows.append((.1,left,body,[.1,0,0],[1,0,0]))
    if attack == 'future_contact': rows.append((1e9,0,body,[0,0,1.7658],[0,0,1]))
    contacts = dict(sim_time=np.array([[r[0]] for r in rows]), body_a=np.array([[r[1]] for r in rows]),
                    body_b=np.array([[r[2]] for r in rows]), force_on_b=np.array([r[3] for r in rows]),
                    normal_on_b=np.array([r[4] for r in rows]), position=np.tile([1.,0,0],(len(rows),1)))
    if attack == 'missing_contacts': contacts = {}
    if attack == 'empty_contacts': contacts = {k:v[:0] for k,v in contacts.items()}
    states = dict(sim_time=t[:,None], qpos=q, qvel=v, ctrl=np.where((t>=.4)&(t<4.3),0.,.04)[:,None])
    monkeypatch.setattr(grader, 'read_recording', lambda _: ({'model_path':str(path)},states,contacts))
    monkeypatch.setattr(grader, 'grade_safety', lambda *a, **kw: dict(checks={},motion_verdict='unverified',deployment_gate='closed'))
    cfg = NS(arm=NS(gripper=NS(assist=False,joint='jaw',open=.04,closed=0.,min_opening=.005,finger_bodies=('left','right'))))
    world = NS(surfaces={'shelf':NS(body='world',box=[.7,-.2,1.3,.2],height=0.)},pickables={'target':NS(surface='shelf')})
    if failed == 'invalid':
        with pytest.raises(ValueError, match='physics steps'):
            grader.grade(tmp_path,cfg,world,{},[0,0],'target','shelf')
        return
    result = grader.grade(tmp_path,cfg,world,{},[0,0],'target','shelf')
    assert result['physical_verdict'] == ('pass' if failed is None else 'fail'), result
    if failed: assert result['checks'][failed] is False
    assert result['motion_verdict']=='unverified' and result['deployment_gate']=='closed'
