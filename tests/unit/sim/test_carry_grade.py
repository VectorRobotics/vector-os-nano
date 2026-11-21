"""Reject transport evidence that moves a base without a continuously supported load."""
from types import SimpleNamespace as NS
import numpy as np
import mujoco as mj
import pytest
from scripts.grade_carry import carry_window

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('attack,failed', [(None, None), ('one_finger', 'sustained_opposing_loads'),
    ('dropped_sample', 'sustained_opposing_loads'), ('empty', 'sustained_opposing_loads'),
    ('no_payload_motion', 'transported'), ('too_short', 'transported'), ('no_return', 'returned'),
    ('table_drag', 'stayed_clear'), ('hidden_support', 'no_other_support')])
def test_transport_requires_complete_continuous_load_evidence(attack, failed):
    m = mj.MjModel.from_xml_string('''<mujoco><option timestep=".01"/><worldbody>
    <body name="base"><freejoint/><geom size=".1"/></body>
    <body name="target" pos="0 0 .2"><freejoint/><geom type="box" size=".05 .04 .05" mass=".18"/></body>
    <body name="left"><geom size=".01"/></body><body name="right"><geom size=".01"/></body>
    </worldbody></mujoco>''')
    t = np.arange(201)*.01; q = np.tile(m.qpos0, (len(t), 1))
    q[:, 0] = .2*(1-abs(t-1)); q[:, 7] = q[:, 0]
    if attack == 'no_payload_motion': q[:, 7] = 0
    if attack == 'too_short': q[:, [0,7]] *= .1
    if attack == 'no_return': q[:, [0,7]] = (t*.1)[:,None]
    if attack == 'table_drag': q[:, 9] = .05
    body, left, right = (m.body(n).id for n in ('target','left','right'))
    rows = [(time, finger, body, force, normal) for time in t for finger,force,normal in
            [(left,[2,0,.8829],[1,0,0]),(right,[-2,0,.8829],[-1,0,0])]]
    if attack == 'one_finger': rows = [r for r in rows if r[1] == left]
    if attack == 'hidden_support': rows.append((1.,0,body,[0,0,.5],[0,0,1]))
    loads = dict(sim_time=np.array([[r[0]] for r in rows]), body_a=np.array([[r[1]] for r in rows]),
        body_b=np.array([[r[2]] for r in rows]), force_on_b=np.array([r[3] for r in rows]),
        normal_on_b=np.array([r[4] for r in rows]))
    if attack == 'empty': loads = {}
    if attack == 'dropped_sample': t, q = np.delete(t,100), np.delete(q,100,axis=0)
    cfg = NS(model=NS(root_body='base'), arm=NS(gripper=NS(assist=False,finger_bodies=('left','right'))))
    world = NS(pickables={'target':NS(surface='shelf')}, surfaces={'shelf':NS(height=0.)})
    checks, metrics = carry_window(m,dict(sim_time=t[:,None],qpos=q),loads,cfg,world,'target',0.,2.)
    assert all(v is True for v in checks.values()) if failed is None else checks[failed] is False
    assert metrics['minimum_clearance_m'] == pytest.approx(0. if attack == 'table_drag' else .15)
    assert metrics['duration_s'] == 2.
