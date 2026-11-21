"""Command profiles and contact-only evidence; physical success is graded separately."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from vector_os_nano.embodiments.config import ArmSpec, GripperSpec, EmbodimentConfigError
from vector_os_nano.interfaces.physics import GraspSpec
from vector_os_nano.sim import grasp
from vector_os_nano.sim.drivers.arm import Arm
from vector_os_nano.sim.drivers.gripper import Gripper
from vector_os_nano.embodiments.config import _parse_arm


class Runtime:
    def __init__(self, q):
        self.q = np.array(q, dtype=float)
        self.sim_time = 0.
        self.samples = [(0., self.q.copy())]
        self.world = SimpleNamespace(scene=SimpleNamespace(joint=lambda n: int(n), joint_q=lambda j: j))
    def call(self, fn):
        return fn(self)
    submit = call
    def snapshot(self):
        return SimpleNamespace(q=self.q.copy())
    def targets(self, joints):
        return self.q[joints]
    def set_targets(self, joints, q):
        self.q[joints] = q
    def wait_sim(self, seconds, **kw):
        self.sim_time += seconds
        self.samples.append((self.sim_time, self.q.copy()))
        return True


def test_quintic_obeys_each_joint_rate_acceleration_jerk(monkeypatch):
    monkeypatch.setattr('vector_os_nano.sim.drivers.arm.ArmKinematics', lambda *a: None)
    limits = dict(velocity=(.3,.2), acceleration=(.4,.3), jerk=(.6,.4))
    arm = Arm(ArmSpec(('0','1'), 'tcp', (0.,0.), **limits), Runtime([0.,0.]))
    assert arm.move_joints([.8,-.5], duration=.01)
    samples=arm._rt.samples
    times=np.array([t for t,q in samples]); q=np.array([q for t,q in samples]);dt=times[1]-times[0]
    assert np.all(np.diff(times)>0) and np.allclose(q[-1],[.8,-.5])
    for order,key in enumerate(('velocity','acceleration','jerk'),1):
        derivative=np.diff(q,n=order,axis=0)/dt**order
        assert np.all(np.max(np.abs(derivative),axis=0)<=np.array(limits[key])+1e-6)
    assert np.max(np.abs(np.diff(q,axis=0)[[0,-1]]/dt)) < .001
    with pytest.raises(ValueError,match='finite'):
        arm.move_joints([float('nan'),0.])


@pytest.mark.parametrize('kw',[dict(velocity=(1.,)),dict(velocity=(1.,),acceleration=(1.,),jerk=(0.,)),dict(velocity=(float('inf'),),acceleration=(1.,),jerk=(1.,))])
def test_invalid_motion_limits_refused(kw):
    with pytest.raises(EmbodimentConfigError):
        ArmSpec(('0',),'tcp',(0.,),**kw)


def test_jaw_commands_have_finite_speed():
    grip=Gripper.__new__(Gripper);grip._rt=Runtime([.04]);grip._jaw=grip._jaw_q=0
    grip._spec=GripperSpec('0',.04,0.,('left','right'),'palm',.002,assist=False,speed=.05)
    grip._servo(0.)
    rows=grip._rt.samples
    assert rows[-1][0]>=.8-1e-9 and rows[-1][1][0]==pytest.approx(0.)
    assert max(abs(b[1][0]-a[1][0])/(b[0]-a[0]) for a,b in zip(rows,rows[1:]))<=.05+1e-9


@pytest.mark.parametrize('speed', [None, .05])
def test_jaw_finishes_at_exact_requested_target(speed):
    grip=Gripper.__new__(Gripper);grip._rt=Runtime([.038123]);grip._jaw=grip._jaw_q=0
    grip._spec=GripperSpec('0',.04,0.,('left','right'),'palm',.002,assist=False,speed=speed)
    target = .0037
    grip._servo(target)
    assert grip._rt.q[0] == target


def test_manifest_parses_contact_and_motion_limits():
    raw=dict(joints=['0'],ee={'site':'tcp'},home=[0.],velocity=[.5],acceleration=[1.],jerk=[2.],
             gripper=dict(joint='jaw',open=.04,closed=0.,finger_bodies=['left','right'],
                          weld_body='palm',min_opening=.002,assist=False,speed=.05))
    spec=_parse_arm({'arm':raw},'test')
    assert spec.velocity==(.5,) and spec.acceleration==(1.,) and spec.jerk==(2.,)
    assert spec.gripper.assist is False and spec.gripper.speed==.05
    raw['gripper']['assist']='false'
    with pytest.raises(EmbodimentConfigError,match='boolean'):
        _parse_arm({'arm':raw},'test')


def test_contact_only_grip_never_attaches_and_detects_slip():
    scene=SimpleNamespace(joint_q=lambda j:0,robot_root=1,subtree=lambda root:{1,2,3},
                          body_ids={'world':0,'palm':1,'left':2,'right':3,'package':4},
                          body_name=lambda b:str(b),grasp_pairs=())
    contacts={2:frozenset({3,4}),3:frozenset({2,4})}
    world=SimpleNamespace(scene=scene,state=lambda:(np.array([.02]),None),contacts=lambda b:contacts[b],
                          attach=lambda *a:pytest.fail('contact-only grip created a constraint'))
    spec=GraspSpec(1,(2,3),0,0.,.002,assist=False)
    assert grasp.grasp(world,spec)==4
    contacts[3]=frozenset({2})
    assert grasp.grasp(world,spec) is None
    contacts[3]=frozenset({2,4})
    assert grasp.grasp(world,replace(spec,min_opening=.03)) is None
    assert grasp.grasp(world,replace(spec,assist=True)) is None  # old mode still requires an authored pair
