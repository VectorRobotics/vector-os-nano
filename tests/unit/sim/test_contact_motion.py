"""Command profiles and contact-only evidence; physical success is graded separately."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from vector_os_nano.schemas.robot import ArmSpec, GripperSpec, EmbodimentConfigError
from vector_os_nano.interfaces.physics import GraspSpec
from vector_os_nano.sim import grasp
from vector_os_nano.sim.drivers.arm import Arm
from vector_os_nano.sim.drivers.gripper import Gripper
from vector_os_nano.schemas.robot import _parse_arm


class Runtime:
    def __init__(self, q):
        self.q = np.array(q, dtype=float)
        self.sim_time = 0.
        self.tick_dt = .001
        self.held_waits = []
        self.clock_releases = 0
        self.samples = [(0., self.q.copy())]
        self.world = SimpleNamespace(scene=SimpleNamespace(joint=lambda n: int(n), joint_q=lambda j: j))
    def call(self, fn):
        return fn(self)
    submit = call
    def snapshot(self):
        return SimpleNamespace(q=self.q.copy())
    def state(self):
        return self.q.copy(), np.zeros_like(self.q)
    def targets(self, joints):
        return self.q[joints]
    def set_targets(self, joints, q):
        self.q[joints] = q
    def wait_sim(self, seconds, **kw):
        self.held_waits.append(kw.get('hold', False))
        self.sim_time += seconds
        self.samples.append((self.sim_time, self.q.copy()))
        return True
    def release_clock(self):
        self.clock_releases += 1


def test_joint_targets_advance_each_physics_tick_and_release_clock(monkeypatch):
    monkeypatch.setattr('vector_os_nano.sim.drivers.arm.ArmKinematics', lambda *a: None)
    rt = Runtime([0.])
    arm = Arm(ArmSpec(('0',), 'tcp', (0.,)), rt)
    assert arm.move_joints([.01], duration=.1015)
    times = np.array([t for t, _ in rt.samples])
    assert np.max(np.diff(times)) <= rt.tick_dt + 1e-12
    assert .1015 <= times[-1] < .1015 + rt.tick_dt
    assert all(rt.held_waits)
    assert rt.clock_releases == 1


def test_interrupted_joint_wait_holds_actual_position_and_releases_clock(monkeypatch):
    monkeypatch.setattr('vector_os_nano.sim.drivers.arm.ArmKinematics', lambda *a: None)
    rt = Runtime([0.])
    arm = Arm(ArmSpec(('0',), 'tcp', (0.,)), rt)
    monkeypatch.setattr(rt, 'wait_sim', lambda *a, **kw: False)
    assert arm.move_joints([.1], duration=.1) is False
    assert rt.q[0] < .1
    assert rt.clock_releases == 1


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


def test_release_unloads_from_previous_target_at_each_physics_tick(monkeypatch):
    grip = Gripper.__new__(Gripper)
    grip._rt = Runtime([.025])
    grip._jaw = grip._jaw_q = 0
    grip._spec = GripperSpec('0', .04, 0., ('left', 'right'), 'palm', .002, assist=False, speed=.02)
    # The package holds the jaws open while the servo still commands a squeeze.
    monkeypatch.setattr(grip._rt, 'targets', lambda joints: np.array([0.]))
    grip._servo(.04)
    rows = grip._rt.samples[1:]
    commands = np.r_[0., [q[0] for _, q in rows]]
    times = np.r_[0., [t for t, _ in rows]]
    assert np.max(np.diff(times)) <= grip._rt.tick_dt + 1e-12
    assert np.max(np.diff(commands)/np.diff(times)) <= .02 + 1e-9
    assert commands[-1] == .04
    assert all(grip._rt.held_waits) and grip._rt.clock_releases == 1


@pytest.mark.parametrize('speed', [None, .05])
def test_jaw_finishes_at_exact_requested_target(speed):
    grip=Gripper.__new__(Gripper);grip._rt=Runtime([.038123]);grip._jaw=grip._jaw_q=0
    grip._spec=GripperSpec('0',.04,0.,('left','right'),'palm',.002,assist=False,speed=speed)
    target = .0037
    grip._servo(target)
    assert grip._rt.q[0] == target


@pytest.mark.parametrize('position', [0., float('nan')])
def test_open_refuses_a_blocked_jaw(monkeypatch, position):
    grip = Gripper.__new__(Gripper); grip._rt = Runtime([position]); grip._jaw = grip._jaw_q = 0
    grip._spec = GripperSpec('0', .04, 0., ('left', 'right'), 'palm', .002, assist=False)
    grip._grasp = SimpleNamespace(weld_body=1)
    grip._held = None
    monkeypatch.setattr(grip, '_servo', lambda q: None)
    assert grip.open() is False


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


def test_arm_manifest_declares_submillimetre_ik_accuracy():
    raw = dict(joints=['0'], ee={'site':'tcp'}, home=[0.], ik_tolerance=.0001)
    assert _parse_arm({'arm':raw}, 'test').ik_tolerance == .0001
    for bad in (0., -.1, float('nan'), float('inf'), True, 'invalid'):
        with pytest.raises(EmbodimentConfigError):
            _parse_arm({'arm':dict(raw, ik_tolerance=bad)}, 'test')


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


def test_jaw_command_counter_counts_actor_writes_even_when_interrupted(monkeypatch):
    from vector_os_nano.sim.runtime import ACTOR_EPOCH
    grip = Gripper.__new__(Gripper); grip._rt = Runtime([.04]); grip._jaw = grip._jaw_q = 0
    grip._ctrl_motion = 0.
    grip._spec = GripperSpec('0', .04, 0., ('left', 'right'), 'palm', .002, assist=False, speed=.05)
    grip._servo(.02)
    assert grip.ctrl_motion() == 0.  # Background movement cannot become actor evidence.
    with ACTOR_EPOCH.tool():
        grip._servo(.02)
    assert grip.ctrl_motion() == 0.
    monkeypatch.setattr(grip._rt, 'wait_sim', lambda *args, **kwargs: False)
    with ACTOR_EPOCH.tool(), pytest.raises(RuntimeError, match='interrupted'):
        grip._servo(0.)
    assert 0 < grip.ctrl_motion() == pytest.approx(abs(grip._rt.q[0]-.02))
