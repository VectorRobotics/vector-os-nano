"""A loaded/contacting gripper cannot bypass transport preparation through base tools."""
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import pytest
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.interfaces.devices import FingerLoad
from vector_os_nano.skills.transport import prepare, parse
from vector_os_nano.skills.navigate import NavigateSkill
from vector_os_nano.skills.go2.walk import WalkSkill
from vector_os_nano.skills.go2.turn import TurnSkill
from vector_os_nano.skills.go2.explore import ExploreSkill
from vector_os_nano.schemas.robot import EmbodimentConfigError


def context(load=1.8):
    arm = Mock(move_joints=Mock(return_value=True))
    grip = Mock(is_holding=Mock(return_value=True))
    grip.get_finger_load.side_effect = [FingerLoad(t, (0., 0., -load)) for t in (1., 2.)]
    cfg = dict(posture='carry', duration=4., minimum_load=.3, max_force=6.)
    return SkillContext(arm=arm, gripper=grip, base=Mock(limits=(.3,.3,.6)),
        config={'skills': {'transport': cfg}, 'arm': {'postures': {'carry': [0.,1.,-1.]}}},
        services={'cancel': threading.Event()})


@pytest.mark.parametrize('tool,params', [(NavigateSkill(), {'x': 1., 'y': 2.}),
    (WalkSkill(), {}), (TurnSkill(), {}), (ExploreSkill(), {'backend': 'coverage'})])
@pytest.mark.parametrize('load', [0., -.8])
def test_support_or_compression_blocks_every_public_base_route(tool, params, load):
    ctx = context(load)
    result = tool.execute(params, ctx)
    assert not result.success and result.diagnosis_code == 'carry_not_ready'
    ctx.arm.move_joints.assert_not_called()
    ctx.base.walk.assert_not_called(); ctx.base.navigate_to.assert_not_called()


@pytest.mark.parametrize('fault', ['missing', 'stale', 'overload', 'slip', 'blocked', 'cancelled', None])
def test_carry_preparation_keeps_the_grip_and_checks_fresh_load(fault):
    ctx = context()
    if fault == 'missing': ctx.gripper.get_finger_load.side_effect = [None]
    if fault == 'stale': ctx.gripper.get_finger_load.side_effect = [FingerLoad(1., (0.,0.,-1.8))]*2
    if fault == 'overload': ctx.gripper.get_finger_load.side_effect = [FingerLoad(1., (0.,0.,-7.))]
    if fault == 'slip': ctx.gripper.is_holding.side_effect = [True, False]
    if fault == 'blocked': ctx.arm.move_joints.return_value = False
    if fault == 'cancelled': ctx.services['cancel'].set()
    result = prepare(ctx)
    assert (result is None) == (fault is None)
    ctx.gripper.open.assert_not_called(); ctx.gripper.close.assert_not_called()
    if fault is None: ctx.arm.move_joints.assert_called_once_with([0.,1.,-1.], duration=4.)


def test_loaded_arm_without_transport_config_refuses():
    ctx = context(); ctx.config['skills'].clear()
    assert prepare(ctx).diagnosis_code == 'not_configured'
    ctx.arm.move_joints.assert_not_called()


def test_device_interruption_preserves_cancellation_and_the_grasp():
    ctx = context()
    def interrupted(*args, **kwargs):
        ctx.services['cancel'].set()
        raise RuntimeError('interrupted wait')
    ctx.arm.move_joints.side_effect = interrupted
    assert prepare(ctx).diagnosis_code == 'cancelled'
    ctx.gripper.open.assert_not_called(); ctx.arm.home.assert_not_called()


@pytest.mark.parametrize('change', [{'posture': 'unknown'}, {'duration': float('nan')},
    {'minimum_load': 1., 'max_force': 6.}, {'extra': True}])
def test_invalid_transport_config_is_rejected(change):
    cfg = SimpleNamespace(base=True, arm=SimpleNamespace(postures={'carry': [0.]},
        gripper=SimpleNamespace(force_feedback=False)))
    with pytest.raises(EmbodimentConfigError):
        parse({'posture': 'carry', 'duration': 4., **change}, cfg, 'transport')


@pytest.mark.parametrize('fault', ['slip', 'unloaded', 'stale', 'overload', 'cancelled'])
def test_transport_monitor_stops_mid_motion_without_releasing(fault):
    from vector_os_nano.skills.transport import CarryGuard
    ctx = context()
    ctx.base.get_sim_time.return_value = 1.
    ctx.gripper.get_finger_load.side_effect = None
    ctx.gripper.get_finger_load.return_value = FingerLoad(1., (0., 0., -1.8))
    monitor = CarryGuard(ctx)
    assert not monitor.is_set()
    # Repeated cancellation polls at one clock instant must not demand a new sensor frame.
    assert not monitor.is_set()
    ctx.base.get_sim_time.return_value = 1.03
    sample = FingerLoad(1.03, (0., 0., -1.8))
    if fault == 'slip': ctx.gripper.is_holding.return_value = False
    if fault == 'unloaded': sample = FingerLoad(1.03, (0., 0., 0.))
    if fault == 'stale': sample = FingerLoad(1., (0., 0., -1.8))
    if fault == 'overload': sample = FingerLoad(1.03, (0., 0., -7.))
    if fault == 'cancelled': ctx.services['cancel'].set()
    ctx.gripper.get_finger_load.return_value = sample
    assert monitor.is_set()
    assert monitor.failure() is not None
    if fault != 'cancelled':
        ctx.base.stop.assert_called_once()
        ctx.gripper.is_holding.return_value = True
        ctx.gripper.get_finger_load.return_value = FingerLoad(2., (0., 0., -1.8))
        assert monitor.is_set(), 'a recovered sample must not resume a failed motion'
    ctx.gripper.open.assert_not_called(); ctx.gripper.close.assert_not_called()
    ctx.arm.move_joints.assert_not_called(); ctx.arm.home.assert_not_called()


@pytest.mark.parametrize('tool,params', [(WalkSkill(), {'distance': 1., 'speed': .1}),
    (TurnSkill(), {'angle': 90}), (NavigateSkill(), {'x': 2., 'y': 0.})])
def test_public_motion_aborts_during_a_lost_grasp(tool, params):
    from tests.harness.kin_base import KinBase
    from vector_os_nano.sim.nav.follow import follow
    from vector_os_nano.schemas.robot import NavSpec
    ctx = context(); base = KinBase()
    base.navigate_to = lambda x, y, **kw: follow(base, (x, y), [],
        nav=NavSpec(body_radius=.3, speed=.2, vyaw_max=.5), fallen=lambda: False, **kw)
    ctx = SkillContext(arm=ctx.arm, gripper=ctx.gripper, base=base,
                       config=ctx.config, services=ctx.services)
    ctx.arm.move_joints.side_effect = lambda *a, **kw: setattr(base, 't', base.t+.1) or True
    ctx.gripper.is_holding.side_effect = lambda: base.t < .25
    ctx.gripper.get_finger_load.side_effect = lambda: FingerLoad(base.t, (0., 0., -1.8))
    result = tool.execute(params, ctx)
    assert not result.success and result.diagnosis_code == 'not_holding', result
    assert .25 <= base.t < .30 and base.cmd == (0., 0., 0.)
    ctx.gripper.open.assert_not_called(); ctx.arm.home.assert_not_called()
