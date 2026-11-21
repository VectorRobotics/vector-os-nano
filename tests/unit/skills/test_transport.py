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
