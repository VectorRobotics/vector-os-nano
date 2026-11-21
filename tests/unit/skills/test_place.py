"""Named placement refuses unknown geometry and checks descent, release and cancellation."""
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import numpy as np
import pytest
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.interfaces.devices import FingerLoad


def test_force_descent_stops_at_support_above_the_nominal_release_height():
    ctx = context(); samples = []
    ctx.config['skills']['place']['support'] = dict(max_descent=.02, step=.002,
        duration=.25, dwell=.5, minimum_load=.2, unload_fraction=.25, max_force=8.)
    def read():
        z = ctx.arm.get_joint_positions.return_value[2]; samples.append(z)
        load = 1.8 if z > .491 else (0. if z >= .488 else -20.)
        return FingerLoad(float(len(samples)), (0., 0., -load))
    ctx.gripper.get_finger_load.side_effect = read
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.success, result
    assert result.result_data['release_position'][2] >= .488
    assert min(samples) >= .488


def test_force_place_realigns_the_hover_after_mount_deflection():
    ctx = context(); offset = [0.]; samples = []
    ctx.config['skills']['place']['support'] = dict(max_descent=.02, step=.002,
        duration=.25, dwell=.5, minimum_load=.2, unload_fraction=.25, max_force=8.)
    ctx.arm.ik_top_down.side_effect = lambda xyz, **kw: [xyz[0]-offset[0], *xyz[1:], 0., 0., 0.]
    ctx.arm.ik.side_effect = ctx.arm.ik_top_down.side_effect
    ctx.arm.fk.side_effect = lambda q: ([q[0]+offset[0], *q[1:3]], np.eye(3))
    move = ctx.arm.move_joints.side_effect
    def deflect(q, duration):
        result = move(q, duration); offset[0] = .025
        return result
    def read():
        samples.append(ctx.arm.get_joint_positions.return_value[2])
        return FingerLoad(float(len(samples)), (0., 0., -1.8 if samples[-1] > .49 else 0.))
    ctx.arm.move_joints.side_effect = deflect
    ctx.gripper.get_finger_load.side_effect = read
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.success, result
    assert ctx.arm.fk(ctx.arm.get_joint_positions())[0][0] == pytest.approx(.35)


def context():
    arm = Mock()
    arm.get_joint_positions.return_value = [0., 0., .7, 0., 0., 0.]
    arm.ik_top_down.side_effect = lambda xyz, current_joints=None: list(xyz)+[0., 0., 0.]
    arm.ik.side_effect = lambda xyz, current_joints=None, **kw: list(xyz)+[0., 0., 0.]
    arm.fk.side_effect = lambda q: (q[:3], np.eye(3))
    def move(q, duration):
        arm.get_joint_positions.return_value = list(q)
        return True
    arm.move_joints.side_effect = move
    grip = Mock(is_holding=Mock(return_value=True))
    grip.open.side_effect = lambda: setattr(grip.is_holding, 'return_value', False) or True
    surface = SimpleNamespace(name='bench', place_point=(.35, -.12), height=.4)
    world = SimpleNamespace(surfaces={'bench': surface}, resolve_surface=lambda s: 'bench' if s == 'bench' else None)
    cfg = dict(hover_height=.12, release_height=.04, retreat_height=.1, durations=[3., 2., 2.],
               settle_duration=.3, reach_m=.7, position_tolerance=.015, keep_orientation=False)
    return SkillContext(arm=arm, gripper=grip, config={'skills': {'place': cfg}},
                        services={'object_world': world, 'cancel': threading.Event()})


@pytest.mark.parametrize('surface', ['', 'missing', 'front', 'bench or somewhere'])
def test_unknown_surface_refuses_before_motion(surface):
    ctx = context()
    result = PlaceSkill().execute({'surface': surface}, ctx)
    assert result.diagnosis_code == 'unknown_surface'
    ctx.arm.move_joints.assert_not_called()
    ctx.gripper.open.assert_not_called()


def test_empty_gripper_does_not_claim_a_placement():
    ctx = context(); ctx.gripper.is_holding.return_value = False
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'not_holding'
    ctx.arm.move_joints.assert_not_called()


def test_place_descends_then_releases_and_never_homes_or_recloses():
    ctx = context()
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.success
    assert result.result_data['surface'] == 'bench'
    points = [call.args[0][:3] for call in ctx.arm.move_joints.call_args_list]
    np.testing.assert_allclose(points[:2], [[.35, -.12, .52], [.35, -.12, .44]])
    assert points[-1][2] == pytest.approx(.54)
    ctx.arm.home.assert_not_called(); ctx.gripper.close.assert_not_called()


def test_failed_release_does_not_report_success_or_retreat_with_the_load():
    ctx = context(); ctx.gripper.open.side_effect = lambda: True
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'release_failed'
    assert ctx.arm.get_joint_positions.return_value[2] == pytest.approx(.44)


def test_unreachable_fixed_arm_refuses_before_motion():
    ctx = context(); ctx.arm.ik_top_down.side_effect = lambda *a, **kw: None
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'out_of_reach'
    ctx.arm.move_joints.assert_not_called(); ctx.gripper.open.assert_not_called()


def test_cancel_during_descent_keeps_the_hold():
    ctx = context(); move = ctx.arm.move_joints.side_effect
    def cancel(q, duration):
        ok = move(q, duration)
        if q[2] == pytest.approx(.44):
            ctx.services['cancel'].set()
        return ok
    ctx.arm.move_joints.side_effect = cancel
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'cancelled'
    ctx.gripper.open.assert_not_called()


def test_tracking_error_prevents_opening_above_the_surface():
    ctx = context(); ctx.arm.move_joints.side_effect = lambda *a, **kw: True
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'move_failed'
    ctx.gripper.open.assert_not_called()


def test_descent_resolves_ik_again_after_mount_moves_under_the_load():
    ctx = context(); offset = [0.]
    move = ctx.arm.move_joints.side_effect
    ctx.arm.ik_top_down.side_effect = lambda xyz, **kw: [xyz[0]-offset[0], *xyz[1:], 0., 0., 0.]
    ctx.arm.fk.side_effect = lambda q: ([q[0]+offset[0], *q[1:3]], np.eye(3))
    def moving_mount(q, duration):
        result = move(q, duration)
        offset[0] = .025
        return result
    ctx.arm.move_joints.side_effect = moving_mount
    assert PlaceSkill().execute({'surface': 'bench'}, ctx).success
    ctx.gripper.open.assert_called_once()


def test_newly_unreachable_descent_keeps_the_grasp():
    ctx = context(); move = ctx.arm.move_joints.side_effect
    def invalidate(q, duration):
        ok = move(q, duration)
        ctx.arm.ik_top_down.side_effect = lambda *a, **kw: None
        return ok
    ctx.arm.move_joints.side_effect = invalidate
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'ik_unreachable'
    ctx.gripper.open.assert_not_called()


def test_nonfinite_measured_position_never_authorizes_a_release():
    ctx = context(); ctx.arm.fk.side_effect = lambda q: ([float('nan'), 0, 0], np.eye(3))
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'move_failed'
    ctx.gripper.open.assert_not_called()


@pytest.mark.parametrize('missing', ['arm', 'gripper'])
def test_missing_device_is_reported(missing):
    ctx = context()
    ctx = SkillContext(arm=None if missing == 'arm' else ctx.arm,
                       gripper=None if missing == 'gripper' else ctx.gripper,
                       config=ctx.config, services=ctx.services)
    assert PlaceSkill().execute({'surface': 'bench'}, ctx).diagnosis_code == f'no_{missing}'


@pytest.mark.parametrize('phase', ['approach', 'descend', 'retreat'])
def test_motor_failure_never_reports_success(phase):
    ctx = context(); move = ctx.arm.move_joints.side_effect
    heights = dict(approach=.52, descend=.44, retreat=.54)
    ctx.arm.move_joints.side_effect = lambda q, duration: False if q[2] == pytest.approx(heights[phase]) else move(q, duration)
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'move_failed'
    assert result.result_data['phase'] == phase


def test_surface_without_a_place_point_does_not_invent_the_origin():
    ctx = context(); ctx.services['object_world'].surfaces['bench'].place_point = None
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.diagnosis_code == 'not_configured'
    ctx.arm.move_joints.assert_not_called()


def test_configured_heights_and_orientation_drive_the_same_skill():
    ctx = context(); cfg = ctx.config['skills']['place']
    cfg.update(hover_height=.20, release_height=.06, keep_orientation=True)
    assert PlaceSkill().execute({'surface': 'bench'}, ctx).success
    ctx.arm.ik_top_down.assert_not_called()
    assert ctx.arm.move_joints.call_args_list[1].args[0][2] == pytest.approx(.46)
    np.testing.assert_array_equal(ctx.arm.ik.call_args.kwargs['target_rotation'], np.eye(3))


@pytest.mark.parametrize('fault', [None, 'missing', 'stale', 'nonfinite', 'unsupported', 'slip', 'overload'])
def test_force_feedback_must_confirm_support_before_opening(fault):
    ctx = context()
    ctx.config['skills']['place']['support'] = dict(max_descent=.02, step=.002,
        duration=.25, dwell=.5, minimum_load=.2, unload_fraction=.25, max_force=8.)
    samples = []
    def read():
        z = ctx.arm.get_joint_positions.return_value[2]
        samples.append(z)
        if fault == 'missing': return None
        force = 1.8 if z > .432 or fault == 'unsupported' else 0.
        if fault == 'nonfinite': force = float('nan')
        if fault == 'overload': force = 20.
        if fault == 'slip' and z <= .432: ctx.gripper.is_holding.return_value = False
        return FingerLoad(0. if fault == 'stale' else len(samples)*.25, (0., 0., -force))
    ctx.gripper.get_finger_load.side_effect = read
    result = PlaceSkill().execute({'surface': 'bench'}, ctx)
    assert result.success == (fault is None)
    if fault:
        ctx.gripper.open.assert_not_called()
    else:
        assert result.result_data['support_confirmed'] is True
        assert sum(z <= .432 for z in samples) >= 3
        ctx.gripper.open.assert_called_once()


@pytest.mark.parametrize('drift', [False, True])
def test_mobile_approach_uses_placement_position_tolerance(drift):
    ctx = context(); base = Mock()
    base.get_sim_time.return_value = 0.
    base.get_position.return_value = [0., 0., 0.]
    base.get_heading.return_value = 0.
    ctx.services['object_world'].surfaces['bench'].place_point = (1., 0.)
    ctx.config['arm'] = {'postures': {'hold': [0.,0.,.7,0.,0.,0.]}}
    ctx.config['skills']['place']['base'] = dict(standoff=.43, lateral_offset=0.,
        carry_posture='hold', yaw_speed=.2, yaw_deadband=.03, step_duration=.5, approach_steps=40)
    ctx.arm.ik_top_down.side_effect = lambda *a, **kw: None
    def navigate(*a, **kw):
        if not drift or base.navigate_to.call_count == 2:
            ctx.arm.ik_top_down.side_effect = lambda xyz, **kw: list(xyz)+[0.,0.,0.]
        return True
    base.navigate_to.side_effect = navigate
    ctx = SkillContext(arm=ctx.arm, gripper=ctx.gripper, base=base, config=ctx.config, services=ctx.services)
    assert PlaceSkill().execute({'surface': 'bench'}, ctx).success
    assert base.navigate_to.call_args.kwargs['tol'] == .015
    assert base.navigate_to.call_count == (2 if drift else 1)


def test_support_search_backs_off_when_a_step_compresses_the_surface():
    ctx = context(); samples = []
    ctx.config['skills']['place']['support'] = dict(max_descent=.02, step=.002,
        duration=.25, dwell=.5, minimum_load=.2, unload_fraction=.25, max_force=8.)
    def read():
        z = ctx.arm.get_joint_positions.return_value[2]; samples.append(z)
        load = 1.8 if z > .4398 else -1.8 if z < .4396 else 0.
        return FingerLoad(len(samples)*.25, (0., 0., -load))
    ctx.gripper.get_finger_load.side_effect = read
    assert PlaceSkill().execute({'surface': 'bench'}, ctx).success
    assert any(b > a for a,b in zip(samples, samples[1:]))
    assert min(samples) >= .42
