"""Frontal grasp uses measured geometry, heading and configured joint groups."""
import math
from unittest.mock import Mock
import numpy as np
import pytest
from tests.unit.skills.test_grasp import context
from vector_os_nano.core.types import Pose3D
from vector_os_nano.skills import grasp
from vector_os_nano.vcli.cognitive import abort


def frontal():
    ctx = context()
    ctx.base = Mock(get_heading=Mock(return_value=math.pi/2))
    ctx.arm.ik.return_value = [.2]*6
    ctx.config['arm']['postures']['ready'] = [.1]*6
    ctx.config['joint_groups'] = {'torso': {'postures': {'prepare': [-.15, .3], 'carry': [-.03, .3]}}}
    ctx.services['joint_groups'] = {'torso': Mock(move_joints=Mock(return_value=True))}
    cfg = ctx.config['skills']['grasp']
    cfg['frontal'] = dict(pre_offset=[-.11, 0, .1], contact_offset=[-.02, 0, .035], pitch=math.pi/4,
                          ready_posture='ready', prepare=dict(group='torso', posture='prepare'),
                          lift=dict(group='torso', posture='carry'))
    return ctx, cfg


def test_frontal_rotations_and_offsets_follow_measured_heading():
    ctx, cfg = frontal()
    pre, contact, rp, rg = grasp._poses(Pose3D(1.5, -2.65, .815), cfg, ctx.base)
    np.testing.assert_allclose(pre, [1.5, -2.76, .915])
    np.testing.assert_allclose(contact, [1.5, -2.67, .85])
    for rotation in (rp, rg):
        np.testing.assert_allclose(np.array(rotation).T @ rotation, np.eye(3), atol=1e-15)
        assert np.linalg.det(rotation) == pytest.approx(1.)
    np.testing.assert_allclose(np.array(rg)[:, 2], [0, 2**-.5, -2**-.5], atol=1e-15)
    ctx.base.get_heading.return_value = None
    with pytest.raises(ValueError, match='measured base heading'):
        grasp._poses(Pose3D(), cfg, ctx.base)


def test_frontal_centre_comes_only_from_observed_extent():
    _, cfg = frontal()
    meta = dict(grasp_world=[1.475, -2.66, .831], visible_bounds=[[1.475, -2.73, .78], [1.525, -2.57, .85]])
    point = grasp._grasp_point(Pose3D(), meta, cfg)
    assert [point.x, point.y, point.z] == pytest.approx([1.5, -2.65, .815])
    assert 'unverified' in meta['geometry']
    with pytest.raises(ValueError, match='observed package bounds'):
        grasp._grasp_point(Pose3D(), {}, cfg)


def test_lateral_mount_offset_does_not_turn_the_base_toward_the_arm():
    ctx, _ = frontal()
    ctx.base.get_position.return_value = [1.7, -3.7, 0.]
    forward, lateral, yaw, _, _ = grasp._offset(ctx.base, Pose3D(1.5, -2.65, .8), .2)
    assert [forward, lateral, yaw] == pytest.approx([1.05, 0., 0.])


def test_too_close_frontal_grasp_refuses_before_extending_the_arm(monkeypatch):
    abort.clear_abort()
    ctx, cfg = frontal()
    cfg['base'] = dict(view_standoff=1.05, lateral_offset=.2, lateral_deadband=.015)
    ctx.base.get_position.return_value = [1.7, -3.33, 0.]
    meta = dict(grasp_world=[1.5, -2.65, .815], visible_bounds=[[1.475,-2.73,.78],[1.525,-2.57,.85]])
    monkeypatch.setattr(grasp, 'point', lambda *a: (Pose3D(1.5,-2.65,.815), meta, None))
    result = grasp.GraspSkill().execute({'query': 'apple'}, ctx)
    assert result.diagnosis_code == 'approach_failed'
    ctx.arm.move_joints.assert_not_called()
    ctx.base.walk.assert_not_called()


@pytest.mark.parametrize('failure', [None, 'prepare', 'lift', 'cancel', 'slip'])
def test_frontal_group_outcomes_and_cancellation_cannot_report_success(failure):
    abort.clear_abort()
    ctx, cfg = frontal(); token = abort.current_token()
    group = ctx.services['joint_groups']['torso']
    if failure == 'prepare':
        group.move_joints.return_value = False
    error = grasp.view(ctx, cfg, token)
    if failure == 'prepare':
        assert error.diagnosis_code == 'move_failed'
        ctx.gripper.close.assert_not_called()
        return
    assert error is None
    if failure == 'lift':
        group.move_joints.return_value = False
    if failure == 'cancel':
        def cancel(*args, **kw):
            token.cancel(); return False
        group.move_joints.side_effect = cancel
    ctx.gripper.is_holding.side_effect = [True, failure != 'slip']
    result = grasp.grip(ctx, Pose3D(1.5, -2.65, .815), cfg, token, 'package')
    if failure:
        assert result.diagnosis_code == {'lift': 'move_failed', 'cancel': 'cancelled', 'slip': 'not_holding'}[failure]
    else:
        assert result is None
    ctx.arm.ik_top_down.assert_not_called()
    ctx.arm.home.assert_not_called()
    ctx.gripper.open.assert_called_once()
    abort.clear_abort()
