# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Target identity, common perception gates, cancellation and checked motor outcomes."""
import ast
import inspect
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.unit.perception.test_object_localizer import _StubPerception
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Pose3D, SkillResult
from vector_os_nano.schemas.robot import load_embodiment_config
from vector_os_nano.skills import grasp, grasp_perception
from vector_os_nano.skills.grasp import GraspSkill
from vector_os_nano.skills.grasp_perception import _target as _resolve, perceive
from vector_os_nano.schemas.world import load_world

def _target(query, world, front):
    return _resolve(query, load_world(world) if world else None, front)
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.gripper import GripperOpenSkill, GripperCloseSkill
from vector_os_nano.vcli.cognitive import abort


@pytest.fixture(autouse=True)
def token():
    abort.clear_abort()
    yield abort.current_token()
    abort.clear_abort()


def context(body="piper"):
    cfg = load_embodiment_config(body)
    arm = Mock(ik_top_down=Mock(return_value=[0.] * cfg.arm.dof),
               get_joint_positions=Mock(return_value=[0.] * cfg.arm.dof),
               fk=Mock(return_value=([0., 0., .4], np.eye(3))),
               move_joints=Mock(return_value=True), home=Mock(return_value=True))
    grip = Mock(open=Mock(return_value=True), close=Mock(return_value=True), is_holding=Mock(return_value=False))
    grip.close.side_effect = lambda: setattr(grip.is_holding, "return_value", True) or True
    return SkillContext(arm=arm, gripper=grip, perception=_StubPerception(),
                        services={"cancel": abort.current_token()}, config={"skills": cfg.skills, "arm": {"postures": cfg.arm.postures}})


@pytest.mark.parametrize("query", ["red box and green bottle", "take the red box twice", "墙", "桌子", "拿起不存在的物体"])
def test_unknown_target_causes_no_motion(query):
    ctx = context()
    ctx.config["world"] = "workbench"
    ctx.services["object_world"] = load_world("workbench")
    result = GraspSkill().execute({"query": query}, ctx)
    assert result.diagnosis_code == "unknown_object"
    ctx.arm.move_joints.assert_not_called()
    ctx.gripper.open.assert_not_called()


def test_colour_is_not_an_identity_when_two_objects_share_it():
    target = _target("red mug", "tabletop", True)
    assert target.label == "mug" and target.front is False
    assert _target("banana", "tabletop", True).front is True


@pytest.mark.parametrize("query", ["前面的东西", "the thing in front", "nearest object"])
def test_deictic_target_uses_space(query):
    target = _target(query, "apartment", True)
    assert target.front and target.colour is None


def test_holding_and_unconfigured_never_open_or_move():
    ctx = context()
    ctx.gripper.is_holding.return_value = True
    assert GraspSkill().execute({"query": "apple"}, ctx).diagnosis_code == "already_holding"
    ctx.config = {}
    assert GraspSkill().execute({"query": "apple"}, ctx).diagnosis_code == "not_configured"
    ctx.arm.move_joints.assert_not_called()
    ctx.gripper.open.assert_not_called()


@pytest.mark.parametrize("phase", ["view", "approach", "pre_grasp", "descent", "lift"])
def test_cancel_between_any_arm_phases_keeps_hold_and_owns_result(phase, token):
    ctx = context()
    def stopped(*args, **kwargs):
        token.cancel()
        return False
    ctx.arm.move_joints.side_effect = stopped
    result = grasp._move(ctx.arm, [0.]*6, 1., phase, token)
    assert result.diagnosis_code == "cancelled"
    ctx.gripper.open.assert_not_called()


@pytest.mark.parametrize("failure", ["open", "close", "descent", "lift", "slip"])
def test_failed_motor_phase_never_reports_grasp_success(failure, token):
    ctx = context()
    if failure in {"open", "close"}:
        getattr(ctx.gripper, failure).side_effect = None
        getattr(ctx.gripper, failure).return_value = False
    if failure in {"descent", "lift"}:
        ctx.arm.move_joints.side_effect = [True, False] if failure == "descent" else [True, True, True, False]
    if failure == "slip":
        ctx.gripper.is_holding.side_effect = [True, False]
    result = grasp.grip(ctx, Pose3D(.3, 0., .4), ctx.config["skills"]["grasp"], token)
    assert result is not None and not result.success
    assert result.diagnosis_code in {"gripper_failed", "move_failed", "not_holding"}


def test_failed_grasp_retains_observation_and_command_for_diagnosis():
    ctx = context()
    ctx.gripper.close.side_effect = lambda: True
    result = GraspSkill().execute({'query': 'red box'}, ctx)
    assert result.diagnosis_code == 'not_holding'
    assert result.result_data['perceived']
    assert len(result.result_data['command_target']) == 3
    assert len(result.result_data['visible_bounds']) == 2


def test_descent_corrects_mount_deflection_before_closing(token):
    ctx = context(); offset = [0.]
    ctx.config['skills']['grasp']['grasp_z_above'] = 0.
    ctx.arm.get_joint_positions.return_value = [0., 0., .7, 0., 0., 0.]
    ctx.arm.ik_top_down.side_effect = lambda p, **kw: [p[0]-offset[0], *p[1:], 0., 0., 0.]
    ctx.arm.fk.side_effect = lambda q: ([q[0]+offset[0], *q[1:3]], np.eye(3))
    def move(q, duration):
        ctx.arm.get_joint_positions.return_value = list(q)
        if q[2] == pytest.approx(.4): offset[0] = .03
        return True
    ctx.arm.move_joints.side_effect = move
    def close():
        tip = ctx.arm.fk(ctx.arm.get_joint_positions())[0]
        ctx.gripper.is_holding.return_value = math.dist(tip, [.3, 0., .4]) < .005
        return True
    ctx.gripper.close.side_effect = close
    assert grasp.grip(ctx, Pose3D(.3, 0., .4), ctx.config['skills']['grasp'], token) is None


def test_home_uses_controller_and_refuses_to_move_a_load():
    ctx = context()
    assert HomeSkill().execute({}, ctx).success
    ctx.arm.home.assert_called_once_with()
    ctx.arm.home.reset_mock()
    ctx.gripper.open.reset_mock()
    ctx.gripper.is_holding.return_value = True
    result = HomeSkill().execute({}, ctx)
    assert result.diagnosis_code == "holding_object"
    ctx.arm.home.assert_not_called()
    ctx.gripper.open.assert_not_called()


@pytest.mark.parametrize("skill,method", [(GripperOpenSkill, "open"), (GripperCloseSkill, "close")])
def test_gripper_failure_is_the_drivers_result(skill, method):
    ctx = context()
    getattr(ctx.gripper, method).side_effect = None
    getattr(ctx.gripper, method).return_value = False
    assert not skill().execute({}, ctx).success


def test_detector_and_passed_box_use_the_same_plausibility_gate():
    p = _StubPerception()
    p._depth[:] = 1.5  # the box is painted on a flat wall
    target = _target("apple", None, False)
    frame = p.capture()
    provenance = {"size": [64, 48], "cam_pos": frame.cam_pos.tolist(), "cam_mat": frame.cam_mat.tolist()}
    for params in ({}, {"bbox": [29, 21, 35, 27], "frame": provenance}):
        point, _, error = perceive(p, "apple", target, params)
        assert point is None and error.diagnosis_code == "implausible_target"


def test_stale_box_requires_current_detection_and_preserves_failure_cause():
    p = _StubPerception()
    p.detect_in = Mock(side_effect=RuntimeError("weights missing at /private/models/model.bin"))
    result = perceive(p, "apple", _target("apple", None, False), {"bbox": [29, 21, 35, 27]})[2]
    assert result.diagnosis_code == "perception_failed"
    assert "weights missing" in result.error_message and "/private/" not in result.error_message


def test_forward_progress_is_heading_relative():
    gp = Pose3D(0., 1., .4)
    base = SimpleNamespace(get_position=lambda: [0., 0., .3], get_heading=lambda: math.pi / 2)
    forward, lateral, angle, _, _ = grasp._offset(base, gp)
    assert forward == pytest.approx(1.) and abs(lateral) < 1e-6 and abs(angle) < 1e-6


def test_declared_failure_vocabulary_is_exactly_what_the_skill_emits():
    emitted = set()
    for module in (grasp, grasp_perception):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_fail":
                emitted.add(ast.literal_eval(node.args[0]))
    assert set(GraspSkill.failure_modes) == emitted


def test_reachable_mobile_target_is_faced_before_fresh_observation(monkeypatch, token):
    ctx = context("go2_piper")
    ctx.base = Mock()
    face = Mock(return_value=None)
    monkeypatch.setattr(grasp, "face", face)
    gp = Pose3D(.3, .1, .4)
    moved, error = grasp.reach(ctx, gp, ctx.config["skills"]["grasp"], token)
    assert moved and error is None
    face.assert_called_once_with(ctx.base, gp, ctx.config["skills"]["grasp"]["base"], token)


def test_wrong_held_identity_is_not_reported_as_success(token):
    ctx = context()
    ctx.gripper.is_holding.side_effect = lambda name=None: name is None or name == "other"
    result = grasp.grip(ctx, Pose3D(.3, 0., .4), ctx.config["skills"]["grasp"], token, "target")
    assert result.diagnosis_code == "not_holding"
    assert ctx.arm.move_joints.call_count == 3  # alignment, but no lift of the wrong object


def test_world_without_injected_resolver_refuses_before_motion():
    ctx = context()
    ctx.config["world"] = "workbench"
    result = GraspSkill().execute({"query": "red box"}, ctx)
    assert result.diagnosis_code == "not_configured"
    ctx.arm.move_joints.assert_not_called()


def test_cancel_while_closing_owns_the_result(token):
    ctx = context()
    ctx.gripper.close.side_effect = lambda: token.cancel() or True
    result = grasp.grip(ctx, Pose3D(.3, 0., .4), ctx.config["skills"]["grasp"], token)
    assert result.diagnosis_code == "cancelled"


def test_ik_retry_changes_the_base_pose_before_trying_again(monkeypatch, token):
    ctx = context("go2_piper")
    ctx.base = Mock(limits=(.5, .3, .6), get_position=Mock(return_value=(0., 0., .3)),
                    get_heading=Mock(return_value=0.), walk=Mock(return_value=True))
    monkeypatch.setattr(grasp, "face", lambda *args: None)
    moved, error = grasp.reach(ctx, Pose3D(.5, 0., .4), ctx.config["skills"]["grasp"], token, reseat=True)
    assert moved and error is None
    ctx.base.walk.assert_called_once()


@pytest.mark.parametrize('target_present', [True, False])
def test_reobservation_does_not_switch_to_a_higher_scoring_different_object(monkeypatch, target_present):
    p = _StubPerception()
    p.detect_in = Mock(return_value=[SimpleNamespace(bbox=[0, 0, 10, 10], confidence=.9),
                                     SimpleNamespace(bbox=[10, 0, 20, 10], confidence=.8)])
    p.segment = Mock(return_value=np.ones((48, 64), dtype=bool))
    cloud = np.array([(x, y, z) for x in (-.02,.02) for y in (-.07,.07) for z in (.78,.85)])
    target = cloud + [1.5, -2.65, 0]
    wrong = target + [0, .35, 0]
    monkeypatch.setattr(grasp_perception, 'object_points', Mock(side_effect=[(wrong, wrong),
                         (target, target) if target_present else (wrong, wrong)]))
    monkeypatch.setattr(grasp_perception, 'implausible', lambda *a, **kw: None)
    monkeypatch.setattr(grasp_perception, 'grasp_point_from_points', lambda cam, world, *a: Pose3D(*world.mean(axis=0)))
    prior = [[1.47,-2.73,.77],[1.53,-2.57,.86]]
    gp, meta, error = perceive(p, 'package', _target('package', None, False), expected_bounds=prior)
    if target_present:
        assert error is None and [gp.x,gp.y,gp.z] == pytest.approx([1.5,-2.65,.815])
        assert meta['association_bounds'] == prior
    else:
        assert gp is None and error.diagnosis_code == 'implausible_target'
        assert 'reacquire its identity' in error.error_message


def test_memory_is_used_only_to_navigate_then_reobserve(monkeypatch, token):
    ctx = context("go2_piper")
    ctx.base = Mock(limits=(.5, .3, .6), get_position=Mock(return_value=(0., 0., .3)),
                    get_heading=Mock(return_value=0.), navigate_to=Mock(return_value=True))
    ctx.config["skills"]["grasp"]["base"]["scan_steps"] = 0
    ctx.services["spatial_memory"] = Mock(find_objects_by_category=Mock(return_value=[SimpleNamespace(x=3., y=1., z=.4)]))
    fresh = Pose3D(3.05, 1.02, .41)
    ctx.base.navigate_to.side_effect = lambda *a, **k: setattr(ctx.base.get_position, "return_value", (2.3, .8, .3)) or True
    perceive = Mock(side_effect=[(None, {}, grasp._fail("no_detections", "not visible")), (fresh, {}, None)])
    monkeypatch.setattr(grasp, "perceive", perceive)
    monkeypatch.setattr(grasp, "face", lambda *args: None)
    gp, meta, error = grasp.point(ctx, "green bottle", _target("green bottle", "apartment", True), {}, ctx.config["skills"]["grasp"], token)
    assert gp is fresh and error is None and meta["memory_used_for_navigation"]
    assert ctx.base.navigate_to.call_args.kwargs["cancel"] is token
    assert perceive.call_count == 2


def test_cancel_in_approach_prevents_another_base_step(token):
    ctx = context("go2_piper")
    ctx.arm.ik_top_down.return_value = None
    ctx.base = Mock(limits=(.5, .3, .6), get_position=Mock(return_value=(0., 0., .3)),
                    get_heading=Mock(return_value=0.))
    ctx.base.walk.side_effect = lambda **k: token.cancel() or False
    _, error = grasp.reach(ctx, Pose3D(.8, 0., .4), ctx.config["skills"]["grasp"], token)
    assert error.diagnosis_code == "cancelled"
    ctx.base.walk.assert_called_once()


@pytest.mark.parametrize('failure', [None, 'slip', 'blocked'])
def test_grasp_without_carry_posture_still_waits_and_checks_the_hold(failure, token):
    ctx = context()
    ctx.config['arm']['postures'] = {}
    cfg = ctx.config['skills']['grasp']
    def move(q, duration):
        if ctx.arm.move_joints.call_count == 5:
            assert duration == cfg['hold_duration'] and q == ctx.arm.get_joint_positions()
            if failure == 'slip': ctx.gripper.is_holding.return_value = False
            if failure == 'blocked': return False
        return True
    ctx.arm.move_joints.side_effect = move
    result = grasp.grip(ctx, Pose3D(.3, 0., .4), cfg, token)
    assert ctx.arm.move_joints.call_count == 5
    assert result is None if failure is None else result.diagnosis_code in {'not_holding', 'move_failed'}
