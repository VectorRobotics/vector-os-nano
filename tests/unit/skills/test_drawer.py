"""Drawer commands stop on unknown identity, bad sensors, lost grip and cancellation."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading

import numpy as np
import pytest
import yaml

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.embodiments.registry import validate
from vector_os_nano.schemas.robot import parse_embodiment_config, EmbodimentConfigError
from vector_os_nano.schemas.world import Articulation, load_world
from vector_os_nano.skills import drawer


def config():
    path = Path(__file__).resolve().parents[3]/'examples/household/embodiments/malo/robot.yaml'
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize('key,value', [('camera', 'missing'), ('view_posture', 'missing'),
    ('speed', float('nan')), ('endstop_force', 9), ('max_extent', 0), ('timeout', True),
    ('rotation', [1]*9), ('homing_margin', .001), ('prepare', {'group': 'torso', 'posture': 'missing'})])
def test_profile_rejects_unsafe_or_unbound_configuration(key, value):
    raw = config(); raw['skills']['drawer'][key] = value
    with pytest.raises(EmbodimentConfigError, match='drawer'):
        validate(parse_embodiment_config(raw))


def test_recorded_profiles_keep_their_original_homing_bound():
    raw=config();raw['skills']['drawer'].pop('homing_margin')
    parsed=validate(parse_embodiment_config(raw)).skills['drawer']
    assert parsed['homing_margin']==parsed['stroke_margin']


def context():
    cfg = validate(parse_embodiment_config(config()))
    return SkillContext(arm=Mock(), base=Mock(), gripper=Mock(is_holding=Mock(return_value=False)),
        services={'cancel': threading.Event(), 'object_world': replace(load_world('workbench'), articulations={})},
        config={'skills': cfg.skills})


@pytest.mark.parametrize('skill', [drawer.OpenDrawerSkill, drawer.CloseDrawerSkill])
def test_unknown_target_and_occupied_gripper_never_move(skill):
    ctx = context()
    assert skill().execute({'query': 'not the drawer'}, ctx).diagnosis_code == 'unknown_articulation'
    ctx.base.stop.assert_not_called(); ctx.arm.move_joints.assert_not_called()
    ctx.services['object_world'].articulations['drawer'] = Articulation('drawer', 'slide', 0, -.27, .003)
    ctx.gripper.is_holding.return_value = True
    assert skill().execute({'query': 'drawer'}, ctx).diagnosis_code == 'already_holding'
    ctx.gripper.open.assert_not_called()


@pytest.mark.parametrize('fault', ['force', 'stale', 'slip', 'cancel', 'timeout'])
def test_feedback_fault_keeps_the_current_grasp(fault):
    ctx = context(); cfg = ctx.config['skills']['drawer']
    ctx.gripper.is_holding.return_value = fault != 'slip'
    ctx.gripper.get_finger_load.side_effect = [NS(sim_time=t, force=f) for t, f in (
        (1., (0., 0., 0.)), (1. if fault == 'stale' else 2., (7., 0., 0.) if fault == 'force' else (0., 0., 0.)))]
    ctx.arm.fk.return_value = ([0., 0., 0.], np.eye(3))
    if fault == 'cancel': ctx.services['cancel'].set()
    def walk(*, duration, velocity_fn):
        velocity_fn(); velocity_fn()
        return False
    ctx.base.walk.side_effect = walk
    error = drawer._drive(ctx, np.array([1., 0., 0.]), .1, cfg, ctx.services['cancel'])
    assert error is not None and not error.success
    ctx.gripper.open.assert_not_called()


def test_endstop_is_only_a_sensor_estimate_and_search_is_bounded():
    ctx = context(); cfg = ctx.config['skills']['drawer']
    ctx.gripper.is_holding.return_value = True
    ctx.gripper.get_finger_load.return_value = NS(sim_time=1., force=(-2.1, 0., 0.))
    ctx.arm.fk.return_value = ([0., 0., 0.], np.eye(3))
    commands = []
    def walk(*, duration, velocity_fn):
        commands.append(velocity_fn())
        return commands[-1] is None
    ctx.base.walk.side_effect = walk
    assert drawer._drive(ctx, np.array([1., 0., 0.]), .28, cfg, ctx.services['cancel'], home=True) is None
    assert commands == [None]
    ctx.arm.fk.side_effect = [([0., 0., 0.], np.eye(3)), ([.29, 0., 0.], np.eye(3))]
    ctx.gripper.get_finger_load.return_value = NS(sim_time=2., force=(0., 0., 0.))
    assert drawer._drive(ctx, np.array([1., 0., 0.]), .28, cfg, ctx.services['cancel'], home=True).diagnosis_code == 'contact_failed'


@pytest.mark.parametrize('opening', [True, False])
def test_homing_compliance_allowance_never_increases_the_opening_stroke(monkeypatch, opening):
    ctx = context()
    ctx.services['object_world'].articulations['drawer'] = Articulation('drawer', 'slide', 0, -.27, .003)
    ctx.config['arm'] = {'postures': {'drawer_view': [0.]*7}}
    ctx.gripper.is_holding.side_effect = [False, True]
    ctx.base.get_heading.return_value = 0.
    ctx.arm.fk.return_value = ([0., 0., 0.], np.eye(3))
    for name in ('_group_move', '_move', '_reach'):
        monkeypatch.setattr(drawer, name, lambda *a, **kw: None)
    monkeypatch.setattr(drawer, 'perceive', lambda *a, **kw:
        (None, {'visible_bounds': [[0., 0., 0.], [.02, .02, .02]]}, None))
    drive = Mock(return_value=None);monkeypatch.setattr(drawer, '_drive', drive)
    assert drawer._operate({'query': 'drawer'}, ctx, opening=opening).success
    distances = [call.args[2] for call in drive.call_args_list]
    assert distances == pytest.approx([.31, -.003, -.28] if opening else [.31])
