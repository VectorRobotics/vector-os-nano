"""Named servo groups cannot overlap devices or manufacture verification evidence."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
import yaml

from vector_os_nano.schemas.robot import parse_embodiment_config, EmbodimentConfigError
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills import get_skills
from vector_os_nano.skills.joint_posture import JointPostureSkill
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.vcli.worlds.sim_oracles import build_namespace
from vector_os_nano.vcli.cognitive.actor_causation import ActorBaseline, _call_caused, capture, commanded_nothing


def config():
    path = Path(__file__).resolve().parents[3]/'examples/household/embodiments/malo/robot.yaml'
    return yaml.safe_load(path.read_text()), str(path)


def test_groups_prefix_register_and_reject_duplicate_or_invalid_control():
    raw, path = config()
    raw['model']['prefix'] = 'mounted_'
    cfg = parse_embodiment_config(raw, path)
    assert cfg.joint_groups['torso'].joints == ('mounted_torso_lift_joint', 'mounted_waist_pitch_joint')
    assert 'joint_posture' in {s.name for s in get_skills(cfg.capabilities)}
    assert 'joint_posture' not in {s.name for s in get_skills(replace(cfg.capabilities, joint_groups=False))}
    raw['joint_groups']['torso']['joints'][0] = 'left_arm_joint_1'
    with pytest.raises(EmbodimentConfigError, match='already controlled'):
        parse_embodiment_config(raw, path)


@pytest.mark.parametrize('key,value', [('velocity', [0, 1]), ('jerk', [1]),
    ('track_tol', float('nan')), ('postures', {'bad': [0, float('inf')]}), ('unknown', True)])
def test_bad_group_configuration_is_refused(key, value):
    raw, path = config()
    raw['joint_groups']['torso'][key] = value
    with pytest.raises(EmbodimentConfigError):
        parse_embodiment_config(raw, path)


def test_pose_skill_cancel_invalid_request_and_stop_preserve_load():
    group, grip = Mock(), Mock()
    group.move_joints.return_value = True
    group.get_joint_positions.return_value = [-.15, .2]
    token = threading.Event()
    ctx = SkillContext(gripper=grip, services={'joint_groups': {'torso': group}, 'cancel': token},
                       config={'joint_groups': {'torso': {'postures': {'shelf': [-.15, .2]}}}})
    skill = JointPostureSkill()
    assert skill.execute({'group': 'missing', 'posture': 'shelf'}, ctx).diagnosis_code == 'unknown_group'
    assert skill.execute({'group': 'torso', 'posture': 'missing'}, ctx).diagnosis_code == 'unknown_posture'
    group.move_joints.assert_not_called()
    assert skill.execute({'group': 'torso', 'posture': 'shelf'}, ctx).success
    group.move_joints.assert_called_once_with([-.15, .2])
    token.set()
    assert skill.execute({'group': 'torso', 'posture': 'shelf'}, ctx).diagnosis_code == 'cancelled'
    assert StopSkill().execute({}, ctx).success
    group.stop.assert_called_once()
    grip.open.assert_not_called()


def test_oracle_reads_ground_truth_and_causation_is_scoped_to_requested_group():
    raw, path = config()
    cfg = replace(parse_embodiment_config(raw, path), arm=None)
    q = [0., 0.]
    gt = SimpleNamespace(has_base=False, joint_positions=lambda names: list(q))
    ns = build_namespace(gt, None, cfg)
    oracle = ns['joint_group_at']
    assert not oracle('torso', 'shelf') and not oracle('torso', 'undefined')
    q[:] = [-.15, .2]
    assert oracle('torso', 'shelf')
    q[0] = float('nan')
    assert not oracle('torso', 'shelf')
    q[:] = [-.15, .2]
    dev = SimpleNamespace(joint_names=['a', 'b'], ctrl_motion=lambda: .35,
                          get_joint_positions=lambda: [999, 999])
    agent = SimpleNamespace(_services={'joint_groups': {'torso': dev}})
    measured = capture(agent, SimpleNamespace(gt=gt, uid='test'))
    assert measured.joint_groups['torso'] == (.35, (-.15, .2))
    before = replace(measured, joint_groups={'torso': (0., (0., 0.)), 'head': (0., (0.,))})
    call = ast.parse("joint_group_at('torso', 'shelf')", mode='eval').body
    def caused(after):
        return _call_caused('joint_group_at', call, 'joints', 'rise', before, after)
    assert caused(measured)
    assert not caused(replace(measured, joint_groups={'torso': (0., (-.15, .2))}))  # teleport
    assert not caused(replace(measured, joint_groups={'torso': (.35, (0., 0.))}))  # blocked
    assert not caused(replace(measured, joint_groups={'head': (.35, (-.15, .2))}))
    assert not commanded_nothing(agent, before, measured)
    assert not commanded_nothing(agent, ActorBaseline(), ActorBaseline())
    assert commanded_nothing(agent, measured, measured)
