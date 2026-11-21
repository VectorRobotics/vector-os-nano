"""Household examples declare physical controls without shipping private CAD."""
from pathlib import Path
import pytest
import yaml

from vector_os_nano.embodiments.registry import validate
from vector_os_nano.schemas.robot import load_embodiment_config, parse_embodiment_config, EmbodimentConfigError
from vector_os_nano.schemas.world import load_world_file


def test_household_manifests_preserve_wheel_contact_and_disable_grasp_assistance(monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'examples/household'
    monkeypatch.setenv('VECTOR_HOME', str(root))
    robot = validate(load_embodiment_config('malo'))
    world = load_world_file(root/'worlds/artvip_home/world.yaml')
    assert robot.world == world.id == 'artvip_home'
    assert robot.base.policy.ref == 'swerve' and len(robot.base.joints) == 8
    assert robot.base.policy.params['wheel_speed'] < 10.47
    assert robot.arm.dof == 7 and not robot.arm.gripper.assist
    assert 0 < robot.initial_joints['head_pitch_joint'] < .30
    assert robot.arm.home[1] == robot.initial_joints['left_arm_joint_2'] > 0
    assert robot.initial_joints['right_arm_joint_2'] > 0
    assert set(world.pickables) == {'tissue_refill', 'sponge_pack_1', 'sponge_pack_2', 'zip_bag_box'}
    assert all(p.surface in world.surfaces for p in world.pickables.values())
    assert not (root/'embodiments/malo/physical.xml').exists()


@pytest.mark.parametrize('initial', [{'head': float('nan')}, {'head': True}, {'': 0}, {'fl_wheel_joint': 1}])
def test_initial_joints_reject_invalid_positions_and_base_overrides(initial):
    path = Path(__file__).resolve().parents[2]/'examples/household/embodiments/malo/robot.yaml'
    raw = yaml.safe_load(path.read_text())
    raw['initial_joints'] = initial
    with pytest.raises(EmbodimentConfigError, match='initial_joints'):
        parse_embodiment_config(raw, str(path))
