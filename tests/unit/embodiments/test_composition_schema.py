"""Folder identity, composition inheritance and namespace contracts without a simulator."""
import pytest
import yaml

from vector_os_nano.schemas import robot
from vector_os_nano.embodiments.assets import model_root, policy_root


@pytest.fixture
def folders(tmp_path, monkeypatch):
    for name in ('go2', 'piper', 'go2_piper'):
        dst = tmp_path / name / 'robot.yaml'
        dst.parent.mkdir()
        dst.write_text((robot._EMBODIMENTS_DIR / name / 'robot.yaml').read_text())
    monkeypatch.setattr(robot, '_EMBODIMENTS_DIR', tmp_path)
    return tmp_path


def change(folder, name, edit):
    path = folder / name / 'robot.yaml'
    raw = yaml.safe_load(path.read_text())
    edit(raw)
    path.write_text(yaml.safe_dump(raw))


def test_composition_inherits_whole_blocks_and_preserves_file_origins(folders):
    cfg = robot.load_embodiment_config('go2_piper')
    assert cfg.id == 'go2_piper' and cfg.base is cfg.compose.base.base
    assert model_root(cfg) == policy_root(cfg) == folders / 'go2'
    assert cfg.arm.joints[0] == 'piper_joint1'
    assert cfg.arm.gripper.finger_bodies == ('piper_link7', 'piper_link8')
    assert cfg.camera.name == 'manipulation_rgbd'  # sensors replace the base's whole list
    change(folders, 'go2_piper', lambda r: r.update(sensors=[], skills={}))
    cfg = robot.load_embodiment_config('go2_piper')
    assert cfg.sensors == () and cfg.skills == {}
    assert not cfg.capabilities.camera and not cfg.capabilities.lidar


def test_child_arm_names_are_prefixed_once(folders):
    change(folders, 'piper', lambda r: r['model'].update(prefix='tool_'))
    change(folders, 'go2_piper', lambda r: r.pop('arm'))
    cfg = robot.load_embodiment_config('go2_piper')
    assert cfg.compose.arm.arm.ee_site == 'tool_ee_site'
    assert cfg.arm.ee_site == 'piper_tool_ee_site'
    assert cfg.arm.joints[0] == 'piper_tool_joint1'


def test_composed_joint_groups_keep_namespaces_and_reject_duplicate_ownership(folders):
    group = dict(joints=['pan'], postures={'center': [0]}, velocity=[1], acceleration=[1], jerk=[1], track_tol=.01)
    change(folders, 'piper', lambda r: r.update(joint_groups={'camera': group}))
    change(folders, 'go2', lambda r: r.update(joint_groups={'torso': {**group, 'joints': ['lift']}}))
    cfg = robot.load_embodiment_config('go2_piper')
    assert cfg.joint_groups['camera'].joints == ('piper_pan',)
    assert cfg.joint_groups['torso'].joints == ('lift',)
    change(folders, 'go2_piper', lambda r: r.update(joint_groups={'camera': group}))
    with pytest.raises(robot.EmbodimentConfigError, match='distinct'):
        robot.load_embodiment_config('go2_piper')


def test_composed_sensor_overrides_follow_the_base_namespace(folders):
    change(folders, 'go2', lambda r: r['model'].update(prefix='base_'))
    cfg = robot.load_embodiment_config('go2_piper')
    assert cfg.model.root_body == 'base_base_link'
    assert cfg.base.joints[0] == 'base_FL_hip_joint'
    assert cfg.camera.body == 'base_d435_camera' and cfg.camera.name == 'base_manipulation_rgbd'
    assert cfg.arm.ee_site == 'piper_ee_site'


@pytest.mark.parametrize('key', ['base', 'arm'])
def test_unknown_composed_child_names_file_key_and_valid_set(folders, key):
    change(folders, 'go2_piper', lambda r: r['compose'].update({key: 'missing'}))
    with pytest.raises(robot.EmbodimentConfigError, match=rf'go2_piper/robot.yaml.compose.{key}:.*valid:'):
        robot.load_embodiment_config('go2_piper')


def test_composition_cycles_are_rejected(folders):
    change(folders, 'go2_piper', lambda r: r['compose'].update(base='go2_piper'))
    with pytest.raises(robot.EmbodimentConfigError, match='compose cycle: go2_piper -> go2_piper'):
        robot.load_embodiment_config('go2_piper')


def test_identity_cannot_be_authored(folders):
    change(folders, 'go2', lambda r: r.update(id='pretend'))
    with pytest.raises(robot.EmbodimentConfigError, match="unknown key 'id'"):
        robot.load_embodiment_config('go2')


@pytest.mark.parametrize('key,value', [('file', None), ('root_body', False), ('prefix', 4), ('path', 'legacy.xml')])
def test_model_keys_and_types_are_strict(folders, key, value):
    change(folders, 'go2', lambda r: r['model'].update({key: value}))
    with pytest.raises(robot.EmbodimentConfigError, match='robot.yaml.model'):
        robot.load_embodiment_config('go2')


def test_depth_is_derived_from_sensor_ranges(folders):
    assert robot.load_embodiment_config('go2').capabilities.depth
    change(folders, 'go2', lambda r: r['sensors'][0].pop('depth'))
    assert not robot.load_embodiment_config('go2').capabilities.depth


@pytest.mark.parametrize('block', [{'unknown': 1}, {'quat': [0, 0, 0, 0]}, {'pos': [float('nan'), 0, 0]}])
def test_bad_mounts_are_rejected(folders, block):
    change(folders, 'go2_piper', lambda r: r['compose']['mount'].update(block))
    with pytest.raises(robot.EmbodimentConfigError, match='compose.mount'):
        robot.load_embodiment_config('go2_piper')
