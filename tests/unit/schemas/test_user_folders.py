"""Local data folders are discovered dynamically, without replacing shipped identities."""
from pathlib import Path

import pytest
import yaml

from vector_os_nano.embodiments import registry
from vector_os_nano.embodiments.assets import model_root
from vector_os_nano.schemas import robot, world


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path))
    return tmp_path


def write(folder, filename, data):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(yaml.safe_dump(data))
    return path


def test_robot_and_composed_robot_are_data_alone(state):
    before = registry.embodiment_ids()
    raw = yaml.safe_load((robot._EMBODIMENTS_DIR / 'piper' / 'robot.yaml').read_text())
    folder = state / 'embodiments' / 'custom_arm'
    write(folder, 'robot.yaml', raw)
    cfg = registry.load('custom_arm')
    assert registry.embodiment_ids() == sorted([*before, 'custom_arm'])
    assert cfg.id == 'custom_arm' and model_root(cfg) == folder
    write(state / 'embodiments' / 'custom_mobile', 'robot.yaml', {'schema': 1,
          'compose': {'base': 'go2', 'arm': 'custom_arm',
                      'mount': {'body': 'base_link', 'pos': [0, 0, .1], 'quat': [1, 0, 0, 0]}}})
    cfg = registry.load('custom_mobile')
    assert cfg.compose.arm.path == folder / 'robot.yaml'
    assert cfg.arm.ee_site == 'custom_arm_ee_site'


def test_new_world_and_edits_are_seen_without_restart(state):
    from vector_os_nano.perception.vocab import vocab

    assert 'custom_home' not in world.world_folders()
    raw = {'schema': 1, 'scene': 'scene.xml', 'spawn': {'xy': [0, 0], 'z': 0, 'heading': 0},
           'vocab': {'categories': {'tissue': ['纸巾包']}}}
    folder = state / 'worlds' / 'custom_home'
    path = write(folder, 'world.yaml', raw)
    assert world.load_world('custom_home').scene == folder / 'scene.xml'
    assert ('纸巾包', 'tissue') in vocab().nouns
    raw['spawn']['xy'] = [2, 3]
    path.write_text(yaml.safe_dump(raw))
    assert world.load_world('custom_home').spawn[:2] == (2., 3.)


@pytest.mark.parametrize('kind,name,filename,load', [
    ('embodiments', 'piper', 'robot.yaml', registry.load),
    ('worlds', 'workbench', 'world.yaml', world.load_world),
])
def test_duplicate_identity_names_both_locations(state, kind, name, filename, load):
    write(state / kind / name, filename, {'schema': 1})
    with pytest.raises(ValueError, match=f'duplicate {kind} id') as exc:
        load(name)
    assert str(state / kind / name / filename) in str(exc.value)
    assert str(Path(robot.__file__).parents[1] / kind / name / filename) in str(exc.value)
