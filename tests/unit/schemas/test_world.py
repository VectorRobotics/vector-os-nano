"""World input validation precedes model compilation or actor use."""
from copy import deepcopy

import pytest

from vector_os_nano.schemas import world


BASE = dict(schema=1, scene='scene.xml', spawn=dict(xy=[0, 0], z=0, heading=0))


@pytest.mark.parametrize('key,value,match', [
    ('schema', 2, 'upgrade vector_os_nano'), ('schema', True, 'schema'),
    ('schema', 0, 'schema'), ('unknown', 0, 'unknown.*valid'),
    ('scene', '../scene.xml', 'scene'), ('scene', '/scene.xml', 'scene'),
    ('rooms', {'kitchen': [float('nan'), 0]}, 'rooms.kitchen'),
    ('boxes', {'table': [0, 0, float('inf'), 1]}, 'boxes.table'),
    ('doors', {'entry': '00'}, 'doors.entry'),
    ('spawn', {'xy': [0, 0], 'z': 0, 'heading': True}, 'spawn.heading'),
    ('spawn', {'xy': [0, 0], 'z': 0, 'heading': 0, 'yaw': 0}, 'yaw.*valid'),
    ('vocab', {'rooms': {'missing': ['room']}}, 'vocab.rooms'),
    ('vocab', {'colours': {'red': 'red'}}, 'vocab.colours.red'),
    ('user', {'xy': [0, 0], 'room': 'kitchen'}, 'room.*valid'),
    ('surfaces', {'table': {'body': 'table', 'box': [0, 0, 1, 1], 'height': '1'}}, 'surfaces.table.height'),
    ('pickables', {'box': {'category': 'box', 'colour': 'red', 'aliases': 'box'}}, 'pickables.box.aliases'),
])
def test_invalid_world_names_file_and_key(tmp_path, key, value, match):
    raw = deepcopy(BASE)
    raw[key] = value
    path = tmp_path / 'world.yaml'
    with pytest.raises(world.WorldManifestError, match=match) as exc:
        world.parse_world(raw, 'test', path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize('key', ['schema', 'scene', 'spawn'])
def test_required_keys(tmp_path, key):
    raw = deepcopy(BASE)
    del raw[key]
    with pytest.raises(world.WorldManifestError, match=f'missing keys.*{key}'):
        world.parse_world(raw, 'test', tmp_path / 'world.yaml')


def test_duplicate_keys_and_path_traversal_rejected(tmp_path, monkeypatch):
    folder = tmp_path / 'test'
    folder.mkdir()
    (folder / 'world.yaml').write_text('schema: 1\nschema: 1\n')
    monkeypatch.setattr(world, 'WORLDS_DIR', tmp_path)
    with pytest.raises(world.WorldManifestError, match='duplicate key schema'):
        world.load_world('test')
    with pytest.raises(world.WorldManifestError, match='unknown world.*valid'):
        world.load_world('../test')


@pytest.mark.parametrize('name', ['apartment', 'tabletop', 'workbench'])
def test_shipped_worlds_are_self_contained(name):
    manifest = world.load_world(name)
    assert manifest.path.name == 'world.yaml'
    assert manifest.path.parent.name == name
    assert manifest.scene.is_file() and manifest.scene.parent == manifest.path.parent
    assert len(manifest.spawn) == 4
