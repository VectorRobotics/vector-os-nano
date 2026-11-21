"""A cache hit compiles nothing, returns an isolated model and tracks content changes."""
from dataclasses import replace
import os
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.scene.compose import compose

pytestmark = pytest.mark.sim


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path / 'state'))
    (tmp_path / 'tet.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    (tmp_path / 'body.xml').write_text('<mujoco><worldbody><body name="base_link"><geom mesh="tet" mass="1"/></body></worldbody></mujoco>')
    robot_file = tmp_path / 'robot.xml'
    robot_file.write_text('<mujoco><asset><mesh name="tet" file="tet.obj"/></asset><include file="body.xml"/></mujoco>')
    world_file = tmp_path / 'world.xml'
    world_file.write_text('<mujoco><worldbody><geom type="plane" size="10 10 .1"/></worldbody></mujoco>')
    cfg = registry.load('piper')
    cfg = replace(cfg, model=replace(cfg.model, file=str(robot_file)), arm=None, sensors=())
    world = replace(load_world('workbench'), scene=world_file, pickables={})
    return cfg, world


def test_cache_hit_never_compiles_and_models_are_not_shared(pair, monkeypatch):
    first = compose(*pair)
    masses = first.model.body_mass.copy()
    first.model.body_mass[:] = 99

    def forbidden(*args, **kwargs):
        raise AssertionError('a cache hit must not invoke a compiler or serializer')

    monkeypatch.setattr(mj.MjSpec, 'compile', forbidden)
    monkeypatch.setattr(mj.MjSpec, 'to_xml', forbidden)
    monkeypatch.setattr(mj.MjModel, 'from_xml_path', forbidden)
    second = compose(*pair)
    assert second.xml_path == first.xml_path
    assert second.model is not first.model
    np.testing.assert_array_equal(second.model.body_mass, masses)
    assert second.xml_path.with_suffix('.mjb').is_file()


@pytest.mark.parametrize('file,old,new', [('tet.obj', 'v 1 0 0', 'v 2 0 0'), ('body.xml', 'mass="1"', 'mass="2"')])
def test_assets_and_includes_are_hashed_by_bytes_even_if_stat_matches(pair, file, old, new):
    first = compose(*pair)
    asset = Path(pair[0].model.file).parent / file
    stat = asset.stat()
    asset.write_text(asset.read_text().replace(old, new))
    os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert asset.stat().st_size == stat.st_size
    second = compose(*pair)
    assert second.xml_path != first.xml_path
    assert not list(first.xml_path.parent.glob('*.tmp'))


def test_manifest_spawn_invalidates_cache(pair):
    cfg, world = pair
    first = compose(cfg, world)
    second = compose(cfg, replace(world, spawn=(2., 3., 0., 0.)))
    assert second.xml_path != first.xml_path
    np.testing.assert_array_equal(second.model.body_pos[second.robot_root], [2., 3., 0.])


def test_session_symlinks_reuse_the_same_compiled_scene(pair, tmp_path):
    cfg, world = pair
    first = compose(cfg, world)
    aliases = []
    for name, path in [('robot', cfg.path), ('world', world.path), ('scene', world.scene)]:
        link = tmp_path/(name+path.suffix)
        link.symlink_to(path.resolve())
        aliases.append(link)
    second = compose(replace(cfg, path=aliases[0]), replace(world, path=aliases[1], scene=aliases[2]))
    assert second.xml_path == first.xml_path


def test_controller_and_semantic_map_changes_do_not_recompile_physics(pair, monkeypatch):
    cfg, world = pair
    first = compose(cfg, world)
    def forbidden(*args, **kwargs):
        raise AssertionError('controller and semantic-map metadata do not alter the compiled scene')
    monkeypatch.setattr(mj.MjSpec, 'compile', forbidden)
    cfg = replace(cfg, skills={'place': {'settle_duration': 2.}})
    world = replace(world, rooms={'desk': (1., 2.)}, surfaces={})
    assert compose(cfg, world).xml_path == first.xml_path


def test_placed_camera_changes_invalidate_cache(pair):
    cfg, world = pair
    camera = replace(registry.load('piper').camera, name='inspection', body=None, pos=(0., 0., 2.))
    first = compose(replace(cfg, sensors=(camera,)), world)
    second = compose(replace(cfg, sensors=(replace(camera, fovy=60.),)), world)
    assert first.xml_path != second.xml_path
    assert second.model.cam_fovy[second.cameras['inspection']] == 60.


def test_physics_timestep_changes_invalidate_cache(pair):
    cfg, world = pair
    base = replace(registry.load('go2').base, joints=(), stance={}, postures={})
    first = compose(replace(cfg, base=base), world)
    step = base.policy.timestep / 2
    second = compose(replace(cfg, base=replace(base, policy=replace(base.policy, timestep=step))), world)
    assert first.xml_path != second.xml_path
    assert second.model.opt.timestep == step


def test_disabling_assisted_grasp_invalidates_cached_constraints(tmp_path, monkeypatch):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path))
    cfg, world = registry.load('piper'), load_world('workbench')
    assisted = compose(cfg, world)
    cfg = replace(cfg, arm=replace(cfg.arm, gripper=replace(cfg.arm.gripper, assist=False)))
    contact = compose(cfg, world)
    assert contact.xml_path != assisted.xml_path
    assert len(assisted.grasp_pairs) == len(world.pickables) and not contact.grasp_pairs
    assert assisted.model.neq - contact.model.neq == len(world.pickables)


def test_corrupt_binary_is_regenerated(pair):
    first = compose(*pair)
    first.xml_path.with_suffix('.mjb').write_bytes(b'interrupted binary')
    second = compose(*pair)
    assert second.xml_path == first.xml_path
    np.testing.assert_array_equal(second.model.body_mass, first.model.body_mass)
