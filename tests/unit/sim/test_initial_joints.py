"""Declared reset joints and mimics start together, through both physics contracts."""
from dataclasses import replace

import numpy as np
import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim import backends
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.sim.session import _initial_joints

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('backend', ['mujoco', 'pybullet'])
def test_reset_sets_positions_and_actuated_targets_without_driving_passive_mimics(backend):
    cfg = replace(registry.load('piper'), initial_joints={'joint1': -.2, 'joint7': .02, 'joint8': -.02})
    scene = compose(cfg, load_world(cfg.world))
    world = backends.get(backend).load(scene, gui=False)
    try:
        _initial_joints(world, cfg)
        q = world.state()[0]
        for name, value in cfg.initial_joints.items():
            assert q[scene.joint_q(scene.joint(name))] == pytest.approx(value)
        assert world.targets([scene.joint('joint1'), scene.joint('joint7')]) == pytest.approx([-.2, .02])
        world.step(100)
        assert np.isfinite(world.state()[0]).all()
        before = world.state()[0]
        with pytest.raises(ValueError, match='outside scalar joint limits'):
            _initial_joints(world, replace(cfg, initial_joints={'joint1': .1, 'joint7': 5}))
        np.testing.assert_array_equal(world.state()[0], before)
    finally:
        world.close()
