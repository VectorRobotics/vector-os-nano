"""The recording is a copy of actual stepping and rendering, not a second simulation."""
import hashlib
import json

import numpy as np
import pytest

from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld

pytestmark = pytest.mark.sim


def test_recording_matches_every_physics_step_and_original_pixels(tmp_path, monkeypatch):
    source = tmp_path/'model.xml'
    source.write_text('''<mujoco><option timestep=".002"/>
      <worldbody><light pos="0 0 3"/><geom type="plane" size="3 3 .1"/>
      <camera name="camera" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/>
      <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    scene = compiled_scene(source)
    reference = MuJoCoWorld(scene)
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(tmp_path/'records'))
    world = MuJoCoWorld(scene)
    try:
        reference.step(15)
        world.step(15)
        for a, b in zip(reference.state(), world.state()):
            np.testing.assert_array_equal(a, b)
        frame = world.render('camera', 64, 48, ('rgb', 'depth', 'seg'))
    finally:
        reference.close()
        world.close()
        world.close()
    directory, = (tmp_path/'records').iterdir()
    meta = json.loads((directory/'recording.json').read_text())
    assert meta['closed'] and meta['rows'] == 16 and meta['frames'] == 1
    assert meta['physical_verdict'] == meta['motion_verdict'] == 'unverified'
    assert meta['model_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    width = sum(size for name, size in meta['fields'])
    rows = np.fromfile(directory/'states.bin', dtype=meta['dtype']).reshape(-1, width)
    np.testing.assert_allclose(rows[:, 0], np.arange(16)*scene.timestep, atol=1e-15)
    np.testing.assert_array_equal(rows[-1, 2:2+scene.model.nq], world.state()[0])
    assert np.all(np.diff(rows[:, 1]) >= 0)
    original = np.load(directory/'frames/00000000.npz')
    for key in ('rgb', 'depth', 'seg', 'K', 'cam_pos', 'cam_mat'):
        np.testing.assert_array_equal(original[key], getattr(frame, key))
    assert original['sim_time'] == frame.sim_time and original['camera'] == frame.camera
