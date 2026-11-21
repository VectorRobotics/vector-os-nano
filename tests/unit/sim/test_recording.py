"""The recording is a copy of actual stepping and rendering, not a second simulation."""
import hashlib
import gzip
import json

import numpy as np
import pytest

from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld
from vector_os_nano.sim.backends.mujoco.recording import read_recording
from scripts.export_household_replay import export, sample_indices

pytestmark = pytest.mark.sim


def test_recording_matches_every_physics_step_and_original_pixels(tmp_path, monkeypatch):
    source = tmp_path/'model.xml'
    source.write_text('''<mujoco><option timestep=".002"/>
      <worldbody><light pos="0 0 3"/><geom type="plane" size="3 3 .1"/>
      <camera name="camera" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/>
      <body name="object" pos="0 0 .1"><freejoint/><geom type="sphere" size=".1"/></body>
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
    with gzip.open(directory/'contacts.bin.gz', 'rb') as stream:
        contacts = np.frombuffer(stream.read(), dtype='<f8').reshape(-1, 13)
    assert len(contacts) == meta['contact_rows'] > 0
    assert np.isfinite(contacts).all() and contacts[:, 8].max() > 0
    _, states, loads = read_recording(directory)
    np.testing.assert_array_equal(states['qpos'][-1], world.state()[0])
    np.testing.assert_array_equal(loads['force_on_b'], contacts[:, 6:9])
    replay = export(directory, tmp_path/'blender', 'object')
    poses = np.load(tmp_path/'blender/poses.npz')
    np.testing.assert_array_equal(poses['time'], states['sim_time'][replay['source_rows'], 0])
    np.testing.assert_array_equal(poses['bodies'][-1, 0], states['qpos'][-1])
    np.testing.assert_allclose(poses['cameras'][-1, 0, :3], frame.cam_pos, atol=1e-14)
    np.testing.assert_allclose(poses['cameras'][-1, 0, 3:], frame.cam_mat.reshape(-1), atol=1e-14)
    assert replay['source_rows'][-1] == len(states['qpos'])-1
    with (directory/'states.bin').open('ab') as stream:
        stream.write(np.array([0.]).tobytes())
    with pytest.raises(ValueError, match='shape/count'):
        read_recording(directory)
    source.write_text(source.read_text().replace('timestep=".002"', 'timestep=".003"'))
    with pytest.raises(ValueError, match='SHA256'):
        read_recording(directory)


def test_replay_sampling_keeps_last_reset_and_final_state():
    assert sample_indices(np.array([0, 0, .01, .02, .03]), 30).tolist() == [1, 4]
    for time, fps in (([0, 1, 0], 30), ([0, 1], 0), ([0, 1], float('nan'))):
        with pytest.raises(ValueError):
            sample_indices(np.array(time), fps)


def test_continuous_original_sensor_video_is_lossless_and_reset_separated(tmp_path, monkeypatch):
    import subprocess
    source = tmp_path/'camera.xml'
    source.write_text('''<mujoco><option timestep=".002"/><worldbody>
      <light pos="0 0 3"/><geom type="plane" size="3 3 .1"/>
      <camera name="head" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/>
      <body name="object" pos="0 0 .5"><freejoint/><geom type="sphere" size=".1" rgba="1 0 0 1"/></body>
      </worldbody></mujoco>''')
    scene = compiled_scene(source)
    reference = MuJoCoWorld(scene)
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(tmp_path/'records'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', 'head')
    world = MuJoCoWorld(scene)
    try:
        world.step(60); reference.step(60)
        for a, b in zip(world.state(), reference.state()): np.testing.assert_array_equal(a, b)
        q, v = world.state(); q[0] += .1
        world.reset(q, v); world.step(25)
    finally:
        world.close(); reference.close()
    directory, = (tmp_path/'records').iterdir()
    meta, states, _ = read_recording(directory)
    assert meta['frames'] == 0  # Periodic video does not pretend to be actor-requested RGB-D frames.
    assert len(meta['sensor_videos']) == 2
    for segment, count in zip(meta['sensor_videos'], (4, 2)):
        assert segment['complete'] and segment['frames'] == count
        stamps = [json.loads(row) for row in (directory/segment['timestamps']).read_text().splitlines()]
        t = np.array([row['sim_time'] for row in stamps])
        assert np.all(np.abs(t-(segment['start_sim_time']+np.arange(count)/30)) <= .002+1e-9)
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(directory/segment['file']),
                                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
        frames = np.frombuffer(raw, np.uint8).reshape(count, 480, 640, 3)
        assert [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames] == [row['rgb_sha256'] for row in stamps]
        assert all(np.min(abs(states['sim_time'][:, 0]-time)) < 1e-9 for time in t)
    assert not list(directory.glob('*.partial.mkv'))


def test_sensor_render_failure_keeps_video_incomplete(tmp_path, monkeypatch):
    source = tmp_path/'camera.xml'
    source.write_text('<mujoco><worldbody><camera name="head"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(tmp_path/'records'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', 'head')
    world = MuJoCoWorld(compiled_scene(source))
    def broken(*args): raise RuntimeError('camera lost')
    monkeypatch.setattr(world, '_render_job', broken)
    try:
        with pytest.raises(RuntimeError, match='camera lost'): world.step(20)
    finally:
        world.close()
    directory, = (tmp_path/'records').iterdir()
    segment, = json.loads((directory/'recording.json').read_text())['sensor_videos']
    assert not segment['complete'] and 'camera lost' in segment['error']


@pytest.mark.parametrize('names', ['missing', 'head,head', 'head,'])
def test_invalid_recording_camera_names_fail_at_start(tmp_path, monkeypatch, names):
    source = tmp_path/'camera.xml'
    source.write_text('<mujoco><worldbody><camera name="head"/></worldbody></mujoco>')
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(tmp_path/'records'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', names)
    with pytest.raises(ValueError, match='unique names'): MuJoCoWorld(compiled_scene(source))
