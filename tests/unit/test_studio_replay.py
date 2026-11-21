"""Recorded-state inspection restores measured poses without advancing dynamics."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import replay_household_studio as replay


def test_recorded_binary_resolves_its_original_compiler_xml(tmp_path, monkeypatch):
    xml = tmp_path/'compiled.xml'
    xml.write_text('<mujoco><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>')
    binary = tmp_path/'model.mjb'
    replay.mj.mj_saveModel(replay.mj.MjModel.from_xml_path(str(xml)), str(binary))
    monkeypatch.setattr(replay, 'read_recording', lambda _: (
        dict(model_path=str(binary), source_model_path=str(xml)), {'sim_time': np.array([[0.], [.04]])}, {}))
    def stop_before_viewer(model, source):
        assert source == xml
        raise RuntimeError('source resolved before viewer launch')
    monkeypatch.setattr(replay, 'visible_copy', stop_before_viewer)
    with pytest.raises(RuntimeError, match='source resolved'):
        replay.replay(tmp_path, max_frames=1)


def test_visible_copy_preserves_visual_mesh_and_recorded_joint_layout(tmp_path):
    path = tmp_path/'scene.xml'
    path.write_text('''<mujoco><asset>
      <mesh name="visual" vertex="0 0 0  .1 0 0  0 .1 0  0 0 .1"/>
      <mesh name="collision" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
      </asset><worldbody><body><freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      <geom type="mesh" mesh="visual" group="1"/>
      <geom type="mesh" mesh="collision" group="3"/>
      </body></worldbody></mujoco>''')
    source = replay.mj.MjModel.from_xml_path(str(path))
    copy = replay.visible_copy(source, path)
    assert (source.ngeom, source.nmesh, source.nq) == (2, 2, 7)
    assert (copy.ngeom, copy.nmesh, copy.nq) == (1, 1, 7)
    assert source.geom_group.tolist() == [1, 3]


def test_visual_package_retains_inertia_inferred_from_hidden_collision_geometry(tmp_path):
    path=tmp_path/'scene.xml'
    path.write_text('''<mujoco><worldbody><body name="package"><freejoint/>
      <geom type="box" size=".025 .08 .035" mass=".18" group="3"/>
      <geom type="box" size=".025 .08 .035" density="0" group="1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    source=replay.mj.MjModel.from_xml_path(str(path))
    copy=replay.visible_copy(source,path)
    assert copy.ngeom==1
    np.testing.assert_array_equal(copy.body_mass,source.body_mass)
    np.testing.assert_allclose(copy.body_inertia,source.body_inertia,atol=1e-12,rtol=0)


def test_restore_preserves_recording_and_never_steps(monkeypatch):
    model = SimpleNamespace(eq_data=np.zeros((1, 11)))
    data = SimpleNamespace(time=0., qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(1), eq_active=np.zeros(1))
    states = {name: np.arange(2*width).reshape(2, width).astype(float)
              for name, width in [('sim_time', 1), ('qpos', 2), ('qvel', 2), ('ctrl', 1), ('eq_active', 1), ('eq_data', 11)]}
    saved = {k: v.copy() for k, v in states.items()}
    calls = []
    monkeypatch.setattr(replay.mj, 'mj_step', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('no re-simulation')))
    monkeypatch.setattr(replay.mj, 'mj_kinematics', lambda m, d: calls.append('kinematics'))
    monkeypatch.setattr(replay.mj, 'mj_camlight', lambda m, d: calls.append('camera'))
    replay.restore(model, data, states, 1)
    assert data.time == 1.
    assert calls == ['kinematics', 'camera']
    for name in ('qpos', 'qvel', 'ctrl', 'eq_active'):
        np.testing.assert_array_equal(getattr(data, name), states[name][1])
    np.testing.assert_array_equal(model.eq_data.ravel(), states['eq_data'][1])
    for name in states:
        np.testing.assert_array_equal(states[name], saved[name])
