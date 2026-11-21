"""Compare composition to the separately preserved generated robot, never itself."""
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.scene.compose import compose, robot_model, robot_spec, normalize_attach_defaults

pytestmark = pytest.mark.sim


def test_composed_model_preserves_every_numeric_field(tmp_path):
    import json

    original = robot_spec(Path('tests/fixtures/models/go2_piper_reference.xml').resolve()).compile()
    spec = robot_model(registry.load('go2_piper'))
    spec.compile()
    actual = mj.MjModel.from_xml_string(normalize_attach_defaults(spec.to_xml()))
    assert original.names == actual.names
    for name in dir(original):
        if name.startswith('n') and isinstance(getattr(original, name), int):
            assert getattr(original, name) == getattr(actual, name), name
    for name in dir(original.opt):
        value = getattr(original.opt, name)
        if isinstance(value, (int, float, np.ndarray)):
            np.testing.assert_array_equal(value, getattr(actual.opt, name), err_msg=name)
    differences = {}
    for name in dir(original):
        old, new = getattr(original, name), getattr(actual, name)
        if isinstance(old, np.ndarray) and not np.array_equal(old, new):
            differences[name] = {'old_shape': list(old.shape), 'new_shape': list(new.shape),
                                 'max_abs': float(np.max(np.abs(old.astype(float) - new))) if old.shape == new.shape and old.size else None}
    report = tmp_path / 'model-differences.json'
    report.write_text(json.dumps(differences, indent=2))
    # The old generated XML stores fitted capsule axes at six significant digits.
    # New compilation refits those meshes; do not hide any other model-field drift.
    bounds = {'geom_quat': 2e-6, 'bvh_aabb': 2e-7, 'geom_pos': 3e-15, 'dof_length': 1e-15}
    if 'actuator_ctrlspec' in differences:
        # MuJoCo 4 labels position shortcuts explicitly; the old general actuators
        # omit that label. For hinge/slide transmissions the force law is unchanged.
        ids = np.flatnonzero(original.actuator_ctrlspec != actual.actuator_ctrlspec)
        assert ids.tolist() == list(range(12, 19))
        assert np.all(original.actuator_ctrlspec[ids] == 0)
        assert np.all(actual.actuator_ctrlspec[ids] == 1)
        del differences['actuator_ctrlspec']
    assert set(differences) <= bounds.keys(), json.dumps(differences, indent=2)
    for name, diff in differences.items():
        assert diff['old_shape'] == diff['new_shape'] and diff['max_abs'] <= bounds[name], (name, diff)
    # Independent evaluation of the actuator law at nonzero joint angles/velocities.
    old_data, new_data = mj.MjData(original), mj.MjData(actual)
    rng = np.random.default_rng(419)
    old_data.qpos[7:] = new_data.qpos[7:] = rng.uniform(-.5, .5, original.nq - 7)
    old_data.qvel[:] = new_data.qvel[:] = rng.uniform(-.2, .2, original.nv)
    old_data.ctrl[:] = new_data.ctrl[:] = rng.uniform(-.5, .5, original.nu)
    mj.mj_forward(original, old_data)
    mj.mj_forward(actual, new_data)
    np.testing.assert_array_equal(old_data.actuator_force, new_data.actuator_force)


@pytest.mark.parametrize('key', ['root', 'joint', 'site', 'finger', 'mount'])
def test_bad_model_names_fail_with_manifest_and_valid_names(key):
    cfg = registry.load('go2_piper')
    if key == 'root':
        cfg = replace(cfg, model=replace(cfg.model, root_body='typo'))
    elif key == 'joint':
        cfg = replace(cfg, arm=replace(cfg.arm, joints=('typo', *cfg.arm.joints[1:])))
    elif key == 'site':
        cfg = replace(cfg, arm=replace(cfg.arm, ee_site='typo'))
    elif key == 'finger':
        cfg = replace(cfg, arm=replace(cfg.arm, gripper=replace(cfg.arm.gripper, finger_bodies=('typo',))))
    else:
        cfg = replace(cfg, compose=replace(cfg.compose, body='typo'))
    with pytest.raises(ValueError, match='go2_piper/robot.yaml.*typo.*valid:'):
        compose(cfg, load_world(cfg.world))


@pytest.mark.parametrize('body', ['go2', 'go2_piper', 'piper', 'so101', 'g1'])
@pytest.mark.parametrize('world', ['apartment', 'tabletop', 'workbench'])
def test_all_supported_robot_world_pairs_use_one_composer(body, world, g1_assets):
    from vector_os_nano.sim import backends

    cfg, manifest = registry.load(body), load_world(world)
    for backend in backends.BACKENDS:
        assert backends.get(backend).supports(cfg, manifest) is None
    scene = compose(cfg, manifest)
    assert scene.body(cfg.model.root_body) == scene.robot_root
    assert all(name in scene.joint_ids for name in cfg.arm.joints) if cfg.arm else True


def test_model_prefix_renames_robot_and_driver_references_together():
    import yaml
    from vector_os_nano.schemas.robot import parse_embodiment_config

    original = registry.load('piper')
    raw = yaml.safe_load(original.path.read_text())
    raw['model']['prefix'] = 'tool_'
    cfg = parse_embodiment_config(raw, str(original.path))
    scene = compose(cfg, load_world(cfg.world))
    assert scene.body('tool_base_link') == scene.robot_root
    assert cfg.arm.joints[0] == 'tool_joint1' and cfg.arm.gripper.weld_body == 'tool_link6'
    assert cfg.camera.name == 'tool_workspace_cam' and cfg.camera.name in scene.cameras
