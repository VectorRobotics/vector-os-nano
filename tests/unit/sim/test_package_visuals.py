"""Printed wrappers retain the original physical envelope and inertial dynamics."""
import mujoco as mj
import numpy as np
import pytest

from scripts.package_visuals import add_wrapper

pytestmark = pytest.mark.sim


def test_wrapper_retains_box_dynamics_and_uvs(tmp_path):
    spec = mj.MjSpec.from_string('''<mujoco><worldbody><geom type="plane" size="1 1 .1"/>
      <body name="package" pos="0 0 .2"><freejoint/>
        <geom name="tissue_refill" type="box" size=".025 .08 .035" mass=".18"
              rgba=".88 .83 .70 1" friction=".8 .005 .0001" condim="4"/>
      </body></worldbody></mujoco>''')
    baseline = spec.compile()
    report = add_wrapper(spec, spec.body('package'), spec.geom('tissue_refill'), tmp_path)
    vertices = np.asarray(spec.mesh('tissue_refill_wrapper').uservert).reshape(6, 4, 3)
    # Every vertical printed face has its texture top above its texture bottom.
    assert np.all(vertices[:4, 2:, 2] > vertices[:4, :2, 2])
    model = spec.compile()
    assert model.ngeom == baseline.ngeom+1 and model.nq == baseline.nq
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
        np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
    for field in ('geom_size', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(model, field)[:baseline.ngeom], getattr(baseline, field))
    visual = model.geom('tissue_refill_visual').id
    assert model.geom_contype[visual] == model.geom_conaffinity[visual] == 0
    data, reference = mj.MjData(model), mj.MjData(baseline)
    mj.mj_forward(model, data)
    points = model.mesh_vert @ data.geom_xmat[visual].reshape(3, 3).T + data.geom_xpos[visual]
    np.testing.assert_allclose(points.min(0), [-.025, -.08, .165], atol=1e-8)
    np.testing.assert_allclose(points.max(0), [.025, .08, .235], atol=1e-8)
    assert model.mesh_texcoord.shape == (24, 2)
    assert np.all((model.mesh_texcoord > 0) & (model.mesh_texcoord < 1))
    for _ in range(300):
        mj.mj_step(model, data)
        mj.mj_step(baseline, reference)
        np.testing.assert_allclose(data.qpos, reference.qpos, rtol=0, atol=1e-14)
    assert len(report['texture_sha256']) == len(report['font_sha256']) == 64
