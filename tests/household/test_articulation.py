"""Declared drawer acceptance must refer to the real passive joint on a free cart."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim


def test_drawer_manifest_matches_the_converted_mechanism():
    import mujoco as mj
    from vector_os_nano.schemas.world import load_world_file

    configured = os.environ.get('HOUSEHOLD_ASSETS')
    if not configured:
        pytest.skip('HOUSEHOLD_ASSETS must name the locally converted private-CAD asset store')
    folder = Path(configured)/'worlds/artvip_home'
    world = load_world_file(folder/'world.yaml')
    model = mj.MjModel.from_binary_path(str(folder/'scene.mjb'))
    spec = world.articulations['storage_drawer']
    joint, body = model.joint(spec.joint).id, model.body(spec.body).id
    assert model.geom(spec.handle_geom).bodyid == body
    assert model.jnt_type[joint] == mj.mjtJoint.mjJNT_SLIDE
    assert model.jnt_bodyid[joint] == body and model.jnt_limited[joint]
    assert all(model.jnt_range[joint, 0] <= q <= model.jnt_range[joint, 1]
               for q in (spec.closed, spec.opened))
    assert model.dof_damping[model.jnt_dofadr[joint]] > 0
    assert model.dof_frictionloss[model.jnt_dofadr[joint]] > 0
    assert not any(model.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT and
                   model.actuator_trnid[a, 0] == joint for a in range(model.nu))
    while body and model.body_jntnum[body] == 0:
        body = int(model.body_parentid[body])
    body = int(model.body_parentid[body])
    while body and model.body_jntnum[body] == 0:
        body = int(model.body_parentid[body])
    assert body and model.jnt_type[model.body_jntadr[body]] == mj.mjtJoint.mjJNT_FREE
