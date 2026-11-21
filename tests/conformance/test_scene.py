# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The canonical compile step, the reader and the PyBullet load guard (no stepping)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conformance.conftest import SCENES, ROBOTS
from vector_os_nano.sim.scene import canonical, compiled_scene, reader

pytestmark = pytest.mark.sim

PAIRS = [("carrier", "parcel"), ("arm", "parcel")]


def _xml(tmp_path: Path, body: str, extra: str = "") -> Path:
    path = tmp_path / "m.xml"
    path.write_text(f"<mujoco><worldbody>{body}</worldbody>{extra}</mujoco>")
    return path


def test_free_joint_code_is_mujoco_s():
    import mujoco as mj

    from vector_os_nano.interfaces import physics

    assert physics._JNT_FREE == int(mj.mjtJoint.mjJNT_FREE) == reader.JNT_FREE


def test_compiled_scene_tables(contract_scene):
    sc = contract_scene
    assert sc.xml_path == SCENES / "contract.xml" and sc.timestep == 0.002
    assert sc.body("die") == sc.body_ids["die"] and sc.body_name(sc.body("die")) == "die"
    assert sc.cameras == {"cam": 0}
    assert sc.grasp_pairs == ((sc.body("carrier"), sc.body("parcel")), (sc.body("arm"), sc.body("parcel")))
    assert set(sc.free_bodies()) == {"die", "parcel", "wand", "slab"}
    assert sc.subtree(sc.body("rig")) == {sc.body(n) for n in ("rig", "spinner", "dragger", "arm", "weak_arm")}
    assert compiled_scene(SCENES / "contract.xml", timestep=0.001).timestep == 0.001
    with pytest.raises(KeyError, match="bodies"):
        sc.body("nope")


@pytest.mark.parametrize("pairs, error", [
    ([("die", "parcel")], "no weld"),          # no such weld equality
    ([("parcel", "carrier")], "no weld"),      # reversed
    ([("carrier", "die")], None),              # a declared inactive weld to a free body: fine
])
def test_grasp_pairs_must_be_compiled_inactive_welds(pairs, error):
    if error is None:
        assert compiled_scene(SCENES / "contract.xml", grasp_pairs=pairs).grasp_pairs
        return
    with pytest.raises(ValueError, match=error):
        compiled_scene(SCENES / "contract.xml", grasp_pairs=pairs)


def test_grasp_pairs_reject_active_welds_and_fixed_children(tmp_path):
    body = ('<body name="a" pos="0 0 1"><freejoint/><geom size=".1"/></body>'
            '<body name="b" pos="1 0 1"><geom size=".1"/></body>'
            '<body name="c" pos="2 0 1"><freejoint/><geom size=".1"/></body>')
    eq = '<equality><weld body1="a" body2="b" active="false"/><weld body1="b" body2="c"/></equality>'
    path = _xml(tmp_path, body, eq)
    with pytest.raises(ValueError, match="not a free body"):
        compiled_scene(path, grasp_pairs=[("a", "b")])
    with pytest.raises(ValueError, match="inactive"):
        compiled_scene(path, grasp_pairs=[("b", "c")])
    with pytest.raises(KeyError):
        compiled_scene(path, robot_root="nope")


def test_reader_records(robot_scene):
    m = robot_scene("piper").model
    acts = {a.name: a for a in reader.actuators(m)}
    assert acts["joint1"].kind == "position" and acts["joint1"].gain == 80 and acts["joint1"].forcerange == (-100, 100)
    assert acts["gripper"].forcerange == (-10, 10)
    (c,) = reader.couplings(m)
    assert (m.joint(c.joint1).name, m.joint(c.joint2).name) == ("joint8", "joint7") and c.polycoef[1] == -1
    pairs = reader.no_collide_pairs(m)
    names = {(m.body(a).name, m.body(b).name) for a, b in pairs}
    assert ("base_link", "link1") in names  # the <exclude>
    assert ("link1", "link2") in names      # parent / child
    assert ("link7", "link8") not in names  # the two fingers do collide
    bodies = reader.bodies(m)
    assert all(b.gravcomp == 1.0 for b in bodies[1:])
    geoms = reader.geoms(m)
    visual = [g for g in geoms if g.visible]
    assert visual and not any(g.collide for g in visual)  # piper: visual meshes never collide
    red = np.array([0.82, 0.15, 0.15, 1.0])
    assert any(np.allclose(g.rgba, red) for g in visual)  # the material's colour, resolved
    for g in (g for g in geoms if g.mesh >= 0):
        v, f = reader.mesh(m, g.mesh)
        assert f.min() >= 0 and f.max() < len(v)
    go2 = robot_scene("go2").model
    assert {a.kind for a in reader.actuators(go2)} == {"motor"}
    assert [c.name for c in reader.cameras(go2)] == ["d435_rgb", "d435_depth"]


def test_canonical_shape_aabbs_are_today_s_ground_truth_boxes(battery_scene):
    sc = battery_scene
    lo, hi, body, colliding = canonical.shape_aabbs(sc, sc.model.qpos0)
    box = sc.body("box")
    own = body == box
    np.testing.assert_allclose(lo[own][0], [0.28, -0.12, 0.10], atol=1e-12)
    np.testing.assert_allclose(hi[own][0], [0.32, -0.08, 0.14], atol=1e-12)
    assert colliding[own].all() and not colliding[body == sc.body("link2")].all()
    pos, mat = canonical.site_poses(sc, sc.model.qpos0)
    assert pos.shape == (1, 3) and mat.shape == (1, 3, 3)


@pytest.mark.parametrize("body, extra, why", [
    ('<body pos="0 0 1"><joint type="ball"/><geom size=".1"/></body>', "", "ball joint"),
    ('<body pos="0 0 1"><joint type="hinge"/><joint type="slide"/><geom size=".1"/></body>', "", "2 joints"),
    ('<body pos="0 0 1"><joint type="hinge"/><geom size=".1"/></body>', "", "world child moves only by a free joint"),
    ('<body pos="0 0 1"><freejoint/><geom type="ellipsoid" size=".1 .2 .3"/></body>', "", "geom type 4"),
    ('<body pos="0 0 1"><freejoint/><geom size=".1" contype="2" conaffinity="1"/></body>', "", "contype/conaffinity"),
    ('<body pos="0 0 1"><freejoint/><geom size=".1"/></body>', '<worldbody><camera name="c" mode="track"/></worldbody>', "not fixed"),
    ('<body name="r"><geom size=".1"/><body name="a"><joint name="j1"/><geom size=".1"/></body>'
     '<body name="b"><joint name="j2"/><geom size=".1"/></body></body>',
     '<equality><joint joint1="j1" joint2="j2" polycoef="0.1 1 0 0 0"/></equality>', "not linear"),
    ('<body name="r"><geom size=".1"/><body name="a"><joint name="j1"/><geom size=".1"/></body></body>',
     '<actuator><velocity joint="j1"/></actuator>', "neither a joint motor"),
])
def test_pybullet_refuses_what_it_cannot_reproduce(tmp_path, body, extra, why):
    pytest.importorskip("pybullet", reason="pybullet is not installed: uv pip install -e '.[pybullet]'")
    from vector_os_nano.sim.backends.pybullet.world import unsupported

    scene = compiled_scene(_xml(tmp_path, body, extra))
    assert why in (unsupported(scene.model) or "")


def test_pybullet_refuses_global_contact_override_before_connecting(tmp_path):
    from dataclasses import replace
    from vector_os_nano.embodiments import registry
    from vector_os_nano.schemas.world import load_world
    from vector_os_nano.sim import backends
    from vector_os_nano.sim.backends.pybullet.world import unsupported

    (tmp_path / 'options.xml').write_text('<mujoco><option><flag override="enable"/></option></mujoco>')
    path = _xml(tmp_path, '<geom type="plane" size="1 1 .1"/>', '<include file="options.xml"/>')
    manifest = replace(load_world('workbench'), scene=path)
    backend = backends.get('pybullet')
    assert 'global contact override' in backend.supports(registry.load('piper'), manifest)
    scene = compiled_scene(path)
    assert 'global contact override' in unsupported(scene.model)
    with pytest.raises(ValueError, match='global contact override'):
        backend.load(scene, gui=False)


def test_pybullet_refuses_a_pickable_that_collides_as_a_mesh(tmp_path):
    """Bullet reads a mesh hull as touching ~3 mm away (S2cV-01): a grasp decided on
    contact would not be MuJoCo's. The same mesh merely seen, or a primitive, loads."""
    pytest.importorskip("pybullet", reason="pybullet is not installed: uv pip install -e '.[pybullet]'")
    from vector_os_nano.sim import backends

    def scene(shape: str):
        return compiled_scene(_xml(tmp_path, f'<body name="hand" pos="0 0 1"><freejoint/><geom size=".02"/></body>'
                                   f'<body name="obj" pos="0 0 .1"><freejoint/>{shape}</body>',
                                   '<asset><mesh name="m" vertex="0 0 0 .04 0 0 0 .04 0 0 0 .04"/></asset>'
                                   '<equality><weld body1="hand" body2="obj" active="false"/></equality>'),
                              grasp_pairs=[("hand", "obj")])

    with pytest.raises(ValueError, match="collides as a mesh"):
        backends.get("pybullet").load(scene('<geom type="mesh" mesh="m"/>'))
    for ok in ('<geom type="mesh" mesh="m" contype="0" conaffinity="0"/><geom size=".02"/>', '<geom size=".02"/>'):
        backends.get("pybullet").load(scene(ok)).close()


def test_pybullet_accepts_every_shipped_body(robot_scene, battery_scene, tabletop_scene, contract_scene,
                                            flat_go2_scene):
    pytest.importorskip("pybullet", reason="pybullet is not installed: uv pip install -e '.[pybullet]'")
    from vector_os_nano.sim.backends.pybullet.world import unsupported

    shipped = [robot_scene(n) for n in ("so101", "piper", "go2", "go2_piper")]
    for sc in shipped + [battery_scene, tabletop_scene, contract_scene, flat_go2_scene]:
        assert unsupported(sc.model, (c for _p, c in sc.grasp_pairs)) is None, sc.xml_path
    assert ROBOTS.is_dir()
