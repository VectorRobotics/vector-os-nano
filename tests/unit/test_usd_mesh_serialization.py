"""USD face-corner UVs must survive the MJCF cache, not just the first compile."""
import numpy as np
import pytest
import mujoco as mj

pytest.importorskip('pxr', reason='offline USD asset-building extra not installed')
from scripts.prepare_usd import write_mjcf


def test_independent_uv_and_normal_indices_survive_mjcf(tmp_path):
    spec = mj.MjSpec()
    mesh = spec.add_mesh(name='cad', uservert=[0,0,0, 1,0,0, 0,1,0, 0,0,1],
                         userface=[0,2,1, 0,1,3, 0,3,2, 1,2,3],
                         usertexcoord=np.linspace(0,1,24), userfacetexcoord=np.arange(12),
                         usernormal=[0,0,-1, 0,-1,0, -1,0,0, 1,1,1],
                         userfacenormal=np.repeat(np.arange(4),3))
    spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname='cad')
    before = spec.compile()
    write_mjcf(spec, tmp_path/'scene.xml')
    assert mesh.file.endswith('.obj')
    after = mj.MjModel.from_xml_path(str(tmp_path/'scene.xml'))
    for model in (before, after):
        assert model.nmesh == 1
    np.testing.assert_allclose(after.mesh_texcoord[after.mesh_facetexcoord],
                               before.mesh_texcoord[before.mesh_facetexcoord], atol=1e-7)
    # 4.0.1 also ignores inline userfacenormal; compare to the authored normals,
    # not that broken intermediate representation.
    data = mj.MjData(after)
    mj.mj_forward(after, data)
    world_normals = after.mesh_normal[after.mesh_facenormal] @ data.geom_xmat[0].reshape(3,3).T
    authored = np.array([[0,0,-1], [0,-1,0], [-1,0,0], [1,1,1]], dtype=float)
    authored /= np.linalg.norm(authored, axis=1, keepdims=True)
    np.testing.assert_allclose(world_normals, np.repeat(authored[:,None,:],3,axis=1), atol=1e-6)
    np.testing.assert_allclose(after.geom_aabb, before.geom_aabb, atol=1e-6)


def test_mislabeled_jpeg_is_encoded_without_changing_source(tmp_path):
    from PIL import Image

    source = tmp_path/'upstream.png'
    Image.new('RGB',(4,4),(80,120,160)).save(source,format='JPEG')
    original = source.read_bytes()
    with Image.open(source) as image:
        expected = image.getpixel((0,0))
    spec = mj.MjSpec.from_string(f'''<mujoco><asset>
      <texture name="albedo" type="2d" file="{source}"/>
      <material name="surface" texture="albedo"/></asset>
      <worldbody><geom type="box" size="1 1 .1" material="surface"/></worldbody></mujoco>''')
    write_mjcf(spec,tmp_path/'derived'/'world.xml')
    model = mj.MjModel.from_xml_path(str(tmp_path/'derived'/'world.xml'))
    assert tuple(model.tex_data[:3]) == expected
    assert source.read_bytes() == original
