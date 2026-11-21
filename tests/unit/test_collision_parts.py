"""Concave scan panels must retain their openings after contact thickening."""
import numpy as np
import pytest

trimesh = pytest.importorskip('trimesh', reason='offline home-assets extra not installed')
pytest.importorskip('coacd', reason='offline home-assets extra not installed')

from scripts.collision_parts import collision_parts


def test_flat_ring_keeps_hole_and_cached_geometry(tmp_path):
    # Four quads forming a square annulus: the centre is empty space.
    vertices = np.array([[-2,-2,0], [2,-2,0], [2,2,0], [-2,2,0],
                         [-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], dtype=float)
    faces = np.array([[i, (i+1)%4, (i+1)%4+4] for i in range(4)]
                     + [[i, (i+1)%4+4, i+4] for i in range(4)])
    parts = collision_parts(vertices, faces, surface_thickness=.002, cache=tmp_path)
    hulls = [trimesh.Trimesh(v, f) for v, f in parts]
    assert sum(h.volume for h in hulls) == pytest.approx(12*.002)
    # A point is outside a convex polyhedron iff it is beyond at least one face.
    assert all(np.any(np.einsum('ij,ij->i', -h.triangles_center, h.face_normals) > 1e-8)
               for h in hulls)
    cached = collision_parts(vertices, faces, surface_thickness=.002, cache=tmp_path)
    assert all(np.array_equal(a, b) for first, second in zip(parts, cached)
               for a, b in zip(first, second))


@pytest.mark.parametrize('thickness', [0, -1, float('nan')])
def test_invalid_thickness_fails(tmp_path, thickness):
    with pytest.raises(ValueError, match='finite and positive'):
        collision_parts(np.eye(3), [[0,1,2]], surface_thickness=thickness, cache=tmp_path)


@pytest.mark.parametrize('faces', [[[0,1,3]], [[0,1,-1]], [[0,1,1.5]]])
def test_invalid_indices_fail(tmp_path, faces):
    with pytest.raises(ValueError, match='in-range triangular faces'):
        collision_parts(np.eye(3), faces, cache=tmp_path)
