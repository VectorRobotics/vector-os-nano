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


def test_merging_contact_patch_removes_internal_seams_without_changing_other_solids():
    from scripts.collision_parts import merge_convex_patch
    pieces = []
    for x in (-.5, .5):
        box = trimesh.creation.box(extents=[1, 2, 2]); box.apply_translation([x, 0, 0])
        pieces.append((box.vertices, box.faces))
    merged, audit = merge_convex_patch(pieces, [(2, .5), (1, .5)], max_extra_volume=0.)
    hulls = [trimesh.Trimesh(v, f, process=False) for v, f in merged]
    assert audit['merged_parts'] == 2 and audit['extra_volume_m3'] == pytest.approx(0., abs=1e-14)
    assert sum(m.volume for m in hulls) == pytest.approx(8.)
    assert hulls[0].bounds.tolist() == [[-1., -1., -1.], [1., .5, .5]]
    assert all(m.is_convex and m.is_watertight for m in hulls)


def test_contact_patch_hull_cannot_fill_a_gap():
    from scripts.collision_parts import merge_convex_patch
    boxes = [trimesh.creation.box() for _ in range(2)]
    boxes[1].apply_translation([2, 0, 0])
    with pytest.raises(ValueError, match='changes volume'):
        merge_convex_patch([(m.vertices, m.faces) for m in boxes], [(2, 1)], max_extra_volume=1e-9)
    with pytest.raises(ValueError, match='empty'):
        merge_convex_patch([(boxes[0].vertices, boxes[0].faces)], [(2, -2)], max_extra_volume=0.)
