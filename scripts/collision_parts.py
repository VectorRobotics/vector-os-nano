"""Offline convex decomposition with metric tolerances and persistent caching."""
import hashlib
from pathlib import Path
import coacd
import numpy as np
import trimesh

coacd.set_log_level('error')


def merge_convex_patch(parts, planes, *, max_extra_volume):
    """Join a nearly convex contact patch, retaining the solids outside its clipping planes.

    Each (axis, limit) keeps coordinates <= limit. The strict volume budget rejects
    a patch whose hull would fill a cavity; original CAD accuracy needs a separate audit.
    """
    if (not planes or any(axis not in (0, 1, 2) or not np.isfinite(limit) for axis, limit in planes)
            or not np.isfinite(max_extra_volume) or max_extra_volume < 0):
        raise ValueError('finite clipping planes and a nonnegative volume budget are required')
    inside = [trimesh.Trimesh(v, f, process=False) for v, f in parts]
    outside = []
    for axis, limit in planes:
        selected = []
        for mesh in inside:
            if mesh.bounds[1, axis] <= limit:
                selected.append(mesh)
            elif mesh.bounds[0, axis] >= limit:
                outside.append(mesh)
            else:
                a, b = mesh.vertices[mesh.edges_unique].transpose(1, 0, 2)
                crossing = (a[:, axis]-limit)*(b[:, axis]-limit) < 0
                a, b = a[crossing], b[crossing]
                edge = a + ((limit-a[:, axis])/(b[:, axis]-a[:, axis]))[:, None]*(b-a)
                pieces = [trimesh.convex.convex_hull(np.concatenate([mesh.vertices[mask], edge]))
                          for mask in (mesh.vertices[:, axis] <= limit, mesh.vertices[:, axis] >= limit)]
                if not np.isclose(sum(abs(p.volume) for p in pieces), abs(mesh.volume), rtol=1e-7, atol=1e-16):
                    raise ValueError('clipping changed the source convex volume')
                selected.append(pieces[0]); outside.append(pieces[1])
        inside = selected
    if not inside:
        raise ValueError('contact patch is empty')
    hull = trimesh.convex.convex_hull(np.concatenate([m.vertices for m in inside]))
    before = sum(abs(m.volume) for m in inside)
    extra = float(hull.volume-before)
    if abs(extra) > max_extra_volume+1e-15:
        raise ValueError(f'contact patch hull changes volume by {extra:g} m^3')
    return [(m.vertices, m.faces) for m in [hull, *outside]], dict(
        merged_parts=len(inside), volume_before_m3=float(before), extra_volume_m3=extra)


def collision_parts(vertices, faces, tolerance=.005, surface_thickness=.002, *, cache):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError('Expected finite vertices and in-range triangular faces')
    faces = faces.astype(np.int32)
    if not all(np.isfinite(v) and v > 0 for v in (tolerance, surface_thickness)):
        raise ValueError('Collision tolerance and surface thickness must be finite and positive')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(b'convex-parts-v3' + vertices.tobytes() + faces.tobytes()
                         + repr((tolerance,surface_thickness)).encode()).hexdigest()
    path = cache / (key + '.npz')
    if path.exists():
        with np.load(path) as data:
            return [(data[f'v{i}'], data[f'f{i}']) for i in range(len(data.files)//2)]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    result = []
    # Repair fills small source holes, including the openings of handles and panels.
    for part in mesh.split(only_watertight=False, repair=False):
        points = np.asarray(part.vertices)
        if len(points) < 3:
            continue
        values = np.linalg.eigvalsh(np.cov(points.T))
        if values[1] < 1e-16:
            continue  # zero-area source faces have no contact surface
        if values[0] >= 1e-10 and part.is_convex:
            hull = part.convex_hull
            result.append((hull.vertices, hull.faces))
        elif values[0] < 1e-10 or not part.is_watertight:
            # Open scan surfaces are panels, not closed volumes to voxel-fill.
            facets = list(part.facets)
            grouped = set(np.concatenate(facets).tolist()) if facets else set()
            facets += [np.array([i]) for i in range(len(part.faces)) if i not in grouped]
            for group in facets:
                triangles = part.triangles[group]
                normal = part.face_normals[group[0]]
                verts = np.unique(triangles.reshape(-1,3),axis=0)
                if np.linalg.norm(normal) < .5:
                    continue
                hull = trimesh.convex.convex_hull(np.concatenate([verts+surface_thickness/2*normal,verts-surface_thickness/2*normal]))
                # A concave panel/handle needs its individual triangles to retain holes.
                if abs(hull.volume/surface_thickness - part.area_faces[group].sum()) > max(1e-8, .01*part.area_faces[group].sum()):
                    for triangle in triangles:
                        h = trimesh.convex.convex_hull(np.concatenate([triangle+surface_thickness/2*normal,triangle-surface_thickness/2*normal]))
                        result.append((h.vertices,h.faces))
                else:
                    result.append((hull.vertices,hull.faces))
        else:
            result.extend(coacd.run_coacd(coacd.Mesh(points, part.faces), threshold=tolerance,
                          real_metric=True, preprocess_resolution=80, mcts_iterations=80,
                          mcts_nodes=12, seed=0, decimate=True, max_ch_vertex=64))
    if not result:
        raise ValueError('Source collider has no nondegenerate surface')
    assert all(np.isfinite(v).all() and len(v)>=4 and len(f)>=4 for v,f in result)
    import tempfile, os
    with tempfile.NamedTemporaryFile(dir=cache,suffix='.npz',delete=False) as file:
        np.savez_compressed(file, **{k:a for i,(v,f) in enumerate(result)
                                     for k,a in ((f'v{i}',v),(f'f{i}',f))})
    os.replace(file.name,path)
    return result
