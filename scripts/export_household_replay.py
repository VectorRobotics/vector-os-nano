"""Export measured household body/camera poses and compiled visual geometry for Blender.

The pinned ArtVIP background is supplied separately as a Blender scene. No physics
is re-simulated, no trajectory is corrected, and private CAD stays in the output folder.
"""
from pathlib import Path
import argparse
import hashlib
import json

import mujoco as mj
import numpy as np

from vector_os_nano.sim.backends.mujoco.recording import read_recording
from vector_os_nano.sim.scene.reader import geom_rgba


def sample_indices(time, fps):
    if not np.isfinite(fps) or fps <= 0 or len(time) < 2 or np.any(np.diff(time) < 0):
        raise ValueError('one continuous recording and positive finite fps required')
    requested = np.arange(time[0], time[-1]+1/fps, 1/fps)
    return np.searchsorted(time, requested, side='right')-1


def export(recording, output, root_body, fps=30):
    meta, states, _contacts = read_recording(recording)
    path = meta['model_path']
    m = mj.MjModel.from_binary_path(path) if path.endswith('.mjb') else mj.MjModel.from_xml_path(path)
    root = m.body(root_body).id
    robot = {root}
    for b in range(root+1, m.nbody):
        if int(m.body_parentid[b]) in robot:
            robot.add(b)
    packages = {'tissue_refill', 'sponge_pack_1', 'sponge_pack_2', 'zip_bag_box'}
    chosen = [g for g in range(m.ngeom) if m.geom_group[g] < 3 and
              (int(m.geom_bodyid[g]) in robot or m.geom(g).name.startswith(('cart_', 'entry_', 'shelf_'))
               or m.body(int(m.geom_bodyid[g])).name in packages)]
    bodies = sorted({int(m.geom_bodyid[g]) for g in chosen} | {root})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meshes, materials, geometry = {}, {}, []
    for g in chosen:
        mid, mat = int(m.geom_dataid[g]), int(m.geom_matid[g])
        kind = int(m.geom_type[g])
        if kind not in (2, 3, 5, 6, 7):
            raise ValueError(f'unsupported visual geometry {kind}: {m.geom(g).name}')
        if kind == 7 and f'v{mid}' not in meshes:
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            f, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            meshes[f'v{mid}'] = m.mesh_vert[a:a+n].copy()
            meshes[f'f{mid}'] = m.mesh_face[f:f+nf].copy()
            uv, nuv = int(m.mesh_texcoordadr[mid]), int(m.mesh_texcoordnum[mid])
            if nuv:
                meshes[f'uv{mid}'] = m.mesh_texcoord[uv:uv+nuv].copy()
                meshes[f'fu{mid}'] = m.mesh_facetexcoord[f:f+nf].copy()
        if mat >= 0 and str(mat) not in materials:
            textures = {}
            for role, tex in enumerate(m.mat_texid[mat]):
                if tex < 0 or m.tex_pathadr[tex] < 0:
                    continue
                file = Path(m.paths[int(m.tex_pathadr[tex]):].split(b'\0', 1)[0].decode())
                textures[str(role)] = dict(path=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            materials[str(mat)] = dict(roughness=float(m.mat_roughness[mat]),
                                       metallic=float(m.mat_metallic[mat]), textures=textures)
        geometry.append(dict(name=m.geom(g).name or f'visual_{g}', body=bodies.index(int(m.geom_bodyid[g])),
                             kind=kind, mesh=mid if kind == 7 else None, material=str(mat),
                             rgba=geom_rgba(m, g).tolist(), size=m.geom_size[g].tolist(),
                             pos=m.geom_pos[g].tolist(), quat=m.geom_quat[g].tolist()))
    time = states['sim_time'][:, 0]
    indices = sample_indices(time, fps)
    d = mj.MjData(m)
    poses, cameras = [], []
    for row in indices:
        d.qpos[:] = states['qpos'][row]
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        poses.append(np.c_[d.xpos[bodies], d.xquat[bodies]])
        cameras.append(np.c_[d.cam_xpos, d.cam_xmat])
    np.savez_compressed(output/'meshes.npz', **meshes)
    np.savez_compressed(output/'poses.npz', time=time[indices], bodies=poses, cameras=cameras)
    result = dict(schema=1, recording=str(Path(recording).resolve()), model_sha256=meta['model_sha256'],
                  fps=fps, frames=len(indices), source_rows=indices.tolist(),
                  bodies=[m.body(b).name for b in bodies], root_body=root_body,
                  cameras=[dict(name=m.camera(i).name, fovy=float(m.cam_fovy[i])) for i in range(m.ncam)],
                  geometry=geometry, materials=materials, physical_verdict='unverified', motion_verdict='unverified',
                  scope='Offline re-render of measured states; not original agent sensor input; no motion edits')
    result['files'] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                       for name in ('meshes.npz', 'poses.npz')}
    (output/'replay.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(frames=len(indices), visual_geometries=len(geometry), bodies=len(bodies))))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--root-body', required=True)
    p.add_argument('--fps', type=float, default=30)
    args = p.parse_args()
    export(args.recording, args.output, args.root_body, args.fps)
