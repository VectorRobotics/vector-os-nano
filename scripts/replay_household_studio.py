"""Inspect recorded states in MuJoCo Studio/Filament; read-only, without re-simulating physics."""
import argparse
import json
from pathlib import Path
import time
import tempfile
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from scripts.export_household_replay import sample_indices
from vector_os_nano.sim.backends.mujoco.recording import read_recording


def visible_copy(model, xml_path):
    """Omit default-hidden collision meshes from the read-only renderer, never from physics."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    # A visible wrapper can be massless; retain inertia supplied by the hidden physical geoms.
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(root, 'compiler')
    compiler.set('inertiafromgeom', 'false')
    for body in root.iter('body'):
        name = body.get('name')
        i = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) if name else -1
        if i < 0 or model.body_mass[i] <= 0:
            continue
        for inertial in body.findall('inertial'):
            body.remove(inertial)
        ET.SubElement(body, 'inertial', mass=str(model.body_mass[i]),
            pos=' '.join(map(str, model.body_ipos[i])), quat=' '.join(map(str, model.body_iquat[i])),
            diaginertia=' '.join(map(str, model.body_inertia[i])))
    hidden = {model.geom(i).name for i in range(model.ngeom) if model.geom_group[i] >= 3}
    for parent in root.iter():
        parent[:] = [el for el in parent if not (el.tag == 'geom' and
            (el.get('name') in hidden or int(el.get('group', '0')) >= 3))]
    used = {el.get('mesh') for el in root.iter('geom')}
    asset = root.find('asset')
    if asset is not None:
        asset[:] = [el for el in asset if el.tag != 'mesh' or el.get('name') in used]
    # Keep relative asset references relative to the retained compiler XML.
    with tempfile.TemporaryDirectory(prefix='studio-visual-', dir=xml_path.parent) as temp:
        path = Path(temp)/'scene.xml'
        compiler = root.find('compiler')
        if compiler is not None:
            for name in ('meshdir', 'texturedir', 'assetdir'):
                value = compiler.get(name)
                if value and not Path(value).is_absolute():
                    compiler.set(name, str((xml_path.parent/value).resolve()))
        for tag in ('mesh', 'texture', 'hfield', 'skin', 'include'):
            for el in root.iter(tag):
                value = el.get('file')
                if value and not Path(value).is_absolute():
                    # Compiler asset directories, when present, already resolve relative files.
                    directory = (compiler.get({'mesh': 'meshdir', 'texture': 'texturedir'}.get(tag, 'assetdir'))
                                 or compiler.get('assetdir')) if compiler is not None else None
                    if not directory:
                        el.set('file', str((xml_path.parent/value).resolve()))
        ET.ElementTree(root).write(path, encoding='utf-8')
        visible = mj.MjModel.from_xml_path(str(path))
    chosen = np.flatnonzero(model.geom_group < 3)
    if len(chosen) != visible.ngeom or (model.nq, model.nv, model.nu) != (visible.nq, visible.nv, visible.nu):
        raise ValueError('visual copy changed state layout or visible geometry count')
    for field in ('qpos0', 'body_pos', 'body_quat', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                  'jnt_pos', 'jnt_axis', 'jnt_qposadr', 'body_parentid',
                  'mat_rgba', 'mat_roughness', 'mat_metallic', 'cam_pos', 'cam_quat', 'cam_fovy', 'tex_data'):
        np.testing.assert_allclose(getattr(model, field), getattr(visible, field), atol=1e-12, rtol=0)
    for field in ('geom_pos', 'geom_quat', 'geom_size', 'geom_bodyid', 'geom_type', 'geom_matid', 'geom_rgba'):
        np.testing.assert_allclose(getattr(model, field)[chosen], getattr(visible, field), atol=1e-12, rtol=0)
    for i in range(visible.nmesh):
        j = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MESH, visible.mesh(i).name)
        if j < 0:
            raise ValueError('visual copy introduced an unknown mesh')
        for field, adr, count in (('mesh_vert', 'mesh_vertadr', 'mesh_vertnum'),
                                  ('mesh_face', 'mesh_faceadr', 'mesh_facenum'),
                                  ('mesh_texcoord', 'mesh_texcoordadr', 'mesh_texcoordnum')):
            a, b = getattr(model, adr)[j], getattr(visible, adr)[i]
            na, nb = getattr(model, count)[j], getattr(visible, count)[i]
            np.testing.assert_array_equal(getattr(model, field)[a:a+na], getattr(visible, field)[b:b+nb])
    return visible


def restore(model, data, states, row):
    """Restore only the recorded visualization state in the viewer's private copy."""
    data.time = float(states['sim_time'][row, 0])
    for name in ('qpos', 'qvel', 'ctrl', 'eq_active'):
        getattr(data, name)[:] = states[name][row]
    model.eq_data[:] = states['eq_data'][row].reshape(model.eq_data.shape)
    mj.mj_kinematics(model, data)
    mj.mj_camlight(model, data)


def replay(recording, *, gfx='opengl_software', camera=None, max_frames=None):
    from mujoco.studio import app, launch_thread, messages, native_viewer, viewer_protocol

    meta, states, _ = read_recording(recording)
    indices = sample_indices(states['sim_time'][:, 0], 30)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        indices = indices[:max_frames]
    path = meta['model_path']
    model = mj.MjModel.from_binary_path(path) if path.endswith('.mjb') else mj.MjModel.from_xml_path(path)
    # Filament allocates resources even for hidden meshes; the full CAD collider set exhausts its arena.
    original_geoms = model.ngeom
    model = visible_copy(model, Path(meta.get('source_model_path', path)).with_suffix('.xml'))
    visible_geoms = model.ngeom
    camera_id = model.camera(camera).id if camera else None
    endpoint, _ = launch_thread.make_thread_endpoints()
    viewer = native_viewer.NativeViewer(viewer_protocol.ViewerConfig(
        title='VECTOR recorded states — read-only Studio / Filament', width=1280, height=720, gfx=gfx),
        endpoint, model=model, plugins=[app.App()])
    del model
    completed, started = 0, time.monotonic()
    try:
        if camera_id is not None:
            viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            viewer.camera.fixedcamid = camera_id
        for row in indices:
            tick = time.monotonic()
            if not viewer.prepare_next_frame() or not viewer.is_running():
                break
            restore(viewer.model, viewer.data, states, row)
            viewer.dispatch(messages.UpdateEvent())
            viewer.dispatch(messages.BuildGuiEvent())
            viewer.sync()
            completed += 1
            time.sleep(max(0., 1/30 - (time.monotonic()-tick)))
    finally:
        viewer.close()
    # Viewer commands have no simulation consumer: reset/drag/step cannot change the source recording.
    return dict(recording=str(Path(recording).resolve()), model_sha256=meta['model_sha256'],
                engine=mj.__version__, renderer='MuJoCo Studio / Filament', gfx=gfx, frames_rendered=completed,
                requested_frames=len(indices), complete=completed == len(indices),
                original_geoms=original_geoms, visible_geoms=visible_geoms,
                wall_seconds=time.monotonic()-started, scope=__doc__, motion_verdict='unverified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    # 4.0.1's direct-window target is broken (upstream #3916); buffer presentation still uses Filament.
    parser.add_argument('--gfx', default='opengl_software')
    parser.add_argument('--camera')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    print(json.dumps(replay(args.recording, gfx=args.gfx, camera=args.camera, max_frames=args.max_frames), indent=2))
