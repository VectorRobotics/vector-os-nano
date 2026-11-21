"""Blender background entry: rebuild recorded household geometry, motion and cameras."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('replay', type=Path)
p.add_argument('background', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--preview', type=Path)
a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
record = json.loads((a.replay/'replay.json').read_text())
for name, digest in record['files'].items():
    assert hashlib.sha256((a.replay/name).read_bytes()).hexdigest() == digest, name
meshes = np.load(a.replay/'meshes.npz')
poses = np.load(a.replay/'poses.npz')
assert len(poses['time']) == record['frames']
bpy.ops.wm.open_mainfile(filepath=str(a.background))
scene = bpy.context.scene
floor_material = None
for obj in scene.objects:
    if obj.type == 'MESH' and obj.data.materials:
        points = [obj.matrix_world@Vector(v) for v in obj.bound_box]
        if (max(v.z for v in points) < .02 and max(v.x for v in points)-min(v.x for v in points) > 5
                and max(v.y for v in points)-min(v.y for v in points) > 5):
            floor_material = obj.data.materials[0]
            floor_material.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value = .6
            break
parents = []
for name in record['bodies']:
    obj = bpy.data.objects.new('Recorded '+name, None)
    scene.collection.objects.link(obj)
    obj.rotation_mode = 'QUATERNION'
    parents.append(obj)


def material(g):
    source = record['materials'].get(g['material'], {})
    mat = bpy.data.materials.new(g['name']+' surface')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = g['rgba']
    bsdf.inputs['Roughness'].default_value = source.get('roughness', .6)
    bsdf.inputs['Metallic'].default_value = source.get('metallic', 0.)
    for role, texture in source.get('textures', {}).items():
        if role not in ('1', '3', '4', '5'):
            continue
        path = Path(texture['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == texture['sha256'], path
        node = nodes.new('ShaderNodeTexImage')
        node.image = bpy.data.images.load(str(path), check_existing=True)
        if role != '1':
            node.image.colorspace_settings.name = 'Non-Color'
        if role == '5':
            normal = nodes.new('ShaderNodeNormalMap')
            links.new(node.outputs['Color'], normal.inputs['Color'])
            links.new(normal.outputs['Normal'], bsdf.inputs['Normal'])
        else:
            links.new(node.outputs['Color'], bsdf.inputs[{'1':'Base Color','3':'Roughness','4':'Metallic'}[role]])
    return mat


for g in record['geometry']:
    kind, size = g['kind'], g['size']
    if kind == 7:
        mid = g['mesh']
        mesh = bpy.data.meshes.new(g['name'])
        mesh.from_pydata(meshes[f'v{mid}'], [], meshes[f'f{mid}'])
        mesh.update()
        if f'uv{mid}' in meshes:
            indices = meshes[f'fu{mid}'].reshape(-1)
            uv = meshes[f'uv{mid}'][np.maximum(0, indices)]
            uv[:, 1] = 1-uv[:, 1]  # MuJoCo stores image-row V; Blender uses bottom-up UV.
            mesh.uv_layers.new(name='Source UV').data.foreach_set('uv', uv.reshape(-1))
        obj = bpy.data.objects.new(g['name'], mesh)
        scene.collection.objects.link(obj)
    else:
        if kind == 6:
            bpy.ops.mesh.primitive_cube_add(size=2)
            obj = bpy.context.object
            obj.scale = size
        elif kind == 5:
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=size[0], depth=2*size[1])
            obj = bpy.context.object
        elif kind == 2:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=size[0])
            obj = bpy.context.object
        else:
            raise ValueError('capsule visual needs an exact capsule mesh exporter')
        obj.name = g['name']
    obj.parent = parents[g['body']]
    obj.location = g['pos']
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Quaternion(g['quat'])
    wood = floor_material and g['name'].startswith(('entry_storage_floor', 'shelf_board'))
    if wood:
        uv = obj.data.uv_layers.active
        uv.name = next(n.uv_map for n in floor_material.node_tree.nodes if n.type == 'UVMAP')
        for polygon in obj.data.polygons:
            axes = ((1, 2), (0, 2), (0, 1))[max(range(3), key=lambda i: abs(polygon.normal[i]))]
            for loop in polygon.loop_indices:
                v = obj.data.vertices[obj.data.loops[loop].vertex_index].co
                uv.data[loop].uv = tuple(v[i]*size[i]+g['pos'][i] for i in axes)
    obj.data.materials.append(floor_material if wood else material(g))

cameras = []
for camera in record['cameras']:
    bpy.ops.object.camera_add()
    obj = bpy.context.object
    obj.name = 'Recorded camera '+camera['name']
    obj.data.sensor_fit = 'VERTICAL'
    obj.data.angle_y = math.radians(camera['fovy'])
    obj.data.clip_start = camera['clip_start']
    obj.data.clip_end = camera['clip_end']
    obj.rotation_mode = 'QUATERNION'
    cameras.append(obj)
views = []
for name in ('Environment follow', 'Third person'):
    bpy.ops.object.camera_add()
    obj = bpy.context.object
    obj.name = name
    obj.data.sensor_fit = 'VERTICAL'
    obj.data.angle_y = math.radians(52 if name == 'Environment follow' else 60)
    obj.data.clip_start = .03
    views.append(obj)
root = record['bodies'].index(record['root_body'])
for frame, (bodies, camera_poses) in enumerate(zip(poses['bodies'], poses['cameras']), 1):
    scene.frame_set(frame)
    for obj, pose in zip(parents, bodies):
        obj.location, obj.rotation_quaternion = pose[:3], Quaternion(pose[3:])
        obj.keyframe_insert('location', frame=frame)
        obj.keyframe_insert('rotation_quaternion', frame=frame)
    for obj, pose in zip(cameras, camera_poses):
        obj.location = pose[:3]
        obj.rotation_quaternion = Matrix(pose[3:].reshape(3, 3)).to_quaternion()
        obj.keyframe_insert('location', frame=frame)
        obj.keyframe_insert('rotation_quaternion', frame=frame)
    pos = Vector(bodies[root, :3])
    heading = Quaternion(bodies[root, 3:])
    focus = pos+heading@Vector((.25, 0, .85))
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    candidates = (((-.85, -1.6, 1.5), (-.85, 1.6, 1.5), (-1.0, -.7, 1.5)),
                  ((-.85, .65, 1.4), (.9, -1.2, 1.55), (-1.2, -.9, 1.55)))
    for obj, offsets in zip(views, candidates):
        for offset in offsets:
            location = pos+heading@Vector(offset)
            sight = focus-location
            if not scene.ray_cast(deps, location, sight.normalized(), distance=max(.05, sight.length-.5))[0]:
                break
        else:
            raise ValueError(f'No unobstructed {obj.name} camera at recorded frame {frame}')
        obj.location = location
        obj.rotation_euler = sight.to_track_quat('-Z', 'Y').to_euler()
        obj.keyframe_insert('location', frame=frame)
        obj.keyframe_insert('rotation_euler', frame=frame)
for pos, energy in (((4, -2, 3), 180), ((2, -3, 2.8), 120)):
    bpy.ops.object.light_add(type='AREA', location=pos)
    bpy.context.object.data.energy = energy
    bpy.context.object.data.size = 3
scene.camera = views[0]
scene['replay_recording'] = record['recording']
scene['replay_scope'] = record['scope']
scene['replay_model_sha256'] = record['model_sha256']
scene['replay_surface_adjustments'] = 'Matte oak roughness 0.6; extension UV uses the source st map in meters'
scene.view_settings.exposure = -.65
scene.view_settings.look = 'AgX - Medium High Contrast'
scene['replay_color_adjustments'] = 'Exposure -0.65 EV, AgX medium high contrast; source geometry/material colours unchanged'
scene.render.engine = 'CYCLES'
scene.cycles.samples = 16
scene.cycles.use_denoising = True
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'OPTIX'
scene.cycles.device = 'GPU'
scene.render.resolution_x, scene.render.resolution_y = 3840, 2160
scene.render.resolution_percentage = 100
scene.render.fps = round(record['fps'])
scene.render.fps_base = scene.render.fps/record['fps']
scene.frame_start, scene.frame_end = 1, record['frames']
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(a.output))
if a.preview:
    scene.render.image_settings.file_format = 'PNG'
    for frame in (1, record['frames']//2, record['frames']):
        scene.frame_set(frame)
        scene.render.filepath = str(a.preview if frame == 1 else a.preview.with_stem(a.preview.stem+f'-{frame}'))
        bpy.ops.render.render(write_still=True)
print('REPLAY READY', record['frames'], len(record['geometry']), flush=True)
