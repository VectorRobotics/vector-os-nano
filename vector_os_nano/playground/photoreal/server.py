# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2024-2026 Vector Robotics
#
# This script runs ONLY under a standalone Blender interpreter
# (`blender --background --python server.py -- --port N`). It imports `bpy`
# and is therefore distributed under Blender's GPL. The repo venv never imports
# it — process isolation keeps the Apache repo GPL-clean (campaign #10, DQ-11).

"""Blender Cycles/OptiX render server for the Vector co-sim photoreal substrate.

Speaks one-JSON-object-per-line over a localhost socket (the #9 bridge wire
protocol). Ops:
  - ``ping``     -> {ok, pong:true}
  - ``render``   -> {ok, png_b64, width, height}   (camera pose is EXACT)
  - ``shutdown`` -> {ok} then exit

A render request carries ``matrix_world`` (16 row-major floats, camera->world,
already in MuJoCo==Blender shared convention), ``fovy_rad``, ``width``,
``height``, ``samples`` and a JSON ``scene`` spec (floor + assets + lighting).
"""
import base64
import json
import os
import socket
import sys
import tempfile
import time

import bpy          # noqa: E402  (only available inside Blender)
import mathutils    # noqa: E402


def _log(msg):
    print(f"[photoreal-server] {msg}", file=sys.stderr, flush=True)


def _parse_port():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    port = 0
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
    return port


def setup_gpu():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    used = "CPU"
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type in (dev_type, "OPTIX", "CUDA")
                scene.cycles.device = "GPU"
                used = f"{dev_type}: " + ", ".join(d.name for d in gpus)
                break
        except Exception as e:  # noqa: BLE001
            _log(f"{dev_type} unavailable: {e}")
    _log(f"GPU backend = {used}")
    return used


def _clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _principled_color(obj, rgb):
    m = bpy.data.materials.new("mat")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
        rgb[0], rgb[1], rgb[2], 1.0)
    obj.data.materials.append(m)


def _textured_material(obj, image_path):
    """Wrap an image texture onto a primitive (a product label on a cylinder).

    Keeps the primitive's SHAPE (so the rendered object matches the physics
    object exactly — campaign #10 R14: depth-at-bbox stays aligned) while giving
    the VLM a recognisable textured surface (a bare colour primitive garbles it)."""
    m = bpy.data.materials.new("tex")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    try:
        tex.image = bpy.data.images.load(image_path, check_existing=True)
    except Exception as e:  # noqa: BLE001
        _log(f"texture load failed ({image_path}): {e}")
        bsdf.inputs[0].default_value = (0.8, 0.2, 0.15, 1.0)
        obj.data.materials.append(m)
        return
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(m)


def build_scene(spec):
    """Build floor + assets + lighting from a JSON spec. Idempotent (clears first)."""
    _clear()
    scene = bpy.context.scene

    floor = spec.get("floor")
    if floor:
        pos = floor.get("pos", [0.0, 0.0, 0.0])
        bpy.ops.mesh.primitive_plane_add(size=floor.get("size", 12), location=pos)
        _principled_color(bpy.context.object, floor.get("color", [0.8, 0.78, 0.74]))

    for w in spec.get("walls", []):
        bpy.ops.mesh.primitive_cube_add(size=1, location=w.get("pos", [0, 0, 0]))
        cube = bpy.context.object
        cube.scale = w.get("scale", [0.1, 3.0, 0.6])
        _principled_color(cube, w.get("color", [0.55, 0.55, 0.58]))

    for a in spec.get("assets", []):
        path = a["path"]
        before = set(bpy.data.objects)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".obj":
            bpy.ops.wm.obj_import(
                filepath=path,
                up_axis=a.get("up_axis", "Y"),
                forward_axis=a.get("forward_axis", "NEGATIVE_Z"))
        elif ext in (".gltf", ".glb"):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            raise ValueError(f"unsupported asset format: {path}")
        new = [o for o in bpy.data.objects if o not in before]
        scale = a.get("scale", 1.0)
        pos = a.get("pos", [0.0, 0.0, 0.0])
        if a.get("ground", False):  # drop so the asset's lowest point sits on z=pos[2]
            zmin = min((o.matrix_world @ mathutils.Vector(c)).z
                       for o in new if o.type == "MESH" for c in o.bound_box)
        else:
            zmin = 0.0
        for o in new:
            if o.parent is None:
                o.scale = (scale, scale, scale)
                o.location = (pos[0], pos[1], pos[2] - zmin)

    # Primitive graspable objects (pick scene's colored cylinders/boxes).
    for o in spec.get("objects", []):
        otype = o.get("type", "cylinder")
        pos = o.get("pos", [0.0, 0.0, 0.0])
        size = o.get("size", [0.03, 0.04])
        if otype == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(
                radius=size[0], depth=size[1] * 2.0, location=pos)
        elif otype == "box":
            bpy.ops.mesh.primitive_cube_add(location=pos)
            bpy.context.object.scale = (size[0], size[1], size[2])
        else:
            raise ValueError(f"unsupported primitive: {otype}")
        if o.get("texture"):
            _textured_material(bpy.context.object, o["texture"])
        else:
            _principled_color(bpy.context.object, o.get("color", [0.7, 0.7, 0.7]))

    sun = spec.get("sun", {"pos": [1.0, 1.0, 4.0], "energy": 3.0})
    bpy.ops.object.light_add(type="SUN", location=sun.get("pos", [1, 1, 4]))
    bpy.context.object.data.energy = sun.get("energy", 3.0)

    sky = spec.get("sky", {"color": [0.6, 0.7, 0.85], "strength": 1.2})
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    c = sky.get("color", [0.6, 0.7, 0.85])
    bg.inputs[0].default_value = (c[0], c[1], c[2], 1.0)
    bg.inputs[1].default_value = sky.get("strength", 1.2)


def render(req):
    scene = bpy.context.scene
    build_scene(req.get("scene", {}))

    # Camera: matrix_world is camera->world in the MuJoCo==Blender shared
    # convention (Z-up world, camera looks down local -Z). Set it verbatim.
    mw = req["matrix_world"]
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.lens_unit = "FOV"
    cam_data.angle = float(req["fovy_rad"])   # vertical FOV (sensor_fit=VERTICAL)
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    cam.matrix_world = mathutils.Matrix((mw[0:4], mw[4:8], mw[8:12], mw[12:16]))
    scene.camera = cam

    scene.render.resolution_x = int(req.get("width", 640))
    scene.render.resolution_y = int(req.get("height", 480))
    scene.render.image_settings.file_format = "PNG"
    scene.cycles.samples = int(req.get("samples", 32))
    scene.cycles.use_denoising = True

    out = os.path.join(tempfile.gettempdir(), f"photoreal_{os.getpid()}.png")
    scene.render.filepath = out
    t0 = time.monotonic()
    bpy.ops.render.render(write_still=True)
    dt = (time.monotonic() - t0) * 1000.0
    with open(out, "rb") as fh:
        png_b64 = base64.b64encode(fh.read()).decode("ascii")
    try:
        os.remove(out)
    except OSError:
        pass
    _log(f"render {scene.render.resolution_x}x{scene.render.resolution_y} "
         f"@{scene.cycles.samples} samples {dt:.0f} ms")
    return {"ok": True, "png_b64": png_b64,
            "width": scene.render.resolution_x, "height": scene.render.resolution_y}


def serve(conn):
    rfile = conn.makefile("r", encoding="utf-8")
    wfile = conn.makefile("w", encoding="utf-8")
    for line in rfile:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                resp = {"ok": True, "pong": True}
            elif op == "render":
                resp = render(req)
            elif op == "shutdown":
                wfile.write(json.dumps({"ok": True}) + "\n")
                wfile.flush()
                return
            else:
                resp = {"ok": False, "error": f"unknown op: {op}"}
        except Exception as exc:  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        wfile.write(json.dumps(resp) + "\n")
        wfile.flush()


def main():
    port = _parse_port()
    setup_gpu()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    bound = srv.getsockname()[1]
    print(f"PORT {bound}", flush=True)        # stdout handshake (bridge reads this)
    _log(f"listening on 127.0.0.1:{bound}")
    conn, _ = srv.accept()
    try:
        serve(conn)
    finally:
        conn.close()
        srv.close()


main()
