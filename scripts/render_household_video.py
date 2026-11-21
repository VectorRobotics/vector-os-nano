"""Blender entry: synchronized 4K measured-state replay, with the actual left wrist frame.

EEVEE rendering is labelled separately from the original captured sensor frames.
Temporary panel images are bounded to one frame, not a second full image sequence.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import bpy
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('replay', type=Path)
p.add_argument('blend', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--end-frame', type=int)
a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
record = json.loads((a.replay/'replay.json').read_text())
assert hashlib.sha256((a.replay/'poses.npz').read_bytes()).hexdigest() == record['files']['poses.npz']
timestamps = np.load(a.replay/'poses.npz')['time']
end = record['frames'] if a.end_frame is None else a.end_frame
assert record['fps'] == 30 and 1 <= end <= record['frames']
partial = a.output.with_suffix('.partial.mp4')
assert not a.output.exists() and not partial.exists(), 'choose a new output path'
bpy.ops.wm.open_mainfile(filepath=str(a.blend))
scene = bpy.context.scene
assert scene['replay_model_sha256'] == record['model_sha256']
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.use_raytracing = True
scene.eevee.taa_render_samples = 32
scene.render.use_motion_blur = False
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.compression = 15
views = [('Environment follow', 3840, 1080), ('Third person', 1920, 1080),
         ('Recorded camera left_wrist', 1920, 1080)]
for name, _, _ in views:
    assert bpy.data.objects[name].type == 'CAMERA'
encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', '3840x2160', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '4',
    '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)],
    stdin=subprocess.PIPE)
start, completed, frame_seconds = time.monotonic(), 0, []
try:
    with tempfile.TemporaryDirectory(prefix='replay-panels-', dir=a.replay) as temporary:
        panels = [str(Path(temporary)/f'{i}.png') for i in range(3)]
        for frame in range(1, end+1):
            t0 = time.monotonic()
            scene.frame_set(frame)
            for (name, width, height), file in zip(views, panels):
                scene.camera = bpy.data.objects[name]
                scene.render.resolution_x, scene.render.resolution_y = width, height
                scene.render.filepath = file
                bpy.ops.render.render(write_still=True)
            font = 'fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65'
            stamp = f'sim {timestamps[frame-1]:07.3f} s'
            graph = ('[1:v][2:v]hstack[bottom];[0:v][bottom]vstack,'+
                     f"drawtext={font}:text='ENVIRONMENT | {stamp}':x=24:y=24,"+
                     f"drawtext={font}:text='THIRD PERSON | {stamp}':x=24:y=1104,"+
                     f"drawtext={font}:text='LEFT WRIST | {stamp}':x=1944:y=1104,"+
                     f"drawtext={font}:text='MEASURED STATE RE-RENDER | EEVEE | not original sensor input | motion unverified':x=24:y=2110")
            command = ['ffmpeg', '-v', 'error', '-threads', '2']
            for file in panels:
                command += ['-i', file]
            command += ['-filter_complex_threads', '2', '-filter_complex', graph,
                        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
            rgb = subprocess.check_output(command)
            assert len(rgb) == 3840*2160*3
            encoder.stdin.write(rgb)
            completed = frame
            frame_seconds.append(time.monotonic()-t0)
            if frame == 1 or frame % 30 == 0:
                print('FRAME', frame, 'of', end, 'wall_s', round(time.monotonic()-start, 1), flush=True)
finally:
    encoder.stdin.close()
    try:
        rc = encoder.wait(timeout=60)
    except subprocess.TimeoutExpired:
        encoder.kill()
        encoder.wait()
        raise
    metadata = dict(
        recording=record['recording'], model_sha256=record['model_sha256'],
        blend_sha256=hashlib.sha256(a.blend.read_bytes()).hexdigest(),
        frames=completed, full_recording_frames=record['frames'], fps=30,
        complete=False, engine='BLENDER_EEVEE',
        wall_s=time.monotonic()-start, frame_wall_s=frame_seconds,
        cameras=[v[0] for v in views], motion_verdict='unverified', scope=__doc__)
    a.output.with_suffix('.render.json').write_text(json.dumps(metadata, indent=2))
assert rc == 0 and completed == end
info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
    '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate',
    '-of', 'json', str(partial)]))['streams'][0]
assert (int(info['nb_read_frames']), info['width'], info['height'], info['r_frame_rate']) == (end, 3840, 2160, '30/1')
partial.rename(a.output)
metadata.update(complete=completed == record['frames'], video_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
a.output.with_suffix('.render.json').write_text(json.dumps(metadata, indent=2))
print('RENDERED', a.output, completed, flush=True)
