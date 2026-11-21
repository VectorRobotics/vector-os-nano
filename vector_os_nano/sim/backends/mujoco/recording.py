"""Opt-in, append-only physics states and original rendered frames for offline replay."""
from __future__ import annotations

import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

import mujoco as mj
import numpy as np


def read_recording(directory: str | Path):
    """Read only a complete, finite recording whose model still has its recorded hash."""
    directory = Path(directory)
    meta = json.loads((directory/'recording.json').read_text())
    if meta.get('schema') != 1 or not meta.get('closed') or meta.get('dtype') != '<f8':
        raise ValueError('incomplete or unsupported recording')
    model = Path(meta['model_path'])
    meta['model_path'] = str((directory/model if not model.is_absolute() else model).resolve())
    with Path(meta['model_path']).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != meta['model_sha256']:
            raise ValueError('recording model SHA256 mismatch')

    def columns(raw, fields, count):
        width = sum(size for name, size in fields)
        if width <= 0 or raw.size != count*width or not np.isfinite(raw).all():
            raise ValueError('recording shape/count or finite-value check failed')
        rows, index, out = raw.reshape(count, width), 0, {}
        for name, size in fields:
            out[name] = rows[:, index:index+size]
            index += size
        return out

    states = columns(np.memmap(directory/'states.bin', dtype='<f8', mode='r'), meta['fields'], meta['rows'])
    contacts = {}
    if 'contact_fields' in meta:
        with gzip.open(directory/'contacts.bin.gz', 'rb') as stream:
            contacts = columns(np.frombuffer(stream.read(), dtype='<f8'), meta['contact_fields'], meta['contact_rows'])
    return meta, states, contacts


class Recording:
    def __init__(self, directory: str, scene) -> None:
        raw = os.environ.get('VECTOR_ROLLOUT_CAMERAS', '')
        self.cameras = tuple(n.strip() for n in raw.split(',')) if raw else ()
        if len(set(self.cameras)) != len(self.cameras) or any(n not in scene.cameras for n in self.cameras):
            raise ValueError(f'rollout cameras must be unique names from {sorted(scene.cameras)}')
        if self.cameras and (not shutil.which('ffmpeg') or not shutil.which('ffprobe')):
            raise RuntimeError('continuous sensor recording requires ffmpeg and ffprobe')
        if self.cameras and scene.timestep > 1/30:
            raise ValueError('30 fps sensor recording requires a physics timestep <= 1/30 s')
        self.path = Path(directory) / f'mujoco-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'frames').mkdir()
        m = scene.model
        source = self.path/'model.mjb'
        mj.mj_saveModel(m, str(source))
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.meta = dict(schema=1, engine='mujoco', version=mj.__version__,
                         model_path=source.name, source_model_path=str(scene.xml_path.resolve()), model_sha256=digest,
                         dtype='<f8', fields=[['sim_time', 1], ['wall_monotonic', 1],
                         ['qpos', m.nq], ['qvel', m.nv], ['ctrl', m.nu],
                         ['actuator_force', m.nu], ['eq_active', m.neq],
                         ['eq_data', m.eq_data.size], ['max_contact_residual', 1], ['max_applied_force', 1]],
                         timestep=scene.timestep, reset_seed=None, rows=0, frames=0,
                         contact_rows=0, contact_fields=[['sim_time', 1], ['body_a', 1], ['body_b', 1],
                         ['position', 3], ['force_on_b', 3], ['distance', 1], ['normal_on_b', 3]],
                         speculative_contacts=bool(m.opt.enableflags & int(getattr(mj.mjtEnableBit, 'mjENBL_SPECULATIVE', 0))),
                         contact_time='last dynamics stage, before the step integration',
                         frame_scope='original simulator renders; includes readiness/evaluator captures, not proof of actor consumption',
                         physical_verdict='unverified', motion_verdict='unverified', closed=False)
        self.meta['sensor_videos'] = []
        self._videos = []
        self._video_next = None
        self._video_segment = 0
        self._states = (self.path / 'states.bin').open('xb', buffering=1024*1024)
        self._contacts = gzip.open(self.path / 'contacts.bin.gz', 'xb', compresslevel=1)
        self._frame_lock = threading.Lock()
        self._metadata()

    def _metadata(self) -> None:
        partial = self.path / 'recording.json.tmp'
        partial.write_text(json.dumps(self.meta, indent=2))
        partial.replace(self.path / 'recording.json')

    def state(self, m, d, loads=()) -> None:
        penetration = max(0., -float(d.contact.dist[:d.ncon].min())) if d.ncon else 0.
        row = np.concatenate(([d.time, time.monotonic()], d.qpos, d.qvel, d.ctrl,
                              d.actuator_force, d.eq_active, m.eq_data.ravel(),
                              [penetration, max(np.abs(d.qfrc_applied).max(initial=0), np.abs(d.xfrc_applied).max(initial=0))]))
        self._states.write(row.astype('<f8', copy=False).tobytes())
        self.meta['rows'] += 1
        if loads:
            contacts = [[d.time, c.body_a, c.body_b, *c.position, *c.force_on_b, c.distance, *c.normal_on_b] for c in loads]
            self._contacts.write(np.asarray(contacts, dtype='<f8').tobytes())
            self.meta['contact_rows'] += len(loads)

    def frame(self, frame) -> None:
        with self._frame_lock:
            index = self.meta['frames']
            self.meta['frames'] += 1
        arrays = {key: getattr(frame, key) for key in ('rgb', 'depth', 'seg', 'K', 'cam_pos', 'cam_mat')
                  if getattr(frame, key) is not None}
        np.savez_compressed(self.path / 'frames' / f'{index:08d}.npz',
                            sim_time=frame.sim_time, camera=frame.camera, **arrays)

    def capture_due(self, world) -> None:
        """Capture actual online pixels, without dropping frames; opt-in recording can slow wall time."""
        try:
            self._capture_due(world)
        except BaseException as exc:
            for *_, meta in self._videos:
                meta['error'] = f'{type(exc).__name__}: {exc}'
            raise

    def _capture_due(self, world) -> None:
        if not self.cameras:
            return
        now = world.time()
        if self._video_next is not None and now < self._video_next-1e-9:
            return
        if not self._videos:
            self._video_next = now
            for i, camera in enumerate(self.cameras):
                name = f'sensor-{self._video_segment:03d}-{i:02d}'
                partial = self.path/(name+'.partial.mkv')
                log = (self.path/(name+'.ffmpeg.log')).open('xb')
                meta = dict(camera=camera, width=640, height=480, fps=30, frames=0,
                            start_sim_time=now, last_sim_time=now, complete=False,
                            file=name+'.mkv', timestamps=name+'.timestamps.jsonl',
                            scope='original online simulator RGB; lossless FFV1; not proof of actor consumption')
                stamps = None
                try:
                    stamps = (self.path/meta['timestamps']).open('x')
                    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                        '-video_size', '640x480', '-framerate', '30', '-i', 'pipe:0', '-an',
                        '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-threads', '2', str(partial)],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                except BaseException:
                    log.close()
                    if stamps is not None:
                        stamps.close()
                    raise
                self._videos.append((proc, log, stamps, partial, meta))
                self.meta['sensor_videos'].append(meta)
        if now-self._video_next > world.scene.timestep*1.001:
            raise RuntimeError('continuous sensor recording missed a scheduled physics frame')
        for proc, _, stamps, _, meta in self._videos:
            name = meta['camera']
            frame = world._pool.submit(world._render_job, name, world.scene.cameras[name],
                                       640, 480, ('rgb',), False).result()
            if abs(frame.sim_time-now) > 1e-9:
                raise RuntimeError('sensor views were not captured at the same physics time')
            pixels = np.ascontiguousarray(frame.rgb, dtype=np.uint8).tobytes()
            proc.stdin.write(pixels)
            stamps.write(json.dumps(dict(frame=meta['frames'], sim_time=frame.sim_time,
                wall_monotonic=time.monotonic(), rgb_sha256=hashlib.sha256(pixels).hexdigest()))+'\n')
            meta['frames'] += 1
            meta['last_sim_time'] = now
        self._video_next += 1/30

    def finish_videos(self) -> None:
        """Close a segment on reset/exit; partial encodes are never advertised as complete."""
        pending, self._videos = self._videos, []
        self._video_next = None
        if pending:
            self._video_segment += 1
        errors = []
        for proc, log, stamps, partial, meta in pending:
            try:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                if proc.wait(timeout=30) != 0:
                    raise RuntimeError(f'sensor encoder failed; inspect {log.name}')
                probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                    '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height',
                    '-of', 'json', str(partial)]))['streams'][0]
                if (int(probe['nb_read_frames']), probe['width'], probe['height']) != (meta['frames'], 640, 480):
                    raise RuntimeError('sensor video frame count or dimensions mismatch')
                partial.rename(self.path/meta['file'])
                meta['complete'] = 'error' not in meta
            except BaseException as exc:
                errors.append(exc)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            finally:
                log.close()
                stamps.close()
        self._metadata()
        if errors:
            raise RuntimeError('sensor video segment failed') from errors[0]

    def close(self) -> None:
        if not self._states.closed:
            self._states.close()
            self._contacts.close()
            self.finish_videos()
            self.meta['closed'] = True
            self._metadata()
