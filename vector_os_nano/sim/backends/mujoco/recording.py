"""Opt-in, append-only physics states and original rendered frames for offline replay."""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
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
        self.path = Path(directory) / f'mujoco-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'frames').mkdir()
        source = scene.xml_path.with_suffix('.mjb')
        if not source.is_file():
            source = scene.xml_path
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        m = scene.model
        self.meta = dict(schema=1, engine='mujoco', version=mj.__version__,
                         model_path=str(source.resolve()), model_sha256=digest,
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

    def close(self) -> None:
        if not self._states.closed:
            self._states.close()
            self._contacts.close()
            self.meta['closed'] = True
            self._metadata()
