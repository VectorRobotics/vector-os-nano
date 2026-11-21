"""Opt-in, append-only physics states and original rendered frames for offline replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import mujoco as mj
import numpy as np


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
                         ['eq_data', m.eq_data.size], ['max_contact_residual', 1]],
                         timestep=scene.timestep, reset_seed=None, rows=0, frames=0,
                         speculative_contacts=bool(m.opt.enableflags & int(getattr(mj.mjtEnableBit, 'mjENBL_SPECULATIVE', 0))),
                         contact_time='last dynamics stage, before the step integration',
                         frame_scope='original simulator renders; includes readiness/evaluator captures, not proof of actor consumption',
                         physical_verdict='unverified', motion_verdict='unverified', closed=False)
        self._states = (self.path / 'states.bin').open('xb', buffering=1024*1024)
        self._frame_lock = threading.Lock()
        self._metadata()

    def _metadata(self) -> None:
        partial = self.path / 'recording.json.tmp'
        partial.write_text(json.dumps(self.meta, indent=2))
        partial.replace(self.path / 'recording.json')

    def state(self, m, d) -> None:
        penetration = max(0., -float(d.contact.dist[:d.ncon].min())) if d.ncon else 0.
        row = np.concatenate(([d.time, time.monotonic()], d.qpos, d.qvel, d.ctrl,
                              d.actuator_force, d.eq_active, m.eq_data.ravel(), [penetration]))
        self._states.write(row.astype('<f8', copy=False).tobytes())
        self.meta['rows'] += 1

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
            self.meta['closed'] = True
            self._metadata()
