"""Lossless archive reads keep the existing recording shape and finite-value checks."""
import gzip
import hashlib
import json
import numpy as np
import pytest

pytest.importorskip('mujoco')
from vector_os_nano.sim.backends.mujoco.recording import read_recording


@pytest.mark.parametrize('compressed', [False, True])
def test_state_archive_preserves_values_and_rejects_bad_rows(tmp_path, compressed):
    model = b'opaque model content: this reader checks its hash, not its physics'
    (tmp_path/'model.mjb').write_bytes(model)
    meta = dict(schema=1, closed=True, dtype='<f8', rows=2,
                fields=[['sim_time', 1], ['qpos', 2]], model_path='model.mjb',
                model_sha256=hashlib.sha256(model).hexdigest())
    (tmp_path/'recording.json').write_text(json.dumps(meta))
    values = np.array([[0., .3, -.7], [.001, .4, -.6]], dtype='<f8')
    path = tmp_path/('states.bin.gz' if compressed else 'states.bin')
    write = (lambda data: path.write_bytes(gzip.compress(data))) if compressed else path.write_bytes
    write(values.tobytes())
    _, states, _ = read_recording(tmp_path)
    np.testing.assert_array_equal(states['qpos'], values[:, 1:])
    for bad in (values[:-1], np.full_like(values, np.nan)):
        write(bad.tobytes())
        with pytest.raises(ValueError, match='shape/count|finite'):
            read_recording(tmp_path)
