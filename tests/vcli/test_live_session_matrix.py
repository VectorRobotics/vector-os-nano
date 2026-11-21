"""Real-model session selections, independently graded; billed only by explicit opt-in."""
import os
from pathlib import Path

import pytest

from tests.harness import live_matrix as lm

pytestmark = [pytest.mark.sim, pytest.mark.live_llm, pytest.mark.timeout(1000)]


@pytest.mark.parametrize('cell_id', [
    'go2-pybullet-room', 'so101-pybullet-grasp', 'piper-pybullet-grasp', 'so101-mujoco-home'])
def test_live_session_selection_is_proven_on_the_requested_engine(cell_id, tmp_path):
    cell = next(c for c in lm.load_cells() if c['id'] == cell_id)
    out = Path(os.environ.get('GATE_OUT', tmp_path)) / 'live_matrix' / cell_id
    report = lm.run([cell], out, n=2)
    row = report['cells'][0]
    assert row.get('n_run') == 2 and report['graded'] == 2, lm.markdown(report)
    assert row['success'] == row['verified'] == 2, lm.markdown(report)
    assert report['false_grounded'] == report['false_reject'] == 0, lm.markdown(report)
