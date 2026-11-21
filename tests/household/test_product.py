"""Private-CAD integration through the ordinary CLI; explicitly selected local asset tier."""
import json
import math
import os
from pathlib import Path

import pytest

from tests.harness import live_matrix as lm
from tests.household.test_grasp import snapshot

pytestmark = [pytest.mark.sim, pytest.mark.timeout(700)]


def _navigate(tmp_path, monkeypatch, *, live=False):
    configured = os.environ.get('HOUSEHOLD_ASSETS')
    if not configured:
        pytest.skip('HOUSEHOLD_ASSETS must name the locally converted private-CAD asset store')
    assets = Path(configured)
    assert (assets/'embodiments/malo/physical.xml').is_file()
    assert (assets/'worlds/artvip_home/scene.xml').is_file()
    (assets/'cache/scenes').mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('VECTOR_HOME', str(assets))
    out = Path(os.environ.get('GATE_OUT', tmp_path))/('household-live' if live else 'household')
    snapshot(out, assets, 'artvip_home')
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(out/'rollouts'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', 'head,left_wrist')
    cell = dict(id='malo-artvip-dining', body='malo', backend='mujoco',
                launch='启动 malo 的 MuJoCo 仿真，使用 artvip_home 世界，无窗口。', task='去餐厅', kind='navigate',
                expect={'base_at': [5.5, 1.3]}, n=1, timeout_s=600)
    script = {'steps': [
        {'when': cell['launch'], 'tool_calls': [{'name': 'start_simulation', 'input': {
            'embodiment': 'malo', 'backend': 'mujoco', 'world': 'artvip_home', 'gui': False}}]},
        {'when': '', 'text': '启动完成', 'tool_calls': []},
        {'when': cell['task'], 'tool_calls': [{'name': 'go_to_room', 'input': {'room': 'dining_room'}}]},
        {'when': 'go_to_room(dining_room)', 'tool_calls': [{'name': 'verify', 'input': {'expr': "in_room('dining_room')"}}]},
        {'when': '', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
    ]}
    rec = lm.run_rep(cell, 0, out, tool_script=None if live else script)
    result = lm.judge(cell, rec)
    report = lm.summarize([cell], [result])
    report['scope'] = 'Development run with an authored map; physical safety and motion acceptance are separate.'
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    assert not rec['timed_out'] and result['status'] == 'ran', lm.markdown(report)
    assert result['grade'] and result['verdict']['verified'], lm.markdown(report)
    assert result['verdict']['goal_check'] == 'proven', lm.markdown(report)
    snapshots = [json.loads(line) for line in Path(rec['dump']).read_text().splitlines()]
    final = next(s for s in reversed(snapshots) if s['phase'] == 'end' and s['goal'] == cell['task'])
    assert final['world'] == 'artvip_home' and final['backend'] == 'mujoco'
    assert math.dist(final['base'][:2], [5.5, 1.3]) <= .15
    recordings = list((out/'rollouts').glob('*/recording.json'))
    assert len(recordings) == 1
    recording = json.loads(recordings[0].read_text())
    assert recording['closed'] and recording['rows'] > 1000 and recording['frames'] > 0
    from scripts.grade_navigation import grade
    from vector_os_nano.embodiments import registry
    from vector_os_nano.schemas.world import load_world
    physical = grade(recordings[0].parent, registry.load('malo'), load_world('artvip_home'),
                     json.loads((assets/'embodiments/malo/audit.json').read_text()), [5.5, 1.3])
    physical.update(reset_seed=0, checkpoint=json.loads((out/'checkpoint.json').read_text()),
                    videos=[str(recordings[0].parent/v['file']) for v in recording['sensor_videos']],
                    motion_observations='No continuous video reviewed.')
    (out/'physical-grade.json').write_text(json.dumps(physical, indent=2))
    assert physical['physical_verdict'] == 'pass', physical['checks']


def test_malo_navigates_in_the_rebuilt_home_through_the_cli(tmp_path, monkeypatch):
    _navigate(tmp_path, monkeypatch)


@pytest.mark.live_llm
def test_real_model_navigates_malo_in_the_rebuilt_home(tmp_path, monkeypatch):
    _navigate(tmp_path, monkeypatch, live=True)
