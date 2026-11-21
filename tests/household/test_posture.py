"""Development CLI integration of measured torso control; no household task claim."""
import json
import os
from pathlib import Path
import pytest
from tests.harness import live_matrix as lm

pytestmark = [pytest.mark.sim, pytest.mark.timeout(700)]


def _posture(tmp_path, monkeypatch, *, live=False):
    assets = os.environ.get('HOUSEHOLD_ASSETS')
    if not assets:
        pytest.skip('HOUSEHOLD_ASSETS must name the locally converted private-CAD asset store')
    monkeypatch.setenv('VECTOR_HOME', assets)
    out = Path(os.environ.get('GATE_OUT', tmp_path))/('posture-live' if live else 'posture')
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(out/'rollouts'))
    cell = dict(id='malo-torso', body='malo', backend='mujoco',
                launch='启动 malo 的 MuJoCo 仿真，使用 artvip_home 世界，无窗口。',
                task='把 torso 调整到 shelf 姿态，然后检查实际关节位置。', timeout_s=600)
    script = {'steps': [
        {'when': cell['launch'], 'tool_calls': [{'name': 'start_simulation', 'input': {
            'embodiment': 'malo', 'backend': 'mujoco', 'world': 'artvip_home', 'gui': False}}]},
        {'when': '', 'text': '启动完成', 'tool_calls': []},
        {'when': cell['task'], 'tool_calls': [{'name': 'joint_posture', 'input': {'group': 'torso', 'posture': 'shelf'}}]},
        {'when': 'joint_posture', 'tool_calls': [{'name': 'verify', 'input': {'expr': "joint_group_at('torso', 'shelf')"}}]},
        {'when': '', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
    ]}
    rec = lm.run_rep(cell, 0, out, tool_script=None if live else script)
    assert not rec['timed_out']
    text = Path(rec['transcript']).read_text()
    assert 'verified=True' in lm._turns(text, cell)[cell['task']]
    assert 'actor=CAUSED' in lm._turns(text, cell)[cell['task']]
    facts = lm.session_facts(Path(rec['session']), cell['task'])
    assert 'joint_posture' in facts['tools'] and 'verify' in facts['tools']
    snapshots = [json.loads(line) for line in Path(rec['dump']).read_text().splitlines()]
    states = [s for s in snapshots if s['goal'] == cell['task']]
    before = next(s for s in states if s['phase'] == 'start')
    after = next(s for s in reversed(states) if s['phase'] == 'end')
    assert before['joint_groups']['torso']['q'] != pytest.approx([-.15, .2], abs=.003)
    assert after['joint_groups']['torso']['q'] == pytest.approx([-.15, .2], abs=.003)
    (out/'measured-posture.json').write_text(json.dumps(dict(
        scope=__doc__, run=rec, before=before['joint_groups'], after=after['joint_groups'],
        physical_task='unverified', motion='unverified', deployment_gate='closed'), indent=2))


def test_configured_torso_posture_through_the_cli(tmp_path, monkeypatch):
    _posture(tmp_path, monkeypatch)


@pytest.mark.live_llm
def test_real_model_adjusts_torso_through_the_cli(tmp_path, monkeypatch):
    _posture(tmp_path, monkeypatch, live=True)
