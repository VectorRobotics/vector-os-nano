"""Development drawer cycle through bare CLI; authored viewing reset, not held-out acceptance."""
import json
import os
import re
from pathlib import Path

import pytest
import yaml

from tests.household.test_grasp import snapshot
from tests.harness.pty_cli import run_repl_session

pytestmark = [pytest.mark.sim, pytest.mark.timeout(3800)]


def _drawer(tmp_path, monkeypatch, *, live=False):
    configured = os.environ.get('HOUSEHOLD_ASSETS')
    if not configured:
        pytest.skip('HOUSEHOLD_ASSETS must name the locally converted private-CAD asset store')
    assets = Path(configured)
    monkeypatch.setenv('VECTOR_HOME', str(assets))
    name = 'artvip_drawer_validation'
    fixture = assets/'worlds'/name
    fixture.mkdir(exist_ok=True)
    source = assets/'worlds/artvip_home'
    raw = yaml.safe_load((source/'world.yaml').read_text())
    raw.update(scene='scene.xml', spawn=dict(xy=[4.72, -2.35], z=0, heading=0))
    scene = fixture/'scene.xml'
    if not scene.exists():
        scene.symlink_to(source/'scene.xml')
    assert scene.resolve() == (source/'scene.xml').resolve()
    manifest = fixture/'world.yaml'
    if manifest.exists():
        assert yaml.safe_load(manifest.read_text()) == raw, 'refusing to overwrite a different validation world'
    else:
        manifest.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
    out = Path(os.environ.get('GATE_OUT', tmp_path))/('drawer-live' if live else 'drawer')
    snapshot(out, assets, name)
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(out/'rollouts'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', 'head,left_wrist')
    launch = f'start malo in {name} without a window'
    opening, closing = 'open the storage drawer', 'close the storage drawer'
    script = {'steps': [
        {'when': launch, 'tool_calls': [{'name': 'start_simulation', 'input': {'embodiment': 'malo', 'world': name, 'backend': 'mujoco', 'gui': False}}]},
        {'when': '', 'text': 'simulation ready', 'tool_calls': []},
        {'when': opening, 'tool_calls': [{'name': 'open_drawer', 'input': {'query': 'storage drawer'}}]},
        {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': "articulation_open('storage_drawer')"}}]},
        {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
        {'when': closing, 'tool_calls': [{'name': 'close_drawer', 'input': {'query': 'storage drawer'}}]},
        {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': "articulation_closed('storage_drawer')"}}]},
        {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
    ]}
    result = run_repl_session([launch, (re.compile('Started malo simulation, ready in'), opening),
        (re.compile('goal proven: articulation_open'), closing)], tool_script=None if live else script,
        live=live, answers=['a']*10, timeout_sec=3600, out_dir=out,
        extra_env={'VECTOR_HOME': str(assets), 'VECTOR_NAV_STACK': ''})
    (out/'transcript.txt').write_text(result.text)
    from scripts.grade_drawer import grade
    from vector_os_nano.embodiments import registry
    from vector_os_nano.schemas.world import load_world
    recording, = (out/'rollouts').glob('*/recording.json')
    report = grade(recording.parent, registry.load('malo'), load_world(name),
        json.loads((assets/'embodiments/malo/audit.json').read_text()), [4.72, -2.35], 'storage_drawer')
    report.update(reset_seed=0, checkpoint=json.loads((out/'checkpoint.json').read_text()),
        model='real' if live else 'scripted', task_scope=__doc__, cli_timed_out=result.timed_out,
        videos=[str(recording.parent/v['file']) for v in json.loads(recording.read_text())['sensor_videos']],
        motion_observations='No continuous video reviewed; physical metrics do not imply a motion pass.')
    (out/'physical-grade.json').write_text(json.dumps(report, indent=2))
    assert not result.timed_out, result.text[-4000:]
    assert result.exit_code == 0, result.text[-4000:]
    for state in ('open', 'closed'):
        assert f"goal proven: articulation_{state}('storage_drawer')" in result.text, result.text[-4000:]
    assert report['physical_verdict'] == 'pass', report['checks']


def test_contact_drawer_cycle_through_cli(tmp_path, monkeypatch):
    _drawer(tmp_path, monkeypatch)


@pytest.mark.live_llm
def test_real_model_contact_drawer_cycle(tmp_path, monkeypatch):
    _drawer(tmp_path, monkeypatch, live=True)
