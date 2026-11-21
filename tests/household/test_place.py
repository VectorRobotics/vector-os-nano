"""Development Malo contact pick/return, with an authored shelf point and independent physical grade."""
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
import pytest
from tests.household.test_grasp import prepare_world, snapshot
from tests.harness.pty_cli import run_repl_session
from tests.harness.gt_audit import grade

pytestmark = [pytest.mark.sim, pytest.mark.timeout(1000)]


def _place(tmp_path, monkeypatch, live=False, carry=False):
    configured = os.environ.get('HOUSEHOLD_ASSETS')
    if not configured:
        pytest.skip('HOUSEHOLD_ASSETS must name the locally converted private-CAD asset store')
    monkeypatch.setenv('VECTOR_HOME', configured)
    world = prepare_world(Path(configured))
    out = Path(os.environ.get('GATE_OUT', tmp_path))/(('place-live' if live else 'place')+('-carry' if carry else ''))
    snapshot(out, Path(configured), world)
    home = out/'home'; home.mkdir(parents=True, exist_ok=True)
    for name in ('assets', 'embodiments', 'worlds', 'cache'):
        if (Path(configured)/name).is_dir():
            (home/name).symlink_to(Path(configured)/name)
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(out/'rollouts'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', 'head,left_wrist')
    launch = f'start malo in {world} without a window'
    pick, put = 'pick up the tissue refill', 'put the tissue refill on the storage shelf'
    transport = 'Step backward 0.2 metres, then forward 0.2 metres at 0.05 metres per second while keeping the tissue refill held.'
    script = {'steps': [
        {'when': launch, 'tool_calls': [{'name': 'start_simulation', 'input': {'embodiment': 'malo', 'world': world, 'backend': 'mujoco', 'gui': False}}]},
        {'when': '', 'text': 'simulation ready', 'tool_calls': []},
        {'when': pick, 'tool_calls': [{'name': 'grasp', 'input': {'query': 'tissue refill'}}]},
        {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': "holding_object('tissue_refill')"}}]},
        {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
        {'when': put, 'tool_calls': [{'name': 'place', 'input': {'surface': 'storage_shelf'}}]},
        {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': "object_on('tissue_refill', 'storage_shelf')"}}]},
        {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
    ]}
    turns = [launch, (re.compile(r'Started malo simulation, ready in'), pick)]
    if carry:
        script['steps'][5:5] = [
            {'when': transport, 'tool_calls': [{'name': 'walk', 'input': {'direction': 'backward', 'distance': .2, 'speed': .05}}]},
            {'when': '', 'tool_calls': [{'name': 'walk', 'input': {'direction': 'forward', 'distance': .2, 'speed': .05}}]},
            {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': "holding_object('tissue_refill')"}}]},
            {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
        ]
        turns.append((re.compile('goal proven: holding_object'), transport))
    turns.append(put if carry else (re.compile('goal proven: holding_object'), put))
    result = run_repl_session(turns, tool_script=None if live else script, live=live, answers=['a']*10,
        timeout_sec=900, out_dir=out, extra_env={'VECTOR_HOME': str(home), 'VECTOR_NAV_STACK': ''})
    (out/'transcript.txt').write_text(result.text)
    checks = {'grasp': grade(Path(result.dump), {'held_lifted': 'tissue_refill'}, pick),
              'place': grade(Path(result.dump), {'object_on': ['tissue_refill', 'storage_shelf']}, put)}
    report = dict(scope=__doc__, reset_seed=0, model='real' if live else 'scripted', checks=checks,
        physical_verdict='unverified', motion_verdict='unverified', deployment_gate='closed')
    (out/'report.json').write_text(json.dumps(report, indent=2))
    from scripts.grade_place import grade as physical_grade
    from vector_os_nano.embodiments import registry
    from vector_os_nano.schemas.world import load_world
    recording, = (out/'rollouts').glob('*/recording.json')
    physical = physical_grade(recording.parent, registry.load('malo'), load_world(world),
        json.loads((Path(configured)/'embodiments/malo/audit.json').read_text()),
        [1.7, -3.33], 'tissue_refill', 'storage_shelf')
    if carry:
        import mujoco as mj
        from scripts.grade_carry import carry_window, CRITERIA
        from vector_os_nano.sim.backends.mujoco.recording import read_recording
        rows = [json.loads(line) for line in Path(result.dump).read_text().splitlines()]
        interval = {r['phase']: r['sim_time'] for r in rows if r['goal'] == transport}
        meta, states, contacts = read_recording(recording.parent)
        checks_carry, metrics = carry_window(mj.MjModel.from_binary_path(meta['model_path']), states, contacts,
            registry.load('malo'), load_world(world), 'tissue_refill', interval['start'], interval['end'])
        physical.update(carry_checks=checks_carry, carry_metrics=metrics, carry_criteria=CRITERIA, carry_interval=interval)
        if not all(v is True for v in checks_carry.values()):
            physical['physical_verdict'] = 'fail'
    meta = json.loads(recording.read_text())
    physical.update(reset_seed=0, timestamp=datetime.now(timezone.utc).isoformat(),
        checkpoint=json.loads((out/'checkpoint.json').read_text()),
        videos=[str(recording.parent/video['file']) for video in meta['sensor_videos']],
        motion_observations='Continuous motion has not been reviewed; metrics are not a motion verdict.')
    (out/'physical-grade.json').write_text(json.dumps(physical, indent=2))
    report.update(physical_verdict=physical['physical_verdict'], physical_grade=str(out/'physical-grade.json'))
    (out/'report.json').write_text(json.dumps(report, indent=2))
    assert not result.timed_out and all(value[0] for value in checks.values()), result.text[-5000:]
    assert "goal proven: object_on('tissue_refill', 'storage_shelf')" in result.text, result.text[-5000:]
    assert physical['physical_verdict'] == 'pass', physical['checks']


def test_contact_grasp_and_named_place_in_the_cli(tmp_path, monkeypatch):
    _place(tmp_path, monkeypatch)


@pytest.mark.live_llm
def test_real_model_contact_pick_and_place(tmp_path, monkeypatch):
    _place(tmp_path, monkeypatch, live=True)


def test_contact_transport_and_place_in_the_cli(tmp_path, monkeypatch):
    _place(tmp_path, monkeypatch, carry=True)


@pytest.mark.live_llm
def test_real_model_contact_transport_and_place(tmp_path, monkeypatch):
    _place(tmp_path, monkeypatch, live=True, carry=True)
