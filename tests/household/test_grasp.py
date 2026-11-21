"""Development CLI grasp from an authored observation reset; no held-out or whole-home claim."""
import json
import os
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pytest
import yaml
from tests.harness import live_matrix as lm

pytestmark = [pytest.mark.sim, pytest.mark.timeout(900)]


def snapshot(out, assets, world):
    """Retain controller, grading and config provenance beside the immutable physics model."""
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    patch = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'vector_os_nano',
                                    'examples/household', 'scripts', 'tests/household'], cwd=repo)
    (out/'controller.patch').write_bytes(patch)
    for source, name in ((assets/'embodiments/malo/robot.yaml', 'robot.yaml'),
                         (assets/'worlds'/world/'world.yaml', 'world.yaml')):
        shutil.copyfile(source, out/name)
    (out/'checkpoint.json').write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        patch_sha256=hashlib.sha256(patch).hexdigest(), reset_seed=0), indent=2))


def prepare_world(assets):
    source = assets/'worlds/artvip_home'
    world_name = 'artvip_grasp_validation'
    fixture = assets/'worlds'/world_name
    fixture.mkdir(exist_ok=True)
    raw = yaml.safe_load((source/'world.yaml').read_text())
    raw['scene'] = 'scene.xml'
    scene = fixture/'scene.xml'
    if not scene.exists():
        scene.symlink_to(source/'scene.xml')
    assert scene.resolve() == (source/'scene.xml').resolve()
    raw['spawn'] = dict(xy=[1.7, -3.7], z=0, heading=1.5707963267948966)
    manifest = fixture/'world.yaml'
    if manifest.exists():
        assert yaml.safe_load(manifest.read_text()) == raw, 'existing validation world differs; do not overwrite it'
    else:
        manifest.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
    return world_name


def _grasp(tmp_path, monkeypatch, *, live=False):
    configured = os.environ.get('HOUSEHOLD_ASSETS')
    if not configured:
        pytest.skip('HOUSEHOLD_ASSETS must name the locally converted private-CAD asset store')
    assets = Path(configured)
    monkeypatch.setenv('VECTOR_HOME', str(assets))
    world_name = prepare_world(assets)
    out = Path(os.environ.get('GATE_OUT', tmp_path))/('grasp-live' if live else 'grasp')
    snapshot(out, assets, world_name)
    monkeypatch.setenv('VECTOR_ROLLOUT_DIR', str(out/'rollouts'))
    monkeypatch.setenv('VECTOR_ROLLOUT_CAMERAS', 'head,left_wrist')
    cell = dict(id='malo-contact-grasp', body='malo', backend='mujoco', kind='grasp',
        launch=f'启动 malo 的 MuJoCo 仿真，使用 {world_name} 世界，无窗口。',
        task='pick up the tissue refill', expect={'held_lifted': 'tissue_refill'}, timeout_s=800)
    script = {'steps': [
        {'when': cell['launch'], 'tool_calls': [{'name': 'start_simulation', 'input': {
            'embodiment': 'malo', 'backend': 'mujoco', 'world': world_name, 'gui': False}}]},
        {'when': '', 'text': '启动完成', 'tool_calls': []},
        {'when': cell['task'], 'tool_calls': [{'name': 'grasp', 'input': {'query': 'tissue refill'}}]},
        {'when': 'grasp', 'tool_calls': [{'name': 'verify', 'input': {'expr': "holding_object('tissue_refill')"}}]},
        {'when': '', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'}]}
    rec = lm.run_rep(cell, 0, out, tool_script=None if live else script)
    result = lm.judge(cell, rec)
    report = lm.summarize([cell], [result]); report['scope'] = __doc__
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    assert not rec['timed_out'] and result['status'] == 'ran', lm.markdown(report)
    assert result['grade'] and result['verdict']['verified'], lm.markdown(report)
    from vector_os_nano.embodiments import registry
    from vector_os_nano.schemas.world import load_world
    from scripts.grade_grasp import grade
    recording, = (out/'rollouts').glob('*/recording.json')
    physical = grade(recording.parent, registry.load('malo'), load_world(world_name),
                     json.loads((assets/'embodiments/malo/audit.json').read_text()), [1.7, -3.33], 'tissue_refill')
    physical.update(scope=__doc__, reset_seed=0)
    (out/'physical-grade.json').write_text(json.dumps(physical, indent=2))
    assert physical['physical_verdict'] == 'pass', physical['checks']
    assert physical['motion_verdict'] == 'unverified' and physical['deployment_gate'] == 'closed'


def test_rgbd_contact_grasp_through_the_cli(tmp_path, monkeypatch):
    _grasp(tmp_path, monkeypatch)


@pytest.mark.live_llm
def test_real_model_selects_contact_grasp(tmp_path, monkeypatch):
    _grasp(tmp_path, monkeypatch, live=True)
