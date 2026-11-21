"""The built wheel, installed away from the checkout, ships every robot/world resource."""
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import pytest


def install_wheel(tmp_path):
    root = Path(__file__).resolve().parents[3]
    uv = shutil.which('uv')
    assert uv, 'the offline wheel install check requires uv (the project environment manager)'
    wheels, installed = tmp_path / 'wheels', tmp_path / 'installed'
    subprocess.run([sys.executable, '-m', 'hatchling', 'build', '-t', 'wheel', '-d', str(wheels)],
                   cwd=root, check=True, capture_output=True, text=True, timeout=120)
    wheel, = wheels.glob('*.whl')
    subprocess.run([uv, 'pip', 'install', '--python', sys.executable, '--target', str(installed),
                    '--no-index', '--no-deps', str(wheel)], check=True, capture_output=True,
                   text=True, timeout=60, env={**os.environ, 'UV_CACHE_DIR': str(tmp_path / 'uv-cache')})
    return root, installed


def test_installed_robot_and_world_resources(tmp_path):
    root, installed = install_wheel(tmp_path)
    files = subprocess.check_output(['git', 'ls-files', 'vector_os_nano/embodiments', 'vector_os_nano/worlds',
                                    'vector_os_nano/core/default.yaml', 'vector_os_nano/sim/nav/nav.yaml',
                                    'vector_os_nano/sim/nav/ros2', 'vector_os_nano/hardware/so101/so101.urdf'],
                                    cwd=root, text=True).splitlines()
    expected = {str(Path(f).relative_to('vector_os_nano')): hashlib.sha256((root / f).read_bytes()).hexdigest()
                for f in files}
    manifest = tmp_path / 'resources.json'
    manifest.write_text(json.dumps(expected))
    code = '''
import hashlib, json, sys
from pathlib import Path
from importlib.resources import files
sys.path.insert(0, sys.argv[1])
import vector_os_nano
from vector_os_nano.schemas.robot import load_embodiment_config
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.nav.follow import arrival_radius
from vector_os_nano.core.config import load_config
assert Path(vector_os_nano.__file__).is_relative_to(sys.argv[1]), vector_os_nano.__file__
assert arrival_radius() == .5
assert load_config()['agent']['max_planning_retries'] == 3
package = files('vector_os_nano')
assert package.joinpath('hardware/so101/so101.urdf').is_file()
nav = package.joinpath('sim/nav/ros2')
for name in ('bridge.py', 'motion.py', 'terrain.py', 'viz.py', 'launch_nav_only.sh',
             'nav_env.sh', 'tare_go2_indoor.yaml', 'vnav.rviz', 'vector-os-dashboard.json'):
    assert nav.joinpath(name).is_file(), f'missing installed navigation resource: {name}'
for name, digest in json.loads(Path(sys.argv[2]).read_text()).items():
    resource = package.joinpath(name)
    assert resource.is_file(), name
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == digest, name
for folder in package.joinpath('embodiments').iterdir():
    if folder.joinpath('robot.yaml').is_file():
        cfg = load_embodiment_config(folder.name)
        assert cfg.path.is_relative_to(sys.argv[1])
        if not cfg.compose and not cfg.assets:
            assert folder.joinpath(cfg.model.file).is_file()
            assert folder.joinpath('LICENSE').is_file()
for folder in package.joinpath('worlds').iterdir():
    if folder.joinpath('world.yaml').is_file():
        assert load_world(folder.name).scene.is_relative_to(sys.argv[1])
print('installed resources verified')
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(installed), str(manifest)],
                            cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
    assert result.stdout.strip() == 'installed resources verified'


@pytest.mark.sim
@pytest.mark.live_llm
@pytest.mark.timeout(700)
@pytest.mark.parametrize('backend', ['mujoco', 'pybullet'])
def test_installed_console_script_with_real_model_outside_the_checkout(tmp_path, backend):
    from dotenv import dotenv_values
    from tests.harness import gt_audit
    from tests.harness.pty_cli import run_repl_session, strip_ansi
    root, installed = install_wheel(tmp_path)
    console = installed/'bin/vector-cli'
    assert console.is_file()
    out = Path(os.environ.get('GATE_OUT', tmp_path))/('installed-'+backend)
    out.mkdir(parents=True, exist_ok=True)
    credentials = {k: v for k, v in dotenv_values(root/'.env').items()
                   if v and k in {'DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'}}
    env = {**credentials, 'PYTHONPATH': str(installed), 'VECTOR_HOME': str(out/'state')}
    launch = f'启动 piper 的 {backend} 仿真，使用 workbench 世界，无窗口。'
    task = 'pick up the red box'
    result = run_repl_session([launch, (re.compile('Started piper simulation, ready'), task)], live=True, executable=str(console),
                              cwd=tmp_path, extra_env=env, answers=['y']*10,
                              timeout_sec=600, out_dir=out)
    transcript = strip_ansi(result.transcript)
    (out/'transcript.txt').write_text(transcript)
    grade, why = gt_audit.grade(Path(result.dump), {'held_lifted': 'red_box'}, task)
    turns = dict(gt_audit.split_turns(transcript, [launch, task]))
    verdict = gt_audit.parse_verdict(turns.get(task, ''))
    report = dict(backend=backend, executable=str(console), cwd=str(tmp_path), dump=result.dump,
                  wheel_sha256=hashlib.sha256(next((tmp_path/'wheels').glob('*.whl')).read_bytes()).hexdigest(),
                  timed_out=result.timed_out, task=task, grade=grade, reason=why, verdict=verdict,
                  scope='Installed wheel with existing runtime dependencies, real model and bare console script.')
    (out/'report.json').write_text(json.dumps(report, indent=2))
    assert not result.timed_out and 'Started piper simulation, ready' in transcript
    assert f'backend={backend}' in transcript
    assert grade is True, why
    assert verdict and verdict['verified'], turns[task]
