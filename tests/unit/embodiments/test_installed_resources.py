"""The built wheel, installed away from the checkout, ships every robot/world resource."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_installed_robot_and_world_resources(tmp_path):
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
    files = subprocess.check_output(['git', 'ls-files', 'vector_os_nano/embodiments', 'vector_os_nano/worlds'],
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
assert Path(vector_os_nano.__file__).is_relative_to(sys.argv[1]), vector_os_nano.__file__
package = files('vector_os_nano')
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
