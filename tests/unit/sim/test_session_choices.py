"""Selected sessions actually tick on both engines; user folders need no Python."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import threading
import time

import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.schemas import paths
from vector_os_nano.schemas.world import WORLDS_DIR
from vector_os_nano.sim import session as mod
from vector_os_nano.sim.session import SimSession, SimSpec, SimStartError
from vector_os_nano.vcli.tools.base import ToolContext
from vector_os_nano.vcli.tools.sim_tool import SimStartTool

pytestmark = pytest.mark.sim


def context(app):
    return ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                       abort=threading.Event(), app_state=app)


@pytest.mark.parametrize('backend', ['mujoco', 'pybullet'])
@pytest.mark.parametrize('body', registry.embodiment_ids())
def test_selected_engine_world_and_declared_sensors_are_live(backend, body, monkeypatch):
    monkeypatch.setenv('VECTOR_NAV_STACK', '')
    session = SimSession.start(SimSpec(body, False, backend))
    try:
        assert session.runtime.world.name == backend
        assert session.world.id == session.spec.world == registry.load(body).world
        assert session.runtime.sim_time >= .5
        assert {'camera', 'depth'} <= session.ready.keys()
        if session.config.lidar:
            assert 'lidar' in session.ready
        for field in (f'embodiment={body}', f'backend={backend}', f'world={session.world.id}', 'policy=', 'capabilities='):
            assert field in session.summary()
        assert session.agent._config['world'] == session.world.id
        assert session.agent._services['layout']['rooms'] == session.world.rooms
    finally:
        session.stop()


def test_defaults_user_folders_and_world_switch(tmp_path, monkeypatch):
    monkeypatch.setenv('VECTOR_HOME', str(tmp_path))
    shutil.copytree(registry.load('piper').path.parent, tmp_path / 'embodiments' / 'my_arm')
    shutil.copytree(WORLDS_DIR / 'workbench', tmp_path / 'worlds' / 'my_room')
    paths.config_file().write_text('schema: 1\ndefaults: {embodiment: my_arm, backend: pybullet, world: my_room, gui: false}\n')
    spec = SimSpec().resolved()
    assert spec == SimSpec('my_arm', False, 'pybullet', 'my_room')
    from vector_os_nano.vcli.cli import parse_args, _requested_embodiment, _wants_window
    args = parse_args(['--sim'])
    assert _requested_embodiment(args) == 'my_arm' and not _wants_window(args)
    tool, app = SimStartTool(), {}
    assert 'my_arm' in tool.input_schema['properties']['embodiment']['enum']
    assert 'my_room' in tool.input_schema['properties']['world']['enum']
    try:
        result = tool.execute({'embodiment': 'my_arm'}, context(app))
        assert not result.is_error, result.content
        assert app['sim_session'].spec == spec
        old = app['sim_session']
        result = tool.execute({'embodiment': 'my_arm', 'backend': 'mujoco', 'world': 'tabletop'}, context(app))
        assert not result.is_error, result.content
        assert old.runtime.closed
        assert app['sim_session'].world.id == 'tabletop'
        assert app['agent']._config['world'] == 'tabletop'
    finally:
        if app.get('sim_session'):
            app['sim_session'].stop()


@pytest.mark.parametrize('choice', [{'embodiment': 'missing'}, {'embodiment': 'piper', 'backend': 'missing'},
                                  {'embodiment': 'piper', 'world': 'missing'}])
def test_bad_selection_keeps_existing_session(choice):
    stopped = []
    old = SimpleNamespace(stop=lambda: stopped.append(True))
    app = {'sim_session': old}
    result = SimStartTool().execute(choice, context(app))
    assert result.is_error and 'missing' in result.content
    assert not stopped and app['sim_session'] is old


def test_backend_refusal_keeps_existing_session(monkeypatch):
    from vector_os_nano.sim import backends
    monkeypatch.setattr(backends.get('pybullet'), 'supports', lambda *_: 'measured unsupported policy')
    stopped = []
    old = SimpleNamespace(stop=lambda: stopped.append(True))
    app = {'sim_session': old}
    result = SimStartTool().execute({'embodiment': 'piper', 'backend': 'pybullet'}, context(app))
    assert result.is_error and 'measured unsupported policy' in result.content
    assert not stopped and app['sim_session'] is old


def test_cancel_and_expired_start_do_not_build(monkeypatch):
    monkeypatch.setattr(mod, '_build', lambda *_: pytest.fail('cancelled start built a body'))
    with pytest.raises(SimStartError, match='cancelled'):
        SimSession.start(SimSpec('piper', False), cancel=lambda: True)
    with pytest.raises(SimStartError, match='deadline'):
        SimSession.start(SimSpec('piper', False), deadline_s=0)


def test_deadline_or_stop_during_readiness_releases_resources(monkeypatch):
    closed, cancelled = [], threading.Event()
    def build(session, _key):
        session.own('fake', lambda: closed.append(True))
        session.runtime = SimpleNamespace(threaded=True, sim_time=0)
        return SimpleNamespace(_base=None, _arm=None)
    monkeypatch.setattr(mod, '_build', build)
    with pytest.raises(SimStartError, match='physics is not stepping'):
        SimSession.start(SimSpec('piper', False), deadline_s=.1)
    timer = threading.Timer(.05, cancelled.set)
    timer.start()
    try:
        with pytest.raises(SimStartError, match='cancelled'):
            SimSession.start(SimSpec('piper', False), cancel=cancelled.is_set)
    finally:
        timer.join()
    assert closed == [True, True]


def test_headless_start_selects_gl_without_user_environment(tmp_path):
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    env.pop('MUJOCO_GL', None)
    code = """
from vector_os_nano.sim.session import SimSession, SimSpec
s = SimSession.start(SimSpec('piper', False, 'pybullet'))
try:
    assert s.ready['camera'] and s.ready['depth']
finally:
    s.stop()
"""
    run = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-3000:]
