"""Bare REPL backend/world switches exercise the same generic session builder."""
import re
import pytest
from tests.harness.pty_cli import run_repl_session
from tests.e2e_repl import READY


@pytest.mark.sim
@pytest.mark.cli_main
def test_body_backend_and_world_switches_in_one_conversation():
    choices = [('go2', 'mujoco', 'apartment'), ('piper', 'mujoco', 'workbench'),
               ('so101', 'mujoco', 'tabletop'), ('g1', 'pybullet', 'apartment'),
               ('go2_piper', 'pybullet', 'apartment'), ('piper', 'pybullet', 'workbench'),
               ('piper', 'pybullet', 'tabletop')]
    sentences, steps = [], []
    for i, (body, backend, world) in enumerate(choices):
        sentence = f'第 {i + 1} 次切换：用 {backend} 在 {world} 启动 {body}，无窗口'
        sentences.append(sentence)
        steps.extend([{'when': sentence, 'tool_calls': [{'name': 'start_simulation', 'input': {
            'embodiment': body, 'backend': backend, 'world': world, 'gui': False}}]},
            {'when': '', 'text': '启动检查已返回', 'tool_calls': []}])
    result = run_repl_session(sentences, tool_script={'steps': steps}, timeout_sec=270)
    text = ' '.join(result.text.split())
    assert not result.timed_out and result.exit_code == 0, text[-4000:]
    assert re.findall(READY, text) == [b for b, _, _ in choices], text[-4000:]
    for body, backend, world in choices:
        assert f'embodiment={body}, backend={backend}, world={world}' in text
    assert 'Failed to start' not in text and 'Cannot start' not in text
