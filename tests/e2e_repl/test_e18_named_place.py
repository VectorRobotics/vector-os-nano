"""Fixed and mobile arms use the same named place skill through the bare CLI on both engines."""
import os
from pathlib import Path
import pytest
from tests.harness.pty_cli import run_repl_session


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.timeout(650)
@pytest.mark.parametrize('backend', ['mujoco', 'pybullet'])
@pytest.mark.parametrize('body,query,obj,surface', [('piper', 'red box', 'red_box', 'bench'),
    ('so101', 'banana', 'banana', 'tray'), ('go2_piper', 'green bottle', 'pickable_bottle_green', 'pick_table')])
def test_named_pick_and_place(body, query, obj, surface, backend, tmp_path):
    launch = f'start the {body} simulation'
    pick = f'pick up the {query}'
    put = f'put the {query} on the {surface}'
    script = {'steps': [
        {'when': launch, 'tool_calls': [{'name': 'start_simulation', 'input': {'embodiment': body, 'backend': backend, 'gui': False}}]},
        {'when': '', 'text': 'simulation ready', 'tool_calls': []},
        {'when': pick, 'tool_calls': [{'name': 'grasp', 'input': {'query': query}}]},
        # One model-side recovery decision; the second grasp observes again itself.
        {'when': 'Gripper closed without a hold', 'tool_calls': [{'name': 'grasp', 'input': {'query': query}}]},
        {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': f"holding_object('{obj}')"}}]},
        {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
        {'when': put, 'tool_calls': [{'name': 'place', 'input': {'surface': surface}}]},
        {'when': '', 'tool_calls': [{'name': 'verify', 'input': {'expr': f"object_on('{obj}', '{surface}')"}}]},
        {'when': 'PASS', 'tool_calls': [{'name': 'finish', 'input': {}}], 'stop_reason': 'end_turn'},
    ]}
    out = Path(os.environ.get('GATE_OUT', tmp_path))/f'{body}-{backend}'
    result = run_repl_session([launch, pick, put], tool_script=script, answers=['a']*10, timeout_sec=600,
                              out_dir=out, extra_env={'VECTOR_HOME': str(out/'home')})
    out.mkdir(parents=True, exist_ok=True)
    (out/'transcript.txt').write_text(result.text)
    assert not result.timed_out, result.text[-5000:]
    assert f"goal proven: holding_object('{obj}')" in result.text, result.text[-5000:]
    assert f"place → verify object_on('{obj}', '{surface}') ✓ (actor=CAUSED)" in result.text, result.text[-5000:]
    assert f"goal proven: object_on('{obj}', '{surface}')" in result.text, result.text[-5000:]
