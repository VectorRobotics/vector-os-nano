"""A sensor controller owns the clock and cannot resume after stop or report a timeout as success."""
import threading

import numpy as np
import pytest

from tests.conformance.test_legged import body

pytestmark = pytest.mark.sim


def test_feedback_motion_reaches_an_observed_goal(body):
    base = body.base
    assert base.set_posture('stand')
    start = base.get_position()[0]
    times = []
    def controller():
        times.append(base.get_sim_time())
        return (.4, 0., 0.) if base.get_position()[0]-start < .02 else None
    assert base.walk(duration=3., velocity_fn=controller)
    assert base.get_position()[0]-start >= .02
    assert len(times) > 2 and np.allclose(np.diff(times), body.rt.tick_dt, atol=1e-9, rtol=0)
    assert base._cmd == (0., 0., 0.) and body.upright()


@pytest.mark.parametrize('failure', ['timeout', 'stop', 'stop_at_goal', 'cancel', 'nonfinite', 'exception'])
def test_feedback_failure_stops_without_claiming_completion(body, failure):
    base = body.base
    token = threading.Event()
    base._cancel.is_set = token.is_set
    calls = 0
    def controller():
        nonlocal calls
        calls += 1
        if calls == 3:
            if failure.startswith('stop'):
                worker = threading.Thread(target=base.stop); worker.start(); worker.join()
                if failure == 'stop_at_goal': return None
            if failure == 'cancel': token.set(); return None
            if failure == 'nonfinite': return (float('nan'), 0., 0.)
            if failure == 'exception': raise ValueError('sensor unavailable')
        return (.1, 0., 0.)
    if failure in ('nonfinite', 'exception'):
        with pytest.raises(ValueError): base.walk(duration=.02, velocity_fn=controller)
    else:
        assert base.walk(duration=.02, velocity_fn=controller) is False
    assert base._cmd == (0., 0., 0.) and not base._owners
    before = base.get_sim_time()
    assert base.wait_sim(.01) and base.get_sim_time() > before


@pytest.mark.parametrize('duration', [0., float('inf'), float('nan')])
def test_invalid_feedback_deadline_never_invokes_the_controller(body, duration):
    def unexpected():
        pytest.fail('controller ran without a finite positive deadline')
    with pytest.raises(ValueError, match='deadline'):
        body.base.walk(duration=duration, velocity_fn=unexpected)
    assert body.base._cmd == (0., 0., 0.)
