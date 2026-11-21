"""A stop is measured after navigation through the ordinary session builder."""
import json
import math
import os
from pathlib import Path

import pytest

from vector_os_nano.sim.session import SimSession, SimSpec

pytestmark = pytest.mark.sim


def test_navigation_stops_before_arrival(backend, monkeypatch):
    monkeypatch.setenv('VECTOR_SIM_LOCKSTEP', '1')
    s = SimSession.start(SimSpec('go2', gui=False, backend=backend))
    trace = []
    try:
        base = s.agent._base
        read = base.get_odometry
        def measured():
            o = read()
            trace.append(o.to_dict())
            return o
        monkeypatch.setattr(base, 'get_odometry', measured)
        result = base.navigate_to(17, 1.7, timeout=50)
        out = Path(os.environ['GATE_OUT'])/f'{backend}-braking.json'
        out.write_text(json.dumps(dict(result=result, trace=trace)))
        assert result, result
        odom = read()
        assert math.hypot(odom.vx, odom.vy) <= .03 and abs(odom.vyaw) <= .05
    finally:
        s.stop()
