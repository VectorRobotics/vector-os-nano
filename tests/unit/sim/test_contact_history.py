"""Adversarial force-history instrumentation; no physical-task acceptance inferred."""
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.interfaces.physics import ContactLoad, GraspSpec
from vector_os_nano.sim.grasp import ContactHistory


def _load(finger, normal=(1., 0., 0.), magnitude=1.):
    return ContactLoad(body_a=finger, body_b=4, position=(0, 0, 0),
                       force_on_b=tuple(magnitude*x for x in normal), distance=0,
                       normal_on_b=normal)


@pytest.fixture
def history():
    w = SimpleNamespace(t=0., loads=(_load(2), _load(3, (-1., 0., 0.))), welds={}, opening=.02)
    w.scene = SimpleNamespace(robot_root=1, free_bodies=lambda: {'object': 4},
                              subtree=lambda _: {1, 2, 3}, joint_q=lambda _: 0)
    w.time = lambda: w.t
    w.state = lambda: (np.array([w.opening]), np.zeros(1))
    w.contact_loads = lambda: w.loads
    w.attached = lambda: w.welds
    h = ContactHistory()
    h.register(GraspSpec(1, (2, 3), 0, 0., .002, assist=False), w)
    return h, w


def test_dwell_loss_reacquisition_and_reset_fail_closed(history):
    h, w = history
    h.observe(w)
    assert not h.held(1, 4)
    w.t = .21
    h.observe(w)
    assert h.held(1, 4, since=0)
    assert not h.held(1, 4, since=-1)  # no coverage before registration
    w.t, w.loads = .22, ()
    h.observe(w)
    assert not h.held(1, 4)
    w.t, w.loads = .23, (_load(2), _load(3, (-1., 0., 0.)))
    h.observe(w)
    w.t = .44
    h.observe(w)
    assert h.held(1, 4) and not h.held(1, 4, since=0)
    assert h.held(1, 4, since=.23)
    w.t = 0
    h.observe(w)
    assert not h.held(1, 4)


@pytest.mark.parametrize('case', ['one_finger', 'proximity', 'same_direction', 'tension', 'weld', 'closed'])
def test_non_grasps_never_accumulate_possession(history, case):
    h, w = history
    if case == 'one_finger':
        w.loads = (_load(2),)
    elif case == 'proximity':
        w.loads = (_load(2, magnitude=.001), _load(3, (-1., 0., 0.), .001))
    elif case == 'same_direction':
        w.loads = (_load(2), _load(3))
    elif case == 'tension':
        w.loads = (_load(2, magnitude=-1), _load(3, (-1., 0., 0.), -1))
    elif case == 'weld':
        w.welds = {4: 1}
    else:
        w.opening = 0
    h.observe(w)
    w.t = 1
    h.observe(w)
    assert not h.held(1, 4)
