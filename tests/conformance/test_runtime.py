# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The runtime's sentences (charter 3.2) on every backend: ONE SimRuntime drives any
PhysicsWorld — its snapshot is ``world.state()``, ``submit`` runs at a step boundary,
and the world-side grasp decision logs at call time."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conformance.test_drivers import Body
from vector_os_nano.sim import grasp as g
from vector_os_nano.sim.runtime import SimRuntime

pytestmark = pytest.mark.sim


def test_the_snapshot_is_world_state(backend, load, contract_scene):
    rt = SimRuntime(load(backend, contract_scene), substeps=3, threaded=False, paced=False, name="conf")
    try:
        for _ in range(2):
            rt.step(4)
            s, (q, qd) = rt.snapshot(), rt.world.state()
            assert np.array_equal(s.q, q) and np.array_equal(s.qd, qd)
            assert s.sim_time == rt.world.time() and s.step % 12 == 0
        q = q.copy()
        q[contract_scene.joint_q(contract_scene.joint("rail"))] = 0.25
        rt.call(lambda w: w.reset(q))  # a mutation republishes, no tick needed
        assert np.array_equal(rt.snapshot().q, rt.world.state()[0])
        assert rt.snapshot().q[contract_scene.joint_q(contract_scene.joint("rail"))] == 0.25
    finally:
        rt.close()


def test_submit_runs_at_a_step_boundary(backend, load, contract_scene):
    """With the stepper thread running, a job sees every tick whole: its controller AND
    all of its substeps, never one without the other."""
    ticks = []
    rt = SimRuntime(load(backend, contract_scene), controller=lambda w: ticks.append(w.time()),
                    substeps=5, threaded=True, paced=False, name="conf")
    seen = []
    try:
        rt.start()
        for _ in range(40):
            seen.append(rt.call(lambda w: (len(ticks), w.time(), ticks[-1] if ticks else None)))
    finally:
        rt.close()
    dt = rt.tick_dt
    assert len({n for n, _t, _last in seen}) > 5, "the stepper never ran between the jobs"
    for n, t, last in seen:
        assert t == pytest.approx(n * dt, abs=1e-9), (n, t)  # n controllers <=> n whole ticks
        assert last is None or last == pytest.approx(t - dt, abs=1e-9)


def test_grasp_and_release_log_at_call_time(backend, load, battery_scene):
    b = Body(load(backend, battery_scene))
    rt, box, spec = b.rt, b.around("box"), b.gripper._grasp
    try:
        b.servo_jaw(0.0)
        t = rt.sim_time
        assert rt.grasp(spec) == box and rt.attachments() == {box: b.hand}
        assert rt.release(b.hand) == [] and rt.attachments() == {}
        assert rt.grasp(spec) == box  # re-grasped at the same instant: both changes logged
        assert list(rt.attach_log) == [(t, box, b.hand), (t, box, None), (t, box, b.hand)]
    finally:
        rt.close()


def test_every_attach_and_detach_inside_one_job_is_logged(backend, load, battery_scene):
    """Adversarial: a detach and a re-attach inside ONE job leave the attachments as they
    were, yet both transitions are logged; so is a release + attach through sim/grasp,
    and a re-attach to the same parent changes nothing and logs nothing."""
    b = Body(load(backend, battery_scene))
    rt, box, hand = b.rt, b.around("box"), b.hand
    try:
        b.servo_jaw(0.0)
        t = rt.sim_time
        rt.call(lambda w: w.attach(hand, box))
        rt.call(lambda w: (w.detach(box), w.attach(hand, box)))
        rt.call(lambda w: (g.release(w, hand), w.attach(hand, box)))
        rt.call(lambda w: w.attach(hand, box))  # already held by that parent: no event
        held, free = (t, box, hand), (t, box, None)
        assert list(rt.attach_log) == [held, free, held, free, held]
        assert rt.attachments() == {box: hand}
    finally:
        rt.close()
