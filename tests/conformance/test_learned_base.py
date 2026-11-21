# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""G1's learned balance policy: the existing locomotion bounds on both engines.

Both engines run the pinned policy through the same driver and observations.
"""
from __future__ import annotations

import threading

import numpy as np

import pytest

from tests.conformance import test_legged as shared
from vector_os_nano.embodiments import registry

pytestmark = pytest.mark.sim


@pytest.fixture(params=["mujoco", "pybullet"])
def body(request, g1_assets, monkeypatch):
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    from vector_os_nano.sim import backends
    assert backends.get(request.param).supports(registry.load("g1"), None) is None
    robot = shared.Body(request.param, registry.load("g1"))
    try:
        yield robot
    finally:
        robot.close()


@pytest.mark.parametrize("check", [
    shared.test_the_base_is_the_protocol_member_by_member,
    shared.test_it_walks_forward,
    shared.test_it_turns,
    shared.test_an_unknown_posture_names_the_valid_ones,
    shared.test_the_lidar_sees_from_its_mount,
])
def test_shared_locomotion_criteria(body, check):
    check(body)


def test_stop_returns_to_stationary_balance(body):
    # A balancing gait sways at zero command; measure translation over whole cycles,
    # and require the stopped peak speed to return to the independently sampled idle envelope.
    period = body.cfg.base.policy.params["obs"][-1]["period"]
    sc = body.rt.world.scene
    assert body.base.set_posture("stand")
    idle = []
    for _ in body.run(2 * period):
        idle.append(np.linalg.norm(body.rt.snapshot().qd[sc.free_qd(sc.robot_root)][:2]))
    body.base.set_velocity(body.cfg.base.limits[0], 0, 0)
    assert all(body.upright(p) for p in body.run(2.0))
    body.base.stop()
    assert all(body.upright(p) for p in body.run(2.0))
    start, speed = body.pose()[:2].copy(), []
    for p in body.run(2 * period):
        assert body.upright(p)
        speed.append(np.linalg.norm(body.rt.snapshot().qd[sc.free_qd(sc.robot_root)][:2]))
    assert np.linalg.norm(body.pose()[:2] - start) / (2 * period) < .05
    assert max(speed) <= max(idle) + .05, (max(speed), max(idle))


def test_tilt_alone_ends_motion_even_above_the_height_threshold(body):
    q = body.rt.snapshot().q.copy()
    root = body.rt.world.scene.free_q(body.rt.world.scene.robot_root)
    q[root.start + 2] = 1.5  # an inverted humanoid clear of the floor, no penetrating reset
    q[root.start + 3:root.stop] = [0., 1., 0., 0.]
    body.rt.call(lambda w: w.reset(q))
    x, y, z = body.base.get_position()
    assert z > body.cfg.base.fall_z
    assert body.base.navigate_to(x + 1, y, timeout=5).reason == "fell"
    assert body.base.walk(.3, 0, 0, .1) is False


def test_standing_oracle_checks_height_and_tilt(body):
    from vector_os_nano.sim.ground_truth import GroundTruth
    from vector_os_nano.vcli.worlds.sim_oracles import build_namespace

    standing = build_namespace(GroundTruth(body.rt), None, body.cfg)["base_standing"]
    assert body.base.set_posture("stand") and standing()
    q = body.rt.snapshot().q.copy()
    root = body.rt.world.scene.free_q(body.rt.world.scene.robot_root)
    q[root.start + 2] = body.cfg.base.height * 1.2
    body.rt.call(lambda w: w.reset(q))
    assert not standing()
    q[root.start + 2] = body.cfg.base.height
    q[root.start + 3:root.stop] = [0., 1., 0., 0.]
    body.rt.call(lambda w: w.reset(q))
    assert not standing()


def test_navigation_and_cancellation(body):
    x, y, _ = body.base.get_position()
    result = body.base.navigate_to(x + 1.0, y, timeout=15)
    assert result and result["dist"] <= .3 and body.upright(), dict(result)
    stop = threading.Event()
    stop.set()
    result = body.base.navigate_to(x + 5.0, y, cancel=stop, timeout=15)
    assert result.reason == "aborted" and body.upright()
