# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The same command sequence gives bit-identical state on a second run, per backend."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conformance.rig import qexp

pytestmark = pytest.mark.sim


def _contract_run(world, sc) -> tuple[np.ndarray, np.ndarray]:
    q, qd = sc.model.qpos0.copy(), np.zeros(sc.model.nv)
    die = sc.body("die")
    q[sc.free_q(die)] = [0.5, -1.0, 0.4, *qexp(np.array([0.3, 0.5, 0.0]))]
    qd[sc.free_qd(die)] = [0.4, 0.1, 0.0, 3.0, -2.0, 1.0]  # it tumbles onto the floor
    world.reset(q, qd)
    for k in range(600):
        if k % 50 == 0:
            world.set_torque([sc.joint("spin"), sc.joint("rail")], [0.4 * math.sin(k), 3.0 * math.cos(k)])
            world.set_targets([sc.joint("servo"), sc.joint("weak")], [0.2 * math.sin(k / 7), 0.1])
        if k == 200:
            world.attach(sc.body("carrier"), sc.body("parcel"))
        if k == 400:
            world.detach(sc.body("parcel"))
        world.step(1)
    return world.state()


def _battery_run(world, sc) -> tuple[np.ndarray, np.ndarray]:
    arm = [sc.joint(f"joint{i}") for i in range(1, 8)]
    for k, pose in enumerate(([0.0, 1.87, -0.98, 0.0, 0.77, 0.0, 0.035], [0.1, 1.9, -1.0, 0.1, 0.8, 0.5, 0.0])):
        world.set_targets(arm, pose)
        world.step(300)
    return world.state()


@pytest.mark.parametrize("scene_name", ["contract_scene", "battery_scene"])
def test_same_commands_same_bits(backend, load, request, scene_name):
    sc = request.getfixturevalue(scene_name)
    run = _contract_run if scene_name == "contract_scene" else _battery_run
    q1, qd1 = run(load(backend, sc), sc)
    q2, qd2 = run(load(backend, sc), sc)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(qd1, qd2)
    assert np.abs(qd1).max() > 0  # something moved
