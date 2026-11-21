# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Learned policy input, cadence, reset and finite-output boundary."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.robot import EmbodimentConfigError, parse_embodiment_config
from vector_os_nano.interfaces.policy import PolicyObs
from vector_os_nano.sim import policies


def raw():
    import yaml
    from vector_os_nano.schemas.robot import _EMBODIMENTS_DIR
    return yaml.safe_load((_EMBODIMENTS_DIR / "g1" / "robot.yaml").read_text())


@pytest.mark.parametrize("change", [
    lambda p: p.update(timestep=0),
    lambda p: p.update(tick_hz=float("nan")),
    lambda p: p.update(decimation=True),
    lambda p: p["obs"][-1].update(period=0),
    lambda p: p["obs"][-1].update(period=float("inf")),
    lambda p: p["obs"][0].update(scale=float("nan")),
    lambda p: p["action"].update(scale=float("inf")),
])
def test_invalid_policy_values_fail_at_manifest_load(change):
    r = raw()
    change(r["base"]["policy"])
    with pytest.raises(EmbodimentConfigError):
        registry.validate(parse_embodiment_config(r))


def test_observation_decimation_reset_and_torque_limits(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = registry.load("g1")
    layer = torch.nn.Linear(47, 12)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(1)
    torch.jit.trace(layer, torch.zeros(1, 47)).save(str(tmp_path / "motion.pt"))
    policy = policies.make("torchscript", cfg.base, cfg.base.policy.params, np.ones(12), tmp_path)
    stance = np.array([cfg.base.stance[j] for j in cfg.base.joints])
    obs = PolicyObs(0, np.zeros(3), np.array([1., 0, 0, 0]), np.zeros(3),
                    np.array([1., 2, 3]), stance, np.zeros(12), np.array([.2, .3, .4]))
    x = policy.observe(obs)
    assert x.dtype == np.float32 and x.shape == (47,)
    np.testing.assert_allclose(x[:9], [.25, .5, .75, 0, 0, -1, .4, .6, .1])
    np.testing.assert_array_equal(x[-2:], [0, 1])
    for _ in range(10):
        np.testing.assert_array_equal(policy.torques(obs), np.zeros(12))
    np.testing.assert_array_equal(policy.torques(obs), np.ones(12))
    np.testing.assert_array_equal(policy.observe(obs)[33:45], np.ones(12))
    policy.reset(obs)
    np.testing.assert_array_equal(policy.torques(obs), np.zeros(12))
    # Inference can become non-finite after a valid load; never pass it to physics.
    policy._net = lambda x: torch.full((1, 12), float("nan"))
    for _ in range(9):
        policy.torques(obs)
    with pytest.raises(ValueError, match="finite"):
        policy.torques(obs)
