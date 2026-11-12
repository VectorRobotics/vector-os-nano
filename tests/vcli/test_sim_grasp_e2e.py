# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Headless sim grasp regression — Step 4 FIX 1 validation.

Asserts that SIM_PICK_CONFIG (z_offset=0.0) produces a REAL grasp in the
SO-101 sim: holding_object() transitions from False to True (finger contact,
the grasp attachment, the lift), and the target object is physically lifted
(z > initial_z + 0.05).
"""
from __future__ import annotations

import pytest

from tests.harness.sim_session import started
from vector_os_nano.skills.pick import SIM_PICK_CONFIG


class TestSimPickConfig:
    """Static contract for SIM_PICK_CONFIG — the single-source constant."""

    def test_z_offset_is_zero(self):
        assert SIM_PICK_CONFIG["z_offset"] == 0.0

    def test_hardware_offsets_disabled(self):
        assert SIM_PICK_CONFIG["hardware_offsets"] is False


@pytest.mark.sim
@pytest.mark.integration
def test_banana_grasp_and_lift():
    """After pick(banana, hold), the banana is held by contact AND lifted > 5cm."""
    with started("so101") as s:
        holding = s.oracles()["holding_object"]
        assert holding() is False, "Expected gripper empty before pick"
        initial_z = s.gt.object_positions()["banana"][2]

        result = s.agent.execute_skill("pick", {"object_label": "banana", "mode": "hold"})
        assert result.success, f"pick skill failed: {result.error_message}"
        assert holding("banana") is True, "holding_object() must be True after successful pick(hold)"
        lift = s.gt.object_positions()["banana"][2] - initial_z
        assert lift > 0.05, f"banana should be lifted > 5cm, lifted {lift * 100:.1f}cm"
