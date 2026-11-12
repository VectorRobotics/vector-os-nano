# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pick binds its target from the call, never from the simulator's oracle.

Two scenarios:
  (a) execute_skill("pick", {}) — no object_label → the skill refuses (no_target)
      and nothing moves; it used to grasp the nearest scene object (banana) read
      from get_object_positions.
  (b) execute_skill("pick", {"object_label": "mug"}) — bound target → the mug
      is gripped (NOT the nearest banana).

The test is headless and deterministic — no LLM, no network.
"""
from __future__ import annotations

import pytest

from tests.harness.sim_session import started

pytestmark = pytest.mark.sim

# The tabletop's nearest object to the arm base (banana at 0.17 m; the mug is 0.23 m).
_NEAREST_OBJECT = "banana"
_BOUND_TARGET = "mug"


@pytest.fixture(scope="module")
def unbound():
    """A fresh so101 session after an unbound pick."""
    with started("so101") as s:
        yield s, s.agent.execute_skill("pick", {"mode": "hold"})


@pytest.fixture(scope="module")
def bound():
    """A fresh so101 session: (session, empty before?, the bound pick's result)."""
    with started("so101") as s:
        empty = s.oracles()["holding_object"]() is False
        yield s, empty, s.agent.execute_skill("pick", {"object_label": _BOUND_TARGET, "mode": "hold"})


@pytest.mark.integration
class TestPickUnboundRefused:
    """R2: pick with no target asks for one; it never reads the nearest object from
    the simulator's oracle (skills-manip:pick-sim-oracle-nearest-object)."""

    def test_unbound_pick_fails_and_names_why(self, unbound) -> None:
        _s, result = unbound
        assert not result.success and result.result_data["diagnosis"] == "no_target"

    def test_nothing_is_held_or_lifted(self, unbound) -> None:
        s, _result = unbound
        assert s.oracles()["holding_object"]() is False
        z = s.gt.object_positions()[_NEAREST_OBJECT][2]
        assert z < 0.15, f"{_NEAREST_OBJECT!r} must stay on the table, got z={z:.3f}m"


@pytest.mark.integration
class TestPickBoundTargetUnchanged:
    """Bound target must NOT be overridden by the nearest-object logic."""

    def test_gripper_empty_before_bound_pick(self, bound) -> None:
        assert bound[1], "Expected empty gripper before bound pick"

    def test_bound_pick_succeeds(self, bound) -> None:
        result = bound[2]
        assert result.success, f"Bound pick({_BOUND_TARGET!r}) failed: {result.error_message}"

    def test_holding_after_bound_pick(self, bound) -> None:
        assert bound[0].oracles()["holding_object"](_BOUND_TARGET) is True

    def test_mug_is_lifted_not_banana(self, bound) -> None:
        """The bound object (mug) is lifted; banana stays on the table."""
        positions = bound[0].gt.object_positions()
        mug_z = positions[_BOUND_TARGET][2]
        assert mug_z > 0.05, f"{_BOUND_TARGET!r} should be lifted > 5cm, got z={mug_z:.3f}m"
        banana_z = positions[_NEAREST_OBJECT][2]
        assert banana_z < 0.15, f"{_NEAREST_OBJECT!r} should stay on table, got z={banana_z:.3f}m"
