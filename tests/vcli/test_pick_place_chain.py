# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pick -> place handoff regression (headless sim).

Exercises the manipulation handoff a "grab X then put it down" plan relies on:
pick(mode='hold') leaves the object in the gripper (holding_object() True), then
place releases it (holding_object() False) and the object ends up moved + resting.
This guards both the pick->place capability and place's grounded verify_hint
(`released()`, WP07), which replaced the trivially-true `placed_count() >= 1` and the
never-grounding negation `not holding_object()`.
"""
from __future__ import annotations
from vector_os_nano.vcli.worlds.sim_oracles import session_namespace

import pytest

from tests.harness.sim_session import started

pytestmark = pytest.mark.sim


def test_pick_hold_then_place_releases_and_moves_object():
    with started("so101") as s:
        agent, ns = s.agent, session_namespace(s)
        holding, released = ns["holding_object"], ns["released"]  # one GT namespace (WP10)
        x0, y0, _ = s.gt.object_positions()["banana"]

        assert holding() is False  # gripper starts empty

        pick = agent.execute_skill("grasp", {"query": "banana"})
        assert pick.success
        assert holding() is True, "pick(mode='hold') must leave the object held"
        assert released() is False and released("banana") is False

        place = agent.execute_skill("place", {"location": "left"})
        assert place.success
        # place's grounded verify: the gripper released the object.
        assert holding() is False, "place must release the held object"
        assert released() is True and released("banana") is True  # place's verify_hint

        x1, y1, _ = s.gt.object_positions()["banana"]
        moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        assert moved > 0.05, f"placed object should have moved (>5cm); moved {moved:.3f}m"
