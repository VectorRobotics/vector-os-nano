# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Grasp hands the held object to named place, independently verified on the tray."""
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
        assert holding() is True, "grasp must leave the object held"
        assert released() is False and released("banana") is False

        place = agent.execute_skill("place", {"surface": "tray"})
        assert place.success
        assert holding() is False, "place must release the held object"
        assert released() is True and released("banana") is True
        assert ns['object_on']('banana', 'tray')

        x1, y1, _ = s.gt.object_positions()["banana"]
        moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        assert moved > 0.05, f"placed object should have moved (>5cm); moved {moved:.3f}m"
