# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RobotWorld sim-oracle grounding on the real headless SO-101 session.

RobotWorld.build_verify_namespace binds the GT oracles of the live sim session
that owns the agent (``vcli/worlds/sim_oracles``); with no session (real hardware,
a bare agent) it binds nothing, so every verify there grades RAN. The engine merges
the world's namespace over its own bindings.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.harness.sim_session import started
from vector_os_nano.vcli.worlds import RobotWorld


@pytest.fixture()
def sim_agent():
    """The agent of a live headless so101 session (its GT oracles bound)."""
    with started("so101") as s:
        yield s.agent


@pytest.mark.sim
def test_namespace_is_the_tabletop_gt_oracle_set(sim_agent) -> None:
    ns = RobotWorld().build_verify_namespace(sim_agent)
    assert set(ns) == {
        "holding_object", "lifted", "kept_holding", "released", "object_at", "object_on",
        "localized", "arm_at_home", "in_view",
    }  # a fixed base: no base oracles; the manifest's overhead camera (640x480): in_view
    assert not ({"detect_objects", "describe_scene", "placed_count"} & set(ns))  # vacuous, gone
    assert all(callable(fn) for fn in ns.values())


@pytest.mark.sim
def test_oracles_read_the_tabletop(sim_agent) -> None:
    sim_agent._arm.home(duration=0.6)  # sim time passes: the objects settle onto the table
    ns = RobotWorld().build_verify_namespace(sim_agent)
    assert ns["holding_object"]() is False
    assert ns["object_on"]("banana", "table") is True
    assert ns["object_on"]("香蕉", "table") is True
    assert ns["released"]() is True
    assert isinstance(ns["arm_at_home"](), bool)


def test_namespace_empty_without_agent() -> None:
    assert RobotWorld().build_verify_namespace(None) == {}


def test_namespace_empty_without_a_live_sim_session() -> None:
    """Real hardware (or any agent no live sim session owns) contributes nothing."""
    agent = MagicMock()
    agent._base = None
    assert RobotWorld().build_verify_namespace(agent) == {}


@pytest.mark.sim
def test_engine_merges_robot_world_grounding(sim_agent) -> None:
    from vector_os_nano.vcli.engine import VectorEngine

    engine = VectorEngine(backend=MagicMock())
    engine._world = RobotWorld()
    engine._agent = sim_agent

    ns = engine._build_verifier_namespace(sim_agent)
    assert "holding_object" in ns and "object_on" in ns
    assert "detect_objects" not in ns
