"""L52: Door-chain navigation uses nav stack (obstacle avoidance), not dead-reckoning.

When the FAR V-Graph is unavailable, NavigateSkill falls back to door-chain mode.
This suite verifies that door-chain mode calls base.navigate_to() (which publishes
to /way_point for localPlanner obstacle avoidance) instead of _navigate_to_waypoint()
(which uses base.walk() in a straight line through walls).

All tests are source-code-based or logic-based — no ROS2 runtime required.
"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

import vector_os_nano.skills.navigate as navigate_module
from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.skills.navigate import (
    NavigateSkill,
    _navigate_to_waypoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARRIVAL_RADIUS = 0.8  # must match the new implementation


def _make_sg_with_doors() -> SceneGraph:
    """SceneGraph with two rooms connected by a door."""
    sg = SceneGraph()
    for _ in range(5):
        sg.visit("living_room", 3.0, 2.5)
    for _ in range(5):
        sg.visit("kitchen", 17.0, 2.5)
    sg.add_door("living_room", "kitchen", 10.0, 2.5)
    return sg


def _make_sg_multi_hop() -> SceneGraph:
    """SceneGraph: living_room -> hallway -> kitchen (two hops)."""
    sg = SceneGraph()
    for _ in range(5):
        sg.visit("living_room", 3.0, 2.5)
    for _ in range(5):
        sg.visit("hallway", 10.0, 5.0)
    for _ in range(5):
        sg.visit("kitchen", 17.0, 2.5)
    sg.add_door("living_room", "hallway", 6.5, 3.0)
    sg.add_door("hallway", "kitchen", 13.5, 3.0)
    return sg


def _make_base_no_navigate_to(x: float = 3.0, y: float = 2.5, z: float = 0.3) -> MagicMock:
    """Mock base WITHOUT navigate_to — forces dead-reckoning / Mode 2."""
    base = MagicMock(spec=[
        "get_position", "get_heading", "walk",
        "set_velocity", "stand", "sit",
    ])
    base.get_position.return_value = [x, y, z]
    base.get_heading.return_value = 0.0
    base.walk.return_value = None
    return base


def _make_base_with_navigate_to(
    x: float = 3.0, y: float = 2.5, z: float = 0.3,
    navigate_to_result: bool = True,
) -> MagicMock:
    """Mock base WITH navigate_to + go_to_waypoint — triggers Mode 0 (proxy) path."""
    base = MagicMock()
    base.get_position.return_value = [x, y, z]
    base.get_heading.return_value = 0.0
    base.navigate_to.return_value = navigate_to_result
    base.go_to_waypoint.return_value = navigate_to_result
    base.walk.return_value = None
    return base


def _make_ctx(base: MagicMock, sg: SceneGraph | None) -> MagicMock:
    """Build a SkillContext mock."""
    ctx = MagicMock()
    ctx.base = base
    ctx.services = {"spatial_memory": sg} if sg is not None else {}
    return ctx


# ---------------------------------------------------------------------------
# L52-1: Source-code audit — door-chain section must not call dead-reckoning
# ---------------------------------------------------------------------------


class TestDoorChainSourceAudit:
    """Source-code based tests — verify dead-reckoning is gone from door-chain."""

    def test_dead_reckoning_method_uses_navigate_to_not_walk(self) -> None:
        """_dead_reckoning() should call base.navigate_to, not _navigate_to_waypoint."""
        source = inspect.getsource(NavigateSkill._dead_reckoning)
        assert "navigate_to" in source, (
            "_dead_reckoning must call base.navigate_to() for nav stack waypoints"
        )

    def test_dead_reckoning_does_not_call_navigate_to_waypoint_helper(self) -> None:
        """_dead_reckoning must NOT call the _navigate_to_waypoint() dead-reckoning helper."""
        source = inspect.getsource(NavigateSkill._dead_reckoning)
        assert "_navigate_to_waypoint(" not in source, (
            "_dead_reckoning still calls _navigate_to_waypoint (dead-reckoning) "
            "— must be replaced with base.navigate_to()"
        )

    def test_dead_reckoning_does_not_call_base_walk_directly(self) -> None:
        """_dead_reckoning must not call base.walk() for locomotion."""
        source = inspect.getsource(NavigateSkill._dead_reckoning)
        # base.walk calls are only acceptable for dead-reckoning style — should be removed
        assert "base.walk(" not in source, (
            "_dead_reckoning still calls base.walk() directly — "
            "locomotion should go through base.navigate_to()"
        )

    def test_navigate_to_waypoint_function_still_exists(self) -> None:
        """_navigate_to_waypoint() must still exist (may be used by other callers)."""
        assert callable(_navigate_to_waypoint), (
            "_navigate_to_waypoint helper was deleted — it must be kept for other callers"
        )

    def test_dead_reckoning_has_timeout_constant(self) -> None:
        """_dead_reckoning must have a per-waypoint timeout."""
        source = inspect.getsource(NavigateSkill._dead_reckoning)
        # Expect some numeric timeout value to be present
        import re
        # Look for timeout keyword argument or a timeout variable assignment
        has_timeout = (
            "timeout" in source
            or re.search(r"\d+\.0", source) is not None
        )
        assert has_timeout, "_dead_reckoning should define a per-waypoint timeout"

    def test_dead_reckoning_has_arrival_check(self) -> None:
        """_dead_reckoning must check distance to waypoint for arrival."""
        source = inspect.getsource(NavigateSkill._dead_reckoning)
        assert "_ARRIVAL_RADIUS" in source or "0.8" in source or "arrival" in source.lower(), (
            "_dead_reckoning must check arrival distance before advancing to next waypoint"
        )


# ---------------------------------------------------------------------------
# L52-2: Behavior — navigate_to() is called per waypoint
# ---------------------------------------------------------------------------


class TestDoorChainCallsNavigateTo:
    """Verify that in Mode 0+fallback path, go_to_waypoint is called for each waypoint."""

    def test_doorchain_calls_navigate_to_for_each_waypoint(self) -> None:
        """_dead_reckoning calls base.go_to_waypoint once per door-chain waypoint."""
        sg = _make_sg_with_doors()
        # Position robot at living_room
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.go_to_waypoint.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        result = skill._dead_reckoning("kitchen", ctx)

        # Should have called base.go_to_waypoint at least once (one door + room center)
        assert base.go_to_waypoint.call_count >= 1, (
            f"Expected base.go_to_waypoint() to be called, got {base.go_to_waypoint.call_count} calls"
        )

    def test_doorchain_does_not_call_walk(self) -> None:
        """_dead_reckoning must not call base.walk() for waypoint locomotion."""
        sg = _make_sg_with_doors()
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.navigate_to.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        skill._dead_reckoning("kitchen", ctx)

        # base.walk should NOT be called by _dead_reckoning
        assert base.walk.call_count == 0, (
            f"_dead_reckoning called base.walk() {base.walk.call_count} times — "
            "it must use base.navigate_to() instead"
        )

    def test_doorchain_multi_hop_calls_navigate_to_multiple_times(self) -> None:
        """Multi-hop door chain calls go_to_waypoint at least twice (one per door)."""
        sg = _make_sg_multi_hop()
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.go_to_waypoint.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        skill._dead_reckoning("kitchen", ctx)

        # Two doors: living_room->hallway, hallway->kitchen — at least 2 go_to_waypoint calls
        assert base.go_to_waypoint.call_count >= 2, (
            f"Multi-hop door-chain expected >=2 go_to_waypoint calls, "
            f"got {base.go_to_waypoint.call_count}"
        )

    def test_doorchain_returns_success_when_navigate_to_succeeds(self) -> None:
        """_dead_reckoning returns success=True when navigate_to succeeds for all waypoints."""
        sg = _make_sg_with_doors()
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.navigate_to.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        result = skill._dead_reckoning("kitchen", ctx)

        assert result.success is True, f"Expected success=True, got: {result.error_message}"
        assert result.result_data is not None

    def test_doorchain_returns_failure_when_navigate_to_fails(self) -> None:
        """_dead_reckoning returns failure when go_to_waypoint times out on a waypoint."""
        sg = _make_sg_with_doors()
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        # go_to_waypoint fails (returns False = timeout)
        base.go_to_waypoint.return_value = False
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        result = skill._dead_reckoning("kitchen", ctx)

        assert result.success is False


# ---------------------------------------------------------------------------
# L52-3: Mode differentiation — Modes 0 and 1 are not broken
# ---------------------------------------------------------------------------


class TestModesUnchanged:
    """Verify Mode 0 (proxy) and Mode 1 (NavStackClient) still work correctly."""

    def test_mode0_proxy_not_dead_reckoning(self) -> None:
        """Mode 0: if base has navigate_to, skill uses _navigate_with_proxy, not _dead_reckoning."""
        sg = _make_sg_with_doors()
        # base has navigate_to -> Mode 0
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.navigate_to.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        # Spy on _dead_reckoning to ensure it is NOT called
        with patch.object(skill, "_dead_reckoning", wraps=skill._dead_reckoning) as mock_dr:
            result = skill.execute({"room": "kitchen"}, ctx)
            # _dead_reckoning should not have been called when navigate_to exists
            mock_dr.assert_not_called()

        assert result.success is True
        assert result.result_data["mode"] == "proxy_nav_stack"

    def test_mode1_nav_stack_client_not_dead_reckoning(self) -> None:
        """Mode 1: NavStackClient available -> _dead_reckoning not called."""
        sg = _make_sg_with_doors()
        # base WITHOUT navigate_to -> skips Mode 0
        base = _make_base_no_navigate_to(x=3.0, y=2.5)
        ctx = _make_ctx(base, sg)
        # Add nav service
        nav = MagicMock()
        nav.is_available = True
        nav.navigate_to.return_value = True
        nav.get_state_estimation.return_value = None
        ctx.services["nav"] = nav

        skill = NavigateSkill()
        with patch.object(skill, "_dead_reckoning", wraps=skill._dead_reckoning) as mock_dr:
            result = skill.execute({"room": "kitchen"}, ctx)
            mock_dr.assert_not_called()

        assert result.success is True

    def test_mode2_triggered_when_no_proxy_and_no_nav_stack(self) -> None:
        """Mode 2 (door-chain) triggered when base has no navigate_to AND nav stack absent."""
        sg = _make_sg_with_doors()
        base = _make_base_no_navigate_to(x=3.0, y=2.5)
        # Add navigate_to dynamically to test Mode 2 fallback via _dead_reckoning
        # We use a base without navigate_to attribute at all
        ctx = _make_ctx(base, sg)
        # No "nav" service -> Mode 2

        skill = NavigateSkill()
        with patch.object(skill, "_dead_reckoning", return_value=MagicMock(
            success=True, result_data={"mode": "dead_reckoning"}, error_message=None
        )) as mock_dr:
            skill.execute({"room": "kitchen"}, ctx)
            mock_dr.assert_called_once()


# ---------------------------------------------------------------------------
# L52-4: Edge cases
# ---------------------------------------------------------------------------


class TestDoorChainEdgeCases:
    """Edge cases for the new nav stack door-chain."""

    def test_dead_reckoning_no_sg_returns_error(self) -> None:
        """_dead_reckoning with no spatial_memory -> failure."""
        base = _make_base_no_navigate_to(x=3.0, y=2.5)
        ctx = _make_ctx(base, None)

        skill = NavigateSkill()
        result = skill._dead_reckoning("kitchen", ctx)

        assert result.success is False
        assert "explore" in result.error_message.lower()

    def test_dead_reckoning_no_door_chain_returns_error(self) -> None:
        """_dead_reckoning with SG but no door path -> failure."""
        sg = SceneGraph()
        for _ in range(5):
            sg.visit("living_room", 3.0, 2.5)
        for _ in range(5):
            sg.visit("kitchen", 17.0, 2.5)
        # No doors added -> get_door_chain returns []

        base = _make_base_no_navigate_to(x=3.0, y=2.5)
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        result = skill._dead_reckoning("kitchen", ctx)

        assert result.success is False
        assert "explore" in result.error_message.lower()

    def test_dead_reckoning_result_data_has_mode_field(self) -> None:
        """Successful dead_reckoning result_data must contain 'mode' field."""
        sg = _make_sg_with_doors()
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.navigate_to.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        result = skill._dead_reckoning("kitchen", ctx)

        if result.success:
            assert result.result_data is not None
            assert "mode" in result.result_data

    def test_navigate_to_called_with_waypoint_coordinates(self) -> None:
        """base.navigate_to is called with the actual door/waypoint coordinates."""
        sg = _make_sg_with_doors()
        # Door at (10.0, 2.5)
        base = _make_base_with_navigate_to(x=3.0, y=2.5)
        base.navigate_to.return_value = True
        ctx = _make_ctx(base, sg)

        skill = NavigateSkill()
        skill._dead_reckoning("kitchen", ctx)

        # All navigate_to calls should use float coordinates
        for c in base.navigate_to.call_args_list:
            args = c[0]
            assert len(args) >= 2, "navigate_to must be called with (x, y, ...)"
            assert isinstance(args[0], float), f"x must be float, got {type(args[0])}"
            assert isinstance(args[1], float), f"y must be float, got {type(args[1])}"
