# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""N4 — navigate transport selection: real nav stack when alive, sim oracle
as the fallback; the oracle stays verify-only inside the nav-stack path."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill


class _OracleBase:
    def __init__(self) -> None:
        self.calls: list = []
        self._nav_feed = None

    def navigate_to(self, x, y, tol=0.2):
        self.calls.append((x, y))
        return {"reached": True, "remaining": 0.1, "pos": [x, y, 0.0]}


class _FakeFeed:
    def __init__(self, active=True, reached=True) -> None:
        self.active = active
        self.calls: list = []
        self._reached = reached

    def nav_stack_active(self) -> bool:
        return self.active

    def navigate_to(self, x, y, tol=0.5, **kw):
        self.calls.append((x, y))
        self.last_tol = tol
        return {
            "reached": self._reached, "remaining": 0.3 if self._reached else 4.2,
            "pos": [x, y, 0.0], "transport": "nav_stack",
            **({} if self._reached else {"reason": "stall"}),
        }


def _ctx(base) -> SkillContext:
    return SkillContext(base=base, world_model=None)


class TestTransportSelection:
    def test_nav_stack_preferred_when_alive(self) -> None:
        base = _OracleBase()
        base._nav_feed = _FakeFeed(active=True)
        r = NavigateToPointSkill().execute({"x": 1.0, "y": 2.0}, _ctx(base))
        assert r.success is True
        assert r.result_data["transport"] == "nav_stack"
        assert base._nav_feed.calls == [(1.0, 2.0)]
        assert base.calls == []  # the oracle never planned

    def test_oracle_fallback_when_stack_down(self) -> None:
        base = _OracleBase()
        base._nav_feed = _FakeFeed(active=False)
        r = NavigateToPointSkill().execute({"x": 1.0, "y": 2.0}, _ctx(base))
        assert r.success is True
        assert r.result_data["transport"] == "sim_oracle"
        assert base.calls == [(1.0, 2.0)]

    def test_oracle_fallback_without_feed(self) -> None:
        base = _OracleBase()
        r = NavigateToPointSkill().execute({"x": 0.5, "y": 0.5}, _ctx(base))
        assert r.success is True
        assert r.result_data["transport"] == "sim_oracle"

    def test_nav_stack_stall_is_honest_failure(self) -> None:
        base = _OracleBase()
        base._nav_feed = _FakeFeed(active=True, reached=False)
        r = NavigateToPointSkill().execute({"x": 9.0, "y": 9.0}, _ctx(base))
        assert r.success is False
        assert r.result_data["diagnosis"] == "stall"
        assert r.result_data["transport"] == "nav_stack"


class TestSemanticStandoff:
    def _wm(self):
        obj = SimpleNamespace(label="sofa", confidence=0.7, x=-2.1, y=-3.8)
        return SimpleNamespace(
            get_objects_by_label=lambda label: [obj] if label == "sofa" else [],
            get_objects=lambda: [obj],
        )

    def test_label_goal_floors_tol(self) -> None:
        # "go to the sofa" must stop at standoff, not on the sofa (N4).
        base = _OracleBase()
        feed = _FakeFeed(active=True)
        base._nav_feed = feed
        ctx = SkillContext(base=base, world_model=self._wm())
        r = NavigateToPointSkill().execute({"label": "sofa"}, ctx)
        assert r.success is True
        assert feed.last_tol >= 1.5

    def test_label_goal_keeps_larger_explicit_tol(self) -> None:
        base = _OracleBase()
        feed = _FakeFeed(active=True)
        base._nav_feed = feed
        ctx = SkillContext(base=base, world_model=self._wm())
        NavigateToPointSkill().execute({"label": "sofa", "tol": 2.0}, ctx)
        assert feed.last_tol == 2.0

    def test_coordinate_goal_tol_unchanged(self) -> None:
        base = _OracleBase()
        feed = _FakeFeed(active=True)
        base._nav_feed = feed
        NavigateToPointSkill().execute({"x": 1.0, "y": 2.0, "tol": 0.3}, _ctx(base))
        assert feed.last_tol == 0.3


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _rclpy_available(), reason="rclpy not importable")
class TestFeedNavigate:
    def _feed(self, base):
        from vector_os_nano.playground.habitat.sysnav_bridge import (
            HabitatSysnavBridge,
        )

        return HabitatSysnavBridge(base, hz=0.01, state_hz=0.0)

    def test_send_waypoint_publishes_point(self) -> None:
        import rclpy
        import time
        from geometry_msgs.msg import PointStamped

        feed = self._feed(SimpleNamespace())
        got: dict = {}
        probe = rclpy.create_node("way_probe")
        probe.create_subscription(
            PointStamped, "/way_point", lambda m: got.setdefault("wp", m), 5
        )
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and not got:
                feed.send_waypoint(-5.3, -3.4)
                rclpy.spin_once(probe, timeout_sec=0.05)
            wp = got["wp"]
            assert wp.header.frame_id == "map"
            assert wp.point.x == pytest.approx(-5.3)
        finally:
            probe.destroy_node()
            feed.destroy()

    def test_navigate_converging_base_reaches(self) -> None:
        pos = [0.0, 0.0, 0.0]
        base = SimpleNamespace(get_position=lambda: list(pos))
        feed = self._feed(base)
        try:
            # the "robot" jumps to the goal after the first poll
            import threading
            import time

            def _move():
                time.sleep(0.3)
                pos[0], pos[1] = 2.0, 1.0

            threading.Thread(target=_move, daemon=True).start()
            out = feed.navigate_to(2.0, 1.0, timeout_s=5.0, poll_s=0.2)
            assert out["reached"] is True
            assert out["transport"] == "nav_stack"
        finally:
            feed.destroy()

    def test_navigate_frozen_base_stalls_honestly(self) -> None:
        base = SimpleNamespace(get_position=lambda: [0.0, 0.0, 0.0])
        feed = self._feed(base)
        try:
            out = feed.navigate_to(
                5.0, 5.0, timeout_s=30.0, stall_s=0.5, poll_s=0.1
            )
            assert out["reached"] is False
            assert out["reason"] == "stall"
        finally:
            feed.destroy()

    def test_nav_stack_active_false_without_follower(self) -> None:
        feed = self._feed(SimpleNamespace())
        try:
            assert feed.nav_stack_active() is False
        finally:
            feed.destroy()


@pytest.mark.skipif(not _rclpy_available(), reason="rclpy not importable")
class TestThreeValueContract:
    """Batch 2 R6 (design review #2/#8/#9): navigation returns MOTION EVIDENCE
    — {reached, already_there, moved_m, elapsed_s} — and arrival is confirmed
    by ONE geodesic check (verify-side oracle) so a through-wall euclidean
    coincidence can never PASS."""

    def _feed(self, base):
        from vector_os_nano.playground.habitat.sysnav_bridge import (
            HabitatSysnavBridge,
        )

        return HabitatSysnavBridge(base, hz=0.01, state_hz=0.0)

    def test_already_there_short_circuits(self) -> None:
        from types import SimpleNamespace

        base = SimpleNamespace(get_position=lambda: [2.0, 1.0, 0.0])
        feed = self._feed(base)
        try:
            out = feed.navigate_to(2.0, 1.0, tol=0.5, timeout_s=5.0)
            assert out["reached"] is True
            assert out["already_there"] is True
            assert out["moved_m"] == 0.0
            assert "elapsed_s" in out
        finally:
            feed.destroy()

    def test_through_wall_euclid_rejected_by_geodesic(self) -> None:
        from types import SimpleNamespace

        pos = [0.0, 0.0, 0.0]
        base = SimpleNamespace(
            get_position=lambda: list(pos),
            geodesic_distance=lambda a, b: 9.9,  # wall between — long way round
        )
        feed = self._feed(base)
        try:
            import threading
            import time as _t

            def _move():
                _t.sleep(0.3)
                pos[0], pos[1] = 2.0, 1.0  # euclid-converged (other side of wall)

            threading.Thread(target=_move, daemon=True).start()
            out = feed.navigate_to(2.0, 1.0, timeout_s=3.0, poll_s=0.2)
            assert out["reached"] is False
            assert out["reason"] == "geodesic_mismatch"
        finally:
            feed.destroy()

    def test_motion_evidence_in_every_outcome(self) -> None:
        from types import SimpleNamespace

        base = SimpleNamespace(get_position=lambda: [0.0, 0.0, 0.0])
        feed = self._feed(base)
        try:
            out = feed.navigate_to(5.0, 5.0, timeout_s=2.0, stall_s=0.3,
                                   poll_s=0.1)
            assert out["reached"] is False
            assert out["moved_m"] == 0.0
            assert out["elapsed_s"] >= 0.0
            assert out["already_there"] is False
        finally:
            feed.destroy()

    def test_skill_passes_motion_evidence_through(self) -> None:
        from types import SimpleNamespace

        class _Base:
            def navigate_to(self, x, y, tol=0.2):
                return {"reached": True, "pos": [x, y, 0.0], "remaining": 0.1,
                        "already_there": True, "moved_m": 0.0,
                        "elapsed_s": 0.01}

        ctx = SimpleNamespace(base=_Base(), world_model=None, agent=None)
        res = NavigateToPointSkill().execute({"x": 1.0, "y": 2.0}, ctx)
        assert res.success
        assert res.result_data["already_there"] is True
        assert res.result_data["moved_m"] == 0.0
