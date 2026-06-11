# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M3 — NavigateToPointSkill (base-generic shortest-path navigation)."""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill


def _ctx(base) -> SimpleNamespace:
    return SimpleNamespace(base=base)


def _nav_base(reached: bool = True, remaining: float = 0.1):
    calls = []

    def navigate_to(x, y, tol=0.2):
        calls.append((x, y, tol))
        return {"reached": reached, "remaining": remaining,
                "pos": [x, y, 0.0], "heading": 0.0}

    return SimpleNamespace(navigate_to=navigate_to, calls=calls)


class TestNavigateToPoint:
    def test_reaches_goal(self) -> None:
        base = _nav_base()
        res = NavigateToPointSkill().execute({"x": 2.0, "y": -1.5}, _ctx(base))
        assert res.success is True
        assert base.calls == [(2.0, -1.5, 0.2)]
        assert res.result_data["diagnosis"] == "ok"
        assert res.result_data["target"] == [2.0, -1.5]

    def test_stuck_navigation_fails_honestly(self) -> None:
        base = _nav_base(reached=False, remaining=1.7)
        res = NavigateToPointSkill().execute({"x": 5.0, "y": 5.0}, _ctx(base))
        assert res.success is False
        assert "1.70m" in res.error_message
        assert res.result_data["diagnosis"] == "nav_stuck"

    def test_no_base_fails_loud(self) -> None:
        res = NavigateToPointSkill().execute({"x": 1, "y": 1}, _ctx(None))
        assert res.success is False
        assert res.result_data["diagnosis"] == "no_base"

    def test_base_without_capability_fails_loud(self) -> None:
        res = NavigateToPointSkill().execute(
            {"x": 1, "y": 1}, _ctx(SimpleNamespace())
        )
        assert res.success is False
        assert res.result_data["diagnosis"] == "no_navigate_support"

    def test_bad_params_fail_loud(self) -> None:
        res = NavigateToPointSkill().execute({"x": "north"}, _ctx(_nav_base()))
        assert res.success is False
        assert res.result_data["diagnosis"] == "bad_params"

    def test_verify_hint_is_the_vln_criterion(self) -> None:
        assert NavigateToPointSkill.verify_hint == "geodesic_dist(x, y) < 0.5"
