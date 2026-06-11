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


class TestSemanticLabelGoal:
    def _wm(self, *objs):
        from vector_os_nano.core.world_model import ObjectState, WorldModel

        wm = WorldModel()
        for o in objs:
            wm.add_object(o)
        return wm

    def test_label_resolves_to_best_confidence_instance(self) -> None:
        from vector_os_nano.core.world_model import ObjectState

        base = _nav_base()
        wm = self._wm(
            ObjectState("sofa_a", "sofa", x=1.0, y=1.0, confidence=0.7),
            ObjectState("sofa_b", "sofa", x=9.0, y=9.0, confidence=0.95),
        )
        ctx = SimpleNamespace(base=base, world_model=wm)
        res = NavigateToPointSkill().execute({"label": "sofa"}, ctx)
        assert res.success is True
        assert base.calls[0][:2] == (9.0, 9.0)  # the 0.95-confidence one

    def test_unknown_label_fails_loud_with_known_set(self) -> None:
        from vector_os_nano.core.world_model import ObjectState

        wm = self._wm(ObjectState("sofa_0", "sofa", x=1.0, y=2.0))
        ctx = SimpleNamespace(base=_nav_base(), world_model=wm)
        res = NavigateToPointSkill().execute({"label": "游泳池"}, ctx)
        assert res.success is False
        assert res.result_data["diagnosis"] == "object_not_found"
        assert "sofa" in res.error_message  # tells the replanner what EXISTS
