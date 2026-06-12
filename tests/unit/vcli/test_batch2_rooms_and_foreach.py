# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 2 part 1 — foreach honesty (#10) + rooms as REGIONS (#3).

(1) A foreach whose source_path does NOT resolve used to run zero iterations
and let the tree PASS silently — the closed loop was structurally broken
there (no FailureRecord, replan could never correct it). Unresolved now
yields a failed step (failure_class=exec_error) carrying the producer's
actually-available keys; a RESOLVED-but-empty list stays an honest zero.

(2) Room landmarks are REGIONS: seed keeps the rect; navigate's room branch
drives INTO the rect (tol from the rect's half-dims, never the 1.5 object
standoff) so 'inside the room' is reachable, and visited('<room>') is the
honest predicate.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.playground.catalog import get_scenario
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.types import ForEachSpec, SubGoal
from vector_os_nano.vcli.habitat_runtime import seed_room_landmarks


class _Selector:
    def select(self, sub_goal):
        from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult

        return StrategyResult("primitive", "noop", {})


def _executor() -> GoalExecutor:
    ex = GoalExecutor(
        strategy_selector=_Selector(),
        verifier=GoalVerifier({}),
        primitives={"noop": lambda: True},
    )
    ex.blackboard = Blackboard()
    return ex


def _foreach_sg() -> SubGoal:
    return SubGoal(
        name="for_each_obj",
        description="iterate",
        verify="True",
        foreach=ForEachSpec(
            source_step="detect_all",
            source_path="objects",
            var="obj",
            body=(SubGoal(name="use_{i}", description="use", verify="True",
                          strategy="noop"),),
        ),
    )


class TestForeachUnresolvedIsLoudFailure:
    def test_unresolved_source_fails_with_exec_error_and_keys(self):
        ex = _executor()
        # producer captured under a DIFFERENT key than the plan referenced
        ex.blackboard.put("detect_all", {"output": {"items": [1, 2]}})
        records = ex._execute_foreach(_foreach_sg())
        assert len(records) == 1
        rec = records[0]
        assert rec.success is False
        assert rec.failure_class == "exec_error"
        assert "items" in rec.error  # the producer's REAL keys, fed to replan

    def test_no_producer_at_all_fails_loud(self):
        ex = _executor()
        records = ex._execute_foreach(_foreach_sg())
        assert len(records) == 1
        assert records[0].success is False
        assert records[0].failure_class == "exec_error"

    def test_resolved_empty_list_is_honest_zero(self):
        ex = _executor()
        ex.blackboard.put("detect_all", {"output": {"objects": []}})
        records = ex._execute_foreach(_foreach_sg())
        assert records == []  # genuinely nothing to do — not an error

    def test_resolved_list_runs_children(self):
        ex = _executor()
        ex.blackboard.put("detect_all", {"output": {"objects": [{"n": 1}]}})
        records = ex._execute_foreach(_foreach_sg())
        assert len(records) == 1 and records[0].success is True


class TestRoomLandmarksAreRegions:
    def test_seed_carries_rect(self):
        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        k = wm.get_objects_by_label("kitchen")[0]
        assert tuple(k.properties["rect"]) == (-3.6, 1.0, 0.2, 2.8)

    def test_room_goal_uses_region_tol_not_object_standoff(self):
        from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        calls: list[tuple] = []

        class _Base:
            def navigate_to(self, x, y, tol=0.2):
                calls.append((x, y, tol))
                return {"reached": True, "pos": [x, y, 0.0], "dist": 0.05}

        ctx = SimpleNamespace(base=_Base(), world_model=wm, agent=None)
        res = NavigateToPointSkill().execute({"label": "kitchen"}, ctx)
        assert res.success
        x, y, tol = calls[0]
        # kitchen rect half-dims (1.9, 0.9): the drive must end INSIDE the
        # rect — tol bounded by the smaller half-dim, never the 1.5 standoff.
        assert (round(x, 2), round(y, 2)) == (-1.7, 1.9)
        assert tol <= 0.9
        assert res.result_data.get("goal_kind") == "room"

    def test_object_goal_keeps_standoff(self):
        from vector_os_nano.core.world_model import ObjectState
        from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

        wm = WorldModel()
        wm.add_object(ObjectState(object_id="sysnav_1", label="sofa",
                                  x=1.0, y=2.0))
        calls: list[tuple] = []

        class _Base:
            def navigate_to(self, x, y, tol=0.2):
                calls.append((x, y, tol))
                return {"reached": True, "pos": [x, y, 0.0], "dist": 0.05}

        ctx = SimpleNamespace(base=_Base(), world_model=wm, agent=None)
        res = NavigateToPointSkill().execute({"label": "sofa"}, ctx)
        assert res.success
        assert calls[0][2] >= 1.5  # furniture standoff unchanged

    def test_verify_hint_teaches_visited_for_rooms(self):
        from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

        hint = NavigateToPointSkill.verify_hint
        assert "visited(" in hint
