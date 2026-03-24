"""Tests for ActionRecord and GoalResult types in core/types.py."""
import pytest
from vector_os_nano.core.types import ActionRecord, GoalResult


class TestActionRecord:
    def test_frozen(self):
        r = ActionRecord(iteration=0, action="pick", params={"object_label": "banana"},
                         skill_success=True, verified=True, reasoning="closest", duration_sec=1.2)
        with pytest.raises(Exception):
            r.action = "place"

    def test_to_dict(self):
        r = ActionRecord(iteration=0, action="scan", params={},
                         skill_success=True, verified=False, reasoning="", duration_sec=0.5)
        d = r.to_dict()
        assert d["action"] == "scan"
        assert d["skill_success"] is True
        assert d["duration_sec"] == 0.5

    def test_defaults(self):
        r = ActionRecord(iteration=0, action="home")
        assert r.params == {}
        assert r.skill_success is False
        assert r.verified is False


class TestGoalResult:
    def test_to_dict(self):
        gr = GoalResult(success=True, goal="clean table", iterations=3,
                        total_duration_sec=12.5, actions=[], summary="Done.", final_world_state={})
        d = gr.to_dict()
        assert d["success"] is True
        assert d["iterations"] == 3
        assert d["summary"] == "Done."

    def test_defaults(self):
        gr = GoalResult(success=False, goal="test", iterations=0, total_duration_sec=0.0)
        assert gr.actions == []
        assert gr.summary == ""
        assert gr.final_world_state == {}

    def test_with_actions(self):
        a = ActionRecord(iteration=0, action="pick", skill_success=True)
        gr = GoalResult(success=True, goal="test", iterations=1,
                        total_duration_sec=1.0, actions=[a])
        d = gr.to_dict()
        assert len(d["actions"]) == 1
        assert d["actions"][0]["action"] == "pick"
