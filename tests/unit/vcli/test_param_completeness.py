# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 2 (campaign #4) — deterministic param-completeness pass + bounded re-ask.

The decomposer validated strategies and verify expressions but never the
PARAMS against the skill's own declared schema — a step missing a required
param sailed into execution, failed bad_params, and burned a whole
replan cycle (R5 GUI: two live occurrences). Now: after parse, each step's
strategy_params are checked against the registry-derived schema (rule 3
single source — required flags + enum sets); on issues the LLM is re-asked
EXACTLY ONCE with the per-step missing/illegal list and the legal sets;
still-broken params stay as-is (the skill's bad_params fail-loud + replan
remain the judge). No silent defaults, no LLM grading — the check itself
is pure schema comparison.
"""
from __future__ import annotations

import json

from vector_os_nano.vcli.cognitive.param_check import collect_param_issues
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal


class _WalkSkill:
    name = "walk"
    parameters = {
        "direction": {"type": "string", "required": True,
                      "enum": ["forward", "backward", "left", "right"]},
        "distance": {"type": "number", "required": True},
        "speed": {"type": "number", "required": False},
    }


class _LookSkill:
    name = "look"
    parameters = {"target": {"type": "string", "required": False}}


class _Registry:
    def __init__(self):
        self._skills = {"walk": _WalkSkill(), "look": _LookSkill()}

    def get(self, name):
        return self._skills.get(name)

    def list_skills(self):
        return list(self._skills)


def _sg(name, strategy, params):
    return SubGoal(name=name, description=name, strategy=strategy,
                   strategy_params=params, verify="True", timeout_sec=10)


def _tree(*sgs):
    return GoalTree(goal="t", sub_goals=tuple(sgs))


class TestCollect:
    def test_missing_required_reported_with_schema(self):
        issues = collect_param_issues(
            _tree(_sg("w", "walk_skill", {"direction": "forward"})),
            _Registry())
        assert len(issues) == 1
        assert issues[0]["step"] == "w"
        assert "distance" in issues[0]["missing"]

    def test_enum_violation_reported_with_legal_set(self):
        issues = collect_param_issues(
            _tree(_sg("w", "walk_skill",
                      {"direction": "sideways", "distance": 1.0})),
            _Registry())
        assert len(issues) == 1
        bad = issues[0]["illegal"][0]
        assert bad["param"] == "direction"
        assert "forward" in bad["legal"]

    def test_complete_params_no_issue(self):
        issues = collect_param_issues(
            _tree(_sg("w", "walk_skill",
                      {"direction": "left", "distance": 2.0})),
            _Registry())
        assert issues == []

    def test_optional_missing_is_fine_and_unknown_strategy_skipped(self):
        issues = collect_param_issues(
            _tree(_sg("l", "look_skill", {}),
                  _sg("x", "made_up_skill", {})),
            _Registry())
        assert issues == []

    def test_answer_and_empty_strategy_skipped(self):
        issues = collect_param_issues(
            _tree(_sg("a", "answer", {}), _sg("b", "", {})),
            _Registry())
        assert issues == []


class _Backend:
    """First call returns a plan with broken params; second returns the fix."""

    def __init__(self, first: dict, second: dict | str | None):
        self._responses = [json.dumps(first)]
        if second is not None:
            self._responses.append(
                second if isinstance(second, str) else json.dumps(second))
        self.calls: list = []

    def call(self, messages, tools, system, max_tokens):
        self.calls.append(messages)
        from types import SimpleNamespace

        return SimpleNamespace(text=self._responses[len(self.calls) - 1])


def _plan(params):
    return {"goal": "walk", "sub_goals": [{
        "name": "w", "description": "walk", "strategy": "walk_skill",
        "strategy_params": params, "verify": "True", "timeout_sec": 10,
    }]}


def _decomposer(backend):
    from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

    return GoalDecomposer(backend, skill_registry=_Registry())


class TestReaskFlow:
    def test_missing_param_triggers_exactly_one_reask(self):
        backend = _Backend(
            _plan({"direction": "forward"}),                 # distance missing
            {"w": {"direction": "forward", "distance": 20.0}})
        tree = _decomposer(backend).decompose("走20米", "ctx")
        assert len(backend.calls) == 2
        sg = tree.sub_goals[0]
        assert sg.strategy_params["distance"] == 20.0
        # the re-ask prompt names the missing param and the step
        reask_text = json.dumps(backend.calls[1], ensure_ascii=False)
        assert "distance" in reask_text and "step 'w'" in reask_text

    def test_enum_violation_reask_names_legal_set(self):
        backend = _Backend(
            _plan({"direction": "sideways", "distance": 1.0}),
            {"w": {"direction": "left", "distance": 1.0}})
        tree = _decomposer(backend).decompose("往左走", "ctx")
        assert tree.sub_goals[0].strategy_params["direction"] == "left"
        reask_text = json.dumps(backend.calls[1], ensure_ascii=False)
        assert "forward" in reask_text                      # legal set taught

    def test_still_broken_after_reask_left_for_bad_params(self):
        backend = _Backend(
            _plan({"direction": "forward"}),
            {"w": {"direction": "forward"}})                 # still no distance
        tree = _decomposer(backend).decompose("走", "ctx")
        assert len(backend.calls) == 2                       # bounded: no 3rd
        assert "distance" not in tree.sub_goals[0].strategy_params

    def test_garbage_reask_response_keeps_original(self):
        backend = _Backend(_plan({"direction": "forward"}), "not json at all")
        tree = _decomposer(backend).decompose("走", "ctx")
        assert len(backend.calls) == 2
        assert tree.sub_goals[0].strategy_params == {"direction": "forward"}

    def test_complete_plan_single_call(self):
        backend = _Backend(
            _plan({"direction": "forward", "distance": 2.0}), None)
        _decomposer(backend).decompose("走2米", "ctx")
        assert len(backend.calls) == 1

    def test_reask_cannot_change_tree_structure(self):
        # the correction may only touch the NAMED step's params
        backend = _Backend(
            _plan({"direction": "forward"}),
            {"w": {"direction": "forward", "distance": 5.0},
             "evil_new_step": {"x": 1}})
        tree = _decomposer(backend).decompose("走", "ctx")
        assert len(tree.sub_goals) == 1
        assert tree.sub_goals[0].strategy_params["distance"] == 5.0
