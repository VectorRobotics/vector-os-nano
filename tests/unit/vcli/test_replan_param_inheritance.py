# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Replan param-degradation guard (owner finding (c), 2026-06-12).

The owner's '走到sofa' replan emitted navigate_to_skill with EMPTY params —
the previous attempt's {"label": "sofa"} was lost, degrading a meaningful
'no object matching sofa' failure into an opaque 'requires a label OR numeric
x and y'. Kernel rule: a replanned step reusing a strategy that previously ran
WITH params never degrades to empty params — the prior binding is inherited.
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import VGGHarness


def _tree(*steps: SubGoal) -> GoalTree:
    return GoalTree(goal="走到sofa", sub_goals=tuple(steps))


def _sg(name: str, strategy: str, params: dict | None = None) -> SubGoal:
    return SubGoal(
        name=name,
        description=name,
        verify="True",
        strategy=strategy,
        strategy_params=params or {},
    )


class TestInheritReplanParams:
    def test_empty_params_inherit_from_prior_same_strategy(self):
        prev = _tree(_sg("navigate_to_sofa", "navigate_to_skill", {"label": "sofa"}))
        new = _tree(_sg("navigate_to_sofa_again", "navigate_to_skill"))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {"label": "sofa"}

    def test_nonempty_new_params_always_win(self):
        prev = _tree(_sg("nav", "navigate_to_skill", {"label": "sofa"}))
        new = _tree(_sg("nav2", "navigate_to_skill", {"x": 1.0, "y": 2.0}))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {"x": 1.0, "y": 2.0}

    def test_different_strategy_never_inherits(self):
        prev = _tree(_sg("nav", "navigate_to_skill", {"label": "sofa"}))
        new = _tree(_sg("turn_a_bit", "turn_skill"))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {}

    def test_no_prior_tree_is_identity(self):
        new = _tree(_sg("nav", "navigate_to_skill"))
        merged = VGGHarness._inherit_replan_params(new, None)
        assert merged is new

    def test_prior_empty_params_is_identity(self):
        prev = _tree(_sg("nav", "navigate_to_skill"))
        new = _tree(_sg("nav2", "navigate_to_skill"))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged is new

    def test_ambiguous_same_strategy_left_empty(self):
        """#18 (campaign #4): two prior navigates with DIFFERENT targets and a
        replanned step matching neither name — the old 'latest wins' silently
        bound BOTH replanned navigates to sofa. Ambiguity now stays EMPTY so
        the step fails bad_params and the replan hears it (fail loud)."""
        prev = _tree(
            _sg("nav_a", "navigate_to_skill", {"label": "table"}),
            _sg("nav_b", "navigate_to_skill", {"label": "sofa"}),
        )
        new = _tree(_sg("nav_c", "navigate_to_skill"))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {}

    def test_name_match_wins_over_ambiguity(self):
        """#18: (strategy, sub_goal name) pairing — each replanned step gets
        ITS OWN prior binding even when the strategy appears twice."""
        prev = _tree(
            _sg("nav_a", "navigate_to_skill", {"label": "table"}),
            _sg("nav_b", "navigate_to_skill", {"label": "sofa"}),
        )
        new = _tree(
            _sg("nav_a", "navigate_to_skill"),
            _sg("nav_b", "navigate_to_skill"),
        )
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {"label": "table"}
        assert merged.sub_goals[1].strategy_params == {"label": "sofa"}

    def test_single_step_strategy_fallback_keeps_working(self):
        """#18: strategy-level fallback survives ONLY when the prior tree has
        exactly one step with that strategy (no ambiguity to guess over)."""
        prev = _tree(_sg("nav_old", "navigate_to_skill", {"label": "desk"}))
        new = _tree(_sg("nav_renamed", "navigate_to_skill"))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {"label": "desk"}

    def test_multiple_new_steps_each_resolved(self):
        prev = _tree(_sg("nav", "navigate_to_skill", {"label": "sofa"}))
        new = _tree(
            _sg("scan", "look_skill"),
            _sg("nav2", "navigate_to_skill"),
        )
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {}
        assert merged.sub_goals[1].strategy_params == {"label": "sofa"}

    def test_suffix_form_variance_still_inherits(self):
        # The LLM alternates 'navigate_to' / 'navigate_to_skill' across
        # replans — both are the same route and must share bindings.
        prev = _tree(_sg("nav", "navigate_to_skill", {"label": "sofa"}))
        new = _tree(_sg("nav2", "navigate_to"))
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.sub_goals[0].strategy_params == {"label": "sofa"}

    def test_tree_metadata_preserved(self):
        prev = _tree(_sg("nav", "navigate_to_skill", {"label": "sofa"}))
        new = GoalTree(
            goal="走到sofa",
            sub_goals=(_sg("nav2", "navigate_to_skill"),),
            context_snapshot="ctx",
        )
        merged = VGGHarness._inherit_replan_params(new, prev)
        assert merged.goal == "走到sofa"
        assert merged.context_snapshot == "ctx"
