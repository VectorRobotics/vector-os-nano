# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Named-target verify strengthening (STATUS backlog #2 part (b)).

A step whose params bind a NAMED target must verify "holding the REQUESTED
object" (target-aware ``holding_object('<name>')``), never "holding
SOMETHING" (bare ``holding_object()``) — the bare form FALSE-PASSES when a
pick grabbed the wrong object. Generic grabs (no target bound) keep the bare
form. The rewrite is structural (param presence), language-neutral, and
stricter-only; it applies at BOTH plan chokepoints (LLM validate + fast path).
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.verify_strengthen import strengthen_target_verify


class _MockBackend:
    def call(self, *a, **k):  # pragma: no cover — never invoked in these tests
        raise AssertionError("backend must not be called")


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------


class TestStrengthenTargetVerify:
    def test_named_target_binds_into_verify(self) -> None:
        out = strengthen_target_verify(
            "holding_object()", {"object_label": "apple"}
        )
        assert out == "holding_object('apple')"

    def test_generic_grab_keeps_bare_form(self) -> None:
        assert strengthen_target_verify("holding_object()", {}) == "holding_object()"
        assert (
            strengthen_target_verify("holding_object()", {"mode": "hold"})
            == "holding_object()"
        )

    def test_already_target_aware_untouched(self) -> None:
        v = "holding_object('apple')"
        assert strengthen_target_verify(v, {"object_label": "pear"}) == v

    def test_non_holding_verify_untouched(self) -> None:
        assert strengthen_target_verify("True", {"object_label": "apple"}) == "True"
        v = "not holding_object()"
        assert strengthen_target_verify(v, {"object_label": "apple"}) == v

    def test_param_priority_object_label_first(self) -> None:
        out = strengthen_target_verify(
            "holding_object()", {"target": "pear", "object_label": "apple"}
        )
        assert out == "holding_object('apple')"

    def test_quote_in_label_stays_ast_safe(self) -> None:
        out = strengthen_target_verify("holding_object()", {"object_label": "a'b"})
        import ast

        ast.parse(out, mode="eval")  # must remain a valid expression
        assert ast.literal_eval(out.removeprefix("holding_object(").removesuffix(")")) == "a'b"

    def test_blackboard_ref_skipped(self) -> None:
        """A ${step.output...} reference is unresolvable inside a verify
        expression — strengthening would compare against the literal ref and
        FALSE-FAIL. Skip it (bare form is the lesser evil)."""
        v = strengthen_target_verify(
            "holding_object()", {"object_label": "${detect.output.objects[0].name}"}
        )
        assert v == "holding_object()"

    def test_dunder_label_skipped(self) -> None:
        """A '__' in the label would be rejected by the AST validator and kill
        the whole step — skip strengthening instead."""
        v = strengthen_target_verify("holding_object()", {"object_label": "a__b"})
        assert v == "holding_object()"

    def test_whitespace_tolerant(self) -> None:
        out = strengthen_target_verify("  holding_object()  ", {"object": " cup "})
        assert out == "holding_object('cup')"


# ---------------------------------------------------------------------------
# LLM-path chokepoint: GoalDecomposer._validate_sub_goal
# ---------------------------------------------------------------------------


class TestDecomposerValidateStrengthens:
    def _validate(self, raw: dict):
        from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

        d = GoalDecomposer(
            _MockBackend(),
            verify_functions={"holding_object"},
            strategies={"pick_skill"},
        )
        return d._validate_sub_goal(raw, valid_names={raw.get("name", "")})

    def test_named_pick_step_gets_target_aware_verify(self) -> None:
        sg = self._validate(
            {
                "name": "grab_apple",
                "description": "pick up the apple",
                "verify": "holding_object()",
                "strategy": "pick_skill",
                "strategy_params": {"object_label": "apple"},
            }
        )
        assert sg is not None
        assert sg.verify == "holding_object('apple')"

    def test_generic_pick_step_keeps_bare_verify(self) -> None:
        sg = self._validate(
            {
                "name": "grab_any",
                "description": "抓个东西",
                "verify": "holding_object()",
                "strategy": "pick_skill",
                "strategy_params": {},
            }
        )
        assert sg is not None
        assert sg.verify == "holding_object()"


# ---------------------------------------------------------------------------
# Fast-path chokepoint: engine 1-step plan
# ---------------------------------------------------------------------------


class TestFastPathPickVerify:
    # Invariant III: the verify comes from the SKILL's own verify_template
    # (single source) — the engine no longer carries a second map.
    def test_pick_template_gives_bare_holding(self) -> None:
        from vector_os_nano.skills.pick import PickSkill
        from vector_os_nano.vcli.engine import VectorEngine

        assert (
            VectorEngine._verify_for_skill("pick", "", PickSkill())
            == "holding_object()"
        )

    def test_fast_path_pick_with_target_is_target_aware(self) -> None:
        from vector_os_nano.skills.pick import PickSkill
        from vector_os_nano.vcli.engine import VectorEngine

        assert (
            VectorEngine._verify_for_skill("pick", "apple", PickSkill())
            == "holding_object('apple')"
        )


class TestBackfillLabelFromVisited:
    """R8 (owner GUI test regression): the LLM emitted a paramless navigate
    with the target ONLY in the verify — `visited('sofa')`. Label-style
    backfill mirrors the coordinate-style repair: deterministic, never
    overwrites, navigate_to* only."""

    def test_visited_label_backfilled(self):
        from vector_os_nano.vcli.cognitive.verify_strengthen import (
            backfill_target_params,
        )

        out = backfill_target_params(
            "navigate_to_skill", {}, "visited('sofa')")
        assert out["label"] == "sofa"

    def test_existing_params_never_overwritten(self):
        from vector_os_nano.vcli.cognitive.verify_strengthen import (
            backfill_target_params,
        )

        out = backfill_target_params(
            "navigate_to_skill", {"label": "kitchen"}, "visited('sofa')")
        assert out["label"] == "kitchen"

    def test_double_quotes_and_at_position_label(self):
        from vector_os_nano.vcli.cognitive.verify_strengthen import (
            backfill_target_params,
        )

        out = backfill_target_params(
            'navigate_to_skill', {}, 'visited("entryway")')
        assert out["label"] == "entryway"

    def test_non_navigate_untouched(self):
        from vector_os_nano.vcli.cognitive.verify_strengthen import (
            backfill_target_params,
        )

        assert backfill_target_params("walk_skill", {}, "visited('sofa')") == {}


class TestBackfillNullValuedKeys:
    """R6 root-cause (campaign #4 batch 2): deepseek-chat emits schema keys
    with null/"" values; ``setdefault`` keeps the poisoned key and ``"x" in
    params`` treats null coords as bound — the backfill 'fired' but returned
    useless params, surfacing two layers later as 'navigate_to requires a
    label OR numeric x and y' (the R5 GUI kitchen failure)."""

    def _bf(self, params, verify):
        from vector_os_nano.vcli.cognitive.verify_strengthen import (
            backfill_target_params,
        )
        return backfill_target_params("navigate_to_skill", params, verify)

    def test_empty_string_label_is_missing(self):
        assert self._bf({"label": ""}, "visited('kitchen')")["label"] == "kitchen"

    def test_null_label_is_missing(self):
        assert self._bf({"label": None}, "visited('kitchen')")["label"] == "kitchen"

    def test_null_coords_are_missing(self):
        out = self._bf({"x": None, "y": None}, "geodesic_dist(2.0, -1.5) < 0.8")
        assert out["x"] == 2.0 and out["y"] == -1.5

    def test_null_label_with_coord_verify(self):
        out = self._bf({"label": None}, "at_position(1.0, 2.0, 0.5)")
        assert out["x"] == 1.0 and out["y"] == 2.0

    def test_real_label_still_never_overwritten(self):
        assert self._bf({"label": "sofa"}, "visited('kitchen')")["label"] == "sofa"


class TestParseSeamNullStripping:
    """Null-valued params are stripped at the decomposer parse seam, so the
    visited()/coords backfill (and every later missing-param check) fires."""

    def _parse(self, params, verify="visited('kitchen')"):
        import json
        from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

        d = GoalDecomposer.__new__(GoalDecomposer)
        d.KNOWN_STRATEGIES = frozenset({"navigate_to_skill"})
        d.VERIFY_FUNCTIONS = (
            GoalDecomposer.VERIFY_FUNCTIONS | {"visited", "geodesic_dist"}
        )
        raw = json.dumps({"goal": "t", "sub_goals": [{
            "name": "nav", "description": "d", "strategy": "navigate_to_skill",
            "strategy_params": params, "verify": verify, "timeout_sec": 60,
        }]})
        return d._parse_and_validate("t", raw).sub_goals[0]

    def test_null_label_stripped_and_backfilled(self):
        sg = self._parse({"label": None})
        assert sg.strategy_params == {"label": "kitchen"}

    def test_empty_string_stripped(self):
        sg = self._parse({"label": ""})
        assert sg.strategy_params == {"label": "kitchen"}

    def test_real_values_survive(self):
        sg = self._parse({"label": "sofa", "speed": 0.5})
        assert sg.strategy_params == {"label": "sofa", "speed": 0.5}

    def test_zero_and_false_are_real_values(self):
        sg = self._parse({"x": 0.0, "y": 0.0})
        assert sg.strategy_params["x"] == 0.0 and sg.strategy_params["y"] == 0.0
