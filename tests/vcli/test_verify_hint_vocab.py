# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Step 3 — LLM-side grasp generalization: per-skill verify_hint + entity binding.

These are DETERMINISTIC teaching-signal tests (NO network / NO live LLM). They
assert the SHAPE of what the planner is taught and that a correctly-bound plan
PASSES the decompose validator — never model behaviour:

  1. ``Skill.to_schemas()`` surfaces each arm skill's declared ``verify_hint``
     (pick -> holding_object(), home -> arm_at_home(), detect ->
     len(detect_objects()) > 0, place -> not holding_object()).
  2. ``build_decompose_vocab`` derives, world-agnostically:
       - per-strategy ``suggested verify:`` lines in strategy_params_help,
       - a few-shot example whose verify expression IS a skill's verify_hint,
       - an object-ish param shown BOUND (non-empty) in that example so the
         planner learns the shape of binding a task target,
       - a planner_intro carrying the bind-the-target + prefer-suggested-verify
         guidance.
  3. A hand-written pick plan (strategy_params={"object_label": "banana"},
     verify "holding_object()") PASSES decompose validation against an arm
     vocab whose allowlist includes holding_object — the strategy is kept, the
     verify is not cleared, and validation_notes is empty.
"""

from __future__ import annotations

import json
from typing import Any

from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.pick import PickSkill
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.vocab_from_registry import build_decompose_vocab

# The arm verify namespace contributed by RobotWorld/Playground Step 2. The vocab
# allowlist is single-sourced from these (keys), so verify_hint predicates must
# all live here (or be the safe ``True`` literal).
_ARM_VERIFY_SIGS: dict[str, str] = {
    "holding_object": "holding_object() -> bool  # gripper holds an object",
    "arm_at_home": "arm_at_home() -> bool  # arm is at its home pose",
    "detect_objects": "detect_objects(query='') -> list  # detections",
    "describe_scene": "describe_scene() -> str  # scene caption",
    "placed_count": "placed_count() -> int  # objects placed in region",
}


def _arm_registry() -> SkillRegistry:
    """A registry of the real arm skills that declare verify_hint."""
    reg = SkillRegistry()
    for s in (HomeSkill(), DetectSkill(), PickSkill(), PlaceSkill()):
        reg.register(s)
    return reg


def _arm_schemas() -> list[dict[str, Any]]:
    return _arm_registry().to_schemas()


# ---------------------------------------------------------------------------
# 1. to_schemas() surfaces each arm skill's declared verify_hint.
# ---------------------------------------------------------------------------


def test_to_schemas_includes_verify_hint_per_skill() -> None:
    by_name = {s["name"]: s for s in _arm_schemas()}
    assert by_name["pick"]["verify_hint"] == "holding_object()"
    assert by_name["home"]["verify_hint"] == "arm_at_home()"
    assert by_name["detect"]["verify_hint"] == "len(detect_objects()) > 0"
    assert by_name["place"]["verify_hint"] == "not holding_object()"


def test_verify_hint_predicates_live_in_arm_verify_namespace() -> None:
    # Every non-trivial verify_hint must reference only predicates present in the
    # arm verify allowlist (Step 2) or the always-safe True literal — otherwise
    # the planner would be taught a verify the validator rejects.
    allowed = set(_ARM_VERIFY_SIGS) | {"True", "len"}
    for s in _arm_schemas():
        hint = s["verify_hint"]
        if hint == "True":
            continue
        # The predicate name is the token before the first "(".
        for fn in ("holding_object", "arm_at_home", "detect_objects", "placed_count"):
            if fn in hint:
                assert fn in allowed
                break
        else:  # pragma: no cover - defensive
            raise AssertionError(f"verify_hint references unknown predicate: {hint!r}")


# ---------------------------------------------------------------------------
# 2. build_decompose_vocab surfaces the teaching signal (world-agnostic).
# ---------------------------------------------------------------------------


def test_params_help_carries_suggested_verify_per_skill() -> None:
    vocab = build_decompose_vocab(_arm_schemas(), _ARM_VERIFY_SIGS, has_base=False)
    help_text = vocab.strategy_params_help
    # Each skill's declared predicate is surfaced as a 'suggested verify' line.
    assert "suggested verify: holding_object()" in help_text
    assert "suggested verify: arm_at_home()" in help_text
    assert "suggested verify: len(detect_objects()) > 0" in help_text
    assert "suggested verify: not holding_object()" in help_text


def test_example_uses_a_verify_hint_as_the_verify_expr() -> None:
    vocab = build_decompose_vocab(_arm_schemas(), _ARM_VERIFY_SIGS, has_base=False)
    ex = vocab.examples
    assert ex  # non-empty
    # The first chosen skill is 'home' (registry order) -> its verify_hint is the
    # example step's verify expression, not the alphabetical _pick_verify_fn.
    assert '"verify": "arm_at_home()"' in ex


def test_example_binds_an_object_param_non_empty() -> None:
    vocab = build_decompose_vocab(_arm_schemas(), _ARM_VERIFY_SIGS, has_base=False)
    ex = vocab.examples
    # An object-ish param shows BOUND (non-empty) so the planner learns the shape
    # of binding a target. detect.query is required + object-ish; pick.object_label
    # is object-ish. At least one carries a concrete sample value.
    payload = json.loads(ex.split("Response:\n", 1)[1])
    bound = []
    for sg in payload["sub_goals"]:
        for pname, pval in sg["strategy_params"].items():
            if pname in {"object", "object_label", "object_id", "query", "target", "label", "item"}:
                bound.append((pname, pval))
    assert bound, "expected at least one object-ish param in the example"
    assert any(v for _, v in bound), "object-ish param must be shown BOUND (non-empty)"
    # The sample value is derived from the schema (detect/pick descriptions say
    # e.g. 'banana'), not a hardcoded domain table.
    assert any(v == "banana" for _, v in bound)


def test_planner_intro_teaches_bind_target_and_prefer_suggested_verify() -> None:
    vocab = build_decompose_vocab(_arm_schemas(), _ARM_VERIFY_SIGS, has_base=False)
    intro = vocab.planner_intro
    assert "copy that target" in intro
    assert "never leave a known target blank" in intro
    assert "suggested verify" in intro
    # Generic — no embodiment-specific noun that would mislead go2/dev worlds.
    for noun in ("banana", "香蕉", "mug", "arm", "gripper"):
        assert noun not in intro


# ---------------------------------------------------------------------------
# 3. Validator acceptance: a bound pick plan PASSES decompose validation.
# ---------------------------------------------------------------------------


class _MockBackend:
    """Returns a fixed JSON plan; records the system prompt it was given."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_system: Any = None

    def call(self, messages, tools, system, max_tokens, on_text=None):  # noqa: ANN001
        self.last_system = system
        resp = self._response

        class _R:
            text = resp

        return _R()


_PICK_PLAN = {
    "goal": "抓香蕉",  # "grab the banana" — target named in CN, bound in EN param
    "sub_goals": [
        {
            "name": "grasp_banana",
            "description": "pick up the banana",
            "verify": "holding_object()",
            "strategy": "pick_skill",
            "depends_on": [],
            "strategy_params": {"object_label": "banana"},
        }
    ],
    "context_snapshot": "",
}


def _decomposer(plan: dict[str, Any]) -> GoalDecomposer:
    vocab = build_decompose_vocab(_arm_schemas(), _ARM_VERIFY_SIGS, has_base=False)
    return GoalDecomposer(
        _MockBackend(json.dumps(plan)),
        skill_registry=_arm_registry(),
        has_base=False,
        **vocab.as_kwargs(),
    )


def test_bound_pick_plan_passes_validation() -> None:
    gd = _decomposer(_PICK_PLAN)
    # The allowlist (single-sourced from the verify signatures) includes
    # holding_object, and pick_skill is a known strategy.
    assert "holding_object" in gd.VERIFY_FUNCTIONS
    assert "pick_skill" in gd.KNOWN_STRATEGIES

    tree = gd.decompose("抓香蕉", "tabletop scene")

    # No validator complaints; the plan survives intact.
    assert tree.validation_notes == ()
    assert len(tree.sub_goals) == 1
    step = tree.sub_goals[0]
    # The verify expression was accepted AND strengthened to the bound target
    # (backlog #2b): a named pick verifies "holding the REQUESTED object",
    # never the bare "holding anything" form that false-passes a wrong grab.
    assert step.verify == "holding_object('banana')"
    # The strategy is kept (no hallucination clearing).
    assert step.strategy == "pick_skill"
    assert step.cleared_strategy == ""
    # The target the planner extracted is BOUND to the skill's object param.
    assert step.strategy_params == {"object_label": "banana"}
