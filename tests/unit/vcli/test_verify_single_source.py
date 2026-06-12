# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Invariant III — verify single source (design review #5/#17, rule 3).

Two hand-written verify vocabularies existed: the engine fast path's
_VERIFY_MAP and the skills' verify_hint — both defaulting to the 'True'
sentinel, so motion commands were structurally unverified on the fast path.
Now: the skill is the ONLY source. Fast path reads `skill.verify_template`
(optional, '{arg}' substitution); `to_schemas` never coerces a missing hint
to 'True' — it tags the schema `unverified: True` and the vocab surfaces
that to the planner instead of putting the sentinel in its mouth.
"""
from __future__ import annotations

from vector_os_nano.core.skill import SkillRegistry, skill
from vector_os_nano.vcli.engine import VectorEngine


@skill
class _Hinted:
    name: str = "hinted"
    description: str = "has a real predicate"
    verify_hint: str = "at_position(1, 2)"
    parameters: dict = {}
    preconditions: list = []
    postconditions: list = []
    effects: dict = {}

    def execute(self, params, context):  # pragma: no cover
        return None


@skill
class _Bare:
    name: str = "bare"
    description: str = "declares nothing"
    parameters: dict = {}
    preconditions: list = []
    postconditions: list = []
    effects: dict = {}

    def execute(self, params, context):  # pragma: no cover
        return None


@skill
class _Sentinel:
    name: str = "sentinel"
    description: str = "declares the sentinel"
    verify_hint: str = "True"
    parameters: dict = {}
    preconditions: list = []
    postconditions: list = []
    effects: dict = {}

    def execute(self, params, context):  # pragma: no cover
        return None


def _schema(cls):
    reg = SkillRegistry()
    reg.register(cls())
    return reg.to_schemas()[0]


class TestToSchemasNoSilentSentinel:
    def test_real_hint_passes_through_unflagged(self):
        s = _schema(_Hinted)
        assert s["verify_hint"] == "at_position(1, 2)"
        assert not s.get("unverified")

    def test_missing_hint_is_tagged_not_coerced(self):
        s = _schema(_Bare)
        assert s["verify_hint"] == ""
        assert s.get("unverified") is True

    def test_sentinel_hint_is_tagged(self):
        s = _schema(_Sentinel)
        assert s.get("unverified") is True


class TestFastPathSingleSource:
    class _Tpl:
        verify_template = "nearest_room() == '{arg}'"

    class _Plain:
        verify_template = "holding_object()"

    class _NoTpl:
        pass

    def test_template_with_arg_substituted(self):
        v = VectorEngine._verify_for_skill("navigate", "kitchen", self._Tpl())
        assert "nearest_room() == 'kitchen'" in v

    def test_arg_template_without_arg_falls_to_sentinel(self):
        # No room extracted -> no honest predicate can be formed.
        assert VectorEngine._verify_for_skill("navigate", "", self._Tpl()) == "True"

    def test_plain_template_used_verbatim(self):
        v = VectorEngine._verify_for_skill("pick", "", self._Plain())
        assert "holding_object()" in v

    def test_no_template_is_honest_sentinel(self):
        assert VectorEngine._verify_for_skill("walk", "", self._NoTpl()) == "True"

    def test_verify_map_is_gone(self):
        # The second hand-written vocabulary must not exist anywhere.
        import inspect

        import vector_os_nano.vcli.engine as eng

        assert "_VERIFY_MAP" not in inspect.getsource(eng)


class TestSkillTemplates:
    def test_motion_adjacent_skills_declare_templates(self):
        from vector_os_nano.skills.go2.look import LookSkill
        from vector_os_nano.skills.navigate import NavigateSkill
        from vector_os_nano.skills.pick import PickSkill

        assert "nearest_room()" in NavigateSkill.verify_template
        assert "describe_scene()" in LookSkill.verify_template
        assert "holding_object()" in PickSkill.verify_template


class TestVocabSurfacesUnverified:
    def test_params_help_marks_unverified_instead_of_true(self):
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            build_decompose_vocab,
        )

        vocab = build_decompose_vocab(
            [{"name": "bare", "description": "d", "parameters": {},
              "verify_hint": "", "unverified": True}],
            {"at_position": "at_position(x, y)"},
            has_base=True, teach_base_primitives=False,
        )
        help_text = vocab.strategy_params_help
        assert "unverified" in help_text
        assert "suggested verify: True" not in help_text


class TestRobotWorldExemptionShrunk:
    def test_skills_with_real_predicates_no_longer_exempt(self):
        from vector_os_nano.vcli.worlds.robot import RobotWorld

        exempt = RobotWorld().evidence_exempt_strategies()
        assert "navigate" not in exempt
        assert "look" not in exempt
        assert "describe_scene" not in exempt
        # Motion skills keep the transitional exemption until batch 2 lands
        # the moved/duration result contract.
        assert "walk" in exempt and "explore" in exempt
