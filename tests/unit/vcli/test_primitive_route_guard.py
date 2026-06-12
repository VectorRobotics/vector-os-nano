# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Primitive-route executable-truth guard (owner finding (c), 2026-06-12).

The owner's '走到sofa' replan emitted strategy 'scan_360' — taught by the
derived vocab (has_base=True) — but NOTHING in production ever calls
init_primitives(), so the module-global primitive raised 'No hardware
connected' and the replan chased a ghost. Strategy routes to base primitives
must check the EXECUTABLE truth (primitives_ready), not just has_base.

Byte-identical guarantees kept:
- registry-less selectors (legacy go2, level44 harness) keep primitive routes;
- vocab teaching keeps old behaviour unless the engine passes the truth.
"""
from __future__ import annotations


from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import SubGoal


class _Registry:
    """Minimal registry double: the habitat mobile set."""

    def __init__(self, names=("walk", "turn", "stop", "navigate_to")):
        self._names = list(names)

    def list_skills(self):
        return list(self._names)

    def match(self, description):
        for n in self._names:
            if n.replace("_", " ") in description or n in description:
                class M:  # noqa: N801 — tiny inline double
                    skill_name = n
                return M()
        return None


def _sg(name, strategy="", params=None, description=""):
    return SubGoal(
        name=name,
        description=description or name,
        strategy=strategy,
        strategy_params=params or {},
        verify="True",
    )


class TestExplicitPrimitiveWithRegistry:
    def test_scan_360_not_executable_resolves_invalid(self):
        # The owner's exact replan step: bare 'scan_360' in a world whose
        # registry has no such skill and whose primitives are unwired.
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: False
        )
        result = sel.select(_sg("scan_area", strategy="scan_360"))
        assert result.executor_type == "invalid"
        assert "navigate_to" in result.params["valid_strategies"]

    def test_turn_not_executable_falls_to_registered_skill(self):
        # 'turn' is BOTH a primitive name and a registered skill — when the
        # primitive layer is unwired the registered skill must win.
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: False
        )
        result = sel.select(_sg("turn_around", strategy="turn", params={"angle": 1.0}))
        assert result.executor_type == "skill"
        assert result.name == "turn"

    def test_executable_primitives_keep_primitive_route(self):
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: True
        )
        result = sel.select(_sg("scan_area", strategy="scan_360"))
        assert result.executor_type == "primitive"

    def test_registryless_selector_byte_identical(self):
        # Legacy go2 path: no registry injected — primitive routes unchanged
        # even when the checker says not ready (level44 harness contract).
        sel = StrategySelector(primitives_executable=lambda: False)
        result = sel.select(_sg("scan_area", strategy="scan_360"))
        assert result.executor_type == "primitive"

    def test_default_checker_is_executable_truth(self):
        # No checker injected -> consults vcli.primitives readiness (unwired
        # in this test process => not ready => invalid for a registry world).
        sel = StrategySelector(skill_registry=_Registry())
        result = sel.select(_sg("scan_area", strategy="scan_360"))
        assert result.executor_type == "invalid"


class TestKeywordLadderWithRegistry:
    def test_walk_keyword_falls_to_skill_when_unwired(self):
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: False
        )
        result = sel.select(_sg("walk_forward_a_bit", description="walk forward"))
        assert result.executor_type == "skill"
        assert result.name == "walk"

    def test_stop_keyword_falls_to_skill_when_unwired(self):
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: False
        )
        result = sel.select(_sg("halt", description="stop moving"))
        assert result.executor_type == "skill"
        assert result.name == "stop"

    def test_keyword_ladder_keeps_primitives_when_wired(self):
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: True
        )
        result = sel.select(_sg("halt", description="stop moving"))
        assert result.executor_type == "primitive"
        assert result.name == "stop"


class TestSkillSuffixGuard:
    def test_suffix_primitive_exemption_requires_executable(self):
        # 'scan_360_skill' with unwired primitives and no such skill -> invalid
        # (the exemption used to accept any has_base primitive name).
        sel = StrategySelector(
            skill_registry=_Registry(), primitives_executable=lambda: False
        )
        result = sel.select(_sg("scan", strategy="scan_360_skill"))
        assert result.executor_type == "invalid"


class TestVocabTeaching:
    def test_teach_base_primitives_false_omits_them(self):
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            build_decompose_vocab,
        )

        schemas = [{"name": "navigate_to", "description": "go", "parameters": {}}]
        vocab = build_decompose_vocab(
            schemas, {"at_position": "at_position(x, y, tol)"},
            has_base=True, teach_base_primitives=False,
        )
        assert "scan_360" not in vocab.strategies
        assert "walk_forward" not in vocab.strategies

    def test_default_none_keeps_old_has_base_behaviour(self):
        from vector_os_nano.vcli.cognitive.vocab_from_registry import (
            build_decompose_vocab,
        )

        schemas = [{"name": "navigate_to", "description": "go", "parameters": {}}]
        vocab = build_decompose_vocab(
            schemas, {}, has_base=True,
        )
        assert "scan_360" in vocab.strategies


class TestPrimitivesReadyHelper:
    def test_unwired_process_reports_not_ready(self):
        from vector_os_nano.vcli.primitives import primitives_ready

        assert primitives_ready() is False

    def test_wired_context_reports_ready(self, monkeypatch):
        from vector_os_nano.vcli import primitives as prims

        monkeypatch.setattr(
            prims, "_ctx", prims.PrimitiveContext(base=object()), raising=True
        )
        assert prims.primitives_ready() is True
