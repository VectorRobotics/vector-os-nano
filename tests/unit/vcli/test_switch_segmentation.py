# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""M4-B (ADR-012 Option A) — controller-level embodiment-switch segmentation.

A compound single sentence that BOTH switches embodiment and acts (e.g.
"切到go2，然后往前走一米") must be split at switch boundaries so the switch runs via
the proven top-level path (rebinding the active agent) BEFORE the residual is
routed on the NEW embodiment. This is the fix for the campaign-#12 M4-B 0/3:
switch is a @tool, structurally unreachable inside a single VGG decompose.

Two layers tested here:
1. ``IntentRouter.split_switch_segments`` — pure text → ordered segments | None.
2. ``VectorEngine._run_switch_segmented`` — runs each segment on the *current*
   (post-rebind) ``app_state['agent']`` (the stale-closure hazard fix), in order.
"""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.intent_router import IntentRouter, _is_embodiment_switch


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter()


class TestSplitSwitchSegments:
    def test_zh_switch_then_walk(self, router: IntentRouter) -> None:
        assert router.split_switch_segments("切到go2，然后往前走一米") == [
            "切到go2", "往前走一米",
        ]

    def test_en_switch_then_navigate(self, router: IntentRouter) -> None:
        assert router.split_switch_segments(
            "switch to g1, then navigate to x=2 y=1.5"
        ) == ["switch to g1", "navigate to x=2 y=1.5"]

    def test_zh_synonym_comma_move(self, router: IntentRouter) -> None:
        # comma-joined, no sequential word — CJK comma is a valid boundary
        assert router.split_switch_segments("换成go2，挪半米") == ["换成go2", "挪半米"]

    def test_switch_in_the_middle(self, router: IntentRouter) -> None:
        assert router.split_switch_segments(
            "往前走一米，然后切到go2，然后往前走一米"
        ) == ["往前走一米", "切到go2", "往前走一米"]

    def test_pure_switch_not_segmented(self, router: IntentRouter) -> None:
        # a single-step switch must stay on the proven top-level path, not segment
        assert router.split_switch_segments("切到go2") is None

    def test_non_switch_compound_not_segmented(self, router: IntentRouter) -> None:
        assert router.split_switch_segments("往前走一米，然后向左转") is None

    def test_switch_word_named_place_not_segmented(self, router: IntentRouter) -> None:
        # "切换机房" is a place named with a switch word — no embodiment target
        assert router.split_switch_segments("去切换机房，然后看一下") is None

    def test_single_action_not_segmented(self, router: IntentRouter) -> None:
        assert router.split_switch_segments("往前走一米") is None

    def test_empty_not_segmented(self, router: IntentRouter) -> None:
        assert router.split_switch_segments("") is None

    def test_coordinate_ascii_comma_not_oversplit(self, router: IntentRouter) -> None:
        # ASCII comma inside a coordinate must NOT be a split point (only CJK
        # punct + sequential word-connectors split) — the coord stays intact.
        assert router.split_switch_segments(
            "切到go2，然后navigate to x=2, y=1.5"
        ) == ["切到go2", "navigate to x=2, y=1.5"]

    def test_each_segment_is_not_itself_compound(self, router: IntentRouter) -> None:
        # no infinite recursion: every returned segment must split to None
        segs = router.split_switch_segments("切到go2，然后往前走一米")
        assert segs is not None
        for s in segs:
            assert router.split_switch_segments(s) is None


class TestEngineSwitchSegmentation:
    """_run_switch_segmented threads the REBOUND agent between segments."""

    def _bare_engine(self):
        from vector_os_nano.vcli.engine import VectorEngine
        return object.__new__(VectorEngine)  # bypass heavy __init__

    def test_segments_run_on_rebound_agent(self) -> None:
        eng = self._bare_engine()
        OLD, NEW = object(), object()
        app = {"agent": OLD}
        calls: list[tuple[str, object]] = []

        def fake_rtu(msg, session, agent=None, app_state=None, **kw):
            calls.append((msg, agent))
            # simulate switch_embodiment rebinding the active agent
            if _is_embodiment_switch(msg.lower()):
                app_state["agent"] = NEW
            return f"ran:{msg}"

        eng.run_turn_unified = fake_rtu  # shadow for the internal recursion
        out = eng._run_switch_segmented(
            ["切到go2", "往前走一米"], None, OLD,
            None, None, None, None, app, None,
        )
        # switch ran on the OLD agent; residual ran on the REBOUND (NEW) agent
        assert calls[0] == ("切到go2", OLD)
        assert calls[1] == ("往前走一米", NEW)
        assert out == "ran:往前走一米"  # returns the LAST segment's result
        assert app["agent"] is NEW

    def test_compound_switch_routes_to_segmentation(self) -> None:
        eng = self._bare_engine()
        eng._intent_router = IntentRouter()
        seen: dict[str, object] = {}

        def fake_seg(segs, *a, **k):
            seen["segs"] = segs
            return "SEG"

        eng._run_switch_segmented = fake_seg
        out = eng.run_turn_unified(
            "切到go2，然后往前走一米", None, app_state={"agent": object()},
        )
        assert out == "SEG"
        assert seen["segs"] == ["切到go2", "往前走一米"]

    def test_non_switch_does_not_segment(self) -> None:
        eng = self._bare_engine()
        eng._intent_router = IntentRouter()
        called = {"seg": False}
        eng._run_switch_segmented = lambda *a, **k: called.__setitem__("seg", True)

        def boom(_msg):
            raise RuntimeError("reached normal path")

        eng.classify_intent = boom
        with pytest.raises(RuntimeError, match="reached normal path"):
            eng.run_turn_unified("往前走一米", None, app_state={"agent": object()})
        assert called["seg"] is False

    def test_no_app_state_does_not_segment(self) -> None:
        # without app_state the agent cannot be rebound between segments, so a
        # compound switch must NOT segment (falls through to normal routing).
        eng = self._bare_engine()
        eng._intent_router = IntentRouter()
        eng._run_switch_segmented = lambda *a, **k: pytest.fail("must not segment")

        def boom(_msg):
            raise RuntimeError("normal path")

        eng.classify_intent = boom
        with pytest.raises(RuntimeError, match="normal path"):
            eng.run_turn_unified("切到go2，然后往前走一米", None, app_state=None)
