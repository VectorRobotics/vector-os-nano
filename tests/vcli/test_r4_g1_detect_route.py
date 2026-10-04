# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R4/R5 — route the learned grounding-dino DETECTOR to g1's HEAD camera (no-arm).

Covers the NON-cognitive wiring that lets a bare-cli detect command reach the
SECOND model family on the SECOND embodiment (cross-EMBODIMENT x cross-MODEL):

  1. ``RobotWorld.build_verify_namespace`` does NOT bind a ``detect_objects`` oracle
     for a camera-only (no-arm) agent (R5 CORRECTION / D61): grounding the verify on
     the detector's OWN stashed output (``agent._last_detection``) was a self-read
     TAUTOLOGY that produced a FALSE GREEN (GROUNDED). With no such oracle, the g1
     detect step grades RAN-honest — the moat never greens a self-read.
  2. The native ``_NativeDetectTool`` routes a query to the registered capability,
     stashes the boxes on the agent, and reports them (no second model load). KEPT:
     this is the genuine cross-EMBODIMENT x cross-MODEL route.
  3. ``_registered_capabilities`` reaches the live CapabilityRegistry the ENGINE owns;
     ``_build_motor_tools`` surfaces ``detect`` only when registered.

The grounding-dino model is FAKED (a stub capability) — these tests prove the
WIRING/route, not the model's perception (that is the real-sim REAL-VERIFY).
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.native_loop import (
    _NativeDetectTool,
    _build_motor_tools,
    _registered_capabilities,
)
from vector_os_nano.vcli.worlds.robot import RobotWorld


# --- fakes -----------------------------------------------------------------
class _Box:
    """A CapabilityResult-shaped stub (success + output)."""

    def __init__(self, success: bool, output: dict[str, Any]):
        self.success = success
        self.output = output
        self.error = ""


class _StubDetectCap:
    """Stands in for the registered grounding-dino DetectorCapability."""

    name = "detect"
    kind = "detector"
    side_effecting = False

    def __init__(self, detections):
        self._detections = detections
        self.last_query = None

    def invoke(self, payload, context):
        self.last_query = payload.get("query")
        if not self._detections:
            return _Box(False, {"boxes": [], "labels": [], "scores": []})
        return _Box(
            True,
            {
                "boxes": [list(d[1]) for d in self._detections],
                "labels": [d[0] for d in self._detections],
                "scores": [d[2] for d in self._detections],
            },
        )


class _G1Base:
    def get_camera_frame(self, width=640, height=480):
        import numpy as np

        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_position(self):
        return (10.0, 3.0, 0.79)

    def get_heading(self):
        return 0.0


class _G1Agent:
    def __init__(self):
        self._arm = None
        self._base = _G1Base()
        self._perception = object()  # presence only; the stub cap ignores it


class _Ctx:
    def __init__(self, agent):
        self.agent = agent


class _Engine:
    def __init__(self, cap=None, registry=None):
        from vector_os_nano.vcli.cognitive.capabilities import CapabilityRegistry

        self._capability_registry = CapabilityRegistry() if cap is not None else None
        if cap is not None:
            self._capability_registry.register(cap)
        self._registry = registry  # for code-tool discovery (None -> none)


# --- 1. verify oracle: NO self-read detect_objects for g1 (R5 CORRECTION) ---
def _g1_namespace():
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim

    agent = _G1Agent()
    sim = bind_fake_sim(agent, "g1")
    try:
        return RobotWorld().build_verify_namespace(agent)
    finally:
        unbind_fake_sim(sim)


def test_g1_verify_namespace_has_no_self_read_detect_objects():
    """R5/D61: a camera-only (no-arm) g1 gets NO detect_objects oracle.

    Grounding the verify on the detector's own stashed output is a self-read
    tautology (the FALSE GREEN red-team caught in R4). WP10: detection is graded by
    the GT ``in_view`` (segmentation of the robot's own camera) instead; base
    predicates remain.
    """
    ns = _g1_namespace()
    assert "detect_objects" not in ns  # self-read oracle removed
    assert "in_view" in ns and "at_position" in ns  # independent GT


def test_g1_detect_verify_classifies_RAN_not_grounded():
    """The moat-level proof: with no detect_objects oracle in the g1 namespace,
    ``len(detect_objects()) > 0`` classifies RAN — fail-closed to the honest
    D50 grade, NEVER a GROUNDED on a self-read."""
    from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr

    oracle_names = frozenset(_g1_namespace())  # single-sourced from the live verify-namespace
    assert classify_verify_expr("len(detect_objects()) > 0", oracle_names) == "RAN"
    # INDEPENDENT oracles (live GT, not the means) still grade GROUNDED.
    assert classify_verify_expr("at_position(10.0, 3.0)", oracle_names) == "GROUNDED"
    assert classify_verify_expr("in_view('red can')", oracle_names) == "GROUNDED"


def test_no_agent_gets_a_vacuous_detect_objects():
    """WP10 (detect-objects-vacuous-oracle): the old GT ``detect_objects`` listed every
    free body (incl. the robot's own base_link), so ``len(detect_objects()) > 0`` was
    always GROUNDED. No world binds it any more — an arm agent gets in_view instead."""
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim

    class _Arm:
        def get_object_positions(self):
            return {"pickable_can_red": (1.0, 2.0, 0.3)}

    class _ArmAgent:
        def __init__(self):
            self._arm = _Arm()
            self._base = _G1Base()
            self._perception = None

    agent = _ArmAgent()
    sim = bind_fake_sim(agent, "go2_piper")
    try:
        ns = RobotWorld().build_verify_namespace(agent)
    finally:
        unbind_fake_sim(sim)
    assert "detect_objects" not in ns and "describe_scene" not in ns
    assert "in_view" in ns and "localized" in ns


# --- 2. native detect tool routes to the capability ------------------------
def test_native_detect_tool_routes_and_does_not_self_stash():
    """The native detect tool ROUTES to the learned capability and surfaces a
    human-readable result, but MUST NOT stash the detector's own output on the
    agent (R6 moat discipline): a ``_last_detection`` stash that a verify oracle
    could read back is self-certification — the FALSE GREEN removed at D61. The
    detector's output flows ONLY to the human-readable tool text, never to verify.
    """
    cap = _StubDetectCap([("a red object", (100, 120, 260, 380), 0.73)])
    tool = _NativeDetectTool(cap)
    agent = _G1Agent()
    res = tool.execute({"query": "找前面的红色的东西"}, _Ctx(agent))
    assert not res.is_error
    assert cap.last_query == "找前面的红色的东西"  # genuinely routed to the learned model
    assert "grounding-dino localized 1 object" in res.content  # output flows to the human
    # WP10: the suggested verify is the GT in_view (segmentation), never a detection count.
    assert "verify(in_view('找前面的红色的东西'))" in res.content
    assert "detect_objects" not in res.content and "detection_matches_gt" not in res.content
    # MOAT: the means' own output is NOT stashed anywhere a verify oracle can read it.
    assert not hasattr(agent, "_last_detection")


def test_native_detect_tool_empty_query_errors():
    tool = _NativeDetectTool(_StubDetectCap([]))
    res = tool.execute({"query": "  "}, _Ctx(_G1Agent()))
    assert res.is_error


def test_native_detect_tool_nothing_localized_is_honest():
    cap = _StubDetectCap([])  # detector ran, found nothing
    tool = _NativeDetectTool(cap)
    agent = _G1Agent()
    res = tool.execute({"query": "red object"}, _Ctx(agent))
    assert not res.is_error
    assert "localized nothing" in res.content  # honest empty, never fabricated
    # WP10: no false promise about the GT ('it will be False'): the object may well be
    # in view even when the detector missed it.
    assert "it will be False" not in res.content and "verify(in_view('red object'))" in res.content
    # MOAT: even on an empty detection, nothing is stashed for a verify oracle.
    assert not hasattr(agent, "_last_detection")


# --- 3. registry reach + motor-tool surfacing ------------------------------
def test_registered_capabilities_reach_the_engine_registry():
    cap = _StubDetectCap([])
    assert _registered_capabilities(_Engine(cap=cap)) == [cap]
    assert _registered_capabilities(_Engine(cap=None)) == []  # no registry


def test_build_motor_tools_surfaces_detect_when_registered():
    cap = _StubDetectCap([("x", (0, 0, 1, 1), 0.5)])
    tools = _build_motor_tools(_G1Agent(), _Engine(cap=cap))
    assert "detect" in tools
    assert isinstance(tools["detect"], _NativeDetectTool)
    assert "navigate" in tools  # g1 has a base -> the avoidance route too


def test_build_motor_tools_no_detect_when_unregistered():
    tools = _build_motor_tools(_G1Agent(), _Engine(cap=None))
    assert "detect" not in tools
