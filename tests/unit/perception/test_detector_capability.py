# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""DetectorCapability — protocol conformance + invoke/estimate (mocked, no torch)."""
from __future__ import annotations

import numpy as np

from vector_os_nano.interfaces.perception import Detection
from vector_os_nano.perception.detector_capability import DetectorCapability
from vector_os_nano.vcli.cognitive.capabilities.types import (
    Capability,
    CapabilityResult,
)


class _FakeDetector:
    """A detector stand-in — never loads a model."""

    def __init__(self, dets=None):
        self._dets = dets if dets is not None else [
            Detection(label="can", bbox=(10.0, 20.0, 30.0, 40.0), confidence=0.9)
        ]
        self.calls: list[tuple] = []

    def detect(self, rgb, query):
        self.calls.append((rgb.shape, query))
        return list(self._dets)


def _rgb():
    return np.zeros((240, 320, 3), dtype=np.uint8)


# --- protocol conformance ---------------------------------------------------


def test_satisfies_capability_protocol():
    cap = DetectorCapability(detector=_FakeDetector())
    assert isinstance(cap, Capability)
    assert cap.name == "detect"
    assert cap.kind == "detector"
    assert cap.side_effecting is False
    assert "required" in cap.input_schema and "query" in cap.input_schema["required"]
    assert "detections" in cap.output_schema["properties"]


def test_estimate_returns_float_pair():
    est = DetectorCapability(detector=_FakeDetector()).estimate({"query": "can"})
    assert isinstance(est, tuple) and len(est) == 2
    assert all(isinstance(v, float) for v in est)


# --- invoke -----------------------------------------------------------------


def test_invoke_missing_query_errors_without_running_detector():
    fake = _FakeDetector()
    cap = DetectorCapability(detector=fake)
    res = cap.invoke({"rgb": _rgb()}, None)
    assert isinstance(res, CapabilityResult)
    assert res.success is False
    assert "query" in res.error
    assert fake.calls == []  # detector never ran


def test_invoke_no_frame_errors():
    res = DetectorCapability(detector=_FakeDetector()).invoke({"query": "can"}, None)
    assert res.success is False
    assert "RGB" in res.error or "rgb" in res.error


def test_invoke_with_rgb_returns_detections():
    fake = _FakeDetector()
    res = DetectorCapability(detector=fake).invoke({"query": "can", "rgb": _rgb()}, None)
    assert res.success is True
    assert len(res.output["detections"]) == 1
    assert res.output["labels"] == ["can"]
    assert res.output["scores"] == [0.9]
    assert res.output["boxes"] == [[10.0, 20.0, 30.0, 40.0]]
    assert fake.calls and fake.calls[0][1] == "can"


def test_invoke_pulls_frame_from_perception_handle():
    fake = _FakeDetector()

    class _Perc:
        def get_color_frame(self):
            return _rgb()

    res = DetectorCapability(detector=fake).invoke(
        {"query": "bottle", "perception": _Perc()}, None
    )
    assert res.success is True
    assert fake.calls and fake.calls[0][0] == (240, 320, 3)


def test_invoke_pulls_frame_from_context():
    fake = _FakeDetector()

    class _Ctx:
        def get_color_frame(self):
            return _rgb()

    res = DetectorCapability(detector=fake).invoke({"query": "cup"}, _Ctx())
    assert res.success is True


def test_invoke_pulls_frame_from_registration_bound_perception():
    """R36: the producer-routed detect step reaches the live camera via the
    REGISTRATION-BOUND perception. The kernel hands the capability a SkillContext
    (no frame), so without this bound source the routed detector returns 'no RGB
    frame' — exactly the gap the real-sim probe first hit before binding."""
    fake = _FakeDetector()

    class _Perc:
        def get_color_frame(self):
            return _rgb()

    cap = DetectorCapability(detector=fake, perception=_Perc())
    # payload has NO rgb/perception, context is None — only the bound source can
    # supply the frame.
    res = cap.invoke({"query": "bottle"}, None)
    assert res.success is True
    assert fake.calls and fake.calls[0][0] == (240, 320, 3)


def test_invoke_pulls_frame_from_skillcontext_perception_attr():
    """A SkillContext exposes its perception via ``.perception`` (not
    get_color_frame directly). The capability unwraps it so a context-only call
    still reaches a frame — the kernel capability path passes a SkillContext."""
    fake = _FakeDetector()

    class _Perc:
        def get_color_frame(self):
            return _rgb()

    class _SkillCtxLike:
        perception = _Perc()

    res = DetectorCapability(detector=fake).invoke({"query": "cup"}, _SkillCtxLike())
    assert res.success is True
    assert fake.calls and fake.calls[0][0] == (240, 320, 3)


def test_payload_rgb_beats_bound_perception():
    """Source priority: an explicit payload rgb wins over the bound perception."""
    fake = _FakeDetector()

    class _Perc:
        called = False

        def get_color_frame(self):
            _Perc.called = True
            return _rgb()

    cap = DetectorCapability(detector=fake, perception=_Perc())
    res = cap.invoke({"query": "x", "rgb": _rgb()}, None)
    assert res.success is True
    assert _Perc.called is False  # bound source never consulted


def test_invoke_empty_detections_reports_ran_but_unsuccessful():
    fake = _FakeDetector(dets=[])
    res = DetectorCapability(detector=fake).invoke({"query": "ghost", "rgb": _rgb()}, None)
    # ran cleanly but found nothing -> success False (the verify decides real outcome)
    assert res.success is False
    assert res.output["detections"] == []


def test_invoke_detector_error_is_failure():
    class _Boom:
        def detect(self, rgb, query):
            raise RuntimeError("cuda oom")

    res = DetectorCapability(detector=_Boom()).invoke({"query": "can", "rgb": _rgb()}, None)
    assert res.success is False
    assert "cuda oom" in res.error


def test_cold_turn_rebind_pulls_live_agent_perception():
    """R37 Task B: registration runs BEFORE the NL sim-start boots the camera, so the
    agent's _perception is None at bind time. Binding the AGENT (not the snapshot)
    lets the capability pull the LIVE _perception lazily — so a cold turn (sim
    started this same turn) perceives with no pre-boot."""
    fake = _FakeDetector()

    class _Perc:
        def get_color_frame(self):
            return _rgb()

    class _Agent:
        _perception = None  # None at registration (cold)

    agent = _Agent()
    # Snapshot perception is None (the cold-turn gap); bind the agent instead.
    cap = DetectorCapability(detector=fake, perception=None, agent=agent)

    # Before the sim-start: no frame source -> the routed detect cannot perceive.
    res_cold = cap.invoke({"query": "bottle"}, None)
    assert res_cold.success is False
    assert "RGB" in res_cold.error or "rgb" in res_cold.error

    # Mid-session NL sim-start populates the agent's live perception.
    agent._perception = _Perc()

    # Same capability instance now perceives via the LIVE agent perception.
    res_warm = cap.invoke({"query": "bottle"}, None)
    assert res_warm.success is True
    assert fake.calls and fake.calls[-1][0] == (240, 320, 3)


def test_snapshot_perception_wins_over_agent():
    """A bound snapshot perception is preferred over the agent's (warm path identical)."""
    fake = _FakeDetector()

    class _Snap:
        used = False

        def get_color_frame(self):
            _Snap.used = True
            return _rgb()

    class _AgentPerc:
        used = False

        def get_color_frame(self):
            _AgentPerc.used = True
            return _rgb()

    class _Agent:
        _perception = _AgentPerc()

    cap = DetectorCapability(detector=fake, perception=_Snap(), agent=_Agent())
    res = cap.invoke({"query": "x"}, None)
    assert res.success is True
    assert _Snap.used is True
    assert _AgentPerc.used is False  # snapshot wins; agent never consulted


def test_detector_signature_reads_only_rgb_and_query():
    """Structural moat: the detector contract takes ONLY (rgb, query) — no GT pose."""
    import inspect

    from vector_os_nano.perception.grounding_dino import GroundingDinoDetector

    params = list(inspect.signature(GroundingDinoDetector.detect).parameters)
    assert params == ["self", "rgb", "query"]
