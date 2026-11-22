# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RobotWorld.register_capabilities — registers the learned 'detect' capability.

Guarded two ways (keeps dev/sensorless/CI byte-identical):
  - only when the agent has a CAMERA to detect on — a perception with
    ``get_color_frame()`` OR a base/arm with ``get_camera_frame()`` (R4: the
    detector needs a camera, NOT an arm — a go2+arm grasps with it, a g1 (camera,
    no arm) localizes with it);
  - only when torch/transformers are importable.
The model is NOT loaded by registration (DetectorCapability is lazy), so this
runs without weights. We monkeypatch around the torch import-guard so the test
is deterministic whether or not torch is installed.
"""
from __future__ import annotations

import sys

from vector_os_nano.vcli.cognitive.capabilities.registry import CapabilityRegistry
from vector_os_nano.vcli.worlds.robot import RobotWorld


class _Perc:
    def get_color_frame(self):
        import numpy as np

        return np.zeros((240, 320, 3), dtype=np.uint8)


class _ArmAgent:
    """go2+arm: an arm AND a bound head perception (CameraPerception in the real
    path always supplies get_color_frame) — the camera the detector grasps with."""

    def __init__(self, perception=None):
        self._arm = object()  # has an arm
        # Real go2+arm always binds a perception with get_color_frame; default to one
        # so the camera-presence gate (R4) sees a frame source, matching production.
        self._perception = perception if perception is not None else _Perc()


class _BaseOnlyAgent:
    def __init__(self):
        self._arm = None  # base only, no arm, NO camera (sensorless base)


class _G1Camera:
    """A MuJoCoG1-shaped base: a head camera (get_camera_frame), no arm."""

    def get_camera_frame(self, width=640, height=480):
        import numpy as np

        return np.zeros((height, width, 3), dtype=np.uint8)


class _G1Agent:
    """R4: g1 — a CAMERA-bearing base, NO arm. Bound perception OR raw base camera."""

    def __init__(self, perception=None, with_base_camera=True):
        self._arm = None
        self._base = _G1Camera() if with_base_camera else None
        self._perception = perception


def _torch_present() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def test_arm_agent_registers_detect_when_torch_present():
    if not _torch_present():
        # Without torch the guard registers nothing — assert that, then done.
        reg = CapabilityRegistry()
        RobotWorld().register_capabilities(reg, _ArmAgent(), backend=None)
        assert "detect" not in reg.names()
        return
    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, _ArmAgent(), backend=None)
    assert "detect" in reg.names()
    cap = reg.get("detect")
    assert cap.kind == "detector" and cap.side_effecting is False
    # registration must NOT have loaded the model (lazy)
    assert cap._detectors == {}


def test_detect_capability_bound_to_agent_perception():
    """R36: registration BINDS the agent's live RGB source into the capability so
    a producer-routed detect step can perceive on the capability-dispatch path
    (the kernel context is a SkillContext with no frame)."""
    if not _torch_present():
        return  # registration is a no-op without torch; covered elsewhere
    perc = _Perc()
    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, _ArmAgent(perception=perc), backend=None)
    cap = reg.get("detect")
    assert cap is not None
    assert getattr(cap, "_perception", None) is perc


def test_sensorless_base_registers_nothing():
    """A base with NO camera (no get_camera_frame, no perception) registers nothing —
    the camera-presence gate keeps a sensorless agent byte-identical."""
    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, _BaseOnlyAgent(), backend=None)
    assert "detect" not in reg.names()
    assert len(reg) == 0


def test_g1_camera_base_registers_detect_via_raw_camera():
    """R4: a camera-bearing base (g1: get_camera_frame, NO arm) registers the detector.
    The camera gate is satisfied by the raw base camera even without a bound perception."""
    if not _torch_present():
        reg = CapabilityRegistry()
        RobotWorld().register_capabilities(reg, _G1Agent(), backend=None)
        assert "detect" not in reg.names()
        return
    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, _G1Agent(), backend=None)
    assert "detect" in reg.names()
    cap = reg.get("detect")
    assert cap.kind == "detector" and cap.side_effecting is False
    assert cap._detectors == {}  # lazy, no model load


def test_g1_bound_perception_wins_as_frame_source():
    """R4: g1's bound head perception (get_color_frame) is the capability's frame source."""
    if not _torch_present():
        return
    perc = _Perc()
    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, _G1Agent(perception=perc), backend=None)
    cap = reg.get("detect")
    assert cap is not None and getattr(cap, "_perception", None) is perc


def test_no_agent_registers_nothing():
    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, None, backend=None)
    assert len(reg) == 0


def test_missing_torch_registers_nothing(monkeypatch):
    """Simulate torch absent: the ImportError guard registers nothing (CI-safe)."""
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"simulated-absent: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    # Drop any already-imported module so the guarded import actually re-runs.
    for mod in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, mod, None)

    reg = CapabilityRegistry()
    RobotWorld().register_capabilities(reg, _ArmAgent(), backend=None)
    assert "detect" not in reg.names()
