# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S5a — single-sourced capability resolver: byte-identical to the 3 ad-hoc gates.

``resolve_capability_profile(agent)`` consolidates three scattered duck-typed
"what can this body do" checks (CLAUDE.md Rule 11 — no capability-by-code drift):
  - ``native_tools._build_motor_tools``  (``agent._base is not None`` -> offer navigate)
  - ``worlds/robot._agent_has_camera``   (perception/base/arm RGB-frame source)
  - ``engine._has_base``                 (``agent._base is not None`` -> nav vocab)

CONTRACT (behavior-preserving): the GATED flags ``has_base`` and ``camera`` MUST equal
the legacy ad-hoc results for EVERY world, so rewiring the three sites onto this
resolver is byte-identical (this test pins it; the unchanged full suite confirms it).
The ENRICHMENT flags ``has_arm`` / ``has_gripper`` / ``lidar`` reconcile runtime-OR-
declared (the go2+Piper runtime-attach: bare go2's manifest declares ``has_arm:false``
yet a Piper is bound at runtime -> the runtime member wins).
"""
from __future__ import annotations

import pytest

from vector_os_nano.embodiments.capability_profile import resolve_capability_profile
from vector_os_nano.embodiments.config import load_embodiment_config


# --- duck-typed fakes (no sim) mirroring the real driver member surface ---------

class _CamLidar:  # bare go2 / its base: RGB head cam + lidar scan
    def get_camera_frame(self):
        return object()

    def get_lidar_scan(self):
        return object()


class _CamOnly:  # g1-like base: camera, no lidar accessor
    def get_camera_frame(self):
        return object()


class _Perc:  # CameraPerception / arm perception: RGB-D color frame
    def get_color_frame(self):
        return object()


class _Agent:
    def __init__(self, base=None, arm=None, gripper=None, perception=None):
        self._base = base
        self._arm = arm
        self._gripper = gripper
        self._perception = perception


# --- the LEGACY ad-hoc checks, copied VERBATIM, as the byte-identical oracle -----

def _legacy_has_base(agent) -> bool:
    return getattr(agent, "_base", None) is not None


def _legacy_has_camera(agent) -> bool:  # verbatim copy of robot._agent_has_camera body
    if agent is None:
        return False
    perception = getattr(agent, "_perception", None)
    if perception is not None and callable(getattr(perception, "get_color_frame", None)):
        return True
    for member in (getattr(agent, "_base", None), getattr(agent, "_arm", None)):
        if member is not None and callable(getattr(member, "get_camera_frame", None)):
            return True
    return False


def _bare_go2():
    return _Agent(base=_CamLidar())


def _go2_arm():
    return _Agent(base=_CamLidar(), arm=object(), gripper=object(), perception=_Perc())


def _g1():
    return _Agent(base=_CamOnly())


def _arm_only():
    return _Agent(arm=object(), gripper=object(), perception=_Perc())


_WORLDS = {
    "bare_go2": _bare_go2,
    "go2_arm": _go2_arm,
    "g1": _g1,
    "arm_only": _arm_only,
    "dev_none": lambda: None,
}


# --- the byte-identical contract -------------------------------------------------

@pytest.mark.parametrize("name", list(_WORLDS))
def test_has_base_byte_identical_to_legacy(name):
    agent = _WORLDS[name]()
    assert resolve_capability_profile(agent).has_base == _legacy_has_base(agent)


@pytest.mark.parametrize("name", list(_WORLDS))
def test_camera_byte_identical_to_legacy(name):
    agent = _WORLDS[name]()
    assert resolve_capability_profile(agent).camera == _legacy_has_camera(agent)


def test_dev_world_all_false():
    prof = resolve_capability_profile(None)
    assert (prof.has_base, prof.has_arm, prof.has_gripper, prof.camera, prof.lidar) == (
        False, False, False, False, False,
    )


# --- the runtime-OR-declared reconcile (the enrichment flags) --------------------

def test_runtime_attach_overrides_declared_no_arm():
    """go2 manifest declares has_arm:false; a Piper bound at runtime -> has_arm True."""
    cfg = load_embodiment_config("go2")
    assert cfg.capabilities.has_arm is False and cfg.capabilities.has_gripper is False

    class _BaseWithCfg:
        _config = cfg

        def get_camera_frame(self):
            return object()

        def get_lidar_scan(self):
            return object()

    agent = _Agent(base=_BaseWithCfg(), arm=object(), gripper=object(), perception=_Perc())
    prof = resolve_capability_profile(agent)
    assert prof.has_arm is True       # runtime Piper wins over the manifest's false
    assert prof.has_gripper is True
    assert prof.has_base is True and prof.camera is True


def test_declared_enriches_when_runtime_absent():
    """A manifest declaring lidar:true, with a base lacking get_lidar_scan -> lidar True."""
    cfg = load_embodiment_config("go2")
    assert cfg.capabilities.lidar is True

    class _BaseDeclaredOnly:
        _config = cfg

        def get_camera_frame(self):
            return object()
        # NO get_lidar_scan at runtime

    prof = resolve_capability_profile(_Agent(base=_BaseDeclaredOnly()))
    assert prof.lidar is True  # declared manifest enriches when runtime can't prove it
