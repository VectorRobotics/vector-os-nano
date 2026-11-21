# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Single-source runtime capability resolver (CLAUDE.md Rule 11 — no drift).

Three scattered sites used to answer "what can this body do" by ad-hoc duck-typed
``getattr`` probing — the exact capability-by-code drift Rule 11 forbids:
  - ``native_loop._build_motor_tools``  (``agent._base is not None`` -> offer navigate)
  - ``worlds/robot._agent_has_camera``   (perception/base/arm RGB-frame source)
  - ``engine._has_base``                 (``agent._base is not None`` -> nav vocab)

This module consolidates them into ONE resolver returning the already-declared
``CapabilityProfile`` (``embodiments/config.py``), so the question is asked in exactly
one place against one schema.

Authority (deliberate, behavior-preserving — S5a):
  - The GATED flags ``has_base`` and ``camera`` are derived from the SAME runtime
    presence the three sites use today, so rewiring them onto this resolver is
    BYTE-IDENTICAL (pinned by ``tests/unit/vcli/test_capability_profile.py`` and the
    unchanged full suite).
  - The ENRICHMENT flags ``has_arm`` / ``has_gripper`` / ``lidar`` reconcile
    runtime-OR-declared: present iff the live agent has the member OR the body's
    ``robot.yaml`` manifest declares it. This makes the go2+Piper runtime-attach
    honest — bare go2's manifest declares ``has_arm:false`` yet a Piper is bound at
    runtime, so the runtime member wins. No current gate reads these flags, so the
    reconcile cannot change today's behavior; it is the forward seam for S5b+.

Shifting the GATED flags' authority to the declared manifest (so a body's offered
toolset follows its robot.yaml even before runtime members bind) is a deliberate
FUTURE step — a behavior change when declared and runtime diverge — NOT this round.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.schemas.robot import CapabilityProfile, EmbodimentConfig


def _has_member(agent: Any, name: str) -> bool:
    return getattr(agent, name, None) is not None


def _runtime_camera(agent: Any) -> bool:
    """True iff *agent* can supply an RGB frame — the ``_agent_has_camera`` logic (R4).

    A frame source is a bound ``_perception`` exposing ``capture()`` (a body camera)
    or ``get_color_frame()``, OR a base/arm exposing ``get_camera_frame()``. Reaching
    it through the SAME duck-typed accessors the detector uses keeps this embodiment-
    agnostic — the detector needs a camera, not an arm.
    """
    perception = getattr(agent, "_perception", None)
    if perception is not None and any(callable(getattr(perception, m, None)) for m in ("capture", "get_color_frame")):
        return True
    for member in (getattr(agent, "_base", None), getattr(agent, "_arm", None)):
        if member is not None and callable(getattr(member, "get_camera_frame", None)):
            return True
    return False


def _runtime_lidar(agent: Any) -> bool:
    """True iff a base/arm exposes ``get_lidar_scan()`` at runtime."""
    for member in (getattr(agent, "_base", None), getattr(agent, "_arm", None)):
        if member is not None and callable(getattr(member, "get_lidar_scan", None)):
            return True
    return False


def _declared_profile(agent: Any) -> CapabilityProfile | None:
    """The body's declared ``CapabilityProfile`` from its robot.yaml, if the driver loaded one.

    The generic driver loads its manifest into ``self._config`` (an ``EmbodimentConfig``);
    a body without a manifest (e.g. the SO-101 arm) has no ``_config`` -> None.
    """
    base = getattr(agent, "_base", None)
    cfg = getattr(base, "_config", None)
    if isinstance(cfg, EmbodimentConfig):
        return cfg.capabilities
    return None


def resolve_capability_profile(agent: Any) -> CapabilityProfile:
    """The EFFECTIVE capabilities of *agent* — the ONE source the capability gates consult.

    ``agent is None`` (the dev world) -> everything False (no robot capabilities).
    """
    if agent is None:
        return CapabilityProfile(
            has_base=False, has_arm=False, has_gripper=False, camera=False, lidar=False
        )
    declared = _declared_profile(agent)
    d_arm = declared.has_arm if declared is not None else False
    d_gripper = declared.has_gripper if declared is not None else False
    d_lidar = declared.lidar if declared is not None else False
    return CapabilityProfile(
        # GATED — runtime authoritative (byte-identical to engine._has_base /
        # native_loop navigate gate / robot._agent_has_camera).
        has_base=_has_member(agent, "_base"),
        camera=_runtime_camera(agent),
        # ENRICHMENT — runtime OR declared (handles the go2+Piper runtime-attach).
        has_arm=_has_member(agent, "_arm") or d_arm,
        has_gripper=_has_member(agent, "_gripper") or d_gripper,
        lidar=_runtime_lidar(agent) or d_lidar,
    )
