# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for _build_go2_perception (Spec 3 — perception wiring guard).

The go2 SimSession (behind NL start_simulation and --sim-go2) must wire a REAL RGB-D + detector + segmenter perception backend
onto agent._perception so look/explore can depth-localize VLM-named objects to
accurate world (x, y, z) via localize_objects_3d.

Guard contract: only wire perception when the base exposes the full RGB-D frame
source (get_camera_frame / get_depth_frame / get_camera_pose).  A base without a
camera (or any construction failure) leaves perception None so a launch never
crashes and non-perception embodiments stay unaffected.

No MuJoCo, no GPU, no network — pure stub bases.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.sim.session import _build_go2_perception as _build


def _build_go2_perception(base: Any) -> Any:
    return _build(base, width=320, height=240)


class _FullCameraBase:
    """Stub base exposing the complete RGB-D frame source."""

    def get_camera_frame(self, width: int = 320, height: int = 240) -> Any:
        return None

    def get_depth_frame(self, width: int = 320, height: int = 240) -> Any:
        return None

    def get_camera_pose(self) -> tuple:
        return ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


class _NoCameraBase:
    """Stub locomotion-only base with NO camera (e.g. a bare driver)."""

    def get_position(self) -> tuple:
        return (0.0, 0.0, 0.0)


class _MissingDepthBase:
    """Has camera + pose but no depth frame — must NOT wire perception."""

    def get_camera_frame(self, width: int = 320, height: int = 240) -> Any:
        return None

    def get_camera_pose(self) -> tuple:
        return ((0.0, 0.0, 0.0), tuple([1.0] + [0.0] * 8))


def test_build_perception_with_full_camera_base():
    """A base with the full RGB-D frame source yields a Go2GraspPerception."""
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception

    base = _FullCameraBase()
    perception = _build_go2_perception(base)
    assert isinstance(perception, Go2GraspPerception)
    # The perception wraps the exact base it was given, at the manifest size.
    assert perception._base is base
    assert (perception._w, perception._h) == (320, 240)


def test_build_perception_none_when_base_is_none():
    """base=None -> perception None (never crash)."""
    assert _build_go2_perception(None) is None


def test_build_perception_none_when_no_camera():
    """A locomotion-only base (no camera) -> perception None."""
    assert _build_go2_perception(_NoCameraBase()) is None


def test_build_perception_none_when_missing_depth():
    """A base missing get_depth_frame -> perception None (guard rejects it)."""
    assert _build_go2_perception(_MissingDepthBase()) is None


def test_built_perception_is_localizer_usable():
    """The wired perception satisfies localize_objects_3d's usability guard
    (exposes every required method), so depth-localization can run end to end."""
    from vector_os_nano.perception.object_localizer import _perception_is_usable

    perception = _build_go2_perception(_FullCameraBase())
    assert perception is not None
    assert _perception_is_usable(perception) is True
