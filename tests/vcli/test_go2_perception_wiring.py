# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The session's perception wiring (sim/session._devices + _perception).

A SimSession wires ONE RGB-D + detector + segmenter perception over the manifest's
camera onto agent._perception, so look/explore can depth-localize VLM-named objects to
world (x, y, z) via localize_objects_3d. A manifest without a camera leaves perception
None, so a launch never crashes and non-perception bodies stay unaffected.

No MuJoCo, no GPU, no network: the camera driver only stores its runtime until a capture.
"""
from __future__ import annotations

from vector_os_nano.embodiments import registry
from vector_os_nano.perception.camera_perception import CameraPerception
from vector_os_nano.sim.drivers.camera import Camera
from vector_os_nano.sim.session import _perception
from vector_os_nano.sim.devices import devices as _devices


def test_a_manifest_camera_yields_one_camera_perception_at_its_size():
    cfg = registry.load("go2")
    arm, gripper, camera = _devices(cfg, rt=object())
    assert arm is None and gripper is None and isinstance(camera, Camera)
    assert (camera._spec.name, camera._spec.width, camera._spec.height) == ("d435_rgb", 640, 480)
    perception = _perception(camera)
    assert isinstance(perception, CameraPerception) and perception._camera is camera


def test_no_camera_means_no_perception():
    assert _perception(None) is None


def test_built_perception_is_localizer_usable():
    """The wired perception satisfies localize_objects_3d's usability guard (it exposes
    every required method), so depth-localization can run end to end."""
    from vector_os_nano.perception.object_localizer import _perception_is_usable

    assert _perception_is_usable(_perception(_devices(registry.load("go2"), rt=object())[2])) is True
