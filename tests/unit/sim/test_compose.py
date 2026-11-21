# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The scene composer's two guards (compile only, no stepping or rendering): a camera
placed on a body the model lacks, and a gripper finger body with no colliding geom."""
from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.sim.scene.compose import check_fingers, compose, model_files
from vector_os_nano.vcli.worlds.world_manifest import load_world


@pytest.fixture
def piper(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cfg = registry.load("piper")
    return cfg, load_world(cfg.world)


def _compose(cfg, world):
    return compose(world.scene, cfg, spawn=world.spawn, pickables=tuple(world.pickables))


@pytest.mark.parametrize("robot", ["piper", "so101"])
def test_the_shipped_fingers_collide(robot, piper):
    cfg = registry.load(robot)
    check_fingers(_compose(cfg, load_world(cfg.world)), cfg.arm.gripper.finger_bodies)


def _numb_link8(cfg, world, tmp_path):
    """*cfg* with link8's two collision pads made non-colliding, and its scene."""
    src, _ = model_files(cfg)
    root = ET.parse(src).getroot()
    root.find("compiler").set("meshdir", str(src.parent / root.find("compiler").get("meshdir")))
    pads = [g for g in root.iter("body") if g.get("name") == "link8"][0].findall("geom")
    for geom in pads:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    model = tmp_path / "piper_numb_link8.xml"
    model.write_text(ET.tostring(root, encoding="unicode"))
    numb = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, path=str(model)))
    return numb, _compose(numb, world)


def test_a_finger_without_a_colliding_geom_is_refused(piper, tmp_path):
    """link8's two collision pads made non-colliding: the guard names link8 (not link7)."""
    cfg, world = piper
    _numb, scene = _numb_link8(cfg, world, tmp_path)
    check_fingers(scene, ("link7",))
    with pytest.raises(ValueError, match="finger body 'link8' has no colliding geom"):
        check_fingers(scene, cfg.arm.gripper.finger_bodies)


def test_the_session_refuses_a_numb_finger(piper, tmp_path):
    """The session's device wiring runs the guard (deleting its call left every test green)."""
    from vector_os_nano.sim.session import _devices

    numb, scene = _numb_link8(*piper, tmp_path)
    with pytest.raises(ValueError, match="finger body 'link8' has no colliding geom"):
        _devices(numb, SimpleNamespace(world=SimpleNamespace(scene=scene)))


def test_a_camera_on_a_body_the_model_lacks_is_refused(piper):
    cfg, world = piper
    cam = dataclasses.replace(cfg.camera, body="no_such_link")
    bad = dataclasses.replace(cfg, sensors=tuple(cam if s is cfg.camera else s for s in cfg.sensors))
    with pytest.raises(KeyError, match="camera 'workspace_cam' body 'no_such_link' is not in"):
        _compose(bad, world)
