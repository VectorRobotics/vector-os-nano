# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The physics conformance suite: ONE suite, parametrized over every row of
``vector_os_nano.sim.backends.BACKENDS`` (charter sections 3.1 and 8 / S2).

A backend whose engine is not installed skips with the install instruction.
Scenes are compiled once per session; worlds are loaded per test and closed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import vector_os_nano
from vector_os_nano.embodiments import registry
from vector_os_nano.sim import backends
from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.sim.scene.compose import compose

SCENES = Path(__file__).parent / "scenes"
SIM_DIR = Path(vector_os_nano.__file__).parent / "hardware" / "sim"


def _engine_missing(name: str) -> str | None:
    if name == "pybullet" and importlib.util.find_spec("pybullet") is None:
        from vector_os_nano.sim.backends.pybullet import INSTALL_HINT

        return INSTALL_HINT
    return None


@pytest.fixture(params=sorted(backends.BACKENDS))
def backend(request: pytest.FixtureRequest) -> str:
    why = _engine_missing(request.param)
    if why:
        pytest.skip(why)
    return request.param


@pytest.fixture
def pybullet_backend() -> str:
    """For the PyBullet-only behavioural bounds."""
    why = _engine_missing("pybullet")
    if why:
        pytest.skip(why)
    return "pybullet"


@pytest.fixture
def both() -> tuple[str, str]:
    """For the cross-backend comparisons."""
    for name in backends.BACKENDS:
        why = _engine_missing(name)
        if why:
            pytest.skip(why)
    return "mujoco", "pybullet"


@pytest.fixture
def load():
    """``load(backend, scene)`` -> a world, closed at teardown."""
    worlds = []

    def _load(backend: str, scene):
        w = backends.get(backend).load(scene, gui=False)
        worlds.append(w)
        return w

    yield _load
    for w in worlds:
        w.close()


@pytest.fixture(scope="session")
def contract_scene():
    return compiled_scene(SCENES / "contract.xml", grasp_pairs=[("carrier", "parcel"), ("arm", "parcel")])


@pytest.fixture(scope="session")
def battery_scene():
    """battery.xml with the `piper` body (its manifest: camera, gripper) mounted at
    (0, 0, 0.2) and its two grasp welds, by the product's scene composition."""
    return compose(SCENES / "battery.xml", registry.load("piper"), spawn=(0.0, 0.0, 0.2, 0.0),
                   pickables=("box", "cylinder"))


@pytest.fixture(scope="session")
def flat_go2_scene():
    """go2 on flat ground, at its policy's timestep."""
    go2 = registry.load("go2")
    return compose(SCENES / "flat.xml", go2, spawn=(0.0, 0.0, 0.0, 0.0), timestep=go2.base.policy.timestep)


def _robot_scene(name: str):
    if name == "so101":
        return compiled_scene(SIM_DIR / "so101_mujoco.xml", robot_root="base_link")
    if name == "piper":
        return compiled_scene(SIM_DIR / "mjcf" / "piper" / "piper.xml", robot_root="base_link")
    if name in ("go2", "go2_piper"):
        from vector_os_nano.hardware.sim.scene_builder import room_scene
        from vector_os_nano.vcli.worlds.world_manifest import load_world

        cfg = registry.load(name)
        return room_scene(cfg, load_world(cfg.world))
    if name == "g1":
        from vector_os_nano.embodiments.assets import AssetsMissing, asset_dir
        from vector_os_nano.embodiments.config import load_embodiment_config

        try:
            folder = asset_dir(load_embodiment_config("g1"))
        except AssetsMissing as exc:
            pytest.skip(str(exc))
        return compiled_scene(folder / "scene.xml", robot_root="pelvis")
    raise KeyError(name)


_ROBOT_SCENES: dict[str, object] = {}


@pytest.fixture
def robot_scene():
    """``robot_scene(name)`` -> the compiled scene of one of the five bodies (cached)."""
    def _get(name: str):
        if name not in _ROBOT_SCENES:
            _ROBOT_SCENES[name] = _robot_scene(name)
        return _ROBOT_SCENES[name]

    return _get
