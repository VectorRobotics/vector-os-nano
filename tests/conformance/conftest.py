# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The physics conformance suite: ONE suite, parametrized over every row of
``vector_os_nano.sim.backends.BACKENDS`` (charter sections 3.1 and 8 / S2).

A backend whose engine is not installed skips with the install instruction.
Scenes are compiled once per session; worlds are loaded per test and closed.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import pytest

import vector_os_nano
from vector_os_nano.embodiments import registry
from vector_os_nano.sim import backends
from vector_os_nano.sim.scene import compiled_scene
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.schemas.world import load_world, Pickable

SCENES = Path(__file__).parent / "scenes"
ROBOTS = Path(vector_os_nano.__file__).parent / "embodiments"


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


def battery_world():
    return replace(load_world("workbench"), id="battery", scene=SCENES / "battery.xml",
                   spawn=(0., 0., .2, 0.),
                   pickables={name: Pickable(name, name, "red", (name,)) for name in ("box", "cylinder")})


@pytest.fixture(scope="session")
def battery_scene():
    """battery.xml with the `piper` body (its manifest: camera, gripper) mounted at
    (0, 0, 0.2) and its two grasp welds, by the product's scene composition."""
    return compose(registry.load("piper"), battery_world())


@pytest.fixture(scope="session")
def tabletop_scene():
    """The `so101` body in its own world (the tabletop scene, its spawn, its six
    pickables), by the product's scene composition."""
    cfg = registry.load("so101")
    world = load_world(cfg.world)
    return compose(cfg, world)


@pytest.fixture(scope="session")
def flat_go2_scene():
    """go2 on flat ground, at its policy's timestep."""
    go2 = registry.load("go2")
    world = replace(load_world("apartment"), scene=SCENES / "flat.xml", spawn=(0., 0., 0., 0.), pickables={})
    return compose(go2, world)


def _inactive_welds(xml_path) -> list[tuple[str, str]]:
    """``(body1, body2)`` names of every inactive body weld in *xml_path*: the grasp
    pairs of a scene whose file declares them by hand (the go2 room)."""
    m = mj.MjModel.from_xml_path(str(xml_path))
    name = lambda b: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(b))  # noqa: E731
    return [
        (name(m.eq_obj1id[i]), name(m.eq_obj2id[i])) for i in range(m.neq)
        if m.eq_type[i] == mj.mjtEq.mjEQ_WELD and m.eq_objtype[i] == mj.mjtObj.mjOBJ_BODY
        and not m.eq_active0[i]
    ]


def _robot_scene(name: str):
    if name in ("so101", "piper"):
        return compiled_scene(ROBOTS / name / "model" / f"{name}.xml", robot_root="base_link")
    if name in ("go2", "go2_piper"):
        from vector_os_nano.sim.scene.compose import compose
        from vector_os_nano.schemas.world import load_world, Pickable

        cfg = registry.load(name)
        return compose(cfg, load_world(cfg.world))
    if name == "g1":
        from vector_os_nano.embodiments.assets import AssetsMissing, asset_dir
        from vector_os_nano.schemas.robot import load_embodiment_config

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
