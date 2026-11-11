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
from vector_os_nano.sim import backends
from vector_os_nano.sim.scene import compiled_scene, reader

SCENES = Path(__file__).parent / "scenes"
SIM_DIR = Path(vector_os_nano.__file__).parent / "hardware" / "sim"
# ponytail: the coordinator's fetched G1 assets; C1 resolves them under $VECTOR_HOME/assets.
G1_DIR = Path("/data/vos-analysis/release-1.0/assets/g1_gait")


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


def _inactive_welds(path: Path) -> list[tuple[str, str]]:
    scene = compiled_scene(path)
    m = scene.model
    return [
        (scene.body_name(int(m.eq_obj1id[i])), scene.body_name(int(m.eq_obj2id[i])))
        for i in range(m.neq) if m.eq_type[i] == reader.EQ_WELD and not m.eq_active0[i]
    ]


@pytest.fixture(scope="session")
def contract_scene():
    return compiled_scene(SCENES / "contract.xml", grasp_pairs=[("carrier", "parcel"), ("arm", "parcel")])


def _compose(out: Path, world: str, robot: Path, at: tuple[float, float, float],
             welds: tuple[tuple[str, str], ...] = ()) -> Path:
    """A test scene: *robot* attached into scenes/*world* at *at*, plus inactive welds."""
    import mujoco as mj

    from vector_os_nano.hardware.sim.scene_builder import normalize_attach_defaults

    spec = mj.MjSpec.from_file(str(SCENES / world))
    model = mj.MjSpec.from_file(str(robot))
    for mesh in model.meshes:
        mesh.file = str((robot.parent / model.meshdir / mesh.file).resolve())
    model.meshdir = ""
    spec.attach(model, prefix="", frame=spec.worldbody.add_frame(pos=list(at)))
    for parent, child in welds:
        eq = spec.add_equality()
        eq.type, eq.objtype = mj.mjtEq.mjEQ_WELD, mj.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2, eq.active = parent, child, False
    spec.compile()
    path = out / f"{Path(world).stem}_{robot.stem}.xml"
    path.write_text(normalize_attach_defaults(spec.to_xml()))
    return path


@pytest.fixture(scope="session")
def battery_scene(tmp_path_factory: pytest.TempPathFactory):
    """battery.xml with the Piper attached at (0, 0, 0.2) and its two grasp welds."""
    pairs = (("link6", "box"), ("link6", "cylinder"))
    path = _compose(tmp_path_factory.mktemp("battery"), "battery.xml",
                    SIM_DIR / "mjcf" / "piper" / "piper.xml", (0.0, 0.0, 0.2), pairs)
    return compiled_scene(path, robot_root="base_link", grasp_pairs=pairs)


@pytest.fixture(scope="session")
def flat_go2_scene(tmp_path_factory: pytest.TempPathFactory):
    """go2.xml on flat ground (the locomotion bounds need room to walk)."""
    path = _compose(tmp_path_factory.mktemp("flat"), "flat.xml", SIM_DIR / "mjcf" / "go2" / "go2.xml", (0.0, 0.0, 0.0))
    return compiled_scene(path, robot_root="base_link")


def _robot_scene(name: str):
    if name == "so101":
        return compiled_scene(SIM_DIR / "so101_mujoco.xml", robot_root="base_link")
    if name == "piper":
        return compiled_scene(SIM_DIR / "mjcf" / "piper" / "piper.xml", robot_root="base_link")
    if name in ("go2", "go2_piper"):
        from vector_os_nano.hardware.sim import mujoco_go2

        path = mujoco_go2._build_room_scene_xml(with_arm=name == "go2_piper")
        return compiled_scene(path, robot_root="base_link", grasp_pairs=_inactive_welds(path),
                              timestep=mujoco_go2._SIM_DT)
    if name == "g1":
        if not (G1_DIR / "scene.xml").exists():
            pytest.skip(f"G1 assets absent ({G1_DIR}): fetch them first")
        return compiled_scene(G1_DIR / "scene.xml", robot_root="pelvis")
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
