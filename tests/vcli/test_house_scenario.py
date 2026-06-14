# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""N0 (campaign #2) — the 'house' preset: ReplicaCAD dataset-config loading.

ReplicaCAD scenes are scene-INSTANCES (stage + furniture + lighting composed
by a scene_dataset_config), not bare .glb stages — and bare habitat_sim does
NOT consume the dataset's navmesh_instances mapping (spike-verified), so the
repo side resolves the authored navmesh explicitly. No habitat dep here:
everything tested is repo-venv code (scenes.py, catalog, bridge argv, base
passthrough, boot wiring).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_os_nano.playground.catalog import SCENARIOS, get_scenario
from vector_os_nano.playground.habitat.scenes import (
    dataset_navmesh_path,
    preflight_scene_instance,
    resolve_scene_dataset_config,
)
from vector_os_nano.playground.scenario import Scenario
from vector_os_nano.playground.world import PlaygroundWorld


def _make_dataset(tmp_path: Path, *, navmesh: bool = True) -> Path:
    """A minimal on-disk dataset following the ReplicaCAD layout."""
    root = tmp_path / "replica_cad"
    (root / "configs" / "scenes").mkdir(parents=True)
    (root / "navmeshes").mkdir()
    (root / "configs" / "scenes" / "apt_1.scene_instance.json").write_text("{}")
    (root / "configs" / "scenes" / "apt_2.scene_instance.json").write_text("{}")
    mapping = {"apt_1": "navmeshes/apt_1.navmesh"} if navmesh else {}
    cfg = root / "replicaCAD.scene_dataset_config.json"
    cfg.write_text(json.dumps({"navmesh_instances": mapping}))
    if navmesh:
        (root / "navmeshes" / "apt_1.navmesh").write_bytes(b"")
    return cfg


class TestScenarioDatasetField:
    def test_additive_default_empty(self) -> None:
        # Frozen dataclass evolves additively: every pre-existing preset
        # keeps an empty scene_dataset_config.
        for sid in ("tabletop", "go2_room", "apartment"):
            assert get_scenario(sid).scene_dataset_config == ""

    def test_field_is_last_with_default(self) -> None:
        s = Scenario(id="x", embodiment="mobile", scene_xml="", task_hint="t")
        assert s.scene_dataset_config == ""


class TestHousePreset:
    def test_registered_with_dataset_config(self) -> None:
        s = get_scenario("house")
        assert s.sim_backend == "habitat"
        assert s.embodiment == "mobile"
        assert s.scene_ref == "apt_1"  # a dataset scene NAME, not a file
        assert s.scene_dataset_config.endswith("replicaCAD.scene_dataset_config.json")
        assert "house" in SCENARIOS

    def test_rooms_are_labeled_and_valid(self) -> None:
        rooms = get_scenario("house").rooms
        assert len(rooms) >= 4  # a HOUSE has named rooms (unlike apartment)
        for name, (x0, y0, x1, y1) in rooms.items():
            assert x0 < x1 and y0 < y1, f"degenerate box for {name}"

    def test_world_has_base_and_predicates(self) -> None:
        w = PlaygroundWorld(get_scenario("house"))
        assert w.has_base() is True
        ns = w.build_verify_namespace(agent=None)
        assert set(ns) >= {"at_position", "facing", "visited", "geodesic_dist"}


class TestDatasetConfigResolution:
    def test_absolute_passthrough(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path)
        assert resolve_scene_dataset_config(str(cfg)) == str(cfg)

    def test_relative_uses_env_root(self, tmp_path, monkeypatch) -> None:
        cfg = _make_dataset(tmp_path)
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        rel = "replica_cad/replicaCAD.scene_dataset_config.json"
        assert resolve_scene_dataset_config(rel) == str(cfg)

    def test_missing_fails_loud(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="VECTOR_HABITAT_DATA"):
            resolve_scene_dataset_config("nope/missing.json")

    def test_empty_fails_loud(self) -> None:
        with pytest.raises(FileNotFoundError, match="empty"):
            resolve_scene_dataset_config("")


class TestSceneInstancePreflight:
    def test_known_scene_passes(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path)
        assert preflight_scene_instance(str(cfg), "apt_1") == "apt_1"

    def test_unknown_scene_fails_with_available_set(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="apt_2"):
            preflight_scene_instance(str(cfg), "apt_99")

    def test_unconventional_layout_skips_check(self, tmp_path) -> None:
        cfg = tmp_path / "other.scene_dataset_config.json"
        cfg.write_text("{}")  # no configs/scenes dir next to it
        assert preflight_scene_instance(str(cfg), "whatever") == "whatever"

    def test_empty_name_fails_loud(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="empty"):
            preflight_scene_instance(str(cfg), "")


class TestDatasetNavmeshPath:
    def test_mapped_and_present(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path)
        p = dataset_navmesh_path(str(cfg), "apt_1")
        assert p.endswith("navmeshes/apt_1.navmesh") and Path(p).exists()

    def test_unmapped_returns_empty(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path, navmesh=False)
        assert dataset_navmesh_path(str(cfg), "apt_1") == ""

    def test_mapped_but_missing_file_returns_empty(self, tmp_path) -> None:
        cfg = _make_dataset(tmp_path)
        (Path(cfg).parent / "navmeshes" / "apt_1.navmesh").unlink()
        assert dataset_navmesh_path(str(cfg), "apt_1") == ""

    def test_unreadable_config_returns_empty(self, tmp_path) -> None:
        assert dataset_navmesh_path(str(tmp_path / "absent.json"), "apt_1") == ""


class TestBridgeArgv:
    def test_dataset_flags_present_when_set(self) -> None:
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        b = HabitatBridge(
            "apt_1", dataset_config="/d/cfg.json", navmesh="/d/nav.navmesh"
        )
        argv = b._server_argv()
        assert argv[argv.index("--dataset-config") + 1] == "/d/cfg.json"
        assert argv[argv.index("--navmesh") + 1] == "/d/nav.navmesh"

    def test_dataset_flags_absent_when_empty(self) -> None:
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        argv = HabitatBridge("/abs/scene.glb")._server_argv()
        assert "--dataset-config" not in argv
        assert "--navmesh" not in argv


class TestBasePassthrough:
    def test_base_threads_dataset_kwargs_to_bridge(self, monkeypatch) -> None:
        import vector_os_nano.playground.habitat.base as base_mod

        seen: dict = {}

        class _FakeBridge:
            def __init__(self, scene, *, gui=False, dataset_config="",
                         navmesh="", **kw):
                seen.update(
                    scene=scene, gui=gui,
                    dataset_config=dataset_config, navmesh=navmesh,
                )

        monkeypatch.setattr(base_mod, "HabitatBridge", _FakeBridge)
        base_mod.HabitatBase(
            scene="apt_1", gui=False,
            dataset_config="/d/cfg.json", navmesh="/d/nav.navmesh",
        )
        assert seen == {
            "scene": "apt_1", "gui": False,
            "dataset_config": "/d/cfg.json", "navmesh": "/d/nav.navmesh",
        }


class TestN3RobotBody:
    def test_bridge_argv_carries_body_flags(self) -> None:
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        argv = HabitatBridge(
            "apt_1", robot_glb="/d/g1.glb", viewer_mode="chase"
        )._server_argv()
        assert argv[argv.index("--robot-glb") + 1] == "/d/g1.glb"
        assert argv[argv.index("--viewer-mode") + 1] == "chase"

    def test_bridge_argv_omits_when_unset(self) -> None:
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        argv = HabitatBridge("/abs/scene.glb")._server_argv()
        assert "--robot-glb" not in argv and "--viewer-mode" not in argv

    def test_resolve_viewer_env_overrides(self, monkeypatch) -> None:
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_viewer

        monkeypatch.setenv("VECTOR_HABITAT_VIEWER", "first")
        assert resolve_habitat_viewer("chase") == "first"
        monkeypatch.delenv("VECTOR_HABITAT_VIEWER")
        assert resolve_habitat_viewer("first") == "first"
        assert resolve_habitat_viewer() == "chase"  # N3 default: SEE the robot

    def test_resolve_robot_glb_detects_asset(self, tmp_path, monkeypatch) -> None:
        from vector_os_nano.vcli.habitat_runtime import resolve_robot_glb

        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        assert resolve_robot_glb() == ""
        glb = tmp_path / "robots" / "unitree_g1" / "g1_rigid.glb"
        glb.parent.mkdir(parents=True)
        glb.write_bytes(b"glTF")
        assert resolve_robot_glb() == str(glb)

    def test_boot_threads_body_to_base(self, tmp_path, monkeypatch) -> None:
        import dataclasses as dc

        import vector_os_nano.playground.habitat as hab
        from vector_os_nano.vcli.habitat_runtime import boot_habitat_agent

        class _FakeBase:
            def __init__(self, scene, gui=False, dataset_config="", navmesh="",
                         robot_glb="", viewer_mode="", viewer_size=0):
                self.robot_glb = robot_glb
                self.viewer_mode = viewer_mode
                self.viewer_size = viewer_size

            def connect(self) -> None:
                pass

        monkeypatch.setattr(hab, "HabitatBase", _FakeBase)
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        monkeypatch.setenv("VECTOR_HABITAT_GUI", "0")
        glb = tmp_path / "robots" / "unitree_g1" / "g1_rigid.glb"
        glb.parent.mkdir(parents=True)
        glb.write_bytes(b"glTF")
        f = tmp_path / "scene.glb"
        f.write_bytes(b"")
        scenario = dc.replace(get_scenario("apartment"), scene_ref=str(f))
        agent = boot_habitat_agent(PlaygroundWorld(scenario))
        assert agent._base.robot_glb == str(glb)
        assert agent._base.viewer_mode == "chase"


class TestBootWiring:
    def test_boot_resolves_dataset_and_navmesh(self, tmp_path, monkeypatch) -> None:
        import vector_os_nano.playground.habitat as hab
        from vector_os_nano.vcli.habitat_runtime import boot_habitat_agent

        cfg = _make_dataset(tmp_path)

        class _FakeBase:
            def __init__(self, scene, gui=False, dataset_config="", navmesh="",
                         **kw):
                self.scene = scene
                self.gui = gui
                self.dataset_config = dataset_config
                self.navmesh = navmesh

            def connect(self) -> None:
                self.connected = True

        monkeypatch.setattr(hab, "HabitatBase", _FakeBase)
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        monkeypatch.setenv("VECTOR_HABITAT_GUI", "0")
        scenario = dataclasses.replace(
            get_scenario("house"),
            scene_dataset_config="replica_cad/replicaCAD.scene_dataset_config.json",
        )
        world = PlaygroundWorld(scenario)
        agent = boot_habitat_agent(world)
        b = agent._base
        assert b.scene == "apt_1"  # the dataset scene NAME passes through
        assert b.dataset_config == str(cfg)
        assert b.navmesh.endswith("navmeshes/apt_1.navmesh")
        assert b.connected is True

    def test_boot_unknown_scene_fails_loud(self, tmp_path, monkeypatch) -> None:
        import vector_os_nano.playground.habitat as hab
        from vector_os_nano.vcli.habitat_runtime import boot_habitat_agent

        _make_dataset(tmp_path)
        monkeypatch.setattr(hab, "HabitatBase", lambda **kw: None)
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        monkeypatch.setenv("VECTOR_HABITAT_GUI", "0")
        scenario = dataclasses.replace(get_scenario("house"), scene_ref="apt_99")
        with pytest.raises(FileNotFoundError, match="apt_99"):
            boot_habitat_agent(PlaygroundWorld(scenario))

    def test_file_ref_path_unchanged(self, tmp_path, monkeypatch) -> None:
        # The pre-N0 bare-glb path (apartment) keeps working byte-identically.
        import vector_os_nano.playground.habitat as hab
        from vector_os_nano.vcli.habitat_runtime import boot_habitat_agent

        class _FakeBase:
            def __init__(self, scene, gui=False, dataset_config="", navmesh="",
                         **kw):
                self.scene = scene
                self.dataset_config = dataset_config
                self.navmesh = navmesh

            def connect(self) -> None:
                pass

        monkeypatch.setattr(hab, "HabitatBase", _FakeBase)
        monkeypatch.setenv("VECTOR_HABITAT_GUI", "0")
        f = tmp_path / "scene.glb"
        f.write_bytes(b"")
        scenario = dataclasses.replace(get_scenario("apartment"), scene_ref=str(f))
        agent = boot_habitat_agent(PlaygroundWorld(scenario))
        assert agent._base.scene == str(f)
        assert agent._base.dataset_config == ""
        assert agent._base.navmesh == ""
