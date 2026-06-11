# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M2 part 2 — habitat scenario/world/predicate wiring (no habitat dep)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from vector_os_nano.playground.catalog import SCENARIOS, get_scenario
from vector_os_nano.playground.habitat.scenes import resolve_scene_ref
from vector_os_nano.playground.verify.base_predicates import make_geodesic_dist
from vector_os_nano.playground.world import PlaygroundWorld


class TestApartmentScenario:
    def test_registered_with_habitat_backend(self) -> None:
        s = get_scenario("apartment")
        assert s.sim_backend == "habitat"
        assert s.embodiment == "mobile"
        assert s.scene_ref.endswith("apartment_1.glb")
        assert s.rooms == {}  # honestly empty until HM3D-Semantics (DQ-1)

    def test_mobile_embodiment_has_base(self) -> None:
        w = PlaygroundWorld(get_scenario("apartment"))
        assert w.has_base() is True
        assert w.is_robot() is True

    def test_base_namespace_includes_geodesic(self) -> None:
        w = PlaygroundWorld(get_scenario("apartment"))
        ns = w.build_verify_namespace(agent=None)
        assert set(ns) >= {"at_position", "facing", "visited", "geodesic_dist"}

    def test_go2_room_unchanged(self) -> None:
        # Additive evolution: the existing go2 preset keeps its contract.
        w = PlaygroundWorld(get_scenario("go2_room"))
        assert w.has_base() is True
        assert "geodesic_dist" in w.build_verify_namespace(agent=None)
        assert SCENARIOS["go2_room"].sim_backend == "mujoco"


class TestGeodesicPredicateFailSafe:
    def test_no_base_is_inf(self) -> None:
        g = make_geodesic_dist(agent=SimpleNamespace(_base=None))
        assert g(1.0, 2.0) == float("inf")

    def test_base_without_oracle_is_inf(self) -> None:
        base = SimpleNamespace(
            get_position=lambda: [0.0, 0.0, 0.0], get_heading=lambda: 0.0,
            is_connected=lambda: True,
        )
        g = make_geodesic_dist(agent=SimpleNamespace(_base=base))
        assert g(1.0, 2.0) == float("inf")

    def test_oracle_base_returns_distance(self) -> None:
        base = SimpleNamespace(
            get_position=lambda: [0.0, 0.0, 0.0], get_heading=lambda: 0.0,
            is_connected=lambda: True,
            geodesic_distance=lambda a, b: math.dist(a[:2], b[:2]),
        )
        g = make_geodesic_dist(agent=SimpleNamespace(_base=base))
        assert g(3.0, 4.0) == pytest.approx(5.0)

    def test_oracle_error_is_inf_never_raises(self) -> None:
        def _boom(a, b):
            raise RuntimeError("navmesh down")

        base = SimpleNamespace(
            get_position=lambda: [0.0, 0.0, 0.0], get_heading=lambda: 0.0,
            is_connected=lambda: True, geodesic_distance=_boom,
        )
        g = make_geodesic_dist(agent=SimpleNamespace(_base=base))
        assert g(1.0, 1.0) == float("inf")


class TestSceneRefResolution:
    def test_absolute_ref_passthrough(self, tmp_path) -> None:
        f = tmp_path / "x.glb"
        f.write_bytes(b"")
        assert resolve_scene_ref(str(f)) == str(f)

    def test_relative_ref_uses_env_root(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "sub").mkdir()
        f = tmp_path / "sub" / "scene.glb"
        f.write_bytes(b"")
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        assert resolve_scene_ref("sub/scene.glb") == str(f)

    def test_missing_scene_fails_loud(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VECTOR_HABITAT_DATA", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="VECTOR_HABITAT_DATA"):
            resolve_scene_ref("nope/missing.glb")

    def test_empty_ref_fails_loud(self) -> None:
        with pytest.raises(FileNotFoundError, match="empty scene_ref"):
            resolve_scene_ref("")


class TestCliHabitatDispatch:
    def test_non_habitat_world_returns_none(self) -> None:
        from vector_os_nano.vcli.cli import _maybe_init_habitat_agent

        w = PlaygroundWorld(get_scenario("go2_room"))
        assert _maybe_init_habitat_agent(SimpleNamespace(), w) is None

    def test_habitat_world_builds_agent_with_base(self, monkeypatch) -> None:
        import vector_os_nano.playground.habitat as hab
        from vector_os_nano.vcli.cli import _maybe_init_habitat_agent

        class _FakeBase:
            def __init__(self, scene: str) -> None:
                self.scene = scene
                self.connected = False

            def connect(self) -> None:
                self.connected = True

        monkeypatch.setattr(hab, "HabitatBase", _FakeBase)
        monkeypatch.setenv("VECTOR_HABITAT_DATA", "/nonexistent")
        # Point the scenario's ref at a real temp file via absolute override.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".glb") as f:
            scenario = get_scenario("apartment")
            import dataclasses

            w = PlaygroundWorld(dataclasses.replace(scenario, scene_ref=f.name))
            agent = _maybe_init_habitat_agent(SimpleNamespace(), w)
            assert agent is not None
            assert agent._base.connected is True
            assert agent._base.scene == f.name
            # The decompose vocab derives from this registry (rule 3): a
            # base-only world teaches ONLY base-capable skills — no arm set.
            taught = set(agent._skill_registry.list_skills())
            assert taught == {"walk", "turn", "stop", "navigate_to"}
