# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Room landmarks in the world model (owner finding 2026-06-12 round 2).

'走到门口' failed because the habitat agent has no SceneGraph and the world
context listed NO rooms — the planner could not know 'entryway' exists, let
alone its coordinates. The scenario's authored rooms are now seeded into the
world model as type=room landmarks at habitat boot: they appear in the
'Objects (live)' context line WITH coordinates, resolve through navigate_to's
label path (semantic standoff), and survive sysnav merges (distinct ids).
"""
from __future__ import annotations

from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.playground.catalog import get_scenario
from vector_os_nano.vcli.habitat_runtime import seed_room_landmarks


class TestSeedRoomLandmarks:
    def test_house_rooms_become_landmarks(self):
        wm = WorldModel()
        n = seed_room_landmarks(wm, get_scenario("house"))
        assert n == 5
        labels = {o.label for o in wm.get_objects()}
        assert {"entryway", "kitchen", "living_room", "dining", "tv_corner"} <= labels

    def test_landmark_is_room_typed_with_rect_center(self):
        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        entry = wm.get_objects_by_label("entryway")[0]
        # catalog rect (1.2, 1.2, 3.2, 2.8) -> center (2.2, 2.0)
        assert (round(entry.x, 2), round(entry.y, 2)) == (2.2, 2.0)
        assert entry.properties.get("type") == "room"
        assert entry.confidence == 1.0

    def test_zh_alias_carried_for_grounding(self):
        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        entry = wm.get_objects_by_label("entryway")[0]
        assert "门口" in str(entry.properties.get("alias", ""))
        kitchen = wm.get_objects_by_label("kitchen")[0]
        assert "厨房" in str(kitchen.properties.get("alias", ""))

    def test_no_rooms_scenario_is_noop(self):
        wm = WorldModel()
        scen = get_scenario("house")
        import dataclasses

        bare = dataclasses.replace(scen, rooms={})
        assert seed_room_landmarks(wm, bare) == 0
        assert wm.get_objects() == []

    def test_idempotent_reseed_never_duplicates(self):
        wm = WorldModel()
        scen = get_scenario("house")
        seed_room_landmarks(wm, scen)
        seed_room_landmarks(wm, scen)
        assert len(wm.get_objects_by_label("kitchen")) == 1

    def test_existing_objects_untouched(self):
        wm = WorldModel()
        wm.add_object(ObjectState(object_id="sysnav_1", label="sofa", x=1.0, y=2.0))
        seed_room_landmarks(wm, get_scenario("house"))
        assert len(wm.get_objects_by_label("sofa")) == 1
        assert len(wm.get_objects()) == 6


class TestNavigateResolvesRoomLandmark:
    def test_label_kitchen_resolves_to_room_center_with_standoff(self):
        from types import SimpleNamespace

        from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        nav_calls: list[tuple] = []

        class _Base:
            def navigate_to(self, x, y, tol=0.2):
                nav_calls.append((x, y, tol))
                return {"reached": True, "pos": [x, y, 0.0], "dist": 0.05}

        ctx = SimpleNamespace(base=_Base(), world_model=wm, agent=None)
        res = NavigateToPointSkill().execute({"label": "kitchen"}, ctx)
        assert res.success, res.error_message
        # catalog rect (-3.6, 1.0, 0.2, 2.8) -> center (-1.7, 1.9). Batch 2 #3:
        # a room is a REGION — tol comes from the rect half-dims (<=0.9 for
        # the kitchen) so the drive ends INSIDE, never the 1.5 object standoff
        # (the original assertion here had FROZEN the wrong contract).
        x, y, tol = nav_calls[0]
        assert (round(x, 2), round(y, 2)) == (-1.7, 1.9)
        assert 0.3 <= tol <= 0.9


class TestViewerSizeResolver:
    def test_env_override_wins(self, monkeypatch):
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_viewer_size

        monkeypatch.setenv("VECTOR_HABITAT_VIEWER_SIZE", "1200")
        assert resolve_habitat_viewer_size(640) == 1200

    def test_default_is_800(self, monkeypatch):
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_viewer_size

        monkeypatch.delenv("VECTOR_HABITAT_VIEWER_SIZE", raising=False)
        assert resolve_habitat_viewer_size() == 800

    def test_bad_env_falls_through(self, monkeypatch):
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_viewer_size

        monkeypatch.setenv("VECTOR_HABITAT_VIEWER_SIZE", "huge")
        assert resolve_habitat_viewer_size(640) == 640


class TestMarkersFromWorldModel:
    def test_rooms_excluded_objects_carried(self):
        from vector_os_nano.vcli.habitat_runtime import (
            markers_from_world_model,
            seed_room_landmarks,
        )

        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        wm.add_object(
            ObjectState(object_id="sysnav_3", label="sofa", x=1.0, y=2.0, z=0.5)
        )
        markers = markers_from_world_model(wm)
        assert markers == [{"label": "sofa", "x": 1.0, "y": 2.0, "z": 0.5}]


class TestBridgeViewerSizeArgv:
    def test_argv_carries_viewer_size(self, monkeypatch):
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        monkeypatch.setenv("VECTOR_HABITAT_PYTHON", "/usr/bin/python3")
        b = HabitatBridge("scene.glb", viewer_size=1024)
        argv = b._server_argv()
        assert "--viewer-size" in argv
        assert argv[argv.index("--viewer-size") + 1] == "1024"

    def test_zero_size_omitted(self, monkeypatch):
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        monkeypatch.setenv("VECTOR_HABITAT_PYTHON", "/usr/bin/python3")
        b = HabitatBridge("scene.glb")
        assert "--viewer-size" not in b._server_argv()


class TestConsumerOnBatchHook:
    def _msg(self, n=1):
        from types import SimpleNamespace

        node = SimpleNamespace(object_id=[7], label="sofa")
        return SimpleNamespace(nodes=[node] * n)

    def _bridge(self, hook):
        from vector_os_nano.integrations.sysnav_bridge.live_bridge import (
            LiveSysnavBridge,
        )

        return LiveSysnavBridge(world_model=WorldModel(), on_batch=hook)

    def test_hook_fires_once_per_nonempty_batch(self):
        calls = []
        b = self._bridge(lambda: calls.append(1))
        b._callback(self._msg(3))
        assert len(calls) == 1

    def test_empty_batch_never_fires(self):
        calls = []
        b = self._bridge(lambda: calls.append(1))
        from types import SimpleNamespace

        b._callback(SimpleNamespace(nodes=[]))
        assert calls == []

    def test_hook_exception_swallowed(self):
        def _boom():
            raise RuntimeError("display died")

        b = self._bridge(_boom)
        b._callback(self._msg())  # must not raise


class TestRoomsReachPlannerContext:
    def test_objects_line_carries_rooms_with_alias_and_coords(self):
        from types import SimpleNamespace

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.habitat_runtime import seed_room_landmarks

        wm = WorldModel()
        seed_room_landmarks(wm, get_scenario("house"))
        agent = SimpleNamespace(_world_model=wm, _arm=None)
        line = VectorEngine._live_objects_line(agent)
        assert "entryway" in line
        assert "门口" in line          # zh grounding for '走到门口'
        assert "x=2.20" in line        # actionable coordinates, not just names
