# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""look / describe_scene write only what they localized, in the room the robot is
in (a real SceneGraph seeded from the apartment manifest)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.perception.object_localizer import Localized
from vector_os_nano.skills.go2.look import DescribeSceneSkill, LookSkill
from vector_os_nano.schemas.world import load_world

_APT = load_world("apartment")


class _Base:
    def get_position(self):
        return (10.0, 3.0, 0.3)  # the hallway, facing the pick table

    def get_heading(self):
        return 0.0

    def get_camera_frame(self):
        return np.zeros((48, 64, 3), np.uint8)


class _Perception:
    def capture(self):
        return SimpleNamespace(rgb=np.zeros((48, 64, 3), np.uint8), depth=None, K=None,
                               cam_pos=None, cam_mat=None)

    def detect_in(self, rgb, q):
        return []

    def segment(self, rgb, box):
        return None


def _vlm(room: str, names: list[str]):
    v = mock.MagicMock()
    v.describe_scene.return_value = SimpleNamespace(
        summary="a hallway", details="", objects=[SimpleNamespace(name=n, description="") for n in names])
    v.identify_room.return_value = SimpleNamespace(room=room, confidence=0.9, reasoning="")
    v.find_objects.return_value = [SimpleNamespace(name=n, description="") for n in names]
    return v


def _ctx(sg, vlm=None):
    return SkillContext(base=_Base(), perception=_Perception(), config={"world": "apartment"},
                        services={"spatial_memory": sg, **({"vlm": vlm} if vlm else {})})


def _sg() -> SceneGraph:
    sg = SceneGraph(resolve=_APT.resolve)
    sg.load_layout(str(_APT.path))
    return sg


def test_the_vlm_room_is_a_hint_never_a_phantom_room_or_a_visit():
    """skills-nav:look-vlm-room-phantoms — look trusted the VLM room string, created
    phantom/'unknown' rooms and visited them."""
    sg = _sg()
    rooms, visits = set(r.room_id for r in sg.get_all_rooms()), sg.get_room("hallway").visit_count
    found = [Localized("green bottle", 10.88, 3.0, 0.32, 0.6, "edgetam")]
    with mock.patch("vector_os_nano.skills.go2.look.localize_objects_3d", return_value=(found, {})):
        res = LookSkill().execute({}, _ctx(sg, _vlm("garage", ["bottle"])))
        assert res.result_data["room"] == "hallway"
        res = LookSkill().execute({}, _ctx(sg, _vlm("kitchen", [])))
    assert res.result_data["room"] == "hallway"  # the floor box wins over the VLM label
    assert set(r.room_id for r in sg.get_all_rooms()) == rooms
    assert sg.get_room("hallway").visit_count == visits
    (o,) = sg.find_objects_by_category("green bottle")
    assert (o.x, o.y, o.room_id, o.n_obs) == (10.88, 3.0, "hallway", 2)


def test_describe_scene_stores_only_localized_objects():
    """skills-nav:describe-scene-zero-objects — describe_scene stored every VLM name
    without coordinates and wrote 'query: ...' as the room description."""
    sg = _sg()
    desc = sg.get_room("hallway").representative_description
    found = [Localized("red can", 10.9, 3.22, 0.32, 0.5, "box_rect")]
    with mock.patch("vector_os_nano.skills.go2.look.localize_objects_3d", return_value=(found, {})):
        res = DescribeSceneSkill().execute({"query": "red things"}, _ctx(sg, _vlm("hallway", ["red can", "red stool"])))
    assert res.success and res.result_data["unlocalized"] == ["red stool"]
    assert [o.category for o in sg.find_objects_by_category("")] == ["red can"]
    assert sg.get_room("hallway").representative_description == desc
    assert res.result_data["objects"][0]["segmenter"] == "box_rect"


def test_look_without_anything_localized_stores_no_object():
    sg = _sg()
    with mock.patch("vector_os_nano.skills.go2.look.localize_objects_3d", return_value=([], {})):
        res = LookSkill().execute({}, _ctx(sg, _vlm("hallway", ["sofa", "lamp"])))
    assert res.success and sg.find_objects_by_category("") == []
    assert sorted(res.result_data["unlocalized"]) == ["lamp", "sofa"]


def test_explore_discovers_without_a_vlm(monkeypatch):
    """vlm-160px-gates-discovery / explore-auto-observe — explore stored objects only
    from the VLM's names; with no VLM it discovered nothing."""
    from vector_os_nano.skills.go2 import explore

    sg = _sg()
    monkeypatch.setattr(explore, "_spatial_memory", sg)
    monkeypatch.setattr(explore, "_perception", _Perception())
    monkeypatch.setattr(explore, "_vlm", None)
    monkeypatch.setattr(explore, "_vocab", _APT.vocabulary())
    found = [Localized("blue bottle", 10.9, 2.78, 0.32, 0.7, "edgetam")]
    with mock.patch("vector_os_nano.skills.go2.look.localize_objects_3d", return_value=(found, {})) as loc:
        explore._discover(_Base())
    assert "red can" in loc.call_args.args[1]
    assert [o.category for o in sg.find_objects_by_category("")] == ["blue bottle"]
    monkeypatch.setattr(explore, "_perception", None)  # nothing to look with: a no-op
    explore._discover(_Base())
