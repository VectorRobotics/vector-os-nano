# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SysNav bridge — ObjectNode → ObjectState mapping tests.

Uses typed shadow dataclasses so the test suite does not depend on a
sourced SysNav workspace or the ``tare_planner.msg`` ROS2 package.
"""
from __future__ import annotations

from vector_os_nano.core.world_model import ObjectState
from vector_os_nano.integrations.sysnav_bridge import (
    ObjectNodeShadow,
    object_node_to_state,
)
from vector_os_nano.integrations.sysnav_bridge.topic_interfaces import (
    _Header,
    _Point,
)


def _node(
    *,
    object_id: tuple[int, ...] = (42,),
    label: str = "blue bottle",
    position: tuple[float, float, float] = (1.5, 2.0, 0.25),
    bbox: tuple[tuple[float, float, float], ...] = (),
    status: bool = True,
    is_asked_vlm: bool = False,
    img_path: str = "",
    viewpoint_id: int = 7,
    stamp_sec: int = 1714000000,
) -> ObjectNodeShadow:
    return ObjectNodeShadow(
        header=_Header(frame_id="map", stamp_sec=stamp_sec, stamp_nanosec=0),
        object_id=object_id,
        label=label,
        position=_Point(*position),
        bbox3d=tuple(_Point(*c) for c in bbox),
        status=status,
        img_path=img_path,
        is_asked_vlm=is_asked_vlm,
        viewpoint_id=viewpoint_id,
    )


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_basic_position_and_label_mapping() -> None:
    state = object_node_to_state(_node())
    assert state.label == "blue bottle"
    assert (state.x, state.y, state.z) == (1.5, 2.0, 0.25)
    assert state.object_id == "sysnav_42"
    assert state.state == "on_table"


def test_label_normalised_lowercase_strip() -> None:
    state = object_node_to_state(_node(label="  Red CAN  "))
    assert state.label == "red can"


def test_canonical_id_uses_first_object_id() -> None:
    state = object_node_to_state(_node(object_id=(99, 100, 7)))
    assert state.object_id == "sysnav_99"
    assert state.properties["object_id_history"] == (99, 100, 7)


def test_empty_object_id_falls_back_to_unknown() -> None:
    state = object_node_to_state(_node(object_id=()))
    assert state.object_id == "sysnav_unknown"


# ---------------------------------------------------------------------------
# confidence semantics
# ---------------------------------------------------------------------------


def test_confidence_low_when_vlm_not_asked() -> None:
    state = object_node_to_state(_node(is_asked_vlm=False))
    assert state.confidence == 0.7


def test_confidence_high_when_vlm_asked() -> None:
    state = object_node_to_state(_node(is_asked_vlm=True))
    assert state.confidence == 1.0


def test_confidence_levels_are_overridable() -> None:
    state = object_node_to_state(
        _node(is_asked_vlm=True),
        asked_vlm_confidence=0.95,
        unverified_confidence=0.5,
    )
    assert state.confidence == 0.95


# ---------------------------------------------------------------------------
# status / state transitions
# ---------------------------------------------------------------------------


def test_inactive_status_maps_to_unknown_state() -> None:
    state = object_node_to_state(_node(status=False))
    assert state.state == "unknown"


def test_grasped_state_preserved_across_update() -> None:
    prior = ObjectState(
        object_id="sysnav_42", label="blue bottle",
        x=0.0, y=0.0, z=0.0, state="grasped",
    )
    state = object_node_to_state(_node(), prior=prior)
    # Even though sysnav reports active/on_table, we keep the skill-set
    # state so a freshly grasped object is not "demoted" to on_table.
    assert state.state == "grasped"


def test_placed_state_preserved_across_update() -> None:
    prior = ObjectState(
        object_id="sysnav_42", label="blue bottle",
        x=0.0, y=0.0, z=0.0, state="placed",
    )
    state = object_node_to_state(_node(), prior=prior)
    assert state.state == "placed"


def test_unknown_prior_state_does_not_block_observation() -> None:
    prior = ObjectState(
        object_id="sysnav_42", label="blue bottle",
        x=0.0, y=0.0, z=0.0, state="unknown",
    )
    state = object_node_to_state(_node(), prior=prior)
    assert state.state == "on_table"


# ---------------------------------------------------------------------------
# auxiliary fields preserved into properties
# ---------------------------------------------------------------------------


def test_bbox_corners_serialised_into_properties() -> None:
    corners = tuple((float(i), float(i + 1), float(i + 2)) for i in range(8))
    state = object_node_to_state(_node(bbox=corners))
    assert state.properties["bbox3d_corners"] == corners


def test_viewpoint_and_img_path_preserved() -> None:
    state = object_node_to_state(
        _node(viewpoint_id=11, img_path="/tmp/sysnav/42.png"),
    )
    assert state.properties["viewpoint_id"] == 11
    assert state.properties["img_path"] == "/tmp/sysnav/42.png"
    assert state.properties["source"] == "sysnav"


def test_last_seen_taken_from_header_stamp() -> None:
    state = object_node_to_state(_node(stamp_sec=1714000000))
    assert state.last_seen == 1714000000.0


# ---------------------------------------------------------------------------
# duck-typing — object without dataclass
# ---------------------------------------------------------------------------


class _FakeMessage:
    """Mimics a live ``tare_planner.msg.ObjectNode`` via plain attributes."""

    class _Pos:
        x = 3.0
        y = 4.0
        z = 0.5

    class _H:
        frame_id = "map"

        @property
        def stamp_seconds(self):
            return 1714000123.0

    object_id = [777]
    label = "GREEN bottle"
    position = _Pos()
    bbox3d = []
    status = True
    img_path = ""
    is_asked_vlm = True
    viewpoint_id = 0
    header = _H()


def test_adapter_accepts_duck_typed_message() -> None:
    state = object_node_to_state(_FakeMessage())
    assert state.object_id == "sysnav_777"
    assert state.label == "green bottle"
    assert state.x == 3.0
    assert state.confidence == 1.0
    assert state.last_seen == 1714000123.0
