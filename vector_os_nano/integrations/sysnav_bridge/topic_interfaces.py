# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Typed shadow dataclasses for SysNav's ``tare_planner`` ROS2 messages.

We do NOT import SysNav at module load time. The shadow types replicate
the public field layout of these messages so that:

1. Unit tests can construct payloads without depending on a sourced
   SysNav workspace.
2. The bridge has a single conversion path
   (``object_node_to_state`` / ``room_node_to_dict``) that takes either
   the live ROS2 message or a shadow object — duck-typed by attribute
   access.
3. Removing the SysNav dependency from this repo is mechanical: nothing
   here imports it.

Field reference: ``src/exploration_planner/tare_planner/msg/*.msg`` in
the SysNav repo (PolyForm-Noncommercial-1.0.0). API/contract is not
copyrightable when reimplemented; the dataclass layout below is the
shadow, not a copy of any upstream source code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from vector_os_nano.core.world_model import ObjectState


# ---------------------------------------------------------------------------
# Geometry helpers — minimum viable typed mirrors for geometry_msgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Point:
    """Mirror of ``geometry_msgs/Point`` — ``x, y, z`` only."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class _Header:
    """Mirror of ``std_msgs/Header`` — frame + stamp fields we actually read."""

    frame_id: str = "map"
    stamp_sec: int = 0
    stamp_nanosec: int = 0

    @property
    def stamp_seconds(self) -> float:
        return float(self.stamp_sec) + float(self.stamp_nanosec) * 1e-9


# ---------------------------------------------------------------------------
# ObjectNode shadow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectNodeShadow:
    """Typed shadow of ``tare_planner/ObjectNode``.

    SysNav merges duplicate detections; ``object_id`` is a list of all
    historic ids that map to this object — index 0 is the canonical /
    most-recent id.
    """

    header: _Header = field(default_factory=_Header)
    object_id: tuple[int, ...] = ()
    label: str = ""
    position: _Point = field(default_factory=_Point)
    bbox3d: tuple[_Point, ...] = ()           # 8 corners — oriented bbox
    cloud: Any = None                          # PointCloud2; opaque here
    status: bool = True                        # SysNav semantics: True = active
    img_path: str = ""
    is_asked_vlm: bool = False
    viewpoint_id: int = -1


@dataclass(frozen=True)
class ObjectNodeListShadow:
    """Typed shadow of ``tare_planner/ObjectNodeList``."""

    header: _Header = field(default_factory=_Header)
    nodes: tuple[ObjectNodeShadow, ...] = ()


# ---------------------------------------------------------------------------
# RoomNode shadow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoomNodeShadow:
    """Typed shadow of ``tare_planner/RoomNode``.

    Only the fields Vector OS Nano consumes are mirrored. ``polygon`` is
    represented as a tuple of (x, y) points — the upstream
    ``PolygonStamped`` carries 3D points but rooms are floor-plane in
    practice.
    """

    id: int = -1
    show_id: int = -1
    polygon: tuple[tuple[float, float], ...] = ()
    centroid: _Point = field(default_factory=_Point)
    neighbors: tuple[int, ...] = ()
    is_connected: bool = True
    area: float = 0.0
    room_mask: Any = None                     # sensor_msgs/Image; opaque


# ---------------------------------------------------------------------------
# Pure-function adapters
# ---------------------------------------------------------------------------


def _state_from_status(sysnav_status: bool, prior: ObjectState | None) -> str:
    """Map SysNav ``status`` to ``ObjectState.state``.

    SysNav internally tracks ``new`` / ``persistent`` / ``moving`` /
    ``disappeared`` but the wire schema collapses to a single bool
    ``status`` (active or not). Without that richer signal here we
    preserve any prior ``grasped`` / ``placed`` state set by skills,
    and otherwise default to ``on_table`` for active observations.
    """
    if not sysnav_status:
        return "unknown"
    if prior is not None and prior.state in ("grasped", "placed"):
        return prior.state
    return "on_table"


def object_node_to_state(
    node: Any,
    *,
    prior: ObjectState | None = None,
    asked_vlm_confidence: float = 1.0,
    unverified_confidence: float = 0.7,
) -> ObjectState:
    """Convert an ``ObjectNode``-shaped payload to an ``ObjectState``.

    Accepts either an :class:`ObjectNodeShadow` or a live
    ``tare_planner.msg.ObjectNode`` — duck-typed by attribute access.

    Args:
        node: The upstream / shadow object.
        prior: An existing ``ObjectState`` for the same id, if one is
            already in the world model. Used to preserve ``grasped`` /
            ``placed`` states.
        asked_vlm_confidence: Confidence to assign once SysNav reports
            ``is_asked_vlm == True`` (VLM has confirmed the label).
        unverified_confidence: Confidence assigned to objects that have
            been geometrically detected but not yet VLM-verified.

    Returns:
        Frozen :class:`ObjectState` ready for ``WorldModel.add_object``.
    """
    object_ids: Sequence[int] = tuple(getattr(node, "object_id", ()) or ())
    canonical_id = (
        f"sysnav_{int(object_ids[0])}" if object_ids else "sysnav_unknown"
    )

    label = str(getattr(node, "label", "") or "").strip().lower()

    pos = getattr(node, "position", None)
    x = float(getattr(pos, "x", 0.0) or 0.0)
    y = float(getattr(pos, "y", 0.0) or 0.0)
    z = float(getattr(pos, "z", 0.0) or 0.0)

    is_asked_vlm = bool(getattr(node, "is_asked_vlm", False))
    confidence = (
        asked_vlm_confidence if is_asked_vlm else unverified_confidence
    )

    state = _state_from_status(bool(getattr(node, "status", True)), prior)

    header = getattr(node, "header", None)
    if header is not None and hasattr(header, "stamp_seconds"):
        last_seen = float(header.stamp_seconds)
    else:
        last_seen = time.time()

    bbox_corners = tuple(getattr(node, "bbox3d", ()) or ())
    bbox_serialised = tuple(
        (float(getattr(c, "x", 0.0)), float(getattr(c, "y", 0.0)),
         float(getattr(c, "z", 0.0)))
        for c in bbox_corners
    )

    properties: dict[str, Any] = {
        "source": "sysnav",
        "object_id_history": tuple(int(o) for o in object_ids),
        "viewpoint_id": int(getattr(node, "viewpoint_id", -1) or -1),
        "is_asked_vlm": is_asked_vlm,
        "img_path": str(getattr(node, "img_path", "") or ""),
        "bbox3d_corners": bbox_serialised,
    }

    return ObjectState(
        object_id=canonical_id,
        label=label,
        x=x,
        y=y,
        z=z,
        confidence=confidence,
        state=state,
        last_seen=last_seen,
        properties=properties,
    )
