# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Rule-5 moat: a SINGLE source of truth for "is this object a verify-only GT
anchor?" plus perceived-only accessors.

Boot seeds the room's targets into the world model with their ground-truth
coordinates as the deterministic VERIFY anchor (``visited``/``at_position`` bind
to them — the honest judge). Those anchors must be invisible to every PLANNER-
or MEANS-facing reader (the prompt serializers and the navigate label resolver),
else the robot "arrives" by reading GT instead of perceiving. This module owns
the one predicate (``is_verify_anchor``) and the perceived-only WorldModel
accessors that every such reader routes through, so the next sibling reader can
not silently re-open the hole (campaign #12 — centralized fix).
"""
from __future__ import annotations

from vector_os_nano.core.world_model import (
    ObjectState,
    WorldModel,
    is_verify_anchor,
)


def _gt(label: str, x: float, y: float, *, flag: bool = True, src: str = "g1_room_ground_truth") -> ObjectState:
    props: dict = {"source": src}
    if flag:
        props["verify_anchor"] = True
    return ObjectState(object_id=f"target_{label}", label=label, x=x, y=y,
                       confidence=1.0, state="placed", properties=props)


def _perceived(label: str, x: float, y: float, conf: float = 0.7) -> ObjectState:
    return ObjectState(object_id=f"{label}#live", label=label, x=x, y=y,
                       confidence=conf, state="detected",
                       properties={"source": "vlm_detect"})


class TestIsVerifyAnchor:
    def test_explicit_flag_is_anchor(self):
        assert is_verify_anchor(_gt("chair", 3.6, 0.0, flag=True, src="")) is True

    def test_ground_truth_source_is_anchor_backcompat(self):
        # objects seeded BEFORE the explicit flag existed carry only the source.
        assert is_verify_anchor(_gt("chair", 3.6, 0.0, flag=False)) is True
        assert is_verify_anchor(_gt("c", 0, 0, flag=False, src="go2_room_ground_truth")) is True

    def test_perceived_is_not_anchor(self):
        assert is_verify_anchor(_perceived("chair", 2.9, 0.1)) is False

    def test_object_without_properties_is_not_anchor(self):
        bare = ObjectState(object_id="x", label="cube", x=1.0, y=1.0)
        assert is_verify_anchor(bare) is False


class TestPerceivedAccessors:
    def test_get_perceived_objects_excludes_anchors(self):
        wm = WorldModel()
        wm.add_object(_gt("chair", 3.6, 0.0))
        wm.add_object(_perceived("sofa", 1.0, 2.0))
        perceived = wm.get_perceived_objects()
        labels = {o.label for o in perceived}
        assert labels == {"sofa"}

    def test_get_perceived_objects_by_label_filters_gt(self):
        wm = WorldModel()
        wm.add_object(_gt("chair", 3.6, 0.0))           # anchor only
        wm.add_object(_perceived("sofa", 1.0, 2.0))
        assert wm.get_perceived_objects_by_label("chair") == []   # GT not a means
        assert [o.label for o in wm.get_perceived_objects_by_label("sofa")] == ["sofa"]

    def test_perceived_label_not_shadowed_by_gt_anchor(self):
        # A GT chair (conf 1.0) must not shadow a PERCEIVED chair (conf 0.7):
        # the anchor is filtered, so only the perceived one is a valid means.
        wm = WorldModel()
        wm.add_object(_gt("chair", 3.6, 0.0))
        wm.add_object(_perceived("chair", 2.9, 0.1))
        got = wm.get_perceived_objects_by_label("chair")
        assert len(got) == 1
        assert got[0].confidence == 0.7
        assert abs(got[0].x - 2.9) < 1e-9      # the perceived position, not GT

    def test_get_objects_still_returns_everything(self):
        # The raw accessor (verify side) is UNCHANGED — it must still see anchors.
        wm = WorldModel()
        wm.add_object(_gt("chair", 3.6, 0.0))
        wm.add_object(_perceived("sofa", 1.0, 2.0))
        assert len(wm.get_objects()) == 2
