# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2-7 long-chain regression: the detect -> foreach -> pick producer/consumer contract.

A "grab everything" plan decomposes to a detect step followed by a ``foreach`` over
the detected objects, whose body binds a per-item field into the pick param
(e.g. ``object_label="${item.name}"``). The robot-world ``DetectSkill`` historically
produced objects keyed by ``label`` only, while the decompose foreach example (and the
playground detect producer) reference ``${item.name}`` — so the binding silently failed
("Cannot locate target object") and the whole grab-everything chain broke on the first
item. The fix makes DetectSkill expose BOTH ``name`` and ``label`` so the binding
resolves whichever field the planner emits. This test guards that contract.
"""
from __future__ import annotations

import pytest

from tests.harness.sim_session import started

pytestmark = pytest.mark.sim


def _detect_objects():
    """Run DetectSkill against the headless sim's object source, return its objects list."""
    from vector_os_nano.skills.detect import DetectSkill

    with started("so101") as s:
        result = DetectSkill().execute({"query": "all objects"}, s.agent._build_context())
        assert result.success
        return list(result.result_data.get("objects", []))


def test_detect_objects_expose_both_name_and_label():
    """Every detected object must carry BOTH 'name' and 'label' (the foreach contract)."""
    objects = _detect_objects()
    assert objects, "the tabletop oracle should detect the scene objects"
    for obj in objects:
        assert "name" in obj, f"object missing 'name' (foreach ${{item.name}} would fail): {obj}"
        assert "label" in obj, f"object missing 'label': {obj}"
        # Both reference the same object identity so either binding resolves identically.
        assert obj["name"] == obj["label"], f"name/label diverged: {obj}"
        assert obj["name"], "name must be a non-empty object identifier"


def test_detected_names_are_locatable_scene_objects():
    """The bound name must be a real scene object the pick skill can re-detect."""
    objects = _detect_objects()
    names = {obj["name"] for obj in objects}
    # The bundled tabletop scene contains these graspables; detect('all objects')
    # must surface their real names (not a generic 'object_N' that pick can't locate).
    expected = {"banana", "mug", "bottle", "screwdriver", "duck", "lego"}
    assert names == expected, f"detected names {names} != scene objects {expected}"
