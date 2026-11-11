# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deictic front-object resolver — segment the salient thing in front."""
from __future__ import annotations

import numpy as np

from vector_os_nano.perception.front_object import front_object_mask, parse_color

_H, _W = 48, 64


def _muted_bg():
    """A muted gray scene (low saturation) — like the table/floor/walls."""
    rgb = np.full((_H, _W, 3), 130, dtype=np.uint8)
    rgb[:, :, 0] += 5  # tiny tint, still low saturation
    return rgb


def _put_blob(rgb, cx, cy, color, r=5):
    for v in range(max(0, cy - r), min(_H, cy + r)):
        for u in range(max(0, cx - r), min(_W, cx + r)):
            rgb[v, u] = color


def test_without_cv2_the_resolver_fails_loud_not_on_a_degenerate_fallback(monkeypatch):
    """R8a-V2: the numpy fallback merged every blob into one component, so without cv2
    the resolver silently picked the wrong object. cv2 is declared by [sim]; its
    absence must be an ImportError, never a wrong mask."""
    import importlib
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.delitem(sys.modules, "vector_os_nano.perception.front_object")
    with pytest.raises(ImportError):
        importlib.import_module("vector_os_nano.perception.front_object")


def test_central_salient_object_found():
    rgb = _muted_bg()
    _put_blob(rgb, _W // 2, _H // 2, (220, 30, 30))  # vivid red, centre
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    m = front_object_mask(rgb, depth)
    assert m is not None
    ys, xs = np.where(m > 0)
    assert abs(xs.mean() - _W / 2) < 6 and abs(ys.mean() - _H / 2) < 6


def test_picks_most_central_of_two():
    rgb = _muted_bg()
    _put_blob(rgb, _W // 2, _H // 2, (30, 220, 30))   # central green
    _put_blob(rgb, 8, 8, (30, 30, 220))               # off-corner blue
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    m = front_object_mask(rgb, depth)
    ys, xs = np.where(m > 0)
    assert abs(xs.mean() - _W / 2) < 8  # the central blob, not the corner


def test_muted_scene_returns_none():
    rgb = _muted_bg()
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    assert front_object_mask(rgb, depth) is None


def test_far_object_rejected_by_depth():
    rgb = _muted_bg()
    _put_blob(rgb, _W // 2, _H // 2, (220, 30, 30))
    depth = np.full((_H, _W), 9.0, dtype=np.float32)  # all far -> beyond max_depth
    assert front_object_mask(rgb, depth, max_depth=4.0) is None


def test_speckle_below_min_blob_rejected():
    rgb = _muted_bg()
    rgb[24, 32] = (250, 0, 0)  # one vivid pixel
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    assert front_object_mask(rgb, depth, min_blob=50) is None


def test_thin_bridge_keeps_central_object_selectable():
    """Regression (D31): a thin saturated table band must not lose the central object.

    The real go2+piper bug: the brown table's saturation reached the threshold in
    a thin chain that 8-connectivity FUSED the central green cylinder into one
    giant blob with the off-centre cylinders + table, so the central object
    stopped existing as its own component and a brown table sliver won (12.2cm
    error → green@2.3cm after the fix; proven on the live sim, not reproducible
    in a clean synthetic frame because the winning sliver depends on real table
    texture). This guards that opening still yields the central GREEN object as a
    distinct component in a multi-object, table-bridged scene.
    """
    rgb = _muted_bg()
    cy = _H // 2
    # A wide, 1px-THIN saturated table band spanning the scene at the cylinders'
    # base — its saturation just clears the threshold (like the real brown table),
    # so without opening it 8-connects every vivid object into one giant blob and
    # the central object stops existing as its own component (a sliver wins).
    rgb[cy + 4, 6 : _W - 6] = (120, 200, 120)
    _put_blob(rgb, _W // 2, cy, (30, 220, 30), r=4)        # central green (target)
    _put_blob(rgb, _W // 2 - 18, cy, (220, 30, 30), r=4)   # left red
    _put_blob(rgb, _W // 2 + 18, cy, (30, 30, 220), r=4)   # right blue
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    m = front_object_mask(rgb, depth)
    assert m is not None
    ys, xs = np.where(m > 0)
    # Selection must land on the CENTRAL green blob, not a fused-blob centroid
    # nor an off-centre object/table sliver.
    assert abs(xs.mean() - _W / 2) < 5, (
        f"selected centroid x={xs.mean():.1f} drifted off the central object "
        "— the thin table band fused the blobs (opening failed)"
    )
    sel = rgb[m > 0]
    assert sel[:, 1].mean() > sel[:, 0].mean() + 20, "selected object is not the green one"


# ============================= ATTRIBUTE (colour) selection (D47) ============

def _three_color_scene():
    """Muted bg with three distinct-hue blobs at known x positions on a flat depth.

    Layout (left→right): RED at x≈12, GREEN central at x=W/2, BLUE at x≈52.
    Vivid rgba mirrors the scene cylinders so the median-hue gate is exercised on
    realistic colours: red (217,64,51), green (64,179,89), blue (51,102,217).
    """
    rgb = _muted_bg()
    cy = _H // 2
    _put_blob(rgb, 12, cy, (217, 64, 51), r=4)        # red, left
    _put_blob(rgb, _W // 2, cy, (64, 179, 89), r=4)   # green, central
    _put_blob(rgb, _W - 12, cy, (51, 102, 217), r=4)  # blue, right
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    return rgb, depth


def test_parse_color_chinese_and_english():
    assert parse_color("抓红色的东西") == "red"
    assert parse_color("抓蓝色的") == "blue"
    assert parse_color("抓绿色的") == "green"
    assert parse_color("grab the red can") == "red"
    assert parse_color("pick the blue bottle") == "blue"


def test_parse_color_deictic_is_none():
    assert parse_color("抓前面的东西") is None
    assert parse_color("前面的东西") is None
    assert parse_color("grab the thing in front") is None
    assert parse_color("") is None
    assert parse_color(None) is None


def test_color_red_selects_red_blob():
    rgb, depth = _three_color_scene()
    m = front_object_mask(rgb, depth, color="red")
    assert m is not None
    xs = np.where(m > 0)[1]
    assert abs(xs.mean() - 12) < 5, f"red selection centroid x={xs.mean():.1f} not on red blob"
    sel = rgb[m > 0]
    assert sel[:, 0].mean() > sel[:, 1].mean() + 20 and sel[:, 0].mean() > sel[:, 2].mean() + 20


def test_color_blue_selects_blue_blob():
    rgb, depth = _three_color_scene()
    m = front_object_mask(rgb, depth, color="blue")
    assert m is not None
    xs = np.where(m > 0)[1]
    assert abs(xs.mean() - (_W - 12)) < 5, f"blue selection centroid x={xs.mean():.1f} not on blue blob"
    sel = rgb[m > 0]
    assert sel[:, 2].mean() > sel[:, 0].mean() + 20 and sel[:, 2].mean() > sel[:, 1].mean() + 20


def test_color_green_selects_green_blob():
    rgb, depth = _three_color_scene()
    m = front_object_mask(rgb, depth, color="green")
    assert m is not None
    xs = np.where(m > 0)[1]
    assert abs(xs.mean() - _W / 2) < 5, f"green selection centroid x={xs.mean():.1f} not on green blob"
    sel = rgb[m > 0]
    assert sel[:, 1].mean() > sel[:, 0].mean() + 20 and sel[:, 1].mean() > sel[:, 2].mean() + 20


def test_color_none_keeps_frontmost_behavior():
    """color=None must reproduce the existing front/central selection (the green centre)."""
    rgb, depth = _three_color_scene()
    m = front_object_mask(rgb, depth, color=None)
    assert m is not None
    xs = np.where(m > 0)[1]
    assert abs(xs.mean() - _W / 2) < 6, "color=None should pick the central blob, unchanged"


def test_color_no_match_fails_loud():
    """A colour query with no blob of that colour returns None — never the front-most."""
    rgb = _muted_bg()
    _put_blob(rgb, _W // 2, _H // 2, (64, 179, 89), r=4)  # only a green blob present
    depth = np.full((_H, _W), 1.0, dtype=np.float32)
    assert front_object_mask(rgb, depth, color="red") is None
    assert front_object_mask(rgb, depth, color="blue") is None
    assert front_object_mask(rgb, depth, color="green") is not None
