# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deictic "front object" resolver — segment the thing in front, from sensors.

The acceptance phrase "抓前面的东西" (grasp the thing in front) is DEICTIC: it
names no object, so it needs no VLM naming. The right tool for a deictic spatial
query is a salient-object + depth resolver, not a language model. This finds the
most central salient blob in the camera's forward FOV (objects are visually
distinct from the muted table/floor/walls) and returns its pixel mask, from which
grasp_point.py derives the 3D grasp point from REAL depth — honest perception,
never a ground-truth pose lookup.

Pluggable: a VLM (named objects) or EdgeTAM (precise instance masks) can replace
this resolver when available; the 3D-point math downstream is identical. Needs
cv2 ([sim] declares it): without it the import fails loud.
"""
from __future__ import annotations

import cv2
import numpy as np

_SAT_MIN = 140      # HSV saturation: vivid objects (cylinders sat~164-194) vs the
                    # muted scene. The brown table is NOT cleanly below this — its
                    # saturation reaches ~160 (med~132, p90~146), so a raw threshold
                    # leaves thin 1-3px table bridges that 8-connectivity FUSES the
                    # vivid cylinders + table into one giant blob (the central object
                    # then no longer exists as its own component → a brown table
                    # sliver wins). _OPEN_KSIZE below severs those bridges; a
                    # VLM/EdgeTAM front-end would be lighting/texture-robust instead.
_VAL_MIN = 40       # reject near-black
_MIN_BLOB = 50      # px; reject speckle
_MAX_DEPTH = 2.0    # m; the front WORKSPACE only — excludes the far room/doorway
_FRONT_BAND = 0.4   # m; blobs within this of the nearest count as "the front shelf"
_OPEN_KSIZE = 3     # px; morphological-opening kernel. Erode→dilate severs the thin
                    # saturation bridges between the table and the vivid cylinders so
                    # each object survives as its OWN connected component, robustly
                    # and threshold-independently (verified stable for sat_min 140-155;
                    # a pure threshold flips central→wrong-object within ~5 sat units).

# --- ATTRIBUTE (colour) selection (D47) ---------------------------------------
# OpenCV hue is 0..180. Red wraps the 0/180 seam so it needs two bands. Bands sit
# around the scene objects' nominal hues (red rgba .85/.25/.20 -> 2, orange
# .95/.50/.10 -> 14, yellow .95/.80/.15 -> 24, green .25/.70/.35 -> 70, blue
# .20/.40/.85 -> 111, purple .55/.25/.75 -> 138); REAL-VERIFY measures the rendered
# blob hues and these are tuned if the render fidelity shifts them.
_COLOR_HUE: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (168, 180)],
    "orange": [(11, 19)],
    "yellow": [(20, 34)],
    "green": [(38, 88)],
    "blue": [(92, 128)],
    "purple": [(129, 160)],
}


def parse_color(query: str | None) -> str | None:
    """The canonical colour *query* names (the shared world vocabulary), else None.

    English colour words match on word boundaries ("infrared" is not red), CJK as
    substrings ("抓红色的东西" -> red). Only colours this resolver has a hue band for
    are returned; a deictic query with no colour word returns None (front-most).
    """
    from vector_os_nano.perception.vocab import vocab

    colour = vocab().colour(query)
    return colour if colour in _COLOR_HUE else None


def colour_matches(rgb: np.ndarray, mask: np.ndarray, color: str, *, min_px: int = 20) -> bool | None:
    """Does the vivid part of *mask* on *rgb* have *color*'s hue? None = cannot tell.

    The median hue of the mask's saturated pixels against the colour's band. None
    when the colour has no band or too few vivid pixels sit under the mask.
    """
    if color not in _COLOR_HUE:
        return None
    sat, val = _saturation(rgb)
    vivid = (mask > 0) & (sat >= _SAT_MIN // 2) & (val >= _VAL_MIN)
    if int(vivid.sum()) < min_px:
        return None
    return _hue_in_color(float(np.median(_hue(rgb)[vivid])), color)


def _hue(rgb: np.ndarray) -> np.ndarray:
    """The HSV hue channel (OpenCV's 0..180 scale) of *rgb*."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 0]


def _hue_in_color(hue_val: float, color: str) -> bool:
    """True iff *hue_val* (0..180) falls in any band of *color*."""
    for lo, hi in _COLOR_HUE.get(color, []):
        if lo <= hue_val <= hi:
            return True
    return False


def _saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(saturation, value) of *rgb*, both 0..255."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1], hsv[:, :, 2]


def _open(binary: np.ndarray, ksize: int = _OPEN_KSIZE) -> np.ndarray:
    """Morphological opening (erode→dilate) — sever thin bridges between blobs.

    The salient mask fuses distinct vivid objects to each other and to the table
    wherever a 1-3px chain of borderline-saturation pixels connects them; after
    8-connectivity that makes one giant blob and the central object stops being a
    selectable component. Opening removes structures thinner than the kernel
    (the bridges) while preserving the solid object bodies.
    """
    if ksize < 2:
        return binary
    return cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((ksize, ksize), np.uint8))


def _components(binary: np.ndarray):
    """8-connected components -> (n, labels, centroids, areas)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return n, labels, centroids, stats[:, cv2.CC_STAT_AREA]


def front_object_mask(
    rgb: np.ndarray,
    depth: np.ndarray | None = None,
    *,
    color: str | None = None,
    sat_min: int = _SAT_MIN,
    val_min: int = _VAL_MIN,
    min_blob: int = _MIN_BLOB,
    central_frac: float = 0.8,
    max_depth: float = _MAX_DEPTH,
) -> np.ndarray | None:
    """Binary mask (H, W) uint8 of the salient object most in front, or None.

    Salient = high HSV saturation (objects are vivid vs the muted scene),
    gated to the central forward FOV and to near, valid depth. Among qualifying
    connected blobs, the one whose centroid is CLOSEST to the image centre wins
    ("前面" = directly ahead). Returns None when nothing salient is in front
    (FAIL LOUD — never fabricate a target).

    ATTRIBUTE selection (D47): when *color* is given ('red'/'green'/'blue'), keep
    only the salient blobs whose MEDIAN hue falls in `_COLOR_HUE[color]` and pick the
    most-central of THOSE — returns None if none match (FAIL LOUD, never falls back
    to the front-most for a colour query). color=None → front-most behaviour, UNCHANGED.
    """
    if rgb is None or rgb.ndim != 3:
        return None
    h, w = rgb.shape[:2]
    sat, val = _saturation(rgb)
    salient = ((sat >= sat_min) & (val >= val_min)).astype(np.uint8)

    # Central forward FOV gate.
    cx0, cx1 = int(w * (1 - central_frac) / 2), int(w - w * (1 - central_frac) / 2)
    cy0, cy1 = int(h * (1 - central_frac) / 2), int(h - h * (1 - central_frac) / 2)
    gate = np.zeros((h, w), dtype=np.uint8)
    gate[cy0:cy1, cx0:cx1] = 1
    salient &= gate

    # Near, valid depth only (the front workspace, not the far room/wall).
    if depth is not None:
        salient &= ((depth > 0) & (depth <= max_depth)).astype(np.uint8)

    # Sever thin saturation bridges so each vivid object survives as its OWN
    # connected component (the table else fuses the cylinders into one blob and
    # the central object vanishes — the bug this resolver previously had).
    salient = _open(salient)

    if int(salient.sum()) < min_blob:
        return None

    n, labels, centroids, areas = _components(salient)
    img_c = np.array([w / 2.0, h / 2.0])

    # ATTRIBUTE selection: when a colour is requested, pre-compute the hue over the
    # salient pixels so each blob can be matched on its median hue (computed over the
    # blob ∩ salient pixels — exactly the vivid object body, not its anti-aliased rim).
    hue_img = _hue(rgb) if color is not None else None
    if color is not None and color not in _COLOR_HUE:
        return None  # FAIL LOUD on an unknown colour, never the front-most

    # Per-blob: centrality + median depth. "前面" = NEAREST (a foreground object
    # at ~1 m beats a same-colored background object at ~3 m, which 'most central'
    # would wrongly pick when the background sits dead-centre in a doorway).
    cands = []
    for i in range(1, n):
        if areas[i] < min_blob:
            continue
        comp = labels == i
        # Colour gate: keep only blobs whose MEDIAN hue is in the requested band.
        if color is not None:
            hv = hue_img[comp]
            if hv.size == 0 or not _hue_in_color(float(np.median(hv)), color):
                continue
        cen = float(np.linalg.norm(centroids[i] - img_c))
        if depth is not None:
            dv = depth[comp & (depth > 0)]
            zmed = float(np.median(dv)) if dv.size else float("inf")
        else:
            zmed = 0.0
        cands.append((i, cen, zmed))
    if not cands:
        return None  # colour query with no matching blob -> FAIL LOUD (None)

    if color is not None:
        # Colour query: among the colour-matched blobs, the most central wins
        # (the target colour need NOT be the front-most object — that is the point).
        best = min(cands, key=lambda c: c[1])
    elif depth is not None:
        z_near = min(c[2] for c in cands)
        front_shelf = [c for c in cands if c[2] <= z_near + _FRONT_BAND]
        best = min(front_shelf, key=lambda c: c[1])  # most central of the nearest shelf
    else:
        best = min(cands, key=lambda c: c[1])         # no depth -> most central
    return (labels == best[0]).astype(np.uint8)
