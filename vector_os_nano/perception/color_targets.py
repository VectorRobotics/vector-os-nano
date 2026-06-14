# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R9 — colour-segmentation target detection.

PURE numpy (no GL / mujoco) — detect the red/blue/green target boxes in a camera
frame by saturated-colour segmentation. Deterministic and dependency-free; for
distinct saturated colours it is robust on MuJoCo's basic OpenGL render (the
DQ-10 substrate-A domain-gap concern is small here). The R10 vision loop turns a
detection's bearing + area into a world-frame goal estimate.

Each detection: ``{label, x_norm, y_norm, area_frac}`` where x_norm/y_norm are
the blob centroid in [-1, 1] (0 = image centre; x_norm>0 = right), and area_frac
is the masked-pixel fraction of the frame (a coarse range proxy: bigger = closer).
"""
from __future__ import annotations

import numpy as np

# (label, predicate) — a pixel belongs to a colour when its dominant channel
# clears an absolute floor AND dominates the other two by a ratio. Saturated
# target colours pass cleanly; the grey walls / blue-ish floor do not.
_MIN_LEVEL = 90
_DOMINANCE = 1.5
_DEFAULT_MIN_AREA_FRAC = 0.004    # reject specks (< ~0.4% of the frame)


def _mask(rgb: np.ndarray, ch: int) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    chan = (r, g, b)[ch]
    others = [c for i, c in enumerate((r, g, b)) if i != ch]
    m = chan >= _MIN_LEVEL
    for o in others:
        m &= chan >= o * _DOMINANCE
    return m


def detect_targets(rgb: np.ndarray,
                   min_area_frac: float = _DEFAULT_MIN_AREA_FRAC
                   ) -> "list[dict]":
    """Detect saturated red/blue/green blobs in an (H, W, 3) uint8 RGB frame.

    Returns one detection dict per colour present (the union of that colour's
    pixels — one box per colour in the room), sorted by descending area_frac.
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return []
    h, w = rgb.shape[:2]
    total = float(h * w)
    out: list = []
    for label, ch in (("red", 0), ("green", 1), ("blue", 2)):
        m = _mask(rgb, ch)
        n = int(m.sum())
        if n == 0 or n / total < min_area_frac:
            continue
        ys, xs = np.nonzero(m)
        cx, cy = float(xs.mean()), float(ys.mean())
        out.append({
            "label": label,
            "x_norm": (cx / (w - 1)) * 2.0 - 1.0 if w > 1 else 0.0,
            "y_norm": (cy / (h - 1)) * 2.0 - 1.0 if h > 1 else 0.0,
            "area_frac": n / total,
        })
    out.sort(key=lambda d: d["area_frac"], reverse=True)
    return out
