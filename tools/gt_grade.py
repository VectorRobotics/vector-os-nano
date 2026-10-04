#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""gt_grade — an independent ground-truth grade of each REPL turn, for auditing verdicts.

Reads the episode dump a sim session writes when ``VECTOR_GT_DUMP=<file>`` is set
(one JSON line at every turn start and end: base pose, every object's position and
weld, the world's surfaces and user location) and re-derives, with its OWN
thresholds, whether the turn delivered an object to the user and/or left it on a
surface. It imports nothing from vector_os_nano — no oracle, classifier or manifest
code — so a bug in the verify spine cannot also hide here (A2: the benchmark audits
every GROUNDED verdict against this).

    python tools/gt_grade.py --dump run.jsonl --object pickable_bottle_green [--surface dining_table]

Prints one JSON line per turn: {"turn", "goal", "moved_m", "delivered", "object_on"}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

DELIVER_R = 1.0     # m (xy) from the user location at turn end, and not already there at start
ON_MAX_ABOVE = 0.15  # m: the object's centre at most this far above the surface top


def load(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def turns(lines: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(start, end) line pairs, matched by turn number; an unfinished turn is dropped."""
    starts = {ln["turn"]: ln for ln in lines if ln["phase"] == "start"}
    return [(starts[ln["turn"]], ln) for ln in lines if ln["phase"] == "end" and ln["turn"] in starts]


def _xy_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def delivered(start: dict[str, Any], end: dict[str, Any], obj: str) -> bool:
    user = end.get("user")
    if not user or obj not in end["objects"]:
        return False
    return (_xy_dist(end["objects"][obj]["pos"], user) <= DELIVER_R
            and _xy_dist(start["objects"][obj]["pos"], user) > DELIVER_R)


def object_on(end: dict[str, Any], obj: str, surface: str) -> bool:
    s, o = end["surfaces"].get(surface), end["objects"].get(obj)
    if s is None or o is None or o["welded"]:
        return False
    x, y, z = o["pos"]
    x0, y0, x1, y1 = s["box"]
    return x0 <= x <= x1 and y0 <= y <= y1 and s["height"] < z <= s["height"] + ON_MAX_ABOVE


def grade(path: str, obj: str, surface: str | None = None) -> list[dict[str, Any]]:
    out = []
    for start, end in turns(load(path)):
        row = {
            "turn": end["turn"], "goal": end.get("goal", ""),
            "moved_m": round(_xy_dist(start["objects"][obj]["pos"], end["objects"][obj]["pos"]), 3)
            if obj in end["objects"] and obj in start["objects"] else None,
            "delivered": delivered(start, end, obj),
        }
        if surface is not None:
            row["object_on"] = object_on(end, obj, surface)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="the VECTOR_GT_DUMP file")
    ap.add_argument("--object", required=True, help="scene body name, e.g. pickable_bottle_green")
    ap.add_argument("--surface", help="surface name for the object_on check, e.g. dining_table")
    a = ap.parse_args(argv)
    for row in grade(a.dump, a.object, a.surface):
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
