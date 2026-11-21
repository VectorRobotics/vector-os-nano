#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""gt_grade — an independent ground-truth grade of each REPL turn, for auditing verdicts.

Reads the episode dump a sim session writes when ``VECTOR_GT_DUMP=<file>`` is set
(one JSON line at every turn start and end: base pose, every object's position and
weld, the world's surfaces and user location) and re-derives, with its OWN
thresholds, whether the turn delivered an object to the user, left it on a surface,
picked it up (held and lifted), or brought the base into a box or to a point. It
imports nothing from vector_os_nano — no oracle, classifier or manifest code — so a
bug in the verify spine cannot also hide here (A2: the benchmark audits every
GROUNDED verdict against this; A10: the live matrix grades every real-model turn).

    python scripts/gt_grade.py --dump run.jsonl --object pickable_bottle_green [--surface dining_table]
    python scripts/gt_grade.py --dump run.jsonl --box 14 0 20 5 --at 3 2.5
    python scripts/gt_grade.py --dump run.jsonl --object banana --goal 抓起香蕉 --nth 2

A dump holds one REPL session: every sim session it started (a body switch starts a
new one, counting its turns from 1 again). --goal grades one chosen turn: the --nth
(default 1) finished turn whose goal is that sentence.

Prints one JSON line per turn graded: {"turn", "goal"} plus, with --object, "moved_m",
"delivered" and "held_lifted"; with --surface, "object_on"; with --box, "base_in";
with --at, "base_at".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

DELIVER_R = 1.0     # m (xy) from the user location at turn end, and not already there at start
ON_MAX_ABOVE = 0.15  # m: the object's centre at most this far above the surface top
LIFT_DZ = 0.03      # m: a picked-up object's centre rose at least this much over the turn
AT_R = 0.5          # m (xy): the base this close to a point is at it


def load(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def turns(lines: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(start, end) line pairs in file order: an end closes the latest open start of its
    turn number (a later sim session reuses the numbers); an unfinished turn is dropped."""
    open_: dict[Any, dict[str, Any]] = {}
    out = []
    for ln in lines:
        if ln["phase"] == "start":
            open_[ln["turn"]] = ln
        elif ln["turn"] in open_:
            out.append((open_.pop(ln["turn"]), ln))
    return out


def _xy_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def delivered(start: dict[str, Any], end: dict[str, Any], obj: str) -> bool:
    user = end.get("user")
    if not user or obj not in end["objects"]:
        return False
    return (_xy_dist(end["objects"][obj]["pos"], user) <= DELIVER_R
            and _xy_dist(start["objects"][obj]["pos"], user) > DELIVER_R)


def _on(line: dict[str, Any], obj: str, surface: str) -> bool:
    s, o = line["surfaces"].get(surface), line["objects"].get(obj)
    if s is None or o is None or o["welded"]:
        return False
    x, y, z = o["pos"]
    x0, y0, x1, y1 = s["box"]
    return x0 <= x <= x1 and y0 <= y <= y1 and s["height"] < z <= s["height"] + ON_MAX_ABOVE


def object_on(start: dict[str, Any], end: dict[str, Any], obj: str, surface: str) -> bool:
    """Left on the surface in this turn: resting on it at the end and not at the start
    (an object that was already there was never placed)."""
    return _on(end, obj, surface) and not _on(start, obj, surface)


def held_lifted(start: dict[str, Any], end: dict[str, Any], obj: str) -> bool:
    """Picked up in this turn: attached at the end and not at the start, and its
    centre rose by LIFT_DZ (an object attached while still resting is no pick)."""
    s, e = start["objects"].get(obj), end["objects"].get(obj)
    return bool(s and e) and e["welded"] and not s["welded"] and e["pos"][2] - s["pos"][2] >= LIFT_DZ


def _inside(line: dict[str, Any], box: list[float]) -> bool:
    b = line.get("base")
    return b is not None and box[0] <= b[0] <= box[2] and box[1] <= b[1] <= box[3]


def base_in(start: dict[str, Any], end: dict[str, Any], box: list[float]) -> bool:
    """The base entered the xy box [x0, y0, x1, y1] in this turn (outside it at the start)."""
    return _inside(end, box) and not _inside(start, box)


def _near(line: dict[str, Any], xy: list[float]) -> bool:
    b = line.get("base")
    return b is not None and _xy_dist(b, xy) <= AT_R


def base_at(start: dict[str, Any], end: dict[str, Any], xy: list[float]) -> bool:
    """The base came within AT_R of (x, y) in this turn (farther at the start)."""
    return _near(end, xy) and not _near(start, xy)


def grade(path: str, obj: str | None = None, surface: str | None = None,
          box: list[float] | None = None, at: list[float] | None = None,
          goal: str | None = None, nth: int = 1) -> list[dict[str, Any]]:
    pairs = turns(load(path))
    if goal is not None:
        pairs = [p for p in pairs if p[1].get("goal") == goal][nth - 1:nth]
    out = []
    for start, end in pairs:
        row: dict[str, Any] = {"turn": end["turn"], "goal": end.get("goal", "")}
        if obj is not None:
            row["moved_m"] = (round(_xy_dist(start["objects"][obj]["pos"], end["objects"][obj]["pos"]), 3)
                              if obj in end["objects"] and obj in start["objects"] else None)
            row["delivered"] = delivered(start, end, obj)
            row["held_lifted"] = held_lifted(start, end, obj)
            if surface is not None:
                row["object_on"] = object_on(start, end, obj, surface)
        if box is not None:
            row["base_in"] = base_in(start, end, box)
        if at is not None:
            row["base_at"] = base_at(start, end, at)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="the VECTOR_GT_DUMP file")
    ap.add_argument("--object", help="scene body name, e.g. pickable_bottle_green")
    ap.add_argument("--surface", help="surface name for the object_on check, e.g. dining_table")
    ap.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="an xy box for the base_in check, e.g. a room's floor")
    ap.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"), help="a point for the base_at check")
    ap.add_argument("--goal", help="grade only the turn whose goal is this sentence")
    ap.add_argument("--nth", type=int, default=1, help="with --goal: its nth finished turn (default 1)")
    a = ap.parse_args(argv)
    if a.surface and not a.object:
        ap.error("--surface needs --object")
    if a.nth < 1:
        ap.error("--nth counts from 1")
    for row in grade(a.dump, a.object, a.surface, a.box, a.at, a.goal, a.nth):
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
