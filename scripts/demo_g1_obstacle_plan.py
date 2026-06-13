# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Demo: show the G1 obstacle planner routing around a box (campaign #7 R1).

Prints an ASCII top-down map of the scene — Start (S), Goal (G), the box
obstacle (#), and the planned detour path (*) — plus the straight-line vs
planned distance. No sim, no GUI, runs in a second. This is the legible
proof that the visibility-graph planner avoids obstacles.

Run:  .venv/bin/python scripts/demo_g1_obstacle_plan.py
"""
from __future__ import annotations

import math

from vector_os_nano.hardware.sim import g1_vgraph as vg


def _sample_path(path, step=0.15):
    """Densely sample the waypoint chain into points for drawing."""
    pts = []
    for a, b in zip(path, path[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / step))
        for k in range(n + 1):
            t = k / n
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return pts


def main() -> int:
    # Scene: a wall blocking the straight line from start to goal.
    start, goal = (0.0, 0.0), (5.0, 0.0)
    wall = vg.box_polygon(2.5, 0.0, 0.3, 1.6)   # centred at (2.5,0), tall
    inflation = 0.45                             # robot radius + clearance

    path, length = vg.plan_path(start, goal, [wall], inflation)
    straight = math.hypot(goal[0] - start[0], goal[1] - start[1])

    print("\n  G1 obstacle planner demo (top-down view, 1 cell = 0.25 m)\n")
    if path is None:
        print("  Goal UNREACHABLE (blocked) — planner honestly refuses.\n")
        return 0

    samples = _sample_path(path)
    inflated = vg.inflate_polygon(wall, inflation)

    # ASCII grid
    xmin, xmax, ymin, ymax = -1.0, 6.0, -2.5, 2.5
    cell = 0.25
    rows = int((ymax - ymin) / cell)
    cols = int((xmax - xmin) / cell)

    def to_cell(x, y):
        return int((x - xmin) / cell), int((ymax - y) / cell)

    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    # obstacle (real box) + inflated margin
    for r in range(rows):
        for c in range(cols):
            x = xmin + c * cell + cell / 2
            y = ymax - r * cell - cell / 2
            if vg.point_in_polygon((x, y), wall):
                grid[r][c] = "#"
            elif vg.point_in_polygon((x, y), inflated):
                grid[r][c] = "."
    # path
    for (x, y) in samples:
        c, r = to_cell(x, y)
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] in " .":
            grid[r][c] = "*"
    sc, sr = to_cell(*start)
    gc, gr = to_cell(*goal)
    grid[sr][sc] = "S"
    grid[gr][gc] = "G"

    for row in grid:
        print("  " + "".join(row))
    print()
    print(f"  legend: S=start  G=goal  #=obstacle  .=safety margin  *=planned path")
    print(f"  waypoints: {[(round(x, 2), round(y, 2)) for x, y in path]}")
    print(f"  straight-line distance : {straight:.2f} m (blocked by the wall)")
    print(f"  planned detour distance: {length:.2f} m (the honest geodesic)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
