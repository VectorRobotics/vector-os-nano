# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Visibility-graph global planner for G1 obstacle navigation (campaign #7).

PURE planar geometry — NO mujoco/GL import, so it unit-tests offline and is
the SINGLE SOURCE OF TRUTH for both the path the follower walks AND the
``geodesic_distance`` the verify predicate reads (the #1 QA gate from the
design review: execution and verify must never use a different graph).

Pipeline: obstacle polygons -> inflate by the robot radius -> visibility
graph over {start, goal, inflated corners} with line-of-sight-clear edges ->
Dijkstra. ``plan_path`` returns the waypoint chain; ``path_length`` the true
geodesic (``inf`` when the goal is inside an obstacle or boxed in — an honest
'unreachable' that fails the verify predicate, never a phantom euclidean
'close', rule 5).

Conventions: a polygon is a list of (x, y) vertices (CCW or CW, convex).
Boxes give 4 corners; cylinders a CIRCUMSCRIBED n-gon (outer approximation —
conservative, never optimistic about clearance).
"""
from __future__ import annotations

import heapq
import math

Point = "tuple[float, float]"
Polygon = "list[tuple[float, float]]"

_EPS = 1e-9


# --------------------------------------------------------------------------
# Primitive shapes
# --------------------------------------------------------------------------

def box_polygon(cx: float, cy: float, hx: float, hy: float,
                yaw: float = 0.0) -> "list[tuple[float, float]]":
    """Four corners of a (possibly yaw-rotated) box centred at (cx, cy) with
    half-extents (hx, hy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for sx, sy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        corners.append((cx + sx * c - sy * s, cy + sx * s + sy * c))
    return corners


def cylinder_polygon(cx: float, cy: float, r: float,
                     n: int = 8) -> "list[tuple[float, float]]":
    """A CIRCUMSCRIBED n-gon around a disc — outer approximation so the
    collision polygon never under-approximates the true disc (conservative)."""
    # circumscribed: vertices sit on radius r / cos(pi/n) so every edge is
    # tangent to (outside) the true circle.
    rr = r / math.cos(math.pi / n)
    return [(cx + rr * math.cos(2 * math.pi * k / n + math.pi / n),
             cy + rr * math.sin(2 * math.pi * k / n + math.pi / n))
            for k in range(n)]


def _centroid(poly: "list[tuple[float, float]]") -> "tuple[float, float]":
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def inflate_polygon(poly: "list[tuple[float, float]]",
                    r: float) -> "list[tuple[float, float]]":
    """Push every vertex radially OUTWARD from the centroid by ``r``.

    Conservative for convex polygons (corners expand at least r; the design's
    clearance>=tol-floor invariant absorbs the slight diagonal over-inflation,
    which only ever makes the planner SAFER)."""
    cx, cy = _centroid(poly)
    out = []
    for x, y in poly:
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d < _EPS:
            out.append((x, y))
        else:
            out.append((x + dx / d * r, y + dy / d * r))
    return out


# --------------------------------------------------------------------------
# Geometry predicates
# --------------------------------------------------------------------------

def point_in_polygon(p: "tuple[float, float]",
                     poly: "list[tuple[float, float]]") -> bool:
    """Ray-cast point-in-polygon (boundary counts as inside, conservative)."""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if abs((xj - xi) * (y - yi) - (yj - yi) * (x - xi)) < _EPS and (
                min(xi, xj) - _EPS <= x <= max(xi, xj) + _EPS and
                min(yi, yj) - _EPS <= y <= max(yi, yj) + _EPS):
            return True  # on an edge
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def point_in_any(p: "tuple[float, float]",
                 polys: "list[list[tuple[float, float]]]") -> bool:
    return any(point_in_polygon(p, poly) for poly in polys)


def _strictly_inside(p: "tuple[float, float]",
                     poly: "list[tuple[float, float]]") -> bool:
    """Ray-cast point-in-polygon with the boundary EXCLUDED.

    For a segment-clearance midpoint test: walking ALONG an inflated boundary
    edge is allowed (the inflation is the safety margin), but a chord through
    the interior (a convex polygon's diagonal) is blocked. Adjacent inflated
    corners connect along the hull; non-adjacent ones (diagonals) do not."""
    x, y = p
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        # on an edge -> NOT strictly inside (boundary excluded)
        if abs((xj - xi) * (y - yi) - (yj - yi) * (x - xi)) < _EPS and (
                min(xi, xj) - _EPS <= x <= max(xi, xj) + _EPS and
                min(yi, yj) - _EPS <= y <= max(yi, yj) + _EPS):
            return False
        j = i
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def _seg_intersect(a, b, c, d) -> bool:
    """True if segment ab properly crosses segment cd."""
    def ccw(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = ccw(c, d, a), ccw(c, d, b)
    d3, d4 = ccw(a, b, c), ccw(a, b, d)
    if ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and \
       ((d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)):
        return True
    return False


def segment_clear(a: "tuple[float, float]", b: "tuple[float, float]",
                  polys: "list[list[tuple[float, float]]]") -> bool:
    """True if segment a-b avoids the INTERIOR of every polygon.

    Touching a vertex/edge (e.g. a graph edge along an inflated corner) is
    allowed; crossing into the interior or having a midpoint inside is not.
    """
    for poly in polys:
        n = len(poly)
        for i in range(n):
            if _seg_intersect(a, b, poly[i], poly[(i + 1) % n]):
                return False
        # midpoint STRICTLY inside catches a chord through the interior (a
        # convex diagonal); a segment ALONG an inflated boundary edge has its
        # midpoint on the boundary and is allowed (the inflation is the margin).
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        if _strictly_inside(mid, poly):
            return False
    return True


# --------------------------------------------------------------------------
# Visibility graph + Dijkstra
# --------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_visibility_graph(
    start: "tuple[float, float]",
    goal: "tuple[float, float]",
    inflated: "list[list[tuple[float, float]]]",
) -> "tuple[list[tuple[float, float]], dict[int, list[tuple[int, float]]]]":
    """Nodes = start, goal, all inflated-polygon corners NOT inside another
    obstacle. Edges = node pairs with a line-of-sight-clear segment.
    Returns (nodes, adjacency) where adjacency[i] = [(j, weight), ...]."""
    nodes: list = [start, goal]
    for pi, poly in enumerate(inflated):
        others = [q for qi, q in enumerate(inflated) if qi != pi]
        for v in poly:
            # a vertex is trivially ON its OWN inflated boundary; only drop it
            # when it lies inside a DIFFERENT obstacle (a useless routing node).
            if not point_in_any(v, others):
                nodes.append(v)
    adj: dict[int, list] = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if segment_clear(nodes[i], nodes[j], inflated):
                w = _dist(nodes[i], nodes[j])
                adj[i].append((j, w))
                adj[j].append((i, w))
    return nodes, adj


def plan_path(
    start: "tuple[float, float]",
    goal: "tuple[float, float]",
    obstacles: "list[list[tuple[float, float]]]",
    inflation: float,
) -> "tuple[list[tuple[float, float]] | None, float]":
    """Shortest obstacle-avoiding path start->goal.

    Returns (waypoints, length). ``(None, inf)`` when the goal (or start) is
    inside an inflated obstacle or no clear route exists — honest unreachable.
    Waypoints include start and goal.
    """
    inflated = [inflate_polygon(p, inflation) for p in obstacles]
    if point_in_any(goal, inflated) or point_in_any(start, inflated):
        return None, float("inf")
    nodes, adj = build_visibility_graph(start, goal, inflated)
    # Dijkstra from node 0 (start) to node 1 (goal)
    n = len(nodes)
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[0] = 0.0
    pq: list = [(0.0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + _EPS:
            continue
        if u == 1:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] - _EPS:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not math.isfinite(dist[1]):
        return None, float("inf")
    # reconstruct
    path_idx = []
    cur = 1
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    return [nodes[i] for i in path_idx], dist[1]


def path_length(
    start: "tuple[float, float]",
    goal: "tuple[float, float]",
    obstacles: "list[list[tuple[float, float]]]",
    inflation: float,
) -> float:
    """The true geodesic distance (visibility-graph shortest path), ``inf``
    when unreachable. The SAME computation ``plan_path`` uses — the single
    source of truth for the verify predicate and the follower."""
    _, length = plan_path(start, goal, obstacles, inflation)
    return length
