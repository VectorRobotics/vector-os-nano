# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Visibility-graph global planner over planar obstacle polygons.

PURE planar geometry (no MuJoCo import: the model reader below only reads arrays
off the model it is handed), so it unit-tests offline and is the ONE planner the
in-process follower walks and any geodesic check reads.

Pipeline: obstacle polygons -> Minkowski inflation by the robot radius ->
visibility graph over {start, goal, inflated corners} with line-of-sight-clear
edges -> Dijkstra. A goal inside the inflated set is snapped to the nearest free
point (``snap_goal=True``: a room centre means "be in the room", not "stand on the
table"); a start inside it escapes straight to its nearest free point first (a dog
that brushed furniture can always leave). ``(None, inf)`` means honestly
unreachable — never a phantom euclidean 'close'.

Conventions: a polygon is a list of (x, y) vertices of a CONVEX shape (either
winding). Boxes give 4 corners; cylinders a CIRCUMSCRIBED n-gon.
"""
from __future__ import annotations

import heapq
import math
from typing import Any

_EPS = 1e-9
# A snapped point sits this far outside the inflated boundary, so it is free.
_SNAP_MARGIN = 0.01
# Ring search fallback for a point buried under overlapping polygons.
_SNAP_RING_STEP = 0.05
_SNAP_RING_MAX = 2.0
_SNAP_RING_ANGLES = 48


# --------------------------------------------------------------------------
# Primitive shapes
# --------------------------------------------------------------------------

def box_polygon(cx: float, cy: float, hx: float, hy: float,
                yaw: float = 0.0) -> "list[tuple[float, float]]":
    """Four corners (CCW) of a yaw-rotated box centred at (cx, cy), half-extents (hx, hy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + sx * c - sy * s, cy + sx * s + sy * c)
            for sx, sy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def cylinder_polygon(cx: float, cy: float, r: float,
                     n: int = 8) -> "list[tuple[float, float]]":
    """A CIRCUMSCRIBED n-gon around a disc: every edge is tangent to (outside) it."""
    rr = r / math.cos(math.pi / n)
    return [(cx + rr * math.cos(2 * math.pi * k / n + math.pi / n),
             cy + rr * math.sin(2 * math.pi * k / n + math.pi / n))
            for k in range(n)]


def _signed_area(poly: "list[tuple[float, float]]") -> float:
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1]
                     for i in range(len(poly)))


def inflate_polygon(poly: "list[tuple[float, float]]",
                    r: float) -> "list[tuple[float, float]]":
    """Minkowski offset of a convex polygon by ``r``, with mitered corners.

    Every edge moves out along its own normal by exactly ``r`` and adjacent offset
    lines are intersected, so the result contains the true Minkowski sum with a
    disc of radius ``r`` (a box becomes the box with half-extents hx+r, hy+r).
    """
    pts = list(poly) if _signed_area(poly) >= 0 else list(reversed(poly))
    n = len(pts)
    normals = []
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        le = math.hypot(ex, ey)
        normals.append((ey / le, -ex / le))  # outward for CCW
    out = []
    for i in range(n):
        (ax, ay), (bx, by) = normals[i - 1], normals[i]
        # convex shapes only: the corner turn is < 180 deg, so 1 + a.b > 0
        k = r / (1.0 + ax * bx + ay * by)
        out.append((pts[i][0] + k * (ax + bx), pts[i][1] + k * (ay + by)))
    return out


# --------------------------------------------------------------------------
# Geometry predicates
# --------------------------------------------------------------------------

def _on_edge(x: float, y: float, xi: float, yi: float, xj: float, yj: float) -> bool:
    return abs((xj - xi) * (y - yi) - (yj - yi) * (x - xi)) < _EPS and (
        min(xi, xj) - _EPS <= x <= max(xi, xj) + _EPS
        and min(yi, yj) - _EPS <= y <= max(yi, yj) + _EPS)


def _crossings(x: float, y: float, poly: "list[tuple[float, float]]") -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def point_in_polygon(p: "tuple[float, float]",
                     poly: "list[tuple[float, float]]") -> bool:
    """Ray-cast point-in-polygon (boundary counts as inside, conservative)."""
    x, y = p
    if any(_on_edge(x, y, *poly[i - 1], *poly[i]) for i in range(len(poly))):
        return True
    return _crossings(x, y, poly)


def point_in_any(p: "tuple[float, float]",
                 polys: "list[list[tuple[float, float]]]") -> bool:
    return any(point_in_polygon(p, poly) for poly in polys)


def _strictly_inside(p: "tuple[float, float]",
                     poly: "list[tuple[float, float]]") -> bool:
    """Point-in-polygon with the boundary EXCLUDED (walking along an edge is allowed)."""
    x, y = p
    if any(_on_edge(x, y, *poly[i - 1], *poly[i]) for i in range(len(poly))):
        return False
    return _crossings(x, y, poly)


def _seg_intersect(a, b, c, d) -> bool:
    """True if segment ab properly crosses segment cd."""
    def ccw(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    d1, d2 = ccw(c, d, a), ccw(c, d, b)
    d3, d4 = ccw(a, b, c), ccw(a, b, d)
    return (((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS))
            and ((d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)))


def _bbox(poly: "list[tuple[float, float]]") -> "tuple[float, float, float, float]":
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def segment_clear(a: "tuple[float, float]", b: "tuple[float, float]",
                  polys: "list[list[tuple[float, float]]]") -> bool:
    """True if segment a-b avoids the INTERIOR of every polygon.

    Touching a vertex/edge (a graph edge along an inflated corner) is allowed;
    crossing into the interior or having a midpoint strictly inside is not.
    """
    lox, hix = min(a[0], b[0]), max(a[0], b[0])
    loy, hiy = min(a[1], b[1]), max(a[1], b[1])
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    for poly in polys:
        x0, y0, x1, y1 = _bbox(poly)
        if x1 < lox or x0 > hix or y1 < loy or y0 > hiy:
            continue
        n = len(poly)
        if any(_seg_intersect(a, b, poly[i], poly[(i + 1) % n]) for i in range(n)):
            return False
        if _strictly_inside(mid, poly):
            return False
    return True


def _closest_on_segment(p, a, b) -> "tuple[float, float]":
    ex, ey = b[0] - a[0], b[1] - a[1]
    le2 = ex * ex + ey * ey
    t = 0.0 if le2 < _EPS else max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / le2))
    return (a[0] + t * ex, a[1] + t * ey)


def distance_to_polygon(p: "tuple[float, float]",
                        poly: "list[tuple[float, float]]") -> float:
    """Euclidean distance from *p* to a polygon (0 inside)."""
    if point_in_polygon(p, poly):
        return 0.0
    return min(math.dist(p, _closest_on_segment(p, poly[i - 1], poly[i]))
               for i in range(len(poly)))


def nearest_free(p: "tuple[float, float]",
                 inflated: "list[list[tuple[float, float]]]") -> "tuple[float, float] | None":
    """*p* itself when free, else the nearest point outside every inflated polygon.

    Exact for one polygon (projection onto its nearest edge, pushed just outside);
    a point buried under overlapping polygons falls back to a ring search out to
    ``_SNAP_RING_MAX`` m. None when nothing free is that close.
    """
    if not point_in_any(p, inflated):
        return p
    best, best_d = None, math.inf
    for poly in inflated:
        if not point_in_polygon(p, poly):
            continue
        cx, cy = sum(v[0] for v in poly) / len(poly), sum(v[1] for v in poly) / len(poly)
        for i in range(len(poly)):
            q = _closest_on_segment(p, poly[i - 1], poly[i])
            # push outward along the edge normal (away from the convex centroid)
            ex, ey = poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]
            le = math.hypot(ex, ey) or 1.0
            nx, ny = ey / le, -ex / le
            if (q[0] - cx) * nx + (q[1] - cy) * ny < 0:
                nx, ny = -nx, -ny
            c = (q[0] + _SNAP_MARGIN * nx, q[1] + _SNAP_MARGIN * ny)
            d = math.dist(p, c)
            if d < best_d and not point_in_any(c, inflated):
                best, best_d = c, d
    if best is not None:
        return best
    r = _SNAP_RING_STEP
    while r <= _SNAP_RING_MAX + _EPS:
        ring = [(p[0] + r * math.cos(2 * math.pi * k / _SNAP_RING_ANGLES),
                 p[1] + r * math.sin(2 * math.pi * k / _SNAP_RING_ANGLES))
                for k in range(_SNAP_RING_ANGLES)]
        free = [c for c in ring if not point_in_any(c, inflated)]
        if free:
            return free[0]
        r += _SNAP_RING_STEP
    return None


# --------------------------------------------------------------------------
# Visibility graph + Dijkstra
# --------------------------------------------------------------------------

def build_visibility_graph(
    start: "tuple[float, float]",
    goal: "tuple[float, float]",
    inflated: "list[list[tuple[float, float]]]",
) -> "tuple[list[tuple[float, float]], dict[int, list[tuple[int, float]]]]":
    """Nodes = start, goal, every inflated corner NOT inside another obstacle.
    Edges = node pairs with a line-of-sight-clear segment."""
    nodes: list = [start, goal]
    for pi, poly in enumerate(inflated):
        others = [q for qi, q in enumerate(inflated) if qi != pi]
        nodes.extend(v for v in poly if not point_in_any(v, others))
    adj: dict[int, list] = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if segment_clear(nodes[i], nodes[j], inflated):
                w = math.dist(nodes[i], nodes[j])
                adj[i].append((j, w))
                adj[j].append((i, w))
    return nodes, adj


def plan_path(
    start: "tuple[float, float]",
    goal: "tuple[float, float]",
    obstacles: "list[list[tuple[float, float]]]",
    inflation: float,
    snap_goal: bool = False,
) -> "tuple[list[tuple[float, float]] | None, float]":
    """Shortest obstacle-avoiding path start -> goal: ``(waypoints, length)``.

    Waypoints include the start and the (possibly snapped) goal. A start inside
    the inflated set first escapes straight to its nearest free point. A goal
    inside it is unreachable unless ``snap_goal`` moves it to its nearest free
    point (the last waypoint then differs from *goal*). ``(None, inf)`` when no
    clear route exists.
    """
    inflated = [inflate_polygon(p, inflation) for p in obstacles]
    if snap_goal:
        goal = nearest_free(goal, inflated)
        if goal is None:
            return None, math.inf
    elif point_in_any(goal, inflated):
        return None, math.inf
    escape = nearest_free(start, inflated)
    if escape is None:
        return None, math.inf
    nodes, adj = build_visibility_graph(escape, goal, inflated)
    dist = [math.inf] * len(nodes)
    prev = [-1] * len(nodes)
    dist[0] = 0.0
    pq: list = [(0.0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + _EPS:
            continue
        if u == 1:
            break
        for v, w in adj[u]:
            if d + w < dist[v] - _EPS:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (d + w, v))
    if not math.isfinite(dist[1]):
        return None, math.inf
    idx, cur = [], 1
    while cur != -1:
        idx.append(cur)
        cur = prev[cur]
    path = [nodes[i] for i in reversed(idx)]
    if escape != start:
        path.insert(0, start)
        return path, dist[1] + math.dist(start, escape)
    return path, dist[1]


def path_length(
    start: "tuple[float, float]",
    goal: "tuple[float, float]",
    obstacles: "list[list[tuple[float, float]]]",
    inflation: float,
) -> float:
    """The geodesic distance (the SAME computation as ``plan_path``), inf if unreachable."""
    return plan_path(start, goal, obstacles, inflation)[1]


# --------------------------------------------------------------------------
# Obstacles from a compiled MuJoCo model (arrays only: no mujoco import)
# --------------------------------------------------------------------------

# mjtGeom / mjtJoint values (stable MuJoCo ABI; pinned against mujoco in tests).
GEOM_CYLINDER = 5
GEOM_BOX = 6
JNT_FREE = 0

# Visual-only or room-spanning names, skipped unconditionally.
_SKIP_NAME_PREFIXES: tuple[str, ...] = (
    "floor", "ceiling", "ground", "art_", "rug_",
    "bb_",   # baseboards — thin, flat, visual only
    "df_",   # door frames — thin, visual only
)
_OBS_MIN_Z_TOP = 0.10       # m: a geom must rise above this (skip flat rugs/mats)
_OBS_MAX_HALF_EXTENT = 9.0  # m: skip room-spanning slabs (both x and y huge)


def obstacles_from_model(
    model: Any,
    data: Any,
    robot_geom_ids: "set[int] | frozenset[int]" = frozenset(),
) -> "list[list[tuple[float, float]]]":
    """Planar routing polygons of the static scene in a compiled, forwarded model.

    Includes every colliding BOX/CYLINDER geom that rises above the floor, except
    the robot's own geoms (*robot_geom_ids*: the body subtree, which also covers a
    mounted arm), freejoint bodies (pickables), visual-only names and room-spanning
    slabs. The caller runs ``mj_forward`` first so ``geom_xpos/xmat`` are current.
    """
    polys: list[list[tuple[float, float]]] = []
    for gid in range(int(model.ngeom)):
        gtype = int(model.geom_type[gid])
        if gtype not in (GEOM_BOX, GEOM_CYLINDER) or gid in robot_geom_ids:
            continue
        if int(model.geom_contype[gid]) == 0:
            continue
        bid = int(model.geom_bodyid[gid])
        jadr = int(model.body_jntadr[bid])
        if jadr >= 0 and int(model.jnt_type[jadr]) == JNT_FREE:
            continue
        names = (model.geom(gid).name, model.body(bid).name)
        if any(n.startswith(_SKIP_NAME_PREFIXES) for n in names):
            continue
        size = model.geom_size[gid]
        if float(size[0]) >= _OBS_MAX_HALF_EXTENT and float(size[1]) >= _OBS_MAX_HALF_EXTENT:
            continue
        cx, cy, cz = (float(v) for v in data.geom_xpos[gid])
        half_h = float(size[2]) if gtype == GEOM_BOX else float(size[1])
        if cz + half_h < _OBS_MIN_Z_TOP:
            continue
        if gtype == GEOM_BOX:
            xm = data.geom_xmat[gid]
            yaw = math.atan2(float(xm[3]), float(xm[0]))
            polys.append(box_polygon(cx, cy, float(size[0]), float(size[1]), yaw))
        else:
            polys.append(cylinder_polygon(cx, cy, float(size[0])))
    return polys


EQ_WELD = 1   # mjtEq
OBJ_BODY = 1  # mjtObj


def _planar_reach(model: Any, data: Any, gids: "list[int]", ox: float, oy: float) -> float:
    """Farthest horizontal distance from (ox, oy) of any corner of the geoms' AABBs."""
    best = 0.0
    for g in gids:
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        rot = data.geom_xmat[g].reshape(3, 3)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    p = data.geom_xpos[g] + rot @ (c + h * (sx, sy, sz))
                    best = max(best, math.hypot(float(p[0]) - ox, float(p[1]) - oy))
    return best


def overhang(model: Any, data: Any, root_bid: int, first_extra_qpos: int) -> float:
    """How far (m) a mounted arm, plus any body welded to it, reaches past the base.

    The arm is every body under *root_bid* whose own joint sits at or beyond
    *first_extra_qpos* (the joints after the legged layout), with its descendants;
    a held object is a body an ACTIVE weld ties to the arm. Returns the arm's
    horizontal reach minus the rest of the robot's, floored at 0 (a stowed arm
    inside the body's footprint adds nothing).
    """
    parent = model.body_parentid

    def chain(b: int):
        while True:
            yield b
            if b == 0:
                return
            b = int(parent[b])

    def is_arm(b: int) -> bool:
        for a in chain(b):
            if a == root_bid:
                return False
            j = int(model.body_jntadr[a])
            if j >= 0 and int(model.jnt_qposadr[j]) >= first_extra_qpos:
                return True
        return False

    robot = {b for b in range(int(model.nbody)) if root_bid in chain(b)}
    arm = {b for b in robot if is_arm(b)}
    for i in range(int(model.neq)):
        if (int(model.eq_type[i]) == EQ_WELD and bool(data.eq_active[i])
                and int(model.eq_objtype[i]) == OBJ_BODY):
            a, b = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
            if a in arm or b in arm:
                arm.add(b if a in arm else a)
    if not arm:
        return 0.0

    def colliding(bodies: set) -> "list[int]":
        return [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in bodies
                and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]

    ox, oy = float(data.xpos[root_bid][0]), float(data.xpos[root_bid][1])
    return max(0.0, _planar_reach(model, data, colliding(arm), ox, oy)
               - _planar_reach(model, data, colliding(robot - arm), ox, oy))
