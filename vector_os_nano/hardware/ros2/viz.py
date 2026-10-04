# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RViz MarkerArray publisher for the three-layer scene graph.

Publishes to /scene_graph_markers (visualization_msgs/MarkerArray) at 1 Hz.
Designed to be embedded in the Go2VNavBridge node or run standalone.

Marker namespaces
-----------------
    rooms           -- filled CUBE rectangles, one per room (8 total)
    room_borders    -- LINE_STRIP outlines for each room boundary
    room_labels     -- TEXT_VIEW_FACING, one per room (8 total)
    viewpoints      -- SPHERE at each observation position (teal/green)
    viewpoint_fovs  -- TRIANGLE_LIST fan shapes showing camera FOV direction
    objects         -- CUBE for each detected ObjectNode (category-colored)
    object_labels   -- TEXT_VIEW_FACING for each ObjectNode
    robot           -- ARROW at current robot position (teal, prominent)
    robot_body      -- CYLINDER showing robot footprint
    trajectory      -- LINE_STRIP showing robot path history
    nav_goal        -- CYLINDER (pulsing red/coral) at navigation target
    nav_goal_label  -- TEXT_VIEW_FACING showing "GOAL" above the cylinder

All markers are in the "map" frame.
"""
from __future__ import annotations

import math
import time
from typing import Any

# Lazy ROS2 imports — this module is only loaded when ROS2 is available.
_rclpy = None
_MarkerArray = None
_Marker = None
_ColorRGBA = None
_Point = None
_Vector3 = None
_Header = None


def _ensure_imports() -> bool:
    """Lazy-import ROS2 message types. Returns True if available."""
    global _rclpy, _MarkerArray, _Marker, _ColorRGBA, _Point, _Vector3, _Header
    if _MarkerArray is not None:
        return True
    try:
        import rclpy as _r
        from visualization_msgs.msg import Marker, MarkerArray
        from std_msgs.msg import ColorRGBA, Header
        from geometry_msgs.msg import Point, Vector3
        _rclpy = _r
        _MarkerArray = MarkerArray
        _Marker = Marker
        _ColorRGBA = ColorRGBA
        _Point = Point
        _Vector3 = Vector3
        _Header = Header
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Room layout (matches go2_room.xml)
# ---------------------------------------------------------------------------

_ROOM_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "living_room":    (0.0,  0.0,  6.0,  5.0),
    "dining_room":    (0.0,  5.0,  6.0,  10.0),
    "kitchen":        (14.0, 0.0,  20.0, 5.0),
    "study":          (14.0, 5.0,  20.0, 10.0),
    "master_bedroom": (0.0,  10.0, 7.0,  14.0),
    "guest_bedroom":  (12.0, 10.0, 20.0, 14.0),
    "bathroom":       (7.0,  10.0, 10.0, 14.0),
    "hallway":        (6.0,  0.0,  14.0, 10.0),
}

# Apple-quality color palette — distinct, harmonious RGBA in [0, 1]
# Alpha is the fill opacity for unvisited/visited states.
_ROOM_COLORS: dict[str, tuple[float, float, float, float]] = {
    "living_room":    (0.20, 0.78, 0.78, 0.25),  # teal
    "dining_room":    (1.00, 0.42, 0.42, 0.25),  # coral
    "kitchen":        (0.60, 0.80, 0.40, 0.25),  # mint green
    "study":          (0.72, 0.53, 0.94, 0.25),  # lavender
    "master_bedroom": (1.00, 0.72, 0.30, 0.25),  # warm gold
    "guest_bedroom":  (0.40, 0.76, 1.00, 0.25),  # sky blue
    "bathroom":       (1.00, 0.60, 0.80, 0.25),  # soft pink
    "hallway":        (0.75, 0.75, 0.75, 0.18),  # neutral grey
}

# Object category color palette — distinct saturated colors
_OBJECT_COLORS: dict[str, tuple[float, float, float]] = {
    "chair":       (1.00, 0.55, 0.00),  # orange
    "sofa":        (0.20, 0.60, 1.00),  # blue
    "fridge":      (0.40, 0.80, 0.40),  # green
    "counter":     (0.90, 0.90, 0.20),  # yellow
    "table":       (1.00, 0.40, 0.40),  # red
    "desk":        (0.80, 0.50, 0.20),  # brown
    "bed":         (0.60, 0.40, 0.80),  # purple
    "lamp":        (1.00, 1.00, 0.40),  # bright yellow
    "tv":          (0.20, 0.80, 0.80),  # cyan
    "door":        (0.60, 0.60, 0.60),  # grey
    "window":      (0.70, 0.85, 1.00),  # light blue
    "plant":       (0.30, 0.70, 0.30),  # dark green
}
_OBJECT_COLOR_DEFAULT = (1.00, 0.55, 0.00)  # orange fallback

# FOV cone parameters — matches scene_graph.py constants
_VIEWPOINT_FOV_DEG: float = 60.0
_VIEWPOINT_RANGE: float = 3.0
_TRAJECTORY_MAX_POINTS: int = 200


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_header(stamp: Any) -> Any:
    h = _Header()
    h.frame_id = "map"
    h.stamp = stamp
    return h


def _make_point(x: float, y: float, z: float) -> Any:
    p = _Point()
    p.x = x
    p.y = y
    p.z = z
    return p


def _make_color(r: float, g: float, b: float, a: float) -> Any:
    c = _ColorRGBA()
    c.r = r
    c.g = g
    c.b = b
    c.a = a
    return c


def _base_marker(header: Any, ns: str, mid: int, mtype: int) -> Any:
    m = _Marker()
    m.header = header
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = _Marker.ADD
    m.frame_locked = False
    m.lifetime.sec = 5
    m.lifetime.nanosec = 0
    return m


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_room_markers(
    header: Any,
    scene_graph: Any,
    start_id: int,
) -> tuple[list[Any], int]:
    """Build room fill (CUBE), border (LINE_STRIP), and label markers."""
    markers: list[Any] = []
    mid = start_id

    for room_name, (x0, y0, x1, y1) in _ROOM_BOUNDS.items():
        color = _ROOM_COLORS.get(room_name, (0.5, 0.5, 0.5, 0.25))
        # Use SceneGraph center when available; fall back to geometric bound center
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if scene_graph is not None:
            room_node = scene_graph.get_room(room_name)
            if room_node is not None and (room_node.center_x != 0.0 or room_node.center_y != 0.0):
                cx, cy = room_node.center_x, room_node.center_y

        # Determine visit/coverage state
        is_visited = False
        coverage = 0.0
        visit_count = 0
        n_objs = 0
        if scene_graph is not None:
            room_node = scene_graph.get_room(room_name)
            if room_node and room_node.visit_count > 0:
                is_visited = True
                visit_count = room_node.visit_count
                coverage = scene_graph.get_room_coverage(room_name)
                n_objs = len(scene_graph.find_objects_in_room(room_name))

        # --- Fill CUBE ---
        fill = _base_marker(header, "rooms", mid, _Marker.CUBE)
        mid += 1
        fill.pose.position.x = (x0 + x1) / 2
        fill.pose.position.y = (y0 + y1) / 2
        fill.pose.position.z = 0.005
        fill.pose.orientation.w = 1.0
        fill.scale.x = x1 - x0
        fill.scale.y = y1 - y0
        fill.scale.z = 0.01
        if is_visited:
            alpha = 0.18 + coverage * 0.35  # 0.18 → 0.53 as coverage grows
            fill.color = _make_color(color[0], color[1], color[2], alpha)
        else:
            fill.color = _make_color(0.35, 0.35, 0.35, 0.10)
        markers.append(fill)

        # --- Border LINE_STRIP ---
        border = _base_marker(header, "room_borders", mid, _Marker.LINE_STRIP)
        mid += 1
        # Close the rectangle: 5 points (last == first)
        corners = [
            (x0, y0, 0.02), (x1, y0, 0.02),
            (x1, y1, 0.02), (x0, y1, 0.02),
            (x0, y0, 0.02),
        ]
        for bx, by, bz in corners:
            border.points.append(_make_point(bx, by, bz))
        if is_visited:
            border.color = _make_color(color[0], color[1], color[2], 0.85)
            border.scale.x = 0.06  # line width
        else:
            border.color = _make_color(0.5, 0.5, 0.5, 0.35)
            border.scale.x = 0.03
        markers.append(border)

        # --- Room label TEXT_VIEW_FACING ---
        label_text = _format_room_label(
            room_name, is_visited, visit_count, coverage, n_objs
        )
        lbl = _base_marker(header, "room_labels", mid, _Marker.TEXT_VIEW_FACING)
        mid += 1
        lbl.pose.position.x = cx
        lbl.pose.position.y = cy
        lbl.pose.position.z = 0.6
        lbl.pose.orientation.w = 1.0
        lbl.text = label_text
        lbl.scale.z = 0.45  # text height in metres
        if is_visited:
            lbl.color = _make_color(1.0, 1.0, 1.0, 0.95)
        else:
            lbl.color = _make_color(0.6, 0.6, 0.6, 0.60)
        markers.append(lbl)

    return markers, mid


def _format_room_label(
    room_name: str,
    is_visited: bool,
    visit_count: int,
    coverage: float,
    n_objs: int,
) -> str:
    """Format room label text with visit stats when available."""
    if not is_visited:
        return room_name
    return f"{room_name}\n{visit_count}x | {coverage:.0%} | {n_objs} obj"


def _build_viewpoint_markers(
    header: Any,
    scene_graph: Any,
    start_id: int,
) -> tuple[list[Any], int]:
    """Build viewpoint sphere + FOV cone triangle-fan markers."""
    markers: list[Any] = []
    mid = start_id

    if scene_graph is None:
        return markers, mid

    for room in scene_graph.get_all_rooms():
        for vp in scene_graph.get_viewpoints_in_room(room.room_id):
            # Sphere at viewpoint position
            sphere = _base_marker(header, "viewpoints", mid, _Marker.SPHERE)
            mid += 1
            sphere.pose.position.x = vp.x
            sphere.pose.position.y = vp.y
            sphere.pose.position.z = 0.20
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.30
            sphere.scale.y = 0.30
            sphere.scale.z = 0.20
            sphere.color = _make_color(0.00, 0.85, 0.65, 0.85)  # teal-green
            markers.append(sphere)

            # FOV cone as TRIANGLE_LIST fan
            fov_marker = _build_fov_cone(
                header, mid, vp.x, vp.y, vp.heading,
                _VIEWPOINT_FOV_DEG, _VIEWPOINT_RANGE,
            )
            if fov_marker is not None:
                mid += 1
                markers.append(fov_marker)

    return markers, mid


def _build_fov_cone(
    header: Any,
    mid: int,
    x: float,
    y: float,
    heading: float,
    fov_deg: float,
    range_m: float,
    n_segments: int = 8,
) -> Any:
    """Build a TRIANGLE_LIST fan showing a camera FOV cone in 2D (flat on z=0.05)."""
    cone = _base_marker(header, "viewpoint_fovs", mid, _Marker.TRIANGLE_LIST)
    cone.pose.position.x = 0.0
    cone.pose.position.y = 0.0
    cone.pose.position.z = 0.0
    cone.pose.orientation.w = 1.0
    cone.scale.x = 1.0
    cone.scale.y = 1.0
    cone.scale.z = 1.0
    cone.color = _make_color(0.00, 0.85, 0.65, 0.22)  # transparent teal

    half_fov = math.radians(fov_deg / 2)
    z_flat = 0.05

    apex = _make_point(x, y, z_flat)

    # Fan of n_segments triangles from apex to arc
    for i in range(n_segments):
        t0 = heading - half_fov + (2 * half_fov * i / n_segments)
        t1 = heading - half_fov + (2 * half_fov * (i + 1) / n_segments)
        p0 = _make_point(
            x + range_m * math.cos(t0),
            y + range_m * math.sin(t0),
            z_flat,
        )
        p1 = _make_point(
            x + range_m * math.cos(t1),
            y + range_m * math.sin(t1),
            z_flat,
        )
        cone.points.append(apex)
        cone.points.append(p0)
        cone.points.append(p1)

    if not cone.points:
        return None
    return cone


def _compute_object_position(
    obj: Any,
    obj_index: int,
    total_objs: int,
    scene_graph: Any,
    room_id: str,
) -> tuple[float, float]:
    """Compute world position for an object marker.

    Priority order:
    1. If the object has world coordinates (obj.x, obj.y are non-zero),
       use them directly.
    2. If a viewpoint is available, place objects in a cluster 2m along the
       viewpoint heading (heuristic for VLM detections).
    3. Fall back to room center.

    All positions are clamped inside room bounds.
    """
    bounds = _ROOM_BOUNDS.get(room_id)
    margin = 0.5

    # --- Best path: detector provided world coordinates via depth projection.
    if obj.x != 0.0 or obj.y != 0.0:
        ox, oy = obj.x, obj.y
        if bounds:
            x0, y0, x1, y1 = bounds
            ox = max(x0 + margin, min(x1 - margin, ox))
            oy = max(y0 + margin, min(y1 - margin, oy))
        return ox, oy

    # --- Heuristic: cluster objects in front of the observing viewpoint.
    vp = None
    if obj.viewpoint_ids and scene_graph is not None:
        for vp_id in obj.viewpoint_ids:
            vps = scene_graph._viewpoints.get(vp_id)
            if vps is not None:
                vp = vps
                break

    if vp is not None and (vp.x != 0.0 or vp.y != 0.0):
        # Scene center = viewpoint position + 2m along heading
        scene_dist = 2.0
        scene_cx = vp.x + scene_dist * math.cos(vp.heading)
        scene_cy = vp.y + scene_dist * math.sin(vp.heading)

        # Distribute objects in a cluster around scene center.
        if total_objs == 1:
            ox, oy = scene_cx, scene_cy
        else:
            cluster_radius = min(0.8, 0.3 + 0.1 * total_objs)
            angle = obj_index * 2.0 * math.pi / total_objs
            ox = scene_cx + cluster_radius * math.cos(angle)
            oy = scene_cy + cluster_radius * math.sin(angle)
    else:
        # Fallback: room center from SceneGraph or geometric bound center
        cx, cy = 0.0, 0.0
        if scene_graph is not None:
            room_node = scene_graph.get_room(room_id)
            if room_node is not None and (room_node.center_x != 0.0 or room_node.center_y != 0.0):
                cx, cy = room_node.center_x, room_node.center_y
        if cx == 0.0 and cy == 0.0 and bounds:
            x0b, y0b, x1b, y1b = bounds
            cx, cy = (x0b + x1b) / 2, (y0b + y1b) / 2
        if total_objs == 1:
            ox, oy = cx, cy
        else:
            angle = obj_index * 2.0 * math.pi / total_objs
            ox = cx + 1.0 * math.cos(angle)
            oy = cy + 1.0 * math.sin(angle)

    # Clamp inside room bounds
    if bounds:
        x0, y0, x1, y1 = bounds
        ox = max(x0 + margin, min(x1 - margin, ox))
        oy = max(y0 + margin, min(y1 - margin, oy))

    return ox, oy


def _build_object_markers(
    header: Any,
    scene_graph: Any,
    start_id: int,
) -> tuple[list[Any], int]:
    """Build object cube + label markers, with category-based colors."""
    markers: list[Any] = []
    mid = start_id

    if scene_graph is None:
        return markers, mid

    for room in scene_graph.get_all_rooms():
        objs = scene_graph.find_objects_in_room(room.room_id)
        if not objs:
            continue

        n = len(objs)
        for i, obj in enumerate(objs):
            ox, oy = _compute_object_position(
                obj, i, n, scene_graph, room.room_id,
            )

            cat_lower = obj.category.lower()
            r, g, b = _OBJECT_COLORS.get(cat_lower, _OBJECT_COLOR_DEFAULT)

            # Cube
            cube = _base_marker(header, "objects", mid, _Marker.CUBE)
            mid += 1
            cube.pose.position.x = ox
            cube.pose.position.y = oy
            cube.pose.position.z = 0.15
            cube.pose.orientation.w = 1.0
            cube.scale.x = 0.28
            cube.scale.y = 0.28
            cube.scale.z = 0.28
            cube.color = _make_color(r, g, b, 0.88)
            markers.append(cube)

            # Label — offset above the cube
            lbl = _base_marker(header, "object_labels", mid, _Marker.TEXT_VIEW_FACING)
            mid += 1
            lbl.pose.position.x = ox
            lbl.pose.position.y = oy
            lbl.pose.position.z = 0.50
            lbl.pose.orientation.w = 1.0
            lbl.text = obj.category
            lbl.scale.z = 0.22
            lbl.color = _make_color(1.0, 1.0, 1.0, 0.95)
            markers.append(lbl)

    return markers, mid


def _build_robot_markers(
    header: Any,
    robot_x: float,
    robot_y: float,
    robot_heading: float,
    start_id: int,
) -> tuple[list[Any], int]:
    """Build robot arrow + footprint cylinder."""
    markers: list[Any] = []
    mid = start_id

    qz = math.sin(robot_heading / 2)
    qw = math.cos(robot_heading / 2)

    # Main direction arrow — prominent teal
    arrow = _base_marker(header, "robot", mid, _Marker.ARROW)
    mid += 1
    arrow.pose.position.x = robot_x
    arrow.pose.position.y = robot_y
    arrow.pose.position.z = 0.25
    arrow.pose.orientation.z = qz
    arrow.pose.orientation.w = qw
    arrow.scale.x = 0.80   # arrow length
    arrow.scale.y = 0.20   # shaft diameter
    arrow.scale.z = 0.20   # head diameter
    arrow.color = _make_color(0.00, 0.82, 0.82, 1.00)  # bright teal
    markers.append(arrow)

    # Footprint cylinder — slightly larger, semi-transparent
    body = _base_marker(header, "robot_body", mid, _Marker.CYLINDER)
    mid += 1
    body.pose.position.x = robot_x
    body.pose.position.y = robot_y
    body.pose.position.z = 0.12
    body.pose.orientation.w = 1.0
    body.scale.x = 0.55   # diameter
    body.scale.y = 0.55
    body.scale.z = 0.24   # height
    body.color = _make_color(0.00, 0.82, 0.82, 0.25)
    markers.append(body)

    return markers, mid


def _build_trajectory_marker(
    header: Any,
    trajectory: list[tuple[float, float]],
    start_id: int,
) -> tuple[list[Any], int]:
    """Build trajectory LINE_STRIP from position history."""
    markers: list[Any] = []
    mid = start_id

    if len(trajectory) < 2:
        return markers, mid

    line = _base_marker(header, "trajectory", mid, _Marker.LINE_STRIP)
    mid += 1
    line.pose.orientation.w = 1.0
    line.scale.x = 0.06  # line width

    n = len(trajectory)
    for idx, (px, py) in enumerate(trajectory):
        # Color fades from grey-blue (old) to bright teal (recent)
        t = idx / max(n - 1, 1)  # 0.0 (oldest) → 1.0 (newest)
        r = 0.20 * (1 - t) + 0.00 * t
        g = 0.50 * (1 - t) + 0.82 * t
        b = 0.70 * (1 - t) + 0.82 * t
        a = 0.30 + 0.60 * t  # fade in toward present
        line.points.append(_make_point(px, py, 0.03))
        line.colors.append(_make_color(r, g, b, a))

    markers.append(line)
    return markers, mid


def _build_nav_goal_markers(
    header: Any,
    nav_goal: tuple[float, float],
    start_id: int,
) -> tuple[list[Any], int]:
    """Build nav goal cylinder + GOAL text label."""
    markers: list[Any] = []
    mid = start_id

    gx, gy = nav_goal

    # Tall cylinder — acts as beacon
    cyl = _base_marker(header, "nav_goal", mid, _Marker.CYLINDER)
    mid += 1
    cyl.pose.position.x = gx
    cyl.pose.position.y = gy
    cyl.pose.position.z = 0.60
    cyl.pose.orientation.w = 1.0
    cyl.scale.x = 0.40
    cyl.scale.y = 0.40
    cyl.scale.z = 1.20  # tall beacon
    cyl.color = _make_color(1.00, 0.20, 0.20, 0.85)  # bold red
    markers.append(cyl)

    # Ring disc at base
    disc = _base_marker(header, "nav_goal", mid, _Marker.CYLINDER)
    mid += 1
    disc.pose.position.x = gx
    disc.pose.position.y = gy
    disc.pose.position.z = 0.01
    disc.pose.orientation.w = 1.0
    disc.scale.x = 0.80
    disc.scale.y = 0.80
    disc.scale.z = 0.02
    disc.color = _make_color(1.00, 0.20, 0.20, 0.40)
    markers.append(disc)

    # "GOAL" text above cylinder
    lbl = _base_marker(header, "nav_goal_label", mid, _Marker.TEXT_VIEW_FACING)
    mid += 1
    lbl.pose.position.x = gx
    lbl.pose.position.y = gy
    lbl.pose.position.z = 1.40
    lbl.pose.orientation.w = 1.0
    lbl.text = "GOAL"
    lbl.scale.z = 0.40
    lbl.color = _make_color(1.00, 0.50, 0.50, 1.00)
    markers.append(lbl)

    return markers, mid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_scene_graph_markers(
    scene_graph: Any,
    stamp: Any = None,
    robot_x: float = 0.0,
    robot_y: float = 0.0,
    robot_heading: float = 0.0,
    nav_goal: tuple[float, float] | None = None,
    trajectory: list[tuple[float, float]] | None = None,
) -> Any:
    """Build a MarkerArray from the current scene graph state.

    Args:
        scene_graph: SceneGraph instance (or None for static room layout only).
        stamp: ROS2 Time stamp. If None, uses current wall time.
        robot_x, robot_y: Current robot position in metres.
        robot_heading: Current heading in radians (yaw, CCW positive).
        nav_goal: Optional (x, y) of current navigation target.
        trajectory: Optional list of (x, y) tuples for path history (oldest first).
                    Capped at _TRAJECTORY_MAX_POINTS most-recent entries.

    Returns:
        visualization_msgs/MarkerArray, or None if ROS2 not available.
    """
    if not _ensure_imports():
        return None

    if stamp is None:
        import builtin_interfaces.msg
        now = time.time()
        stamp = builtin_interfaces.msg.Time()
        stamp.sec = int(now)
        stamp.nanosec = int((now % 1) * 1e9)

    header = _make_header(stamp)
    markers: list[Any] = []
    mid = 0

    # Rooms (fill + borders + labels)
    room_markers, mid = _build_room_markers(header, scene_graph, mid)
    markers.extend(room_markers)

    # Viewpoints + FOV cones
    vp_markers, mid = _build_viewpoint_markers(header, scene_graph, mid)
    markers.extend(vp_markers)

    # Objects + labels
    obj_markers, mid = _build_object_markers(header, scene_graph, mid)
    markers.extend(obj_markers)

    # Robot arrow + body
    robot_markers, mid = _build_robot_markers(
        header, robot_x, robot_y, robot_heading, mid
    )
    markers.extend(robot_markers)

    # Trajectory trail
    if trajectory:
        # Trim to most-recent N points
        traj = list(trajectory[-_TRAJECTORY_MAX_POINTS:])
        traj_markers, mid = _build_trajectory_marker(header, traj, mid)
        markers.extend(traj_markers)

    # Nav goal beacon
    if nav_goal is not None:
        goal_markers, mid = _build_nav_goal_markers(header, nav_goal, mid)
        markers.extend(goal_markers)

    ma = _MarkerArray()
    ma.markers = markers
    return ma
