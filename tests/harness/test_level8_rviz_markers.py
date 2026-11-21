# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 8 — RViz MarkerArray: scene graph visualisation markers.

Tests ``build_scene_graph_markers()`` from
``vector_os_nano.sim.nav.ros2.viz``.

The function builds a ``visualization_msgs/MarkerArray`` from a live
``SceneGraph`` instance.  All tests are pure Python and run fast — no
MuJoCo, no real API calls, no ROS2 node spin-up required.  The entire
test class is skipped when ``visualization_msgs`` is not installed so
the suite stays green in bare-Python environments.

Marker namespaces under test
----------------------------
    rooms           -- filled CUBE rectangles, one per room (8 total)
    room_borders    -- LINE_STRIP outlines for each room boundary (8 total)
    room_labels     -- TEXT_VIEW_FACING, one per room (8 total)
    viewpoints      -- SPHERE at each ViewpointNode position
    viewpoint_fovs  -- TRIANGLE_LIST FOV cones per viewpoint
    objects         -- CUBE for each detected ObjectNode
    object_labels   -- TEXT_VIEW_FACING for each ObjectNode
    robot           -- ARROW at current robot position (teal)
    robot_body      -- CYLINDER showing robot footprint
    trajectory      -- LINE_STRIP path history
    nav_goal        -- CYLINDER beacon at navigation target (only when provided)
    nav_goal_label  -- TEXT_VIEW_FACING "GOAL" label above the cylinder

Reference: ``vector_os_nano/sim/nav/ros2/viz.py``
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# ROS2 availability guard
# ---------------------------------------------------------------------------

def _ros2_available() -> bool:
    try:
        from visualization_msgs.msg import MarkerArray  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test class — skipped entirely when ROS2 message types are absent
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ros2_available(), reason="ROS2 not available")
class TestLevel8RVizMarkers:
    """MarkerArray generation tests for the scene graph visualiser."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_markers(sg, **kwargs):
        """Call build_scene_graph_markers and assert it returns something."""
        from vector_os_nano.sim.nav.ros2.viz import build_scene_graph_markers
        ma = build_scene_graph_markers(sg, **kwargs)
        assert ma is not None, "build_scene_graph_markers returned None (ROS2 missing?)"
        return ma

    @staticmethod
    def _fresh_sg():
        from vector_os_nano.core.scene_graph import SceneGraph
        return SceneGraph()

    # ------------------------------------------------------------------
    # T8-0  Empty scene graph — room layout always present
    # ------------------------------------------------------------------

    def test_empty_scene_graph_has_room_markers(self):
        """Even with empty SceneGraph, room boundaries and labels are generated."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg, robot_x=10.0, robot_y=3.0)
        # 8 rooms * 3 (fill + border + label) + 2 robot (arrow + body) = 26 minimum
        assert len(ma.markers) >= 17

    # ------------------------------------------------------------------
    # T8-1  Room namespace correctness
    # ------------------------------------------------------------------

    def test_room_boundary_namespaces(self):
        """Room markers use 'rooms', 'room_borders', and 'room_labels' namespaces."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg)

        room_markers = [m for m in ma.markers if m.ns == "rooms"]
        border_markers = [m for m in ma.markers if m.ns == "room_borders"]
        label_markers = [m for m in ma.markers if m.ns == "room_labels"]

        # Exactly 8 rooms defined in the static layout
        assert len(room_markers) == 8
        assert len(border_markers) == 8
        assert len(label_markers) == 8

    def test_room_fill_is_cube(self):
        """Room fill markers use CUBE type."""
        from visualization_msgs.msg import Marker
        sg = self._fresh_sg()
        ma = self._make_markers(sg)
        room_fills = [m for m in ma.markers if m.ns == "rooms"]
        for m in room_fills:
            assert m.type == Marker.CUBE, f"Room fill should be CUBE, got {m.type}"

    def test_room_border_is_line_strip(self):
        """Room border markers use LINE_STRIP type."""
        from visualization_msgs.msg import Marker
        sg = self._fresh_sg()
        ma = self._make_markers(sg)
        borders = [m for m in ma.markers if m.ns == "room_borders"]
        assert borders, "Expected room border markers"
        for m in borders:
            assert m.type == Marker.LINE_STRIP, (
                f"Room border should be LINE_STRIP, got {m.type}"
            )

    def test_visited_room_fill_has_color(self):
        """Visited room fill uses the room palette color (not grey)."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        ma = self._make_markers(sg)
        kitchen_fills = [
            m for m in ma.markers
            if m.ns == "rooms" and m.pose.position.x == pytest.approx(17.0, abs=0.1)
        ]
        assert kitchen_fills, "Expected a fill marker near kitchen center"
        m = kitchen_fills[0]
        # Kitchen uses mint green (0.60, 0.80, 0.40) — green channel dominant
        assert m.color.g > 0.5, f"Kitchen fill should be greenish, got g={m.color.g}"
        assert m.color.a > 0.15, "Visited room fill should be more opaque"

    def test_unvisited_room_fill_is_grey(self):
        """Unvisited room fill is grey with low alpha."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg)
        fills = [m for m in ma.markers if m.ns == "rooms"]
        for m in fills:
            # All channels should be equal (grey) for unvisited rooms
            assert m.color.r == pytest.approx(m.color.g, abs=0.05)
            assert m.color.g == pytest.approx(m.color.b, abs=0.05)
            assert m.color.a <= 0.15

    # ------------------------------------------------------------------
    # T8-2  Viewpoint markers
    # ------------------------------------------------------------------

    def test_viewpoint_markers_appear(self):
        """After adding viewpoints, green sphere markers appear in 'viewpoints' ns."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe_with_viewpoint(
            "kitchen", 16.5, 2.0, 0.5, ["fridge"], "kitchen scene"
        )
        ma = self._make_markers(sg)

        vp_markers = [m for m in ma.markers if m.ns == "viewpoints"]
        assert len(vp_markers) >= 1

    def test_viewpoint_marker_is_green_sphere(self):
        """Viewpoint markers are SPHERE type with green/teal colour."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe_with_viewpoint(
            "kitchen", 16.5, 2.0, 0.5, ["fridge"], "kitchen scene"
        )
        ma = self._make_markers(sg)

        vp_markers = [m for m in ma.markers if m.ns == "viewpoints"]
        assert vp_markers, "Expected at least one viewpoint marker"
        m = vp_markers[0]
        assert m.type == Marker.SPHERE
        assert m.color.g > 0.5, "Viewpoint marker should be predominantly green"
        assert m.color.r < 0.1

    def test_viewpoint_fov_cone_appears(self):
        """FOV cone TRIANGLE_LIST marker appears alongside each viewpoint sphere."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe_with_viewpoint(
            "kitchen", 16.5, 2.0, 0.5, ["fridge"], "kitchen scene"
        )
        ma = self._make_markers(sg)

        fov_markers = [m for m in ma.markers if m.ns == "viewpoint_fovs"]
        assert len(fov_markers) >= 1, "Expected at least one FOV cone marker"

    def test_viewpoint_fov_cone_is_triangle_list(self):
        """FOV cone markers use TRIANGLE_LIST type with multiple triangle points."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe_with_viewpoint(
            "kitchen", 16.5, 2.0, 1.0, ["fridge"], "kitchen scene"
        )
        ma = self._make_markers(sg)

        fov_markers = [m for m in ma.markers if m.ns == "viewpoint_fovs"]
        assert fov_markers
        m = fov_markers[0]
        assert m.type == Marker.TRIANGLE_LIST
        # n_segments=8 triangles × 3 points each = 24 points
        assert len(m.points) == 24, (
            f"Expected 24 FOV cone points (8 triangles), got {len(m.points)}"
        )

    def test_viewpoint_fov_cone_is_semi_transparent(self):
        """FOV cone has low alpha (semi-transparent fill)."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe_with_viewpoint(
            "kitchen", 16.5, 2.0, 0.5, ["fridge"], "kitchen scene"
        )
        ma = self._make_markers(sg)

        fov_markers = [m for m in ma.markers if m.ns == "viewpoint_fovs"]
        assert fov_markers
        m = fov_markers[0]
        assert m.color.a < 0.5, (
            f"FOV cone should be semi-transparent (a<0.5), got {m.color.a}"
        )

    # ------------------------------------------------------------------
    # T8-3  Object markers
    # ------------------------------------------------------------------

    def test_object_markers_appear(self):
        """After observing objects, orange cube + text markers appear."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge", "counter"], "Kitchen scene")
        ma = self._make_markers(sg)

        obj_markers = [m for m in ma.markers if m.ns == "objects"]
        label_markers = [m for m in ma.markers if m.ns == "object_labels"]
        assert len(obj_markers) >= 2
        assert len(label_markers) >= 2

    def test_object_marker_is_orange_cube(self):
        """Object cube markers have orange colour (r=1.0, g~0.55, b=0.0)."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge"], "Kitchen scene")
        ma = self._make_markers(sg)

        obj_markers = [m for m in ma.markers if m.ns == "objects"]
        assert obj_markers, "Expected at least one object marker"
        m = obj_markers[0]
        assert m.type == Marker.CUBE
        # fridge is mapped to green in the palette
        assert m.color.g > 0.3, "Expected green-dominant fridge marker"

    def test_object_label_text_matches_category(self):
        """Object label text equals the detected object category."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge"], "Kitchen scene")
        ma = self._make_markers(sg)

        label_markers = [m for m in ma.markers if m.ns == "object_labels"]
        texts = {m.text for m in label_markers}
        assert "fridge" in texts

    def test_object_cube_is_high_alpha(self):
        """Object cubes are highly opaque (alpha >= 0.8) for visual prominence."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["chair"], "Kitchen scene")
        ma = self._make_markers(sg)

        obj_markers = [m for m in ma.markers if m.ns == "objects"]
        assert obj_markers
        for m in obj_markers:
            assert m.color.a >= 0.80, (
                f"Object cube alpha should be >= 0.8, got {m.color.a}"
            )

    # ------------------------------------------------------------------
    # T8-4  Robot arrow marker
    # ------------------------------------------------------------------

    def test_robot_arrow_marker(self):
        """Robot position is shown as a teal ARROW in the 'robot' namespace."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        ma = self._make_markers(sg, robot_x=5.0, robot_y=3.0, robot_heading=1.57)

        robot_markers = [m for m in ma.markers if m.ns == "robot"]
        assert len(robot_markers) == 1

        m = robot_markers[0]
        assert m.type == Marker.ARROW
        assert m.pose.position.x == pytest.approx(5.0)
        assert m.pose.position.y == pytest.approx(3.0)
        # Teal: r=0, g>0.5, b>0.5
        assert m.color.r == pytest.approx(0.0, abs=0.05)
        assert m.color.g > 0.5
        assert m.color.b > 0.5

    def test_robot_arrow_heading_encoded_in_quaternion(self):
        """Robot heading is encoded as a quaternion on the arrow marker."""
        sg = self._fresh_sg()
        heading = math.pi / 4  # 45 degrees
        ma = self._make_markers(sg, robot_heading=heading)

        robot_markers = [m for m in ma.markers if m.ns == "robot"]
        assert robot_markers
        m = robot_markers[0]
        expected_z = math.sin(heading / 2)
        expected_w = math.cos(heading / 2)
        assert m.pose.orientation.z == pytest.approx(expected_z, abs=1e-6)
        assert m.pose.orientation.w == pytest.approx(expected_w, abs=1e-6)

    def test_robot_body_cylinder_present(self):
        """A 'robot_body' CYLINDER marker appears at the robot position."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        ma = self._make_markers(sg, robot_x=3.0, robot_y=4.0)

        body_markers = [m for m in ma.markers if m.ns == "robot_body"]
        assert len(body_markers) == 1
        m = body_markers[0]
        assert m.type == Marker.CYLINDER
        assert m.pose.position.x == pytest.approx(3.0)
        assert m.pose.position.y == pytest.approx(4.0)

    def test_robot_body_cylinder_is_semi_transparent(self):
        """Robot body cylinder is semi-transparent (teal, low alpha)."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg)

        body_markers = [m for m in ma.markers if m.ns == "robot_body"]
        assert body_markers
        m = body_markers[0]
        assert m.color.a < 0.50, (
            f"Robot body should be semi-transparent, got alpha={m.color.a}"
        )
        assert m.color.g > 0.5, "Robot body should be teal (green channel dominant)"

    # ------------------------------------------------------------------
    # T8-5  Navigation goal marker
    # ------------------------------------------------------------------

    def test_nav_goal_marker_absent_by_default(self):
        """No nav_goal marker appears when nav_goal is not provided."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg)

        goal_markers = [m for m in ma.markers if m.ns == "nav_goal"]
        assert len(goal_markers) == 0

    def test_nav_goal_marker(self):
        """Navigation goal is shown as a CYLINDER beacon in 'nav_goal' namespace."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        ma = self._make_markers(sg, nav_goal=(12.0, 7.5))

        goal_markers = [m for m in ma.markers if m.ns == "nav_goal"]
        assert len(goal_markers) >= 1

        # Find the tall beacon cylinder (scale.z > 1.0)
        beacon = next(
            (m for m in goal_markers if m.scale.z > 1.0), None
        )
        assert beacon is not None, "Expected a tall CYLINDER beacon for nav_goal"
        assert beacon.type == Marker.CYLINDER
        assert beacon.pose.position.x == pytest.approx(12.0)
        assert beacon.pose.position.y == pytest.approx(7.5)
        # Red beacon
        assert beacon.color.r == pytest.approx(1.0, abs=0.05)
        assert beacon.color.g < 0.3
        assert beacon.color.b < 0.3

    def test_nav_goal_label_appears(self):
        """A 'GOAL' text label appears in 'nav_goal_label' namespace when goal is set."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg, nav_goal=(8.0, 3.0))

        label_markers = [m for m in ma.markers if m.ns == "nav_goal_label"]
        assert len(label_markers) == 1, "Expected exactly one GOAL label"
        assert label_markers[0].text == "GOAL"

    def test_nav_goal_label_absent_without_goal(self):
        """No nav_goal_label appears when no nav_goal is provided."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg)
        label_markers = [m for m in ma.markers if m.ns == "nav_goal_label"]
        assert len(label_markers) == 0

    # ------------------------------------------------------------------
    # T8-6  Trajectory marker
    # ------------------------------------------------------------------

    def test_trajectory_absent_without_data(self):
        """No trajectory marker when trajectory=None."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg)
        traj = [m for m in ma.markers if m.ns == "trajectory"]
        assert len(traj) == 0

    def test_trajectory_absent_with_single_point(self):
        """No trajectory marker for a single-point list (need >= 2 to draw line)."""
        sg = self._fresh_sg()
        ma = self._make_markers(sg, trajectory=[(1.0, 2.0)])
        traj = [m for m in ma.markers if m.ns == "trajectory"]
        assert len(traj) == 0

    def test_trajectory_line_strip_appears(self):
        """With >= 2 trajectory points, a LINE_STRIP appears in 'trajectory' ns."""
        from visualization_msgs.msg import Marker

        sg = self._fresh_sg()
        traj = [(float(i), float(i) * 0.5) for i in range(10)]
        ma = self._make_markers(sg, trajectory=traj)

        traj_markers = [m for m in ma.markers if m.ns == "trajectory"]
        assert len(traj_markers) == 1
        m = traj_markers[0]
        assert m.type == Marker.LINE_STRIP

    def test_trajectory_line_strip_has_correct_point_count(self):
        """LINE_STRIP has the same number of points as trajectory entries."""
        sg = self._fresh_sg()
        traj = [(float(i), 0.0) for i in range(15)]
        ma = self._make_markers(sg, trajectory=traj)

        traj_markers = [m for m in ma.markers if m.ns == "trajectory"]
        assert traj_markers
        assert len(traj_markers[0].points) == 15

    def test_trajectory_capped_at_max_points(self):
        """Trajectory is capped at _TRAJECTORY_MAX_POINTS (200) most-recent entries."""
        from vector_os_nano.sim.nav.ros2.viz import _TRAJECTORY_MAX_POINTS

        sg = self._fresh_sg()
        # Pass more than the cap
        traj = [(float(i), 0.0) for i in range(_TRAJECTORY_MAX_POINTS + 50)]
        ma = self._make_markers(sg, trajectory=traj)

        traj_markers = [m for m in ma.markers if m.ns == "trajectory"]
        assert traj_markers
        # Should be capped at _TRAJECTORY_MAX_POINTS
        assert len(traj_markers[0].points) == _TRAJECTORY_MAX_POINTS

    def test_trajectory_per_point_colors(self):
        """LINE_STRIP trajectory uses per-point colors (color count == point count)."""
        sg = self._fresh_sg()
        traj = [(float(i), 0.0) for i in range(5)]
        ma = self._make_markers(sg, trajectory=traj)

        traj_markers = [m for m in ma.markers if m.ns == "trajectory"]
        assert traj_markers
        m = traj_markers[0]
        assert len(m.colors) == len(m.points), (
            "LINE_STRIP should have one color entry per point"
        )

    def test_trajectory_newest_point_is_teal(self):
        """The last (newest) trajectory point has a teal color (g>0.5, b>0.5)."""
        sg = self._fresh_sg()
        traj = [(float(i), 0.0) for i in range(10)]
        ma = self._make_markers(sg, trajectory=traj)

        traj_markers = [m for m in ma.markers if m.ns == "trajectory"]
        assert traj_markers
        last_color = traj_markers[0].colors[-1]
        assert last_color.g > 0.5
        assert last_color.b > 0.5
        assert last_color.a > 0.7

    # ------------------------------------------------------------------
    # T8-7  Room label content
    # ------------------------------------------------------------------

    def test_room_label_shows_visit_info(self):
        """Visited room label includes visit count and coverage."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge"], "A kitchen")
        ma = self._make_markers(sg)

        labels = [
            m for m in ma.markers
            if m.ns == "room_labels" and "kitchen" in m.text
        ]
        assert labels, "Expected a room_labels marker for kitchen"
        label_text = labels[0].text
        assert "1x" in label_text, f"Expected visit count '1x' in: {label_text!r}"

    def test_unvisited_room_label_is_plain(self):
        """Unvisited room label shows only the room name, no visit stats."""
        sg = self._fresh_sg()
        # Do not visit any room
        ma = self._make_markers(sg)

        labels = [
            m for m in ma.markers
            if m.ns == "room_labels" and m.text.strip() == "hallway"
        ]
        # hallway label should just be "hallway" when never visited
        assert labels, "Expected a plain 'hallway' label marker"

    def test_coverage_affects_room_label(self):
        """Room coverage percentage appears in label after a viewpoint is added."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe_with_viewpoint(
            "kitchen", 16.5, 2.0, 0.0, ["fridge"], "A kitchen"
        )
        ma = self._make_markers(sg)

        labels = [
            m for m in ma.markers
            if m.ns == "room_labels" and "kitchen" in m.text
        ]
        assert labels
        label_text = labels[0].text
        # Coverage is formatted with % (e.g. "20%")
        assert "%" in label_text, f"Expected coverage % in label: {label_text!r}"

    # ------------------------------------------------------------------
    # T8-8  Structural invariants
    # ------------------------------------------------------------------

    def test_marker_frame_is_map(self):
        """All markers have frame_id='map'."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge"], "A kitchen")
        ma = self._make_markers(sg)

        for m in ma.markers:
            assert m.header.frame_id == "map", (
                f"Marker ns={m.ns!r} id={m.id} has frame_id={m.header.frame_id!r}"
            )

    def test_all_marker_ids_unique(self):
        """No duplicate (ns, id) pairs in the marker array."""
        sg = self._fresh_sg()
        sg.visit("kitchen", 17.0, 2.5)
        sg.observe("kitchen", ["fridge", "counter", "table"], "A busy kitchen")
        sg.visit("living_room", 3.0, 2.5)
        sg.observe("living_room", ["sofa", "tv"], "A living room")
        traj = [(float(i), float(i)) for i in range(5)]
        ma = self._make_markers(sg, nav_goal=(10.0, 5.0), trajectory=traj)

        seen = set()
        for m in ma.markers:
            key = (m.ns, m.id)
            assert key not in seen, f"Duplicate marker key: ns={m.ns!r}, id={m.id}"
            seen.add(key)

    def test_marker_count_grows_with_objects(self):
        """Total marker count increases when objects are added to multiple rooms."""
        sg_empty = self._fresh_sg()
        ma_empty = self._make_markers(sg_empty)
        base_count = len(ma_empty.markers)

        sg_with_objs = self._fresh_sg()
        sg_with_objs.visit("kitchen", 17.0, 2.5)
        sg_with_objs.observe("kitchen", ["fridge", "counter"], "Kitchen")
        sg_with_objs.visit("study", 17.0, 7.5)
        sg_with_objs.observe("study", ["desk", "chair", "lamp"], "Study")
        ma_with_objs = self._make_markers(sg_with_objs)

        # 5 new objects -> 5 cube + 5 label + viewpoints = at least 10 extra markers
        assert len(ma_with_objs.markers) >= base_count + 10

    def test_marker_count_grows_with_trajectory(self):
        """Adding trajectory increases marker count by exactly 1 LINE_STRIP."""
        sg = self._fresh_sg()
        ma_no_traj = self._make_markers(sg)
        base_count = len(ma_no_traj.markers)

        traj = [(float(i), 0.0) for i in range(5)]
        ma_with_traj = self._make_markers(sg, trajectory=traj)
        assert len(ma_with_traj.markers) == base_count + 1

    def test_marker_count_grows_with_nav_goal(self):
        """Adding nav_goal increases marker count (cylinder + disc + label = +3)."""
        sg = self._fresh_sg()
        ma_no_goal = self._make_markers(sg)
        base_count = len(ma_no_goal.markers)

        ma_with_goal = self._make_markers(sg, nav_goal=(5.0, 5.0))
        # Cylinder + disc + label = 3 extra
        assert len(ma_with_goal.markers) == base_count + 3
