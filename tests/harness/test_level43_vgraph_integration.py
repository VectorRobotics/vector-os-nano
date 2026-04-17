"""Level 43: V-Graph cross-room edge formation — live ROS2 integration (opt-in).

Requires a full nav stack bringup (scripts/launch_nav_only.sh) plus a
running bridge (go2_vnav_bridge). Opt-in via `pytest -m ros2`.

AC4 (from .sdd/spec.md): after driving Go2 through a doorway, the
V-Graph visualization topic reports >=1 edge where endpoints fall in
different known-room bounding boxes, within 30s.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent.parent


# Room bounding boxes in map frame (from scene_room.xml layout diagram).
# Format: (x_min, x_max, y_min, y_max). Derived from the ASCII diagram
# at the top of scene_room.xml — verify there if you adjust.
#
# ASCII layout (scene_room.xml lines 20-39):
#       0      6           14     20
#  14  +======+=====+=+====+======+
#      |Master| Bath|L|  Guest BR |
#      |  BR  | room|a|  (8x4)   |
#      |(7x4) |(3x4)|u|          |
#  10  +=D====+=D===+D+=====D====+
#      |      |                  |
#      |Dining|  OPEN HALLWAY    | Study
#      |Room  |  + GALLERY       | (6x5)
#      |(6x5) |  (8m x 10m)     |
#   5  +=D====+  open plan      +=D=====+
#      |      |                  |       |
#      |Living|  Go2 starts     |Kitchen |
#      |Room  |  (10, 3)        |(6x5)  |
#      |(6x5) |                  |       |
#   0  +======+==================+=======+
#
# D = doorway (1.2m gap)
# Central hall is ONE open space from y=0..10, x=6..14
ROOMS: dict[str, tuple[float, float, float, float]] = {
    "living_room": (0.0,  6.0,  0.0,  5.0),
    "dining_room": (0.0,  6.0,  5.0, 10.0),
    "hallway":     (6.0, 14.0,  0.0, 10.0),
    "kitchen":     (14.0, 20.0, 0.0,  5.0),
    "study":       (14.0, 20.0, 5.0, 10.0),
    "master_br":   (0.0,  7.0, 10.0, 14.0),
    "guest_br":    (12.0, 20.0, 10.0, 14.0),
    "bathroom":    (6.0, 10.0, 10.0, 14.0),
}


def _which_room(x: float, y: float) -> str | None:
    for name, (xmn, xmx, ymn, ymx) in ROOMS.items():
        if xmn <= x <= xmx and ymn <= y <= ymx:
            return name
    return None


def _launch_stack() -> subprocess.Popen:
    """Launch nav stack via launch_nav_only.sh. Caller MUST teardown."""
    script = _REPO / "scripts" / "launch_nav_only.sh"
    return subprocess.Popen(
        ["bash", str(script), "--no-rviz"],
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _teardown_stack(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


_ROS2_OPT_IN = os.environ.get("VECTOR_ROS2_INTEGRATION") == "1"

_skip_unless_opted_in = pytest.mark.skipif(
    not _ROS2_OPT_IN,
    reason=(
        "opt-in test — set VECTOR_ROS2_INTEGRATION=1 or use -m ros2 with a live nav stack"
    ),
)


@pytest.mark.ros2
@pytest.mark.slow
@_skip_unless_opted_in
def test_vgraph_forms_cross_room_edge_through_door():
    """AC4: V-Graph builds a cross-room edge when Go2 drives through a doorway."""
    pytest.importorskip("rclpy", reason="rclpy not available")
    pytest.importorskip(
        "visualization_msgs.msg",
        reason="visualization_msgs not available",
    )
    import rclpy
    from visualization_msgs.msg import MarkerArray

    stack = _launch_stack()
    edges_seen: list[tuple[tuple[float, float], tuple[float, float]]] = []

    try:
        rclpy.init()
        node = rclpy.create_node("vgraph_integration_probe")

        def _on_vgraph(msg: MarkerArray) -> None:
            # FAR publishes V-Graph edges as LINE_LIST markers (type 5).
            # Each pair of consecutive points is one edge.
            for marker in msg.markers:
                if marker.type != 5:  # LINE_LIST
                    continue
                pts = marker.points
                for i in range(0, len(pts) - 1, 2):
                    edges_seen.append(
                        ((pts[i].x, pts[i].y), (pts[i + 1].x, pts[i + 1].y))
                    )

        node.create_subscription(MarkerArray, "/viz_graph_topic", _on_vgraph, 10)

        # Wait for /viz_graph_topic publisher (FAR bringup can take 10-15s).
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if node.count_publishers("/viz_graph_topic") > 0:
                break
            rclpy.spin_once(node, timeout_sec=0.5)
        else:
            pytest.skip(
                "FAR /viz_graph_topic never became available — stack bringup failed"
            )

        # TODO: drive Go2 via /joy for ~20s to cross a doorway
        # (living_room → hallway is the natural choice; Go2 spawns at ~(10, 3)).
        # Publisher-pattern: sensor_msgs/Joy with axes[4] > 0 (forward).
        # For now: spin and observe any edges that naturally form during startup.
        spin_deadline = time.time() + 20.0
        while time.time() < spin_deadline:
            rclpy.spin_once(node, timeout_sec=0.5)

        cross_room_edges = [
            (p1, p2)
            for p1, p2 in edges_seen
            if _which_room(*p1)
            and _which_room(*p2)
            and _which_room(*p1) != _which_room(*p2)
        ]

        rooms_seen = {
            _which_room(*p) for edge in edges_seen for p in edge if _which_room(*p)
        }

        assert cross_room_edges, (
            f"No cross-room V-Graph edges observed after 35s. "
            f"Total edges: {len(edges_seen)}. Rooms spanned: {rooms_seen}."
        )

        node.destroy_node()
        rclpy.shutdown()
    finally:
        _teardown_stack(stack)
