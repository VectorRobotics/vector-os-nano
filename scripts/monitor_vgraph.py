#!/usr/bin/env python3
"""Monitor FAR V-Graph state in real-time.

Subscribes to FAR's output topics and prints V-Graph metrics.
Does NOT modify any nav stack config — read-only observer.

Usage (while nav stack is running):
    source /opt/ros/jazzy/setup.bash
    python3 scripts/monitor_vgraph.py

Shows:
    - /robot_vgraph: node count, edge count
    - /viz_graph_topic: visualization marker count
    - /registered_scan: point count per frame
"""
from __future__ import annotations

import sys
import time


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError:
        print("ROS2 not available. Source /opt/ros/jazzy/setup.bash first.")
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node("vgraph_monitor")
    qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
    qos_rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=5)

    stats: dict[str, int] = {
        "vgraph_msgs": 0,
        "viz_msgs": 0,
        "scan_msgs": 0,
        "last_scan_pts": 0,
    }

    # /robot_vgraph — FAR's encoded graph
    try:
        from visibility_graph_msg.msg import Graph
        def _vgraph_cb(msg: Graph) -> None:
            stats["vgraph_msgs"] += 1
            n_nodes = len(msg.nodes) if hasattr(msg, "nodes") else -1
            print(f"[VGRAPH] msg #{stats['vgraph_msgs']} nodes={n_nodes}")
        node.create_subscription(Graph, "/robot_vgraph", _vgraph_cb, qos_rel)
        print("Subscribed to /robot_vgraph (visibility_graph_msg/Graph)")
    except ImportError:
        print("visibility_graph_msg not available — skipping /robot_vgraph")

    # /viz_graph_topic — V-Graph edge visualization
    try:
        from visualization_msgs.msg import MarkerArray
        def _viz_cb(msg: MarkerArray) -> None:
            stats["viz_msgs"] += 1
            n_markers = len(msg.markers)
            if stats["viz_msgs"] % 5 == 0 or n_markers > 0:
                print(f"[VIZ] msg #{stats['viz_msgs']} markers={n_markers}")
        node.create_subscription(MarkerArray, "/viz_graph_topic", _viz_cb, qos_be)
        print("Subscribed to /viz_graph_topic (MarkerArray)")

        # Also monitor contour and node topics
        def _contour_cb(msg: MarkerArray) -> None:
            pass  # silent — just count
        def _node_cb(msg: MarkerArray) -> None:
            if stats["viz_msgs"] % 10 == 0:
                print(f"  [NODES] {len(msg.markers)} node markers")
        node.create_subscription(MarkerArray, "/viz_contour_topic", _contour_cb, qos_be)
        node.create_subscription(MarkerArray, "/viz_node_topic", _node_cb, qos_be)
    except ImportError:
        print("visualization_msgs not available — skipping /viz_graph_topic")

    # /registered_scan — obstacle pointcloud
    try:
        from sensor_msgs.msg import PointCloud2
        def _scan_cb(msg: PointCloud2) -> None:
            stats["scan_msgs"] += 1
            n_pts = msg.width * msg.height
            stats["last_scan_pts"] = n_pts
            if stats["scan_msgs"] % 25 == 0:  # every ~5s at 5Hz
                print(f"[SCAN] {n_pts} pts in /registered_scan")
        node.create_subscription(PointCloud2, "/registered_scan", _scan_cb, qos_rel)
        print("Subscribed to /registered_scan")
    except ImportError:
        print("sensor_msgs not available — skipping /registered_scan")

    print("\nMonitoring V-Graph. Ctrl+C to stop.\n")
    try:
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
            elapsed = time.time() - start
            # Periodic summary every 30s
            if elapsed > 0 and int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(
                    f"\n--- {int(elapsed)}s: vgraph={stats['vgraph_msgs']} "
                    f"viz={stats['viz_msgs']} scans={stats['scan_msgs']} "
                    f"last_pts={stats['last_scan_pts']} ---\n"
                )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
