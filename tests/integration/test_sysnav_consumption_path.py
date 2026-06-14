# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M4 Part B — the /object_nodes_list CONSUMPTION path, end to end.

A stub publisher (standing in for SysNav's semantic_mapping_node, whose
perception weights are a gated download — DQ-3) publishes a real
``tare_planner/ObjectNodeList``; the EXISTING ``LiveSysnavBridge`` (v2.4,
retained in-tree, spins its own daemon executor) consumes it into a
``WorldModel``. Proves every line WE own; a real SysNav only changes who
fills the message.

Requires the SysNav sibling workspace's tare_planner msgs on the
PYTHONPATH (source its install) — skipped wherever they are absent.
"""
from __future__ import annotations

import time

import pytest


def _tare_msgs_available() -> bool:
    try:
        from tare_planner.msg import ObjectNodeList  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _tare_msgs_available(),
    reason="tare_planner msgs not on PYTHONPATH (source the SysNav install)",
)


def test_object_nodes_reach_world_model() -> None:
    import rclpy
    from geometry_msgs.msg import Point
    from tare_planner.msg import ObjectNode, ObjectNodeList

    from vector_os_nano.core.world_model import WorldModel
    from vector_os_nano.integrations.sysnav_bridge.live_bridge import (
        LiveSysnavBridge,
    )

    if not rclpy.ok():
        rclpy.init()

    wm = WorldModel()
    bridge = LiveSysnavBridge(world_model=wm)
    assert bridge.start() is True, "LiveSysnavBridge failed to start"

    pub_node = rclpy.create_node("sysnav_stub_publisher")
    pub = pub_node.create_publisher(ObjectNodeList, "/object_nodes_list", 10)

    def _node(obj_id: int, label: str, x: float, y: float, z: float) -> ObjectNode:
        n = ObjectNode()
        n.object_id = [obj_id]
        n.label = label
        n.status = True
        n.is_asked_vlm = True
        n.position = Point(x=x, y=y, z=z)
        return n

    msg = ObjectNodeList()
    msg.nodes = [
        _node(0, "sofa", 1.0, 2.0, 0.4),
        _node(1, "table", -0.5, 1.0, 0.6),
    ]

    try:
        deadline = time.time() + 10.0
        while time.time() < deadline and len(wm.get_objects()) < 2:
            pub.publish(msg)
            time.sleep(0.1)  # the bridge spins its own daemon executor

        objs = {o.label: o for o in wm.get_objects()}
        assert set(objs) == {"sofa", "table"}, f"world model got: {list(objs)}"
        assert objs["sofa"].x == pytest.approx(1.0)
        assert objs["sofa"].y == pytest.approx(2.0)
        # v2.4 contract: SysNav ids map to sysnav_<id0> canonical object ids,
        # and a VLM-confirmed node carries full confidence.
        assert objs["sofa"].object_id == "sysnav_0"
        assert objs["sofa"].confidence == pytest.approx(1.0)
    finally:
        pub_node.destroy_node()
        bridge.stop()
