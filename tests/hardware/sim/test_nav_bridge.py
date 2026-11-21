"""The packaged external bridge creates real ROS entities; no navigation-success claim."""
from unittest.mock import Mock
import struct
import pytest

pytestmark = pytest.mark.sim


def test_packaged_bridge_binds_callbacks_and_publishes_terrain(tmp_path, monkeypatch):
    import rclpy
    from vector_os_nano.sim.nav.ros2.bridge import NavigationBridge
    monkeypatch.setenv('VECTOR_RUN_DIR', str(tmp_path))
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    node = None
    try:
        node = NavigationBridge(Mock(), None, None, Mock(), terrain_map=tmp_path/'terrain.npz', quiet=True)
        message = node._build_terrain_pc2([(1., 2., .4, .4)])
        assert message.header.frame_id == 'map'
        assert message.width == 1
        assert struct.unpack('<ffff', message.data) == pytest.approx((1., 2., .4, .4))
        assert callable(node._follow_path) and callable(node._stuck_detector)
    finally:
        if node is not None:
            node.destroy_node()
        if owned:
            rclpy.shutdown()
