"""Tests for NavStackClient -- navigation stack interface wrapper."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestNavStackClientNoROS2:
    def test_import_without_rclpy(self):
        """NavStackClient module should import even without rclpy."""
        from vector_os_nano.core.nav_client import NavStackClient
        assert NavStackClient is not None

    def test_not_available_without_node(self):
        from vector_os_nano.core.nav_client import NavStackClient
        client = NavStackClient(node=None)
        assert not client.is_available

    def test_navigate_to_without_node_returns_false(self):
        from vector_os_nano.core.nav_client import NavStackClient
        client = NavStackClient(node=None)
        result = client.navigate_to(5.0, 3.0)
        assert result is False

    def test_cancel_without_node_is_noop(self):
        """cancel() with no node should not raise."""
        from vector_os_nano.core.nav_client import NavStackClient
        client = NavStackClient(node=None)
        client.cancel()  # must not raise

    def test_get_state_estimation_without_node_returns_none(self):
        from vector_os_nano.core.nav_client import NavStackClient
        client = NavStackClient(node=None)
        assert client.get_state_estimation() is None

    def test_default_timeout_is_positive(self):
        from vector_os_nano.core.nav_client import NavStackClient
        client = NavStackClient(node=None)
        assert client._timeout > 0


class TestNavStackClientWithMockNode:
    """Tests using a fully mocked ROS2 node (no real rclpy needed)."""

    def _make_client(self):
        """Return a NavStackClient wired to a MagicMock node."""
        from vector_os_nano.core.nav_client import NavStackClient

        mock_node = MagicMock()
        mock_publisher = MagicMock()
        mock_node.create_publisher.return_value = mock_publisher
        mock_node.create_subscription = MagicMock()

        # Lazy ROS2 imports inside _setup_ros2 need geometry_msgs etc.
        # Patch them so the setup succeeds without a real ROS2 install.
        with patch.dict("sys.modules", {
            "geometry_msgs": MagicMock(),
            "geometry_msgs.msg": MagicMock(),
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
            "nav_msgs": MagicMock(),
            "nav_msgs.msg": MagicMock(),
        }):
            client = NavStackClient(node=mock_node)

        return client, mock_node, mock_publisher

    def test_is_available_with_node(self):
        client, _, _ = self._make_client()
        assert client.is_available

    def test_navigate_to_publishes_waypoint(self):
        """navigate_to publishes to /way_point and returns True when goal_reached."""
        from vector_os_nano.core.nav_client import NavStackClient

        client, mock_node, mock_publisher = self._make_client()

        # Side-effect: set _goal_reached when publish is called, simulating
        # immediate callback from the navigation stack.
        def _on_publish(msg: object) -> None:
            client._goal_reached = True

        mock_publisher.publish.side_effect = _on_publish

        with patch.dict("sys.modules", {
            "geometry_msgs": MagicMock(),
            "geometry_msgs.msg": MagicMock(),
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
        }):
            result = client.navigate_to(10.0, 5.0)

        assert result is True
        # Waypoint publisher must have been called.
        assert mock_publisher.publish.called

    def test_navigate_to_returns_false_when_not_available(self):
        from vector_os_nano.core.nav_client import NavStackClient
        client = NavStackClient(node=None)
        assert client.navigate_to(1.0, 2.0) is False

    def test_cancel_publishes(self):
        client, mock_node, mock_publisher = self._make_client()

        with patch.dict("sys.modules", {
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
        }):
            client.cancel()

        # cancel_pub.publish must have been called.
        assert mock_publisher.publish.called

    def test_get_state_estimation_returns_odometry(self):
        from vector_os_nano.core.nav_client import NavStackClient
        from vector_os_nano.core.types import Odometry

        client, _, _ = self._make_client()
        # Inject a mock odometry snapshot.
        client._last_odom = Odometry(timestamp=1.0, x=5.0, y=3.0)
        odom = client.get_state_estimation()
        assert isinstance(odom, Odometry)
        assert odom.x == 5.0
        assert odom.y == 3.0

    def test_get_state_estimation_returns_none_before_first_message(self):
        client, _, _ = self._make_client()
        # No odometry received yet.
        assert client.get_state_estimation() is None

    def test_on_goal_reached_updates_flag(self):
        client, _, _ = self._make_client()
        mock_msg = MagicMock()
        mock_msg.data = True
        client._on_goal_reached(mock_msg)
        assert client._goal_reached is True

    def test_on_goal_reached_false_clears_flag(self):
        client, _, _ = self._make_client()
        client._goal_reached = True
        mock_msg = MagicMock()
        mock_msg.data = False
        client._on_goal_reached(mock_msg)
        assert client._goal_reached is False

    def test_navigate_to_timeout_returns_false(self):
        """navigate_to should return False if goal not reached within timeout."""
        client, mock_node, mock_publisher = self._make_client()
        # goal_reached stays False; use a very short timeout.
        client._goal_reached = False

        with patch.dict("sys.modules", {
            "geometry_msgs": MagicMock(),
            "geometry_msgs.msg": MagicMock(),
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
        }):
            result = client.navigate_to(1.0, 2.0, timeout=0.05)

        assert result is False

    def test_custom_timeout_overrides_default(self):
        """navigate_to timeout kwarg overrides instance _timeout."""
        client, _, _ = self._make_client()
        client._goal_reached = False
        import time
        start = time.time()
        with patch.dict("sys.modules", {
            "geometry_msgs": MagicMock(),
            "geometry_msgs.msg": MagicMock(),
            "std_msgs": MagicMock(),
            "std_msgs.msg": MagicMock(),
        }):
            client.navigate_to(0.0, 0.0, timeout=0.05)
        elapsed = time.time() - start
        # Should complete in roughly 0.05 s, well under default 60 s.
        assert elapsed < 5.0
