"""Tests for NavStackClient Nav2 action mode.

Tests run WITHOUT rclpy / nav2_msgs installed by patching every ROS2 import
at the sys.modules level.  Each test class documents which aspect it covers.

Coverage map
------------
TestModeSelection           -- constructor mode resolution (cmu / nav2 / auto)
TestNav2Navigate            -- _nav2_navigate success / failure / timeout paths
TestNav2Cancel              -- cancel() in nav2 mode
TestNav2Feedback            -- _on_nav2_feedback callback + nav2_feedback property
TestNav2StateEstimation     -- odom topic selection by mode
TestBackwardCompatibility   -- all existing CMU behaviour unchanged
TestEdgeCases               -- zero timeout, sequential goals, overlapping goals
"""
from __future__ import annotations

import sys
import time
import types
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATUS_SUCCEEDED = 4
STATUS_ABORTED = 6
STATUS_CANCELED = 5


def _make_ros2_modules():
    """Return (sys.modules patch dict, action_client_instance mock).

    The action_client_instance mock is the object returned by
    ActionClient(node, NavigateToPose, topic) and is what nav_client.py
    stores as self._nav2_client.

    By default wait_for_server returns False (no server found).
    Tests that want Nav2 to be found must set:
        action_client_instance.wait_for_server.return_value = True
    """
    # --- action_msgs ---
    action_msgs = types.ModuleType("action_msgs")
    action_msgs_msg = types.ModuleType("action_msgs.msg")

    class _GoalStatus:
        STATUS_SUCCEEDED = STATUS_SUCCEEDED
        STATUS_ABORTED = STATUS_ABORTED
        STATUS_CANCELED = STATUS_CANCELED

    action_msgs_msg.GoalStatus = _GoalStatus
    action_msgs.msg = action_msgs_msg

    # --- geometry_msgs ---
    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    # Use MagicMock() instances as constructors so we can inspect calls
    PointStamped_cls = MagicMock(name="PointStamped")
    PoseStamped_cls = MagicMock(name="PoseStamped")
    geometry_msgs_msg.PointStamped = PointStamped_cls
    geometry_msgs_msg.PoseStamped = PoseStamped_cls
    geometry_msgs.msg = geometry_msgs_msg

    # --- std_msgs ---
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    Bool_cls = MagicMock(name="Bool")
    std_msgs_msg.Bool = Bool_cls
    std_msgs.msg = std_msgs_msg

    # --- nav_msgs ---
    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    Odometry_cls = MagicMock(name="Odometry")
    nav_msgs_msg.Odometry = Odometry_cls
    nav_msgs.msg = nav_msgs_msg

    # --- nav2_msgs ---
    nav2_msgs = types.ModuleType("nav2_msgs")
    nav2_msgs_action = types.ModuleType("nav2_msgs.action")

    # NavigateToPose.Goal() must return a real MagicMock with chainable attributes
    goal_instance = MagicMock(name="NavigateToPoseGoal")
    NavigateToPose_cls = MagicMock(name="NavigateToPose")
    NavigateToPose_cls.Goal.return_value = goal_instance
    nav2_msgs_action.NavigateToPose = NavigateToPose_cls
    nav2_msgs.action = nav2_msgs_action

    # --- rclpy.action ---
    rclpy = types.ModuleType("rclpy")
    rclpy_action = types.ModuleType("rclpy.action")

    # action_client_instance is what ActionClient(node, type, topic) returns
    action_client_instance = MagicMock(name="ActionClientInstance")
    action_client_instance.wait_for_server.return_value = False  # default: no server

    ActionClient_cls = MagicMock(name="ActionClient")
    ActionClient_cls.return_value = action_client_instance

    rclpy_action.ActionClient = ActionClient_cls
    rclpy.action = rclpy_action

    mods = {
        "action_msgs": action_msgs,
        "action_msgs.msg": action_msgs_msg,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "nav_msgs": nav_msgs,
        "nav_msgs.msg": nav_msgs_msg,
        "nav2_msgs": nav2_msgs,
        "nav2_msgs.action": nav2_msgs_action,
        "rclpy": rclpy,
        "rclpy.action": rclpy_action,
    }
    return mods, action_client_instance


def _make_mock_node() -> MagicMock:
    node = MagicMock(name="RclpyNode")
    node.create_publisher.return_value = MagicMock(name="Publisher")
    node.create_subscription = MagicMock()
    clock = MagicMock()
    clock.now.return_value.to_msg.return_value = MagicMock()
    node.get_clock.return_value = clock
    return node


def _make_succeeded_futures():
    """Return (send_future, goal_handle, result_future) all immediately done, STATUS_SUCCEEDED."""
    result = MagicMock(name="Result")
    result.status = STATUS_SUCCEEDED

    result_future = MagicMock(name="ResultFuture")
    result_future.done.return_value = True
    result_future.result.return_value = result

    goal_handle = MagicMock(name="GoalHandle")
    goal_handle.accepted = True
    goal_handle.get_result_async.return_value = result_future

    send_future = MagicMock(name="SendFuture")
    send_future.done.return_value = True
    send_future.result.return_value = goal_handle

    return send_future, goal_handle, result_future


def _make_nav2_client_with_mods(mode: str = "nav2"):
    """Return (client, action_client_instance, mods_dict).

    The action_client_instance is the mock stored as client._nav2_client.
    """
    from vector_os_nano.core.nav_client import NavStackClient

    mods, ac_instance = _make_ros2_modules()
    if mode == "nav2":
        # In explicit nav2 mode the code does NOT call wait_for_server
        pass
    elif mode == "auto":
        ac_instance.wait_for_server.return_value = True

    with patch.dict(sys.modules, mods):
        node = _make_mock_node()
        client = NavStackClient(node=node, mode=mode)

    return client, ac_instance, mods


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ros2_mods():
    """Patch all ROS2 modules; yield (mods_dict, action_client_instance)."""
    mods, ac_instance = _make_ros2_modules()
    with patch.dict(sys.modules, mods):
        yield mods, ac_instance


# ---------------------------------------------------------------------------
# TestModeSelection
# ---------------------------------------------------------------------------

class TestModeSelection:
    def test_mode_nav2_creates_action_client(self, ros2_mods):
        """mode='nav2' should create an ActionClient for NavigateToPose."""
        mods, ac_instance = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")

        assert client._nav2_client is not None
        assert client._active_mode == "nav2"

    def test_mode_cmu_uses_topic_interface(self, ros2_mods):
        """mode='cmu' should create publishers and subscribers."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="cmu")

        assert client._active_mode == "cmu"
        assert client._waypoint_pub is not None
        assert client._cancel_pub is not None
        assert client._nav2_client is None

    def test_mode_auto_detects_nav2_when_available(self, ros2_mods):
        """mode='auto' should use Nav2 when the action server is reachable."""
        mods, ac_instance = ros2_mods
        # Make server available
        ac_instance.wait_for_server.return_value = True

        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="auto")

        assert client._active_mode == "nav2"
        assert client._nav2_client is not None

    def test_mode_auto_falls_back_to_cmu(self, ros2_mods):
        """mode='auto' should fallback to CMU when Nav2 is not reachable."""
        mods, ac_instance = ros2_mods
        ac_instance.wait_for_server.return_value = False  # default, but explicit

        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="auto")

        assert client._active_mode == "cmu"
        assert client._nav2_client is None
        assert client._waypoint_pub is not None

    def test_mode_auto_timeout_is_2_seconds(self, ros2_mods):
        """Auto detection must call wait_for_server with timeout_sec=2.0."""
        mods, ac_instance = ros2_mods
        ac_instance.wait_for_server.return_value = False

        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        NavStackClient(node=node, mode="auto")

        ac_instance.wait_for_server.assert_called_once_with(timeout_sec=2.0)

    def test_mode_property_returns_nav2(self, ros2_mods):
        """mode property should return 'nav2' when nav2 active."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")
        assert client.mode == "nav2"

    def test_mode_property_returns_cmu(self, ros2_mods):
        """mode property should return 'cmu' when cmu active."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="cmu")
        assert client.mode == "cmu"

    def test_mode_property_returns_none_without_node(self):
        """mode property should return None before setup."""
        from vector_os_nano.core.nav_client import NavStackClient

        client = NavStackClient(node=None)
        assert client.mode is None

    def test_nav2_mode_without_nav2_msgs_fails_gracefully(self):
        """If nav2_msgs is not installed, mode='nav2' should not crash."""
        # Use only geometry/std/nav mocks but exclude nav2 and rclpy.action
        base_mods, _ = _make_ros2_modules()
        trimmed = {
            k: v for k, v in base_mods.items()
            if "nav2" not in k and "rclpy" not in k
        }
        # Explicitly remove them to force ImportError
        trimmed["nav2_msgs"] = None  # type: ignore[assignment]
        trimmed["nav2_msgs.action"] = None  # type: ignore[assignment]
        trimmed["rclpy"] = None  # type: ignore[assignment]
        trimmed["rclpy.action"] = None  # type: ignore[assignment]

        with patch.dict(sys.modules, trimmed):
            from vector_os_nano.core.nav_client import NavStackClient

            node = _make_mock_node()
            client = NavStackClient(node=node, mode="nav2")
            assert not client.is_available

    def test_auto_mode_without_nav2_msgs_uses_cmu(self, ros2_mods):
        """If nav2_msgs is unavailable, mode='auto' should fall through to CMU."""
        mods, ac_instance = ros2_mods
        # Make ActionClient constructor raise ImportError
        mods["rclpy.action"].ActionClient.side_effect = ImportError("no nav2")

        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="auto")

        assert client._active_mode == "cmu"


# ---------------------------------------------------------------------------
# TestNav2Navigate
# ---------------------------------------------------------------------------

class TestNav2Navigate:
    def _make_client(self, ros2_mods):
        mods, ac_instance = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")
        return client, ac_instance

    def test_navigate_to_success(self, ros2_mods):
        """Goal accepted + STATUS_SUCCEEDED should return True."""
        client, ac_instance = self._make_client(ros2_mods)
        send_future, _, _ = _make_succeeded_futures()
        ac_instance.send_goal_async.return_value = send_future

        result = client.navigate_to(3.0, 4.0)
        assert result is True

    def test_navigate_to_goal_rejected(self, ros2_mods):
        """Goal rejected by Nav2 action server should return False."""
        client, ac_instance = self._make_client(ros2_mods)

        goal_handle = MagicMock(name="GoalHandle")
        goal_handle.accepted = False
        send_future = MagicMock(name="SendFuture")
        send_future.done.return_value = True
        send_future.result.return_value = goal_handle
        ac_instance.send_goal_async.return_value = send_future

        result = client.navigate_to(1.0, 2.0)
        assert result is False

    def test_navigate_to_goal_aborted(self, ros2_mods):
        """STATUS_ABORTED should return False."""
        client, ac_instance = self._make_client(ros2_mods)
        send_future, _, result_future = _make_succeeded_futures()
        result_future.result.return_value.status = STATUS_ABORTED
        ac_instance.send_goal_async.return_value = send_future

        result = client.navigate_to(1.0, 2.0)
        assert result is False

    def test_navigate_to_timeout_sends_cancel(self, ros2_mods):
        """When the result future never completes, cancel_goal_async must be called."""
        client, ac_instance = self._make_client(ros2_mods)

        goal_handle = MagicMock(name="GoalHandle")
        goal_handle.accepted = True
        result_future = MagicMock(name="ResultFuture")
        result_future.done.return_value = False  # never done
        goal_handle.get_result_async.return_value = result_future

        send_future = MagicMock(name="SendFuture")
        send_future.done.return_value = True
        send_future.result.return_value = goal_handle
        ac_instance.send_goal_async.return_value = send_future

        result = client.navigate_to(1.0, 2.0, timeout=0.05)

        assert result is False
        goal_handle.cancel_goal_async.assert_called_once()

    def test_navigate_to_send_timeout(self, ros2_mods):
        """If send_goal_async future never resolves, return False."""
        client, ac_instance = self._make_client(ros2_mods)

        send_future = MagicMock(name="SendFuture")
        send_future.done.return_value = False  # never done
        ac_instance.send_goal_async.return_value = send_future

        result = client.navigate_to(1.0, 2.0, timeout=0.05)
        assert result is False

    def test_navigate_to_correct_pose(self, ros2_mods):
        """Goal pose should set x, y, frame_id='map', orientation.w=1."""
        client, ac_instance = self._make_client(ros2_mods)
        send_future, _, _ = _make_succeeded_futures()
        ac_instance.send_goal_async.return_value = send_future

        client.navigate_to(7.5, -2.3)

        sent_goal = ac_instance.send_goal_async.call_args[0][0]
        assert float(sent_goal.pose.pose.position.x) == 7.5
        assert float(sent_goal.pose.pose.position.y) == -2.3
        assert float(sent_goal.pose.pose.position.z) == 0.0
        assert float(sent_goal.pose.pose.orientation.w) == 1.0
        assert sent_goal.pose.header.frame_id == "map"

    def test_navigate_to_custom_timeout(self, ros2_mods):
        """A custom timeout that expires before result should return False quickly."""
        client, ac_instance = self._make_client(ros2_mods)

        send_future = MagicMock(name="SendFuture")
        send_future.done.return_value = False
        ac_instance.send_goal_async.return_value = send_future

        start = time.time()
        result = client.navigate_to(0.0, 0.0, timeout=0.12)
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 2.0  # must not block for 60 s

    def test_navigate_to_not_available(self):
        """navigate_to with no action client returns False."""
        from vector_os_nano.core.nav_client import NavStackClient

        client = NavStackClient(node=None, mode="nav2")
        assert client.navigate_to(1.0, 1.0) is False

    def test_feedback_callback_registered(self, ros2_mods):
        """send_goal_async should be called with feedback_callback set."""
        client, ac_instance = self._make_client(ros2_mods)
        send_future, _, _ = _make_succeeded_futures()
        ac_instance.send_goal_async.return_value = send_future

        client.navigate_to(1.0, 2.0)

        kwargs = ac_instance.send_goal_async.call_args[1]
        assert "feedback_callback" in kwargs
        assert kwargs["feedback_callback"] == client._on_nav2_feedback

    def test_goal_handle_cleared_after_success(self, ros2_mods):
        """After successful navigation, _nav2_goal_handle should be None."""
        client, ac_instance = self._make_client(ros2_mods)
        send_future, _, _ = _make_succeeded_futures()
        ac_instance.send_goal_async.return_value = send_future

        client.navigate_to(1.0, 2.0)
        assert client._nav2_goal_handle is None

    def test_goal_handle_cleared_after_failure(self, ros2_mods):
        """After a failed navigation (aborted), _nav2_goal_handle should be None."""
        client, ac_instance = self._make_client(ros2_mods)
        send_future, _, result_future = _make_succeeded_futures()
        result_future.result.return_value.status = STATUS_ABORTED
        ac_instance.send_goal_async.return_value = send_future

        client.navigate_to(1.0, 2.0)
        assert client._nav2_goal_handle is None


# ---------------------------------------------------------------------------
# TestNav2Cancel
# ---------------------------------------------------------------------------

class TestNav2Cancel:
    def _make_nav2_client(self, ros2_mods):
        mods, ac_instance = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        return NavStackClient(node=node, mode="nav2"), ac_instance

    def test_cancel_active_goal(self, ros2_mods):
        """cancel() should call cancel_goal_async on the active goal handle."""
        client, _ = self._make_nav2_client(ros2_mods)
        mock_handle = MagicMock(name="GoalHandle")
        client._nav2_goal_handle = mock_handle

        client.cancel()

        mock_handle.cancel_goal_async.assert_called_once()

    def test_cancel_no_active_goal_does_not_raise(self, ros2_mods):
        """cancel() with no active goal should silently do nothing."""
        client, _ = self._make_nav2_client(ros2_mods)
        client._nav2_goal_handle = None

        client.cancel()  # must not raise

    def test_cancel_resets_goal_handle(self, ros2_mods):
        """After cancel, _nav2_goal_handle should be None."""
        client, _ = self._make_nav2_client(ros2_mods)
        client._nav2_goal_handle = MagicMock(name="GoalHandle")

        client.cancel()

        assert client._nav2_goal_handle is None

    def test_cancel_exception_does_not_propagate(self, ros2_mods):
        """If cancel_goal_async raises, cancel() should log and not re-raise."""
        client, _ = self._make_nav2_client(ros2_mods)
        mock_handle = MagicMock(name="GoalHandle")
        mock_handle.cancel_goal_async.side_effect = RuntimeError("oops")
        client._nav2_goal_handle = mock_handle

        client.cancel()  # must not raise
        assert client._nav2_goal_handle is None


# ---------------------------------------------------------------------------
# TestNav2Feedback
# ---------------------------------------------------------------------------

class TestNav2Feedback:
    def _make_nav2_client(self, ros2_mods):
        mods, ac_instance = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        return NavStackClient(node=node, mode="nav2"), ac_instance

    def test_feedback_callback_stores_feedback(self, ros2_mods):
        """_on_nav2_feedback should store feedback_msg.feedback."""
        client, _ = self._make_nav2_client(ros2_mods)
        feedback_msg = MagicMock()
        feedback_msg.feedback.distance_remaining = 3.5

        client._on_nav2_feedback(feedback_msg)

        assert client._nav2_feedback is feedback_msg.feedback
        assert client._nav2_feedback.distance_remaining == 3.5

    def test_nav2_feedback_property_initial_none(self, ros2_mods):
        """nav2_feedback should be None before any feedback arrives."""
        client, _ = self._make_nav2_client(ros2_mods)
        assert client.nav2_feedback is None

    def test_nav2_feedback_property_reflects_latest(self, ros2_mods):
        """nav2_feedback property should return the last stored feedback."""
        client, _ = self._make_nav2_client(ros2_mods)
        fb1 = MagicMock(name="Feedback1")
        fb2 = MagicMock(name="Feedback2")

        msg1 = MagicMock()
        msg1.feedback = fb1
        msg2 = MagicMock()
        msg2.feedback = fb2

        client._on_nav2_feedback(msg1)
        client._on_nav2_feedback(msg2)
        assert client.nav2_feedback is fb2

    def test_feedback_reset_before_new_goal(self, ros2_mods):
        """_nav2_navigate should reset _nav2_feedback to None before sending goal."""
        client, ac_instance = self._make_nav2_client(ros2_mods)
        # Pre-load stale feedback
        client._nav2_feedback = MagicMock(name="StaleFeedback")

        # Track when send_goal_async is called and check feedback at that point
        feedback_at_send = []

        def _capture_feedback_state(goal, feedback_callback):
            feedback_at_send.append(client._nav2_feedback)
            send_future, _, _ = _make_succeeded_futures()
            return send_future

        ac_instance.send_goal_async.side_effect = _capture_feedback_state

        client.navigate_to(1.0, 2.0)

        # Feedback must have been None when send_goal_async was called
        assert len(feedback_at_send) == 1
        assert feedback_at_send[0] is None


# ---------------------------------------------------------------------------
# TestNav2StateEstimation
# ---------------------------------------------------------------------------

class TestNav2StateEstimation:
    def test_nav2_subscribes_to_odom(self, ros2_mods):
        """Nav2 mode should subscribe to /odom."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        NavStackClient(node=node, mode="nav2")

        subscribed_topics = [c[0][1] for c in node.create_subscription.call_args_list]
        assert "/odom" in subscribed_topics

    def test_cmu_subscribes_to_state_estimation(self, ros2_mods):
        """CMU mode should subscribe to /state_estimation."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        NavStackClient(node=node, mode="cmu")

        subscribed_topics = [c[0][1] for c in node.create_subscription.call_args_list]
        assert "/state_estimation" in subscribed_topics

    def test_nav2_does_not_subscribe_to_state_estimation(self, ros2_mods):
        """Nav2 mode must NOT subscribe to /state_estimation."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        NavStackClient(node=node, mode="nav2")

        subscribed_topics = [c[0][1] for c in node.create_subscription.call_args_list]
        assert "/state_estimation" not in subscribed_topics

    def test_get_state_estimation_returns_odometry(self, ros2_mods):
        """get_state_estimation should return the stored Odometry snapshot."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient
        from vector_os_nano.core.types import Odometry

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")
        client._last_odom = Odometry(timestamp=0.0, x=1.0, y=2.0)

        odom = client.get_state_estimation()
        assert isinstance(odom, Odometry)
        assert odom.x == 1.0
        assert odom.y == 2.0

    def test_on_state_estimation_populates_last_odom(self, ros2_mods):
        """_on_state_estimation callback should populate _last_odom."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")

        msg = MagicMock()
        msg.pose.pose.position.x = 5.0
        msg.pose.pose.position.y = 6.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 0.1
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.z = 0.0

        client._on_state_estimation(msg)

        assert client._last_odom is not None
        assert client._last_odom.x == 5.0
        assert client._last_odom.y == 6.0


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def _make_cmu_client(self, ros2_mods):
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        return NavStackClient(node=node, mode="cmu")

    def test_default_mode_is_auto(self):
        """NavStackClient() default mode kwarg should be 'auto'."""
        from vector_os_nano.core.nav_client import NavStackClient

        import inspect
        sig = inspect.signature(NavStackClient.__init__)
        assert sig.parameters["mode"].default == "auto"

    def test_cmu_navigate_publishes_waypoint(self, ros2_mods):
        """CMU navigate_to should publish a PointStamped to /way_point."""
        client = self._make_cmu_client(ros2_mods)

        def _set_reached(msg: object) -> None:
            client._goal_reached = True

        client._waypoint_pub.publish.side_effect = _set_reached

        result = client.navigate_to(2.0, 3.0)
        assert result is True
        assert client._waypoint_pub.publish.called

    def test_cmu_navigate_timeout_returns_false(self, ros2_mods):
        """CMU navigate_to with short timeout returns False when goal never reached."""
        client = self._make_cmu_client(ros2_mods)
        client._goal_reached = False

        result = client.navigate_to(1.0, 1.0, timeout=0.05)
        assert result is False

    def test_cmu_cancel_publishes_bool(self, ros2_mods):
        """CMU cancel() should publish Bool(data=True) to /cancel_goal."""
        client = self._make_cmu_client(ros2_mods)

        client.cancel()

        assert client._cancel_pub.publish.called

    def test_is_available_cmu_checks_publisher(self, ros2_mods):
        """is_available in CMU mode checks _waypoint_pub not None."""
        client = self._make_cmu_client(ros2_mods)
        assert client.is_available
        client._waypoint_pub = None
        assert not client.is_available

    def test_is_available_nav2_checks_action_client(self, ros2_mods):
        """is_available in Nav2 mode checks _nav2_client not None."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")
        assert client.is_available
        client._nav2_client = None
        assert not client.is_available

    def test_no_node_is_not_available(self):
        """NavStackClient(node=None) must not be available."""
        from vector_os_nano.core.nav_client import NavStackClient

        client = NavStackClient(node=None)
        assert not client.is_available

    def test_no_node_navigate_returns_false(self):
        """navigate_to with no node returns False in any mode."""
        from vector_os_nano.core.nav_client import NavStackClient

        client = NavStackClient(node=None)
        assert client.navigate_to(0.0, 0.0) is False

    def test_no_node_cancel_is_noop(self):
        """cancel() with no node should not raise."""
        from vector_os_nano.core.nav_client import NavStackClient

        client = NavStackClient(node=None)
        client.cancel()  # must not raise

    def test_get_state_estimation_returns_none_initially(self, ros2_mods):
        """Before any odom message, get_state_estimation should return None."""
        client = self._make_cmu_client(ros2_mods)
        assert client.get_state_estimation() is None

    def test_on_goal_reached_sets_flag(self, ros2_mods):
        """_on_goal_reached callback should set _goal_reached."""
        client = self._make_cmu_client(ros2_mods)
        msg = MagicMock()
        msg.data = True
        client._on_goal_reached(msg)
        assert client._goal_reached is True

    def test_on_goal_reached_false_clears_flag(self, ros2_mods):
        """_on_goal_reached with data=False should clear _goal_reached."""
        client = self._make_cmu_client(ros2_mods)
        client._goal_reached = True
        msg = MagicMock()
        msg.data = False
        client._on_goal_reached(msg)
        assert client._goal_reached is False


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def _make_nav2_client(self, ros2_mods):
        mods, ac_instance = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node = _make_mock_node()
        client = NavStackClient(node=node, mode="nav2")
        return client, ac_instance

    def test_navigate_to_zero_timeout(self, ros2_mods):
        """Zero timeout should return almost immediately."""
        client, ac_instance = self._make_nav2_client(ros2_mods)

        send_future = MagicMock(name="SendFuture")
        send_future.done.return_value = False  # never done
        ac_instance.send_goal_async.return_value = send_future

        start = time.time()
        result = client.navigate_to(1.0, 1.0, timeout=0.0)
        elapsed = time.time() - start

        assert result is False
        assert elapsed < 1.0

    def test_multiple_sequential_goals(self, ros2_mods):
        """Multiple successive navigate_to calls should all succeed."""
        client, ac_instance = self._make_nav2_client(ros2_mods)

        for x, y in [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]:
            send_future, _, _ = _make_succeeded_futures()
            ac_instance.send_goal_async.return_value = send_future
            result = client.navigate_to(x, y)
            assert result is True

    def test_goal_handle_none_between_goals(self, ros2_mods):
        """Between navigate_to calls, _nav2_goal_handle should be None."""
        client, ac_instance = self._make_nav2_client(ros2_mods)

        send_future, _, _ = _make_succeeded_futures()
        ac_instance.send_goal_async.return_value = send_future
        client.navigate_to(1.0, 2.0)

        assert client._nav2_goal_handle is None

    def test_navigate_with_large_coordinates(self, ros2_mods):
        """navigate_to should handle large coordinate values without error."""
        client, ac_instance = self._make_nav2_client(ros2_mods)

        send_future, _, _ = _make_succeeded_futures()
        ac_instance.send_goal_async.return_value = send_future

        result = client.navigate_to(1e6, -1e6)
        assert result is True

        sent_goal = ac_instance.send_goal_async.call_args[0][0]
        assert sent_goal.pose.pose.position.x == 1e6
        assert sent_goal.pose.pose.position.y == -1e6

    def test_cmu_and_nav2_modes_independent(self, ros2_mods):
        """CMU and Nav2 clients created from separate NavStackClient instances don't interfere."""
        mods, _ = ros2_mods
        from vector_os_nano.core.nav_client import NavStackClient

        node_a = _make_mock_node()
        node_b = _make_mock_node()

        cmu = NavStackClient(node=node_a, mode="cmu")
        nav2 = NavStackClient(node=node_b, mode="nav2")

        assert cmu._active_mode == "cmu"
        assert nav2._active_mode == "nav2"
        assert cmu._nav2_client is None
        assert nav2._waypoint_pub is None

    def test_module_imports_without_ros2(self):
        """nav_client module must be importable with zero ROS2 packages present."""
        from vector_os_nano.core import nav_client  # noqa: F401
        assert nav_client.NavStackClient is not None

    def test_default_timeout_60_seconds(self):
        """Default timeout should be 60.0 seconds."""
        from vector_os_nano.core.nav_client import NavStackClient

        client = NavStackClient(node=None)
        assert client._timeout == 60.0
