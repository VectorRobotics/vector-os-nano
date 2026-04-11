"""Gazebo Harmonic Go2 proxy — controls Go2 in Gz Sim via ROS2 topics.

Inherits from Go2ROS2Proxy for odometry/camera subscriptions and nav methods.
Overrides motion commands to use unitree_guide_controller's /control_input
topic (control_input_msgs/msg/Inputs) instead of /cmd_vel_nav (Twist).

FSM commands (unitree_guide_controller):
    1 = PASSIVE     (motors off)
    2 = FIXEDDOWN   (lie down)
    3 = FIXEDSTAND  (stand rigid)
    4 = FREESTAND   (stand with compliance)
    5 = TROTTING    (walk — velocity via lx/ly/rx)
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

logger = logging.getLogger(__name__)

# FSM state enum values (from controller_common/enumClass.h)
_FSM_PASSIVE = 1
_FSM_FIXEDDOWN = 2
_FSM_FIXEDSTAND = 3
_FSM_FREESTAND = 4
_FSM_TROTTING = 5


class GazeboGo2Proxy(Go2ROS2Proxy):
    """BaseProtocol for Go2 in Gazebo Harmonic with unitree_guide_controller.

    Subscribes to the same topics as Go2ROS2Proxy (/state_estimation,
    /camera/image, /camera/depth). Publishes motion commands to
    /control_input instead of /cmd_vel_nav.
    """

    _NODE_NAME: str = "gazebo_go2_proxy"

    def __init__(self) -> None:
        super().__init__()
        self._ctrl_pub: Any = None
        self._current_fsm: int = _FSM_PASSIVE

    @property
    def name(self) -> str:
        return "gazebo_go2"

    @property
    def supports_lidar(self) -> bool:
        return True

    def connect(self) -> None:
        """Connect to Gazebo Go2 — sets up /control_input publisher."""
        if not self.is_gazebo_running():
            raise ConnectionError(
                "Gazebo not running — /clock topic not found. "
                "Start with: bash scripts/launch_gazebo.sh"
            )
        logger.info("Gazebo confirmed running, connecting via ROS2...")
        super().connect()

        # Create /control_input publisher for unitree_guide_controller
        if self._node is not None:
            try:
                from control_input_msgs.msg import Inputs
                self._ctrl_pub = self._node.create_publisher(Inputs, "/control_input", 10)
                # Command stand on connect
                self._send_fsm(_FSM_FIXEDSTAND)
                logger.info("GazeboGo2Proxy: /control_input publisher ready, FIXEDSTAND sent")
            except ImportError:
                logger.warning(
                    "control_input_msgs not found — falling back to /cmd_vel_nav. "
                    "Source quadruped_ros2_control install first."
                )
                self._ctrl_pub = None

        logger.info("GazeboGo2Proxy connected (node=%s)", self._NODE_NAME)

    # ------------------------------------------------------------------
    # FSM command interface
    # ------------------------------------------------------------------

    def _send_fsm(self, command: int, lx: float = 0.0, ly: float = 0.0,
                  rx: float = 0.0, ry: float = 0.0) -> None:
        """Publish a control_input_msgs/Inputs message."""
        if self._ctrl_pub is None:
            return
        try:
            from control_input_msgs.msg import Inputs
            msg = Inputs()
            msg.command = command
            msg.lx = float(lx)
            msg.ly = float(ly)
            msg.rx = float(rx)
            msg.ry = float(ry)
            self._ctrl_pub.publish(msg)
            self._current_fsm = command
        except Exception as exc:
            logger.warning("Failed to publish /control_input: %s", exc)

    # ------------------------------------------------------------------
    # Motion overrides — use FSM commands instead of Twist
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Send velocity via unitree_guide_controller FSM.

        Switches to TROTTING mode and sets velocity via lx/ly/rx.
        lx = forward speed, ly = lateral speed, rx = yaw rate.
        """
        if self._ctrl_pub is not None:
            if vx == 0.0 and vy == 0.0 and vyaw == 0.0:
                # Zero velocity — switch to stand
                self._send_fsm(_FSM_FIXEDSTAND)
            else:
                self._send_fsm(_FSM_TROTTING, lx=vx, ly=vy, rx=vyaw)
        else:
            # Fallback to /cmd_vel_nav if control_input not available
            super().set_velocity(vx, vy, vyaw)

    def walk(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0,
             duration: float = 1.0) -> bool:
        """Walk at velocity for duration, then stop."""
        deadline = time.time() + duration
        while time.time() < deadline:
            self.set_velocity(vx, vy, vyaw)
            time.sleep(0.1)  # 10 Hz command rate
        self.set_velocity(0.0, 0.0, 0.0)
        return True

    def stop(self) -> None:
        """Emergency stop — switch to stand."""
        self._send_fsm(_FSM_FIXEDSTAND)

    def stand(self, duration: float = 1.0) -> bool:
        """Command fixed stand posture."""
        self._send_fsm(_FSM_FIXEDSTAND)
        time.sleep(duration)
        return True

    def sit(self, duration: float = 1.0) -> bool:
        """Command lie down (closest to sit for quadruped)."""
        self._send_fsm(_FSM_FIXEDDOWN)
        time.sleep(duration)
        return True

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @staticmethod
    def is_gazebo_running() -> bool:
        """Check if Gazebo is running by detecting the /clock topic."""
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True, text=True, timeout=5,
            )
            return "/clock" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
            logger.debug("is_gazebo_running check failed: %s", exc)
            return False
