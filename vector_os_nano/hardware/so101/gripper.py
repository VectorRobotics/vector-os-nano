# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SO-101 gripper driver — pure Python, no ROS2.

Implements the GripperProtocol interface using SerialBus for low-level
SCS communication. The gripper is motor 6 (id=6 in JOINT_CONFIG).

Encoder constants ported from vector_ws/src/so101_hardware/hardware_bridge.py:
    GRIPPER_OPEN_ENC   = 2500
    GRIPPER_CLOSED_ENC = 1332

The gripper requires a SerialBus that is already connected (typically
the same bus used by SO101Arm).
"""

import logging
import time

from vector_os_nano.hardware.so101.joint_config import JOINT_CONFIG, enc_to_rad
from vector_os_nano.hardware.so101.serial_bus import SerialBus

logger = logging.getLogger(__name__)

# Encoder positions for fully open and fully closed gripper
GRIPPER_OPEN_ENC: int = 2500
GRIPPER_CLOSED_ENC: int = 1332

# Encoder threshold for is_holding() — if gripper stopped between these
# limits it is holding an object.
_HOLDING_THRESHOLD = 50   # counts away from open/closed endpoints


class SO101Gripper:
    """SO-101 gripper driver.

    Uses the SerialBus shared with SO101Arm (motor 6).

    Usage:
        bus = SerialBus(port="/dev/ttyACM0")
        bus.connect()
        gripper = SO101Gripper(bus)
        gripper.open()
        gripper.close()
        holding = gripper.is_holding()
        bus.disconnect()
    """

    def __init__(self, serial_bus: SerialBus) -> None:
        self._bus = serial_bus
        self._motor_id: int = JOINT_CONFIG["gripper"]["id"]  # 6

    # ------------------------------------------------------------------
    # GripperProtocol implementation
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the gripper fully.

        Returns True if the write command was acknowledged.
        """
        ok = self._bus.write_position(self._motor_id, GRIPPER_OPEN_ENC)
        if ok:
            logger.debug("SO101Gripper: open command sent (enc=%d)", GRIPPER_OPEN_ENC)
        else:
            logger.warning("SO101Gripper: open command failed")
        return ok

    def close(self) -> bool:
        """Close the gripper.

        Sends the close command 3 times with 0.2 s intervals (ported from
        hardware_bridge GripperCommand execute: write + 0.5s + write).
        This improves reliability on the STS3215 servo bus.

        Returns True if at least the first write succeeded.
        """
        ok = self._bus.write_position(self._motor_id, GRIPPER_CLOSED_ENC)
        if ok:
            logger.debug("SO101Gripper: close command sent (enc=%d)", GRIPPER_CLOSED_ENC)
        else:
            logger.warning("SO101Gripper: close command failed (attempt 1)")
            return False

        # Send 2 more times with 0.2 s interval for reliability
        for attempt in range(2, 4):
            time.sleep(0.2)
            retry_ok = self._bus.write_position(self._motor_id, GRIPPER_CLOSED_ENC)
            if not retry_ok:
                logger.debug(
                    "SO101Gripper: close retry %d failed (non-fatal)", attempt
                )
        return True

    def is_holding(self, name: str | None = None) -> bool:
        """Check whether the gripper has grasped an object.

        Reads the current encoder position. If it stopped between the
        fully-closed endpoint and the fully-open endpoint (with a threshold
        margin), it is likely holding an object.

        Returns:
            True if an object appears to be gripped.
        """
        if name is not None:
            return False  # Encoders cannot identify an object.
        enc = self._bus.read_position(self._motor_id)
        if enc < 0:
            logger.warning("SO101Gripper: cannot read encoder for is_holding check")
            return False

        # Gripper is holding if encoder stopped meaningfully above CLOSED
        # (i.e., an object is preventing full closure) and below the OPEN
        # position minus the threshold (i.e., not wide open).
        above_closed = enc > GRIPPER_CLOSED_ENC + _HOLDING_THRESHOLD
        below_open = enc < GRIPPER_OPEN_ENC - _HOLDING_THRESHOLD
        holding = above_closed and below_open
        logger.debug(
            "SO101Gripper: is_holding=%s enc=%d (closed=%d open=%d)",
            holding, enc, GRIPPER_CLOSED_ENC, GRIPPER_OPEN_ENC
        )
        return holding

    def get_position(self) -> float:
        """Return normalized gripper position.

        Returns:
            0.0 = fully closed (GRIPPER_CLOSED_ENC)
            1.0 = fully open  (GRIPPER_OPEN_ENC)
            Values are clamped to [0.0, 1.0].
        """
        enc = self._bus.read_position(self._motor_id)
        if enc < 0:
            raise RuntimeError('SO101Gripper: cannot read encoder position')

        enc_range = GRIPPER_OPEN_ENC - GRIPPER_CLOSED_ENC
        normalized = (enc - GRIPPER_CLOSED_ENC) / enc_range
        return float(max(0.0, min(1.0, normalized)))

    def get_force(self) -> None:
        """Grip force in Newtons.

        Returns None — force sensing not available on STS3215 in v0.1.
        Future: use ADDR_PRESENT_CURRENT or ADDR_PRESENT_LOAD.
        """
        return None
