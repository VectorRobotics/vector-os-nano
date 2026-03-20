"""Low-level serial communication with Feetech STS3215 servos.

Extracted from vector_ws/src/so101_hardware/so101_hardware/hardware_bridge.py.
Uses scservo_sdk (imported at runtime — may not be installed in test env).
No ROS2 imports — pure Python with stdlib logging.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# SCS Protocol constants (STS3215 control table addresses)
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_CURRENT = 69   # For future grasp detection
ADDR_PRESENT_LOAD = 71      # For future grasp detection
PROTOCOL_END = 0             # STS protocol variant


class SerialBus:
    """Low-level serial communication with Feetech STS3215 servos.

    Wraps scservo_sdk PortHandler + PacketHandler. scservo_sdk is imported
    lazily (at connect() time) so that importing this module in a test
    environment without hardware does not fail.

    All methods are safe to call from a single thread. Thread safety is the
    caller's responsibility.
    """

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 1000000) -> None:
        self.port = port
        self.baudrate = baudrate
        self._port_handler = None
        self._packet_handler = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the serial port and initialise protocol handler.

        Returns True on success, False if the port cannot be opened.
        Does NOT enable torque — callers should call set_torque() per motor
        after performing a read-before-write to prevent startup jump.
        """
        try:
            import scservo_sdk as scs
        except ImportError as exc:
            logger.error("scservo_sdk not installed: %s", exc)
            return False

        port_handler = scs.PortHandler(self.port)
        packet_handler = scs.PacketHandler(PROTOCOL_END)

        if not port_handler.openPort():
            logger.error("Failed to open serial port %s", self.port)
            return False

        if not port_handler.setBaudRate(self.baudrate):
            logger.error("Failed to set baudrate %d on %s", self.baudrate, self.port)
            port_handler.closePort()
            return False

        self._port_handler = port_handler
        self._packet_handler = packet_handler
        self._connected = True
        logger.info("SerialBus connected: port=%s baudrate=%d", self.port, self.baudrate)
        return True

    def disconnect(self) -> None:
        """Close the serial port. Safe to call when already disconnected."""
        if self._port_handler is not None:
            try:
                self._port_handler.closePort()
            except Exception as exc:
                logger.warning("Error closing port: %s", exc)
            self._port_handler = None
            self._packet_handler = None
        self._connected = False
        logger.info("SerialBus disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Read / Write helpers
    # ------------------------------------------------------------------

    def read_position(self, motor_id: int) -> int:
        """Read current encoder position for a motor.

        Returns the raw encoder count on success, or -1 on communication error.
        """
        if not self._connected or self._packet_handler is None:
            logger.warning("read_position called while disconnected (motor_id=%d)", motor_id)
            return -1

        try:
            import scservo_sdk as scs
        except ImportError:
            return -1

        pos, result, _ = self._packet_handler.read2ByteTxRx(
            self._port_handler, motor_id, ADDR_PRESENT_POSITION
        )
        if result != scs.COMM_SUCCESS:
            logger.debug(
                "read_position failed: motor_id=%d result=%d", motor_id, result
            )
            return -1
        return pos

    def write_position(self, motor_id: int, position: int) -> bool:
        """Write a goal encoder position to a motor.

        Returns True if the write succeeded, False on communication error.
        """
        if not self._connected or self._packet_handler is None:
            logger.warning("write_position called while disconnected (motor_id=%d)", motor_id)
            return False

        result, error = self._packet_handler.write2ByteTxRx(
            self._port_handler, motor_id, ADDR_GOAL_POSITION, position
        )

        try:
            import scservo_sdk as scs
            if result != scs.COMM_SUCCESS:
                logger.debug(
                    "write_position failed: motor_id=%d position=%d result=%d",
                    motor_id, position, result
                )
                return False
        except ImportError:
            pass

        return True

    def set_torque(self, motor_id: int, enable: bool) -> bool:
        """Enable or disable torque on a motor.

        Returns True on success, False on error.
        """
        if not self._connected or self._packet_handler is None:
            logger.warning("set_torque called while disconnected (motor_id=%d)", motor_id)
            return False

        value = 1 if enable else 0
        result, error = self._packet_handler.write1ByteTxRx(
            self._port_handler, motor_id, ADDR_TORQUE_ENABLE, value
        )

        try:
            import scservo_sdk as scs
            if result != scs.COMM_SUCCESS:
                logger.warning(
                    "set_torque failed: motor_id=%d enable=%s result=%d",
                    motor_id, enable, result
                )
                return False
        except ImportError:
            pass

        return True

    def read_load(self, motor_id: int) -> int:
        """Read current load value for a motor.

        Used for future grasp detection. Returns -1 on error.
        """
        if not self._connected or self._packet_handler is None:
            return -1

        try:
            import scservo_sdk as scs
        except ImportError:
            return -1

        load, result, _ = self._packet_handler.read2ByteTxRx(
            self._port_handler, motor_id, ADDR_PRESENT_LOAD
        )
        if result != scs.COMM_SUCCESS:
            return -1
        return load
