"""Isaac Sim Arm Proxy — controls a 6-DOF arm in Isaac Sim Docker via ROS2 topics.

Implements ArmProtocol for an arm running inside an Isaac Sim Docker container.
Communicates exclusively via ROS2 topics — no direct Isaac Sim SDK dependency.

Topic interface:
    Subscribes: /arm/joint_states      (sensor_msgs/JointState)   → cache positions
                /arm/end_effector_pose (geometry_msgs/PoseStamped) → cache EE pose
    Publishes:  /arm/joint_commands    (std_msgs/Float64MultiArray) → target positions

DH parameters are placeholders — exact values to be provided when the Isaac Sim
URDF is finalised. IK is stubbed (returns None) until the Isaac Sim Lula IK
service is available.
"""
from __future__ import annotations

import logging
import math
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

_DOF: int = 6

# Convergence threshold in radians for move_joints polling.
_JOINT_CONVERGENCE_TOL: float = 0.01  # ~0.57 deg

# Maximum time to wait for joint convergence per move_joints call.
_MOVE_TIMEOUT_DEFAULT: float = 10.0

# Poll interval while waiting for joint convergence (seconds).
_POLL_INTERVAL: float = 0.05

# Isaac Sim Docker container name (must match scripts/launch_isaac.sh).
_ISAAC_CONTAINER_NAME: str = "vector-isaac-sim"

# ---------------------------------------------------------------------------
# Placeholder DH parameters (a, alpha, d, theta_offset) for 6-DOF UR-like arm.
# These are approximate — update when the Isaac Sim URDF is finalised.
# Format: (a_m, alpha_rad, d_m, theta_offset_rad)
# ---------------------------------------------------------------------------
_DH_PARAMS: list[tuple[float, float, float, float]] = [
    (0.000,  math.pi / 2,  0.1625, 0.0),   # Joint 1: shoulder_pan
    (-0.425, 0.000,        0.000,  0.0),   # Joint 2: shoulder_lift
    (-0.392, 0.000,        0.000,  0.0),   # Joint 3: elbow
    (0.000,  math.pi / 2,  0.1333, 0.0),   # Joint 4: wrist_1
    (0.000, -math.pi / 2,  0.0997, 0.0),   # Joint 5: wrist_2
    (0.000,  0.000,        0.0996, 0.0),   # Joint 6: wrist_3
]


# ---------------------------------------------------------------------------
# DH helpers
# ---------------------------------------------------------------------------

def _dh_transform(a: float, alpha: float, d: float, theta: float) -> list[list[float]]:
    """Compute a single DH homogeneous transform T = Rot_z(theta) * Trans_z(d)
    * Trans_x(a) * Rot_x(alpha) as a 4x4 row-major list-of-lists."""
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return [
        [ct,       -st * ca,   st * sa,   a * ct],
        [st,        ct * ca,  -ct * sa,   a * st],
        [0.0,       sa,         ca,        d     ],
        [0.0,       0.0,        0.0,       1.0   ],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """4x4 matrix multiplication (no numpy dependency)."""
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _fk_dh(joint_positions: list[float]) -> tuple[list[float], list[list[float]]]:
    """Forward kinematics via DH convention.

    Returns:
        (position_xyz, rotation_3x3) — end-effector in base frame.
    """
    t = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]  # identity
    for idx, theta in enumerate(joint_positions):
        a, alpha, d, offset = _DH_PARAMS[idx]
        ti = _dh_transform(a, alpha, d, theta + offset)
        t = _mat_mul(t, ti)

    position_xyz = [t[0][3], t[1][3], t[2][3]]
    rotation_3x3 = [
        [t[0][0], t[0][1], t[0][2]],
        [t[1][0], t[1][1], t[1][2]],
        [t[2][0], t[2][1], t[2][2]],
    ]
    return position_xyz, rotation_3x3


# ---------------------------------------------------------------------------
# IsaacSimArmProxy
# ---------------------------------------------------------------------------


class IsaacSimArmProxy:
    """ArmProtocol implementation for a 6-DOF arm in Isaac Sim Docker.

    Communicates via ROS2 topics. rclpy is imported lazily inside connect()
    so this module is safe to import on systems without ROS2 installed.

    Thread safety: ROS2 callbacks run in a daemon background thread. All
    cached state (_joint_positions, _ee_pose) is written from that thread
    and read from the caller's thread. Python's GIL provides sufficient
    protection for the simple assignments used here.
    """

    def __init__(self) -> None:
        self._node: Any = None
        self._cmd_pub: Any = None
        self._joint_positions: list[float] = [0.0] * _DOF
        self._ee_pose: Any = None          # geometry_msgs/PoseStamped (cached)
        self._connected: bool = False
        self._spin_thread: Any = None

    # ------------------------------------------------------------------
    # ArmProtocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "isaac_sim_arm"

    @property
    def joint_names(self) -> list[str]:
        return list(_JOINT_NAMES)

    @property
    def dof(self) -> int:
        return _DOF

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise rclpy node, publisher, and subscribers.

        Verifies the Isaac Sim Docker container is running first.

        Raises:
            ConnectionError: If the Isaac Sim container is not running or
                             rclpy cannot be initialised.
        """
        if not self.is_isaac_sim_running():
            raise ConnectionError(
                f"Isaac Sim container '{_ISAAC_CONTAINER_NAME}' not running. "
                "Start with: ./scripts/launch_isaac.sh"
            )

        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            import threading

            if not rclpy.ok():
                rclpy.init()

            self._node = Node("isaac_sim_arm_proxy")

            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                depth=10,
            )

            from std_msgs.msg import Float64MultiArray
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import PoseStamped

            self._cmd_pub = self._node.create_publisher(
                Float64MultiArray, "/arm/joint_commands", reliable_qos
            )
            self._node.create_subscription(
                JointState, "/arm/joint_states", self._joint_state_cb, reliable_qos
            )
            self._node.create_subscription(
                PoseStamped, "/arm/end_effector_pose", self._ee_pose_cb, reliable_qos
            )

            # Spin in a background daemon thread — caller is not blocked.
            self._spin_thread = threading.Thread(
                target=lambda: rclpy.spin(self._node), daemon=True
            )
            self._spin_thread.start()
            self._connected = True

            logger.info("IsaacSimArmProxy connected to Isaac Sim arm via ROS2")

        except ImportError as exc:
            raise ConnectionError(
                "rclpy not available — ROS2 must be sourced before using IsaacSimArmProxy"
            ) from exc
        except Exception as exc:
            logger.error("Failed to connect IsaacSimArmProxy: %s", exc)
            self._connected = False
            raise ConnectionError(f"IsaacSimArmProxy connect failed: {exc}") from exc

    def disconnect(self) -> None:
        """Destroy the rclpy node and mark proxy as disconnected.

        Safe to call even if not connected (idempotent).
        """
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._connected = False
        logger.info("IsaacSimArmProxy disconnected")

    # ------------------------------------------------------------------
    # ROS2 callbacks (run in background spin thread)
    # ------------------------------------------------------------------

    def _joint_state_cb(self, msg: Any) -> None:
        """Cache joint positions from /arm/joint_states.

        Reorders incoming positions to match _JOINT_NAMES order.  If a joint
        name is missing in the message, its cached value is retained.
        """
        if not msg.name:
            return
        name_to_pos: dict[str, float] = dict(zip(msg.name, msg.position))
        updated = list(self._joint_positions)
        for i, jname in enumerate(_JOINT_NAMES):
            if jname in name_to_pos:
                updated[i] = float(name_to_pos[jname])
        self._joint_positions = updated

    def _ee_pose_cb(self, msg: Any) -> None:
        """Cache end-effector pose from /arm/end_effector_pose."""
        self._ee_pose = msg

    # ------------------------------------------------------------------
    # ArmProtocol — state accessors
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> list[float]:
        """Return cached joint positions (radians), length == dof.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected:
            raise RuntimeError("IsaacSimArmProxy is not connected")
        return list(self._joint_positions)

    # ------------------------------------------------------------------
    # ArmProtocol — motion interface
    # ------------------------------------------------------------------

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Publish target joint positions and poll until convergence or timeout.

        Args:
            positions: Target joint positions in radians, length == dof.
            duration: Motion duration hint (seconds). Used as timeout.

        Returns:
            True when all joints converge within _JOINT_CONVERGENCE_TOL,
            False on timeout.

        Raises:
            ValueError: If len(positions) != dof.
            RuntimeError: If not connected.
        """
        if not self._connected:
            raise RuntimeError("IsaacSimArmProxy is not connected")
        if len(positions) != _DOF:
            raise ValueError(
                f"Expected {_DOF} joint positions, got {len(positions)}"
            )

        self._publish_command(positions)

        timeout = max(duration, _MOVE_TIMEOUT_DEFAULT)
        deadline = time.time() + timeout

        while time.time() < deadline:
            current = self._joint_positions
            if all(
                abs(current[i] - positions[i]) < _JOINT_CONVERGENCE_TOL
                for i in range(_DOF)
            ):
                logger.debug("move_joints: converged in %.2fs", timeout - (deadline - time.time()))
                return True
            time.sleep(_POLL_INTERVAL)

        logger.warning(
            "move_joints: timeout after %.1fs — positions may not have converged", timeout
        )
        return False

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        """Move end-effector to Cartesian target via IK.

        Returns False because IK is not yet implemented (Lula IK service pending).
        """
        solution = self.ik(target_xyz)
        if solution is None:
            logger.warning(
                "move_cartesian: IK returned no solution for target %s", target_xyz
            )
            return False
        return self.move_joints(solution, duration=duration)

    def stop(self) -> None:
        """Hold position by re-publishing current joint positions.

        Publishes the cached joint positions as the new target so the
        Isaac Sim controller maintains the current pose. Does not raise
        even under error conditions.
        """
        try:
            if self._connected and self._node is not None:
                self._publish_command(self._joint_positions)
                logger.debug("stop: holding at current joint positions")
        except Exception as exc:
            logger.error("stop: failed to publish hold command: %s", exc)

    # ------------------------------------------------------------------
    # ArmProtocol — kinematics
    # ------------------------------------------------------------------

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics via DH parameters.

        DH parameters are approximate placeholders. Update _DH_PARAMS when
        the exact Isaac Sim URDF is finalised.

        Args:
            joint_positions: Joint positions in radians, length == dof.

        Returns:
            (position_xyz, rotation_3x3) in the robot base frame.
        """
        if len(joint_positions) != _DOF:
            raise ValueError(
                f"Expected {_DOF} joint positions, got {len(joint_positions)}"
            )
        return _fk_dh(joint_positions)

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Inverse kinematics — not yet implemented.

        Will delegate to Isaac Sim Lula IK service in a future sprint.

        Returns:
            None always (IK not yet available).
        """
        logger.debug(
            "ik: Lula IK service not yet implemented — returning None for target %s",
            target_xyz,
        )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_command(self, positions: list[float]) -> None:
        """Publish joint target positions to /arm/joint_commands."""
        if self._node is None or self._cmd_pub is None:
            return
        try:
            from std_msgs.msg import Float64MultiArray

            msg = Float64MultiArray()
            msg.data = [float(p) for p in positions]
            self._cmd_pub.publish(msg)
        except Exception as exc:
            logger.warning("_publish_command: failed to publish: %s", exc)

    # ------------------------------------------------------------------
    # Docker health check
    # ------------------------------------------------------------------

    @staticmethod
    def is_isaac_sim_running() -> bool:
        """Check if the Isaac Sim Docker container is running."""
        try:
            result = subprocess.run(
                [
                    "docker", "ps",
                    "--filter", f"name={_ISAAC_CONTAINER_NAME}",
                    "--filter", "status=running",
                    "--format", "{{.Names}}",
                ],
                capture_output=True, text=True, timeout=5,
            )
            return _ISAAC_CONTAINER_NAME in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
            logger.debug("is_isaac_sim_running check failed: %s", exc)
            return False
