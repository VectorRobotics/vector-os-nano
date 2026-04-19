"""ROS2 proxies for Piper arm + gripper (main-process side).

The Piper MuJoCo actuators live inside the bridge subprocess (where
MuJoCoGo2 runs). We control them via ROS2 topics:

    publish /piper/joint_cmd      (Float64MultiArray, 6 arm positions)
    publish /piper/gripper_cmd    (Float64, 0.0=closed .. 1.0=open)
    subscribe /piper/joint_state  (JointState, 6 arm + piper_joint7)

FK / IK run LOCALLY in the main process on an isolated ``MjModel`` loaded
from the same MJCF — this keeps the skill's IK calls blocking and
immediate without a round-trip service. The dog base pose needed for FK
is read from an attached Go2ROS2Proxy (xy + yaw, extended to a yaw-only
quaternion — sufficient on flat floors).

Usage::

    base = Go2ROS2Proxy(); base.connect()
    piper = PiperROS2Proxy(base, scene_xml_path); piper.connect()
    gripper = PiperGripperROS2Proxy(base); gripper.connect()
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (duplicate of MuJoCoPiper's — kept in sync)
# ---------------------------------------------------------------------------

_ARM_JOINT_NAMES: list[str] = [
    "piper_joint1", "piper_joint2", "piper_joint3",
    "piper_joint4", "piper_joint5", "piper_joint6",
]
_EE_SITE_NAME: str = "piper_ee_site"
_GRIPPER_JOINT_NAME: str = "piper_joint7"

_IK_MAX_ITER: int = 200
_IK_POS_TOL: float = 2e-3
_IK_ROT_TOL: float = 2e-2
_IK_STEP_SIZE: float = 0.4
_IK_DAMPING: float = 5e-3

_R_TOP_DOWN: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64,
)

_IK_TOP_DOWN_SEEDS: list[list[float] | None] = [
    [0.0, 1.57, -1.57, 0.0, 1.57, 0.0],
    [0.0, 2.48, -1.75, 0.0, 1.00, 3.13],
    [0.0, 1.57, -1.57, 0.0, 1.57, 3.14],
    None,
]

_HOME_JOINTS: list[float] = [0.0] * 6
_MOVE_UPDATE_HZ: float = 50.0

# Gripper ctrl range (see piper.xml: joint7 range [0, 0.035])
_GRIPPER_OPEN_CMD: float = 1.0   # normalized
_GRIPPER_CLOSED_CMD: float = 0.0


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """Yaw-only rotation as a (w, x, y, z) quaternion — MuJoCo convention."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


# ---------------------------------------------------------------------------
# PiperROS2Proxy  (ArmProtocol)
# ---------------------------------------------------------------------------


class PiperROS2Proxy:
    """ArmProtocol backed by ROS2 topics; IK/FK computed locally.

    Args:
        base_proxy: A connected Go2ROS2Proxy (supplies dog world pose for
            FK/IK at the moment of the call).
        scene_xml_path: Absolute path to scene_room_piper.xml (or any MJCF
            that contains the Piper bodies + piper_ee_site).
        node_name: ROS2 node name; defaults to "piper_agent_proxy".
    """

    _NODE_NAME: str = "piper_agent_proxy"

    def __init__(
        self,
        base_proxy: Any,
        scene_xml_path: str,
        node_name: str | None = None,
    ) -> None:
        self._base = base_proxy
        self._scene_xml_path = scene_xml_path
        self._node_name = node_name or self._NODE_NAME

        self._connected: bool = False
        self._node: Any = None
        self._joint_cmd_pub: Any = None

        self._last_joint_state: list[float] = [0.0] * 7  # 6 arm + 1 gripper
        self._last_joint_state_ts: float = 0.0
        self._state_lock = threading.Lock()

        # Isolated IK model + cached indices (loaded on connect)
        self._ik_model: Any = None
        self._ik_data: Any = None
        self._ik_arm_qpos_adr: list[int] = []
        self._ik_arm_dof_adr: list[int] = []
        self._ik_arm_joint_ids: list[int] = []
        self._ik_ee_site_id: int = -1

        self._spin_thread: Any = None

    # ------------------------------------------------------------------
    # ArmProtocol surface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "piper_ros2_proxy"

    @property
    def joint_names(self) -> list[str]:
        return list(_ARM_JOINT_NAMES)

    @property
    def dof(self) -> int:
        return 6

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Init ROS2 node, load isolated IK model, wait for first joint_state."""
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(self._node_name)
        self._joint_cmd_pub = self._node.create_publisher(
            Float64MultiArray, "/piper/joint_cmd", 10
        )
        self._node.create_subscription(
            JointState, "/piper/joint_state", self._joint_state_cb, 10
        )

        # Load isolated IK model
        import mujoco
        if not os.path.exists(self._scene_xml_path):
            raise FileNotFoundError(
                f"PiperROS2Proxy: scene xml not found: {self._scene_xml_path}"
            )
        self._ik_model = mujoco.MjModel.from_xml_path(self._scene_xml_path)
        self._ik_data = mujoco.MjData(self._ik_model)
        m = self._ik_model

        self._ik_arm_joint_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in _ARM_JOINT_NAMES
        ]
        if any(j < 0 for j in self._ik_arm_joint_ids):
            raise RuntimeError(
                "PiperROS2Proxy: arm joints missing from MJCF — "
                "ensure VECTOR_SIM_WITH_ARM=1 scene was generated."
            )
        self._ik_arm_qpos_adr = [int(m.jnt_qposadr[j]) for j in self._ik_arm_joint_ids]
        self._ik_arm_dof_adr = [int(m.jnt_dofadr[j]) for j in self._ik_arm_joint_ids]
        self._ik_ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, _EE_SITE_NAME)
        if self._ik_ee_site_id < 0:
            raise RuntimeError(f"PiperROS2Proxy: {_EE_SITE_NAME!r} missing from MJCF")

        self._spin_thread = threading.Thread(
            target=lambda: rclpy.spin(self._node), daemon=True,
        )
        self._spin_thread.start()

        # Wait up to 3 s for first joint_state (bridge advertises at 20 Hz)
        for _ in range(60):
            if self._last_joint_state_ts > 0:
                break
            time.sleep(0.05)

        self._connected = True
        logger.info(
            "PiperROS2Proxy connected (%s). First state: %s after %.2fs",
            self._node_name,
            "received" if self._last_joint_state_ts > 0 else "NOT received (bridge missing?)",
            time.monotonic() - (self._last_joint_state_ts or time.monotonic()),
        )

    def disconnect(self) -> None:
        """Destroy the ROS2 node. Does not shut down rclpy (shared)."""
        self._connected = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("PiperROS2Proxy: not connected")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _joint_state_cb(self, msg: Any) -> None:
        """Cache the latest 6 arm + 1 gripper positions."""
        with self._state_lock:
            self._last_joint_state = [float(p) for p in msg.position][:7]
            self._last_joint_state_ts = time.monotonic()

    def get_joint_positions(self) -> list[float]:
        """Return the 6 arm joint positions (radians) from /piper/joint_state."""
        self._require_connected()
        with self._state_lock:
            return list(self._last_joint_state[:6])

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Linearly interpolate the 6 arm targets and publish at 50 Hz.

        Blocks for ``duration`` seconds. Returns True when the final target
        has been published. The bridge writes whatever the latest message
        was to the piper ctrl array immediately on receipt.
        """
        self._require_connected()
        if len(positions) != self.dof:
            raise ValueError(
                f"PiperROS2Proxy.move_joints: expected 6 positions, got {len(positions)}"
            )
        start = self.get_joint_positions()
        steps = max(1, int(duration * _MOVE_UPDATE_HZ))
        dt = duration / steps

        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        for i in range(1, steps + 1):
            t = i / steps
            msg.data = [float(start[j] + t * (positions[j] - start[j])) for j in range(6)]
            self._joint_cmd_pub.publish(msg)
            time.sleep(dt)
        return True

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        q = self.ik_top_down(target_xyz)
        if q is None:
            return False
        return self.move_joints(q, duration=duration)

    def stop(self) -> None:
        """Hold current joint positions by publishing them once as the target."""
        self._require_connected()
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = self.get_joint_positions()
        self._joint_cmd_pub.publish(msg)

    def home(self, duration: float = 3.0) -> bool:
        return self.move_joints(list(_HOME_JOINTS), duration=duration)

    # ------------------------------------------------------------------
    # FK / IK — isolated model, dog base pose pulled from base_proxy
    # ------------------------------------------------------------------

    def _sync_ik_base(self, arm_joints: list[float] | None) -> None:
        """Write dog world pose (from base_proxy) into the IK data."""
        x, y, z = self._base.get_position()
        yaw = float(self._base.get_heading())
        qw, qx, qy, qz = _yaw_to_quat_wxyz(yaw)

        self._ik_data.qpos[0] = float(x)
        self._ik_data.qpos[1] = float(y)
        self._ik_data.qpos[2] = float(z)
        self._ik_data.qpos[3] = qw
        self._ik_data.qpos[4] = qx
        self._ik_data.qpos[5] = qy
        self._ik_data.qpos[6] = qz
        # Leg positions are irrelevant for Piper FK (Piper is mounted to
        # base_link, not to any leg). Leave whatever default was loaded.

        if arm_joints is not None:
            for adr, q in zip(self._ik_arm_qpos_adr, arm_joints):
                self._ik_data.qpos[adr] = float(q)

        self._ik_data.qvel[:] = 0.0
        self._ik_data.qacc[:] = 0.0

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """FK via the isolated model + current dog pose."""
        self._require_connected()
        if len(joint_positions) != self.dof:
            raise ValueError(
                f"PiperROS2Proxy.fk: expected 6 joints, got {len(joint_positions)}"
            )
        import mujoco
        self._sync_ik_base(joint_positions)
        mujoco.mj_forward(self._ik_model, self._ik_data)
        pos = [float(v) for v in self._ik_data.site_xpos[self._ik_ee_site_id]]
        mat = self._ik_data.site_xmat[self._ik_ee_site_id].reshape(3, 3)
        rot = [[float(v) for v in row] for row in mat]
        return pos, rot

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Position-only (3-DoF) IK."""
        self._require_connected()
        self._sync_ik_base(arm_joints=None)
        return self._ik_iterate(target_xyz, target_rot=None,
                                seed_joints=current_joints)

    def ik_top_down(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """6-DoF IK with gripper z-axis pointing world −Z."""
        self._require_connected()
        seeds: list[list[float] | None] = []
        if current_joints is not None:
            seeds.append(list(current_joints))
        seeds.extend(_IK_TOP_DOWN_SEEDS)

        self._sync_ik_base(arm_joints=None)
        for seed in seeds:
            sol = self._ik_iterate(target_xyz, target_rot=_R_TOP_DOWN,
                                   seed_joints=seed)
            if sol is not None:
                return sol
        return None

    def _ik_iterate(
        self,
        target_xyz: tuple[float, float, float],
        target_rot: np.ndarray | None,
        seed_joints: list[float] | None,
    ) -> list[float] | None:
        import mujoco
        m = self._ik_model
        d = self._ik_data

        if seed_joints is not None:
            for adr, q in zip(self._ik_arm_qpos_adr, seed_joints):
                d.qpos[adr] = float(q)

        target_pos = np.array(target_xyz, dtype=np.float64)
        use_rot = target_rot is not None

        jacp = np.zeros((3, m.nv), dtype=np.float64)
        jacr = np.zeros((3, m.nv), dtype=np.float64) if use_rot else None

        converged = False
        for _ in range(_IK_MAX_ITER):
            mujoco.mj_forward(m, d)
            ee_pos = d.site_xpos[self._ik_ee_site_id].copy()
            pos_err = target_pos - ee_pos
            if use_rot:
                ee_mat = d.site_xmat[self._ik_ee_site_id].reshape(3, 3).copy()
                dR = target_rot @ ee_mat.T
                rot_err = 0.5 * np.array(
                    [dR[2, 1] - dR[1, 2],
                     dR[0, 2] - dR[2, 0],
                     dR[1, 0] - dR[0, 1]],
                    dtype=np.float64,
                )
                err = np.concatenate([pos_err, rot_err])
                if (np.linalg.norm(pos_err) < _IK_POS_TOL
                        and np.linalg.norm(rot_err) < _IK_ROT_TOL):
                    converged = True
                    break
            else:
                err = pos_err
                if np.linalg.norm(pos_err) < _IK_POS_TOL:
                    converged = True
                    break

            mujoco.mj_jacSite(m, d, jacp, jacr, self._ik_ee_site_id)
            if use_rot:
                J = np.vstack([
                    jacp[:, self._ik_arm_dof_adr],
                    jacr[:, self._ik_arm_dof_adr],
                ])
                JJt = J @ J.T + _IK_DAMPING * np.eye(6)
            else:
                J = jacp[:, self._ik_arm_dof_adr]
                JJt = J @ J.T + _IK_DAMPING * np.eye(3)

            try:
                dq = J.T @ np.linalg.solve(JJt, err)
            except np.linalg.LinAlgError:
                return None

            for i, (adr, jid) in enumerate(
                zip(self._ik_arm_qpos_adr, self._ik_arm_joint_ids)
            ):
                new_q = float(d.qpos[adr] + _IK_STEP_SIZE * dq[i])
                lo = float(m.jnt_range[jid, 0])
                hi = float(m.jnt_range[jid, 1])
                d.qpos[adr] = float(np.clip(new_q, lo, hi))

        if not converged:
            return None
        return [float(d.qpos[adr]) for adr in self._ik_arm_qpos_adr]


# ---------------------------------------------------------------------------
# PiperGripperROS2Proxy  (GripperProtocol)
# ---------------------------------------------------------------------------


class PiperGripperROS2Proxy:
    """Gripper proxy via /piper/gripper_cmd + /piper/joint_state.

    We reuse the joint_state stream from PiperROS2Proxy for gripper
    position reads — avoiding a second topic. To keep this class usable
    standalone, we also accept an optional ``arm_proxy`` that exposes
    the cached state; otherwise we create our own subscription.
    """

    _NODE_NAME: str = "piper_gripper_proxy"

    def __init__(
        self,
        arm_proxy: PiperROS2Proxy | None = None,
        node_name: str | None = None,
    ) -> None:
        self._arm_proxy = arm_proxy
        self._node_name = node_name or self._NODE_NAME
        self._connected: bool = False
        self._node: Any = None
        self._cmd_pub: Any = None
        self._last_gripper_pos: float = 0.0
        self._last_gripper_cmd: float = _GRIPPER_OPEN_CMD
        self._state_lock = threading.Lock()

    def connect(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64

        if not rclpy.ok():
            rclpy.init()
        self._node = Node(self._node_name)
        self._cmd_pub = self._node.create_publisher(Float64, "/piper/gripper_cmd", 10)

        # Independent joint_state subscriber (simpler than sharing with arm_proxy)
        self._node.create_subscription(
            JointState, "/piper/joint_state", self._joint_state_cb, 10
        )
        self._spin_thread = threading.Thread(
            target=lambda: rclpy.spin(self._node), daemon=True,
        )
        self._spin_thread.start()

        self._connected = True
        logger.info("PiperGripperROS2Proxy connected (%s)", self._node_name)

    def disconnect(self) -> None:
        self._connected = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

    def _joint_state_cb(self, msg: Any) -> None:
        # piper_joint7 is element index 6 (0-5 = arm, 6 = gripper)
        if len(msg.position) >= 7:
            with self._state_lock:
                self._last_gripper_pos = float(msg.position[6])

    # --- GripperProtocol ---

    def open(self) -> bool:
        if not self._connected:
            return False
        from std_msgs.msg import Float64
        self._last_gripper_cmd = _GRIPPER_OPEN_CMD
        msg = Float64(); msg.data = _GRIPPER_OPEN_CMD
        self._cmd_pub.publish(msg)
        return True

    def close(self) -> bool:
        if not self._connected:
            return False
        from std_msgs.msg import Float64
        self._last_gripper_cmd = _GRIPPER_CLOSED_CMD
        msg = Float64(); msg.data = _GRIPPER_CLOSED_CMD
        self._cmd_pub.publish(msg)
        return True

    def is_holding(self) -> bool:
        """Commanded closed AND joint7 > 5 mm (jaws held open by contact)."""
        if not self._connected:
            return False
        with self._state_lock:
            pos = self._last_gripper_pos
        return self._last_gripper_cmd <= 0.01 and pos > 0.005

    def get_position(self) -> float:
        """Normalized: 0.0=closed, 1.0=open."""
        with self._state_lock:
            p = self._last_gripper_pos
        return max(0.0, min(1.0, p / 0.035))

    def get_force(self) -> float | None:
        return None  # no sensor in sim
