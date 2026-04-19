"""MuJoCo Piper parallel-jaw gripper.

Implements GripperProtocol. Piper's gripper is a single-actuator parallel
jaw: one position actuator on joint7 (piper_gripper), and joint8 follows via
an equality constraint defined in the Menagerie MJCF. Range is 0 (fully
closed) to 0.035m (fully open, ~35 mm jaw separation).

Shares MjModel/MjData with the parent MuJoCoGo2 — this class just reads and
writes indices into the shared data buffers.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACTUATOR_NAME: str = "piper_gripper"
_JOINT_NAME: str = "piper_joint7"

# joint7 range per piper.xml is [0, 0.035]; ctrl uses the same units (meters
# of jaw half-opening). Fully closed commands exactly zero; fully open is
# the upper joint limit.
_CLOSED_POS: float = 0.0
_OPEN_POS: float = 0.035

# Heuristic: jaws considered "holding something" when commanded closed but
# the physics-driven joint position is kept open by the gripped object.
_HOLDING_THRESHOLD: float = 0.005  # 5 mm jaw separation while cmd=closed

# Lazy MuJoCo import (same pattern as the other sim modules)
_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


# ---------------------------------------------------------------------------
# MuJoCoPiperGripper
# ---------------------------------------------------------------------------


class MuJoCoPiperGripper:
    """Piper parallel-jaw gripper in MuJoCo.

    Args:
        go2: A connected MuJoCoGo2 instance. Must have been launched with
            with_arm=True so the piper_gripper actuator exists.
    """

    def __init__(self, go2: "MuJoCoGo2") -> None:
        self._go2 = go2
        self._connected: bool = False
        self._actuator_id: int = -1
        self._joint_qpos_adr: int = -1

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if not getattr(self._go2, "_connected", False):
            raise RuntimeError(
                "MuJoCoPiperGripper: parent MuJoCoGo2 must be connected first"
            )
        mj = _get_mujoco()
        model = self._go2._mj.model

        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, _ACTUATOR_NAME)
        if aid < 0:
            raise RuntimeError(
                f"MuJoCoPiperGripper: actuator {_ACTUATOR_NAME!r} not in loaded "
                "MJCF. Ensure sim was launched with with_arm=True."
            )
        self._actuator_id = aid

        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, _JOINT_NAME)
        if jid < 0:
            raise RuntimeError(
                f"MuJoCoPiperGripper: joint {_JOINT_NAME!r} not in loaded MJCF."
            )
        self._joint_qpos_adr = int(model.jnt_qposadr[jid])

        self._connected = True
        logger.info(
            "MuJoCoPiperGripper connected (actuator=%s, closed=%.3f, open=%.3f)",
            _ACTUATOR_NAME, _CLOSED_POS, _OPEN_POS,
        )

    def disconnect(self) -> None:
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "MuJoCoPiperGripper: not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # GripperProtocol
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the jaws fully (35 mm separation)."""
        self._require_connection()
        self._go2._mj.data.ctrl[self._actuator_id] = _OPEN_POS
        return True

    def close(self) -> bool:
        """Command jaws closed. The position controller drives toward 0;
        if an object blocks closure it stops against the object.
        """
        self._require_connection()
        self._go2._mj.data.ctrl[self._actuator_id] = _CLOSED_POS
        return True

    def is_holding(self) -> bool:
        """Position heuristic: commanded closed AND jaws held open by contact.

        Piper has no force sensor; we infer grasp from steady-state position
        error. Safe because the physics thread is running continuously and the
        joint position settles within ~0.1 s of the ctrl change.
        """
        self._require_connection()
        data = self._go2._mj.data
        cmd = float(data.ctrl[self._actuator_id])
        pos = float(data.qpos[self._joint_qpos_adr])
        return cmd <= _HOLDING_THRESHOLD and pos > _HOLDING_THRESHOLD

    def get_position(self) -> float:
        """Normalized jaw position: 0.0 closed, 1.0 fully open."""
        self._require_connection()
        pos = float(self._go2._mj.data.qpos[self._joint_qpos_adr])
        return max(0.0, min(1.0, pos / _OPEN_POS))

    def get_force(self) -> float | None:
        """Piper has no force sensor in the Menagerie MJCF — return None."""
        return None
