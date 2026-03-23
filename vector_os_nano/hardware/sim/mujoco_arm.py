"""MuJoCo-based simulated SO-101 arm.

Implements ArmProtocol -- drop-in replacement for SO101Arm.
MuJoCo is imported lazily so the module is safe to import on systems
without mujoco installed (import error deferred until connect()).

Usage:
    arm = MuJoCoArm(gui=True)    # with viewer
    arm.connect()
    arm.move_joints([0.0, -1.2, 0.5, 0.8, 0.3])
    arm.disconnect()
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MuJoCo import
# ---------------------------------------------------------------------------

_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

_ARM_ACTUATOR_NAMES = [
    "act_shoulder_pan",
    "act_shoulder_lift",
    "act_elbow_flex",
    "act_wrist_flex",
    "act_wrist_roll",
]

_DEFAULT_SCENE_XML = Path(__file__).parent / "so101_mujoco.xml"

# EE site name in the MJCF (matches gripper_frame_joint from URDF)
_EE_SITE_NAME = "ee_site"

# IK defaults
_IK_MAX_ITER = 100
_IK_TOL = 1e-3  # 1 mm
_IK_STEP_SIZE = 0.5
_IK_DAMPING = 1e-4


# ---------------------------------------------------------------------------
# MuJoCoArm
# ---------------------------------------------------------------------------


class MuJoCoArm:
    """SO-101 arm in MuJoCo simulation.

    ArmProtocol-compatible drop-in replacement for SO101Arm. Runs a full
    rigid-body dynamics simulation with MuJoCo.

    Args:
        gui: Open an interactive viewer on connect().
        scene_xml: Path to the MJCF scene XML. Defaults to the bundled
            so101_mujoco.xml with table and graspable objects.
    """

    def __init__(
        self,
        gui: bool = False,
        scene_xml: str | None = None,
    ) -> None:
        self._gui: bool = gui
        self._scene_xml: str = (
            scene_xml if scene_xml is not None else str(_DEFAULT_SCENE_XML)
        )
        self._model: Any = None
        self._data: Any = None
        self._viewer: Any = None
        self._connected: bool = False

        # Cached joint / actuator / site IDs (set on connect)
        self._arm_joint_ids: list[int] = []
        self._arm_actuator_ids: list[int] = []
        self._ee_site_id: int = -1

    # ------------------------------------------------------------------
    # ArmProtocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mujoco_so101"

    @property
    def joint_names(self) -> list[str]:
        return list(_ARM_JOINT_NAMES)

    @property
    def dof(self) -> int:
        return 5

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Load MJCF, create MjData, optionally launch viewer."""
        mj = _get_mujoco()

        self._model = mj.MjModel.from_xml_path(self._scene_xml)
        self._data = mj.MjData(self._model)

        # Cache IDs
        self._arm_joint_ids = [
            mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_JOINT, n)
            for n in _ARM_JOINT_NAMES
        ]
        self._arm_actuator_ids = [
            mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _ARM_ACTUATOR_NAMES
        ]
        self._ee_site_id = mj.mj_name2id(
            self._model, mj.mjtObj.mjOBJ_SITE, _EE_SITE_NAME
        )

        # Initial forward pass
        mj.mj_forward(self._model, self._data)

        # GUI viewer (passive — does not block)
        if self._gui:
            try:
                import mujoco.viewer  # noqa: PLC0415

                self._viewer = mujoco.viewer.launch_passive(
                    self._model, self._data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
                # Resize window via GLFW (MuJoCo viewer doesn't accept size params)
                try:
                    import glfw
                    # Find the MuJoCo window and resize it
                    for _ in range(10):
                        import time as _t; _t.sleep(0.05)
                        win = glfw.get_current_context()
                        if win:
                            glfw.set_window_size(win, 640, 480)
                            glfw.set_window_pos(win, 50, 50)
                            break
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("MuJoCo viewer failed to launch: %s", exc)
                self._viewer = None

        self._connected = True
        logger.info(
            "MuJoCoArm connected (gui=%s), joints=%s",
            self._gui,
            _ARM_JOINT_NAMES,
        )

    def disconnect(self) -> None:
        """Shut down viewer and release model. Idempotent."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None
        self._model = None
        self._data = None
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "MuJoCoArm: not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> list[float]:
        """Read current joint positions in radians."""
        self._require_connection()
        return [
            float(self._data.joint(_ARM_JOINT_NAMES[i]).qpos[0])
            for i in range(self.dof)
        ]

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------

    def move_joints(
        self,
        positions: list[float],
        duration: float = 3.0,
    ) -> bool:
        """Move to target joint positions over duration seconds.

        Interpolates actuator targets linearly from current to target over
        the full duration, producing smooth natural-speed motion.
        When GUI is active, paces the simulation to real-time.
        """
        self._require_connection()
        if len(positions) != self.dof:
            raise ValueError(
                f"MuJoCoArm.move_joints: expected {self.dof} positions, "
                f"got {len(positions)}"
            )

        mj = _get_mujoco()
        import time as _time  # noqa: PLC0415

        dt = self._model.opt.timestep
        steps = max(1, int(duration / dt))
        sync_interval = max(1, int(1.0 / 60.0 / dt))

        # Read current actuator targets as start
        start = [float(self._data.ctrl[aid]) for aid in self._arm_actuator_ids]

        if self._viewer is not None:
            wall_start = _time.monotonic()
            for i in range(steps):
                # Linear interpolation of actuator targets
                t = (i + 1) / steps
                for j, (act_id, s, g) in enumerate(
                    zip(self._arm_actuator_ids, start, positions)
                ):
                    self._data.ctrl[act_id] = s + t * (g - s)

                mj.mj_step(self._model, self._data)

                if i % sync_interval == 0:
                    self._viewer.sync()
                    sim_elapsed = (i + 1) * dt
                    wall_elapsed = _time.monotonic() - wall_start
                    sleep = sim_elapsed - wall_elapsed
                    if sleep > 0:
                        _time.sleep(sleep)
        else:
            for i in range(steps):
                t = (i + 1) / steps
                for act_id, s, g in zip(self._arm_actuator_ids, start, positions):
                    self._data.ctrl[act_id] = s + t * (g - s)
                mj.mj_step(self._model, self._data)

        return True

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        """Move end-effector to target position via IK."""
        self._require_connection()
        solution = self.ik(target_xyz)
        if solution is None:
            return False
        return self.move_joints(solution, duration=duration)

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics via MuJoCo.

        Temporarily sets joints, runs mj_forward, reads ee_site, restores.
        """
        self._require_connection()
        mj = _get_mujoco()

        # Save full state
        old_qpos = self._data.qpos.copy()
        old_qvel = self._data.qvel.copy()

        # Set requested joints
        for i, pos in enumerate(joint_positions):
            self._data.joint(_ARM_JOINT_NAMES[i]).qpos[0] = pos

        mj.mj_forward(self._model, self._data)

        # Read EE site
        ee_pos = list(self._data.site_xpos[self._ee_site_id].copy())
        ee_rot = (
            self._data.site_xmat[self._ee_site_id]
            .reshape(3, 3)
            .tolist()
        )

        # Restore
        self._data.qpos[:] = old_qpos
        self._data.qvel[:] = old_qvel
        mj.mj_forward(self._model, self._data)

        return ee_pos, ee_rot

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Inverse kinematics using Jacobian-based iterative solver.

        Uses MuJoCo's mj_jac to compute the site Jacobian and solves
        via damped least-squares.

        Returns joint positions if converged within 1 mm, else None.
        """
        self._require_connection()
        mj = _get_mujoco()
        import numpy as np  # noqa: PLC0415

        target = np.array(target_xyz, dtype=np.float64)

        # Save state
        old_qpos = self._data.qpos.copy()
        old_qvel = self._data.qvel.copy()

        # Seed from current or provided joints
        seed = current_joints if current_joints is not None else self.get_joint_positions()
        for i, pos in enumerate(seed):
            self._data.joint(_ARM_JOINT_NAMES[i]).qpos[0] = pos

        # Map arm joint qpos indices (for indexing into full Jacobian)
        arm_qpos_adrs = [
            self._model.jnt_qposadr[jid] for jid in self._arm_joint_ids
        ]
        arm_dof_adrs = [
            self._model.jnt_dofadr[jid] for jid in self._arm_joint_ids
        ]

        success = False
        for _ in range(_IK_MAX_ITER):
            mj.mj_forward(self._model, self._data)

            # Current EE position
            ee_pos = self._data.site_xpos[self._ee_site_id].copy()
            err = target - ee_pos
            if np.linalg.norm(err) < _IK_TOL:
                success = True
                break

            # Compute site Jacobian (3 x nv for position)
            jacp = np.zeros((3, self._model.nv), dtype=np.float64)
            mj.mj_jacSite(self._model, self._data, jacp, None, self._ee_site_id)

            # Extract columns for arm DOFs only
            J = jacp[:, arm_dof_adrs]

            # Damped least-squares: dq = J^T (J J^T + lambda I)^{-1} err
            JJt = J @ J.T + _IK_DAMPING * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)

            # Apply
            for i, adr in enumerate(arm_qpos_adrs):
                self._data.qpos[adr] += _IK_STEP_SIZE * dq[i]
                # Clamp to joint limits
                lo = self._model.jnt_range[self._arm_joint_ids[i], 0]
                hi = self._model.jnt_range[self._arm_joint_ids[i], 1]
                self._data.qpos[adr] = float(
                    np.clip(self._data.qpos[adr], lo, hi)
                )

        if success:
            result = [
                float(self._data.qpos[adr]) for adr in arm_qpos_adrs
            ]
        else:
            result = None

        # Restore
        self._data.qpos[:] = old_qpos
        self._data.qvel[:] = old_qvel
        mj.mj_forward(self._model, self._data)

        return result

    def stop(self) -> None:
        """Stop all motion by targeting current positions."""
        self._require_connection()
        positions = self.get_joint_positions()
        self.move_joints(positions, duration=0.05)

    # ------------------------------------------------------------------
    # Scene management (simulation-specific)
    # ------------------------------------------------------------------

    def add_object(
        self,
        name: str,
        position: list[float],
        color: list[float] | None = None,
        size: float = 0.03,
    ) -> int:
        """MuJoCo scene objects are defined in the MJCF XML.

        For dynamic object spawning, modify the XML and reload.
        This method logs a warning — use the scene XML to define objects.

        Returns -1 (not supported at runtime in MuJoCo).
        """
        logger.warning(
            "MuJoCoArm.add_object: runtime object spawning not supported. "
            "Define objects in the MJCF scene XML instead."
        )
        return -1

    def get_object_positions(self) -> dict[str, list[float]]:
        """Return positions of all free-body objects in the scene.

        Returns:
            Dict mapping body name to [x, y, z] position.
        """
        self._require_connection()
        mj = _get_mujoco()
        result: dict[str, list[float]] = {}
        for i in range(self._model.nbody):
            body_name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, i)
            if body_name is None:
                continue
            # Free bodies have a freejoint — check via parent joint type
            jnt_start = self._model.body_jntadr[i]
            if jnt_start < 0:
                continue
            jnt_type = self._model.jnt_type[jnt_start]
            if jnt_type == mj.mjtJoint.mjJNT_FREE:
                pos = list(self._data.body(body_name).xpos)
                result[body_name] = pos
        return result

    def step(self, n: int = 1) -> None:
        """Advance simulation by n timesteps. Syncs viewer if active."""
        self._require_connection()
        mj = _get_mujoco()
        for _ in range(n):
            mj.mj_step(self._model, self._data)
        if self._viewer is not None:
            self._viewer.sync()

    def render(
        self,
        camera_name: str = "overhead",
        width: int = 640,
        height: int = 480,
    ) -> Any:
        """Render an RGB image from a named camera.

        Returns:
            numpy array (H, W, 3) uint8 BGR image, or None on failure.
        """
        self._require_connection()
        mj = _get_mujoco()
        import numpy as np  # noqa: PLC0415

        try:
            renderer = mj.Renderer(self._model, height=height, width=width)
            renderer.update_scene(self._data, camera=camera_name)
            rgb = renderer.render()
            renderer.close()
            # MuJoCo returns RGB; convert to BGR for OpenCV compatibility
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            return bgr
        except Exception as exc:
            logger.warning("MuJoCoArm.render failed: %s", exc)
            return None
