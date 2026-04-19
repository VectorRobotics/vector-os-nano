"""MuJoCo-based AgileX Piper 6-DoF arm.

Implements ArmProtocol. The Piper is a nested body tree inside the combined
Go2+Piper MJCF, so we share MjModel/MjData with the parent MuJoCoGo2 rather
than owning our own. Phase B of the v2.1 manipulation stack (Phase A mounted
the arm; Phase C is the PickTopDownSkill that drives this).

Thread model
------------
The MuJoCoGo2 physics thread runs at 1 kHz and owns stepping. This class:
- reads data.qpos[...] directly (atomic, safe)
- writes data.ctrl[...] (atomic, safe — physics thread reads on next tick)
- performs fk/ik on a scratch MjData clone (no race with live qpos)

We never call mj_step from here.

Usage:
    go2 = MuJoCoGo2(..., room=True)
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    piper.move_joints([0.0] * 6, duration=2.0)
    top_down_q = piper.ik_top_down((10.4, 3.0, 0.30))
    if top_down_q is not None:
        piper.move_joints(top_down_q, duration=3.0)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MuJoCo import (matches mujoco_go2.py / mujoco_arm.py pattern)
# ---------------------------------------------------------------------------

_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


# ---------------------------------------------------------------------------
# Joint / actuator / site names in the combined go2_piper model
# ---------------------------------------------------------------------------

_ARM_JOINT_NAMES: list[str] = [
    "piper_joint1",
    "piper_joint2",
    "piper_joint3",
    "piper_joint4",
    "piper_joint5",
    "piper_joint6",
]
_ARM_ACTUATOR_NAMES: list[str] = [
    "piper_joint1",
    "piper_joint2",
    "piper_joint3",
    "piper_joint4",
    "piper_joint5",
    "piper_joint6",
]
_EE_SITE_NAME: str = "piper_ee_site"

# ---------------------------------------------------------------------------
# IK tuning
# ---------------------------------------------------------------------------

_IK_MAX_ITER: int = 200
_IK_POS_TOL: float = 2e-3          # 2 mm
_IK_ROT_TOL: float = 2e-2          # ~1.1 deg — loose enough for top-down demo
_IK_STEP_SIZE: float = 0.4
_IK_DAMPING: float = 5e-3

# Pre-canned seed poses for ik_top_down. Solving 6-DoF IK from the URDF-zero
# config fails often because the arm is fully extended forward (gripper
# pointing +X) and a top-down target needs the gripper pointing -Z — that's
# nearly a 90° rotation of the whole wrist chain. Seeding from a pose that
# already has the wrist bent 90° down converges in ~30 iterations.
# Format: list of 6-joint configurations.
_IK_TOP_DOWN_SEEDS: list[list[float]] = [
    # Generic top-down ready: shoulder up, elbow bent back, wrist pitched
    # 90° so gripper points down.
    [0.0,  1.57, -1.57, 0.0,  1.57, 0.0],
    # Seed when target is forward+low (known working from prior test)
    [0.0,  2.48, -1.75, 0.0,  1.00, 3.13],
    # Seed with opposite wrist roll (helps when joint6 saturates)
    [0.0,  1.57, -1.57, 0.0,  1.57, 3.14],
    # Finally, the current live state — useful for small adjustments
    None,  # sentinel: use current_joints / live state
]

# Top-down grasp target rotation: gripper z-axis = world -Z (fingers down),
# local x stays aligned with world +X, local y flips. Valid rotation (det=+1).
_R_TOP_DOWN: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# Home pose (URDF zero configuration, matches _PIPER_STOW_QPOS on MuJoCoGo2)
_HOME_JOINTS: list[float] = [0.0] * 6

# move_joints interpolation rate (writes ctrl at ~50 Hz servo rate)
_MOVE_UPDATE_HZ: float = 50.0


# ---------------------------------------------------------------------------
# MuJoCoPiper
# ---------------------------------------------------------------------------


class MuJoCoPiper:
    """AgileX Piper arm in MuJoCo, sharing state with the parent Go2.

    Args:
        go2: A connected MuJoCoGo2 instance. Its MjModel must contain the
            Piper bodies (loaded via scene_room_piper.xml / go2_piper.xml).
            If the model has no piper_joint1, connect() will raise.
    """

    def __init__(self, go2: "MuJoCoGo2") -> None:
        self._go2 = go2
        self._connected: bool = False

        # Cached IDs into the live go2 model (set on connect)
        self._arm_joint_qpos_adr: list[int] = []
        self._arm_joint_dof_adr: list[int] = []
        self._arm_joint_ids: list[int] = []
        self._arm_actuator_ids: list[int] = []

        # IK uses a completely isolated (separately loaded) MjModel + MjData
        # for kinematics. Sharing the live model with the 1 kHz physics
        # thread — even in read-only mode for IK — has caused intermittent
        # segfaults under extended testing. The IK model must match the
        # live model structurally (same joints/actuators/sites); we load it
        # from the same MJCF file at connect time.
        self._ik_model: Any = None
        self._ik_data: Any = None
        # IK-model-specific IDs
        self._ik_arm_qpos_adr: list[int] = []
        self._ik_arm_dof_adr: list[int] = []
        self._ik_arm_joint_ids: list[int] = []
        self._ik_ee_site_id: int = -1

        # Lock for ctrl writes against concurrent stop() / skill interleaving
        self._ctrl_lock = threading.Lock()

    # ------------------------------------------------------------------
    # ArmProtocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mujoco_piper"

    @property
    def joint_names(self) -> list[str]:
        return list(_ARM_JOINT_NAMES)

    @property
    def dof(self) -> int:
        return 6

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Resolve joint / actuator / site IDs against the shared model.

        Raises:
            RuntimeError: If the parent MuJoCoGo2 is not connected, or if the
                loaded MJCF does not contain the Piper joints (i.e. with_arm
                mode was not enabled).
        """
        if not getattr(self._go2, "_connected", False):
            raise RuntimeError("MuJoCoPiper: parent MuJoCoGo2 must be connected first")

        mj = _get_mujoco()
        model = self._go2._mj.model

        joint_ids: list[int] = []
        for name in _ARM_JOINT_NAMES:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(
                    f"MuJoCoPiper: joint {name!r} not in loaded MJCF. "
                    "Ensure sim was launched with with_arm=True."
                )
            joint_ids.append(jid)
        self._arm_joint_ids = joint_ids
        self._arm_joint_qpos_adr = [int(model.jnt_qposadr[j]) for j in joint_ids]
        self._arm_joint_dof_adr = [int(model.jnt_dofadr[j]) for j in joint_ids]

        actuator_ids: list[int] = []
        for name in _ARM_ACTUATOR_NAMES:
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"MuJoCoPiper: actuator {name!r} not in loaded MJCF.")
            actuator_ids.append(aid)
        self._arm_actuator_ids = actuator_ids

        # (Live-model site id is not cached — IK uses the isolated model.)
        live_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, _EE_SITE_NAME)
        if live_sid < 0:
            raise RuntimeError(
                f"MuJoCoPiper: site {_EE_SITE_NAME!r} not in loaded MJCF. "
                "Regenerate go2_piper.xml via build_go2_piper.py."
            )

        # Allocate an isolated IK MjModel + MjData. Loading from the same
        # MJCF ensures structural equivalence (ids / qpos layout match). The
        # isolated model is never touched by the physics thread, so IK and
        # mj_step cannot race on shared MuJoCo internals.
        self._allocate_ik_model(mj)

        self._connected = True
        logger.info(
            "MuJoCoPiper connected: joints=%s, ee_site=%s (isolated IK model loaded)",
            _ARM_JOINT_NAMES, _EE_SITE_NAME,
        )

    def _allocate_ik_model(self, mj: Any) -> None:
        """Load a private MjModel+MjData from the same MJCF as the live model.

        The loaded scene path is read back from the already-loaded live model
        via mj.mj_id2name or the model's registered files; in practice we use
        the same XML that _build_room_scene_xml produced, which is recorded on
        the go2 instance as ``go2._scene_xml_path`` when MuJoCoGo2 loads it.
        """
        scene_path = getattr(self._go2, "_scene_xml_path", None)
        if scene_path is None:
            # Fallback: reconstruct via the builder helper. This keeps us safe
            # even if MuJoCoGo2 ever stops exposing the attribute.
            from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml
            scene_path = str(_build_room_scene_xml(with_arm=True))

        self._ik_model = mj.MjModel.from_xml_path(str(scene_path))
        self._ik_data = mj.MjData(self._ik_model)

        # Re-resolve ids against the private model.
        m = self._ik_model
        self._ik_arm_joint_ids = [
            mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in _ARM_JOINT_NAMES
        ]
        self._ik_arm_qpos_adr = [int(m.jnt_qposadr[j]) for j in self._ik_arm_joint_ids]
        self._ik_arm_dof_adr = [int(m.jnt_dofadr[j]) for j in self._ik_arm_joint_ids]
        self._ik_ee_site_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, _EE_SITE_NAME)

    def disconnect(self) -> None:
        """Mark disconnected. Release the isolated IK model. Does not affect
        the parent Go2. Idempotent.
        """
        self._connected = False
        self._ik_model = None
        self._ik_data = None

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("MuJoCoPiper: not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> list[float]:
        """Read the 6 arm joint positions in radians."""
        self._require_connection()
        data = self._go2._mj.data
        return [float(data.qpos[adr]) for adr in self._arm_joint_qpos_adr]

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Interpolate ctrl targets from current to positions over duration.

        The MuJoCoGo2 physics thread (1 kHz) runs the position-PD actuators
        at their usual rate; we just update their set-points at ~50 Hz wall
        time. Returns True once interpolation ends (does not verify the arm
        actually reached the target — position PD can have steady-state error
        near joint limits; use get_joint_positions() to verify).
        """
        self._require_connection()
        if len(positions) != self.dof:
            raise ValueError(
                f"MuJoCoPiper.move_joints: expected {self.dof} positions, "
                f"got {len(positions)}"
            )

        data = self._go2._mj.data
        start = [float(data.ctrl[aid]) for aid in self._arm_actuator_ids]

        steps = max(1, int(duration * _MOVE_UPDATE_HZ))
        dt = duration / steps

        for i in range(1, steps + 1):
            t = i / steps
            with self._ctrl_lock:
                for aid, s, g in zip(self._arm_actuator_ids, start, positions):
                    data.ctrl[aid] = s + t * (g - s)
            time.sleep(dt)

        return True

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        """Top-down IK + move_joints. Returns False if IK unreachable."""
        self._require_connection()
        q = self.ik_top_down(target_xyz)
        if q is None:
            return False
        return self.move_joints(q, duration=duration)

    def stop(self) -> None:
        """Hold current positions (freeze motion quickly)."""
        self._require_connection()
        data = self._go2._mj.data
        with self._ctrl_lock:
            for adr, aid in zip(self._arm_joint_qpos_adr, self._arm_actuator_ids):
                data.ctrl[aid] = float(data.qpos[adr])

    def home(self, duration: float = 3.0) -> bool:
        """Move to the URDF zero configuration."""
        return self.move_joints(list(_HOME_JOINTS), duration=duration)

    # ------------------------------------------------------------------
    # Kinematics — all run on a scratch MjData clone
    # ------------------------------------------------------------------

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics via the isolated IK model.

        Returns:
            (position_xyz_world, rotation_3x3_row_major) for piper_ee_site,
            computed with the live dog base pose but the supplied arm joints.
        """
        self._require_connection()
        if len(joint_positions) != self.dof:
            raise ValueError(
                f"MuJoCoPiper.fk: expected {self.dof} joints, got {len(joint_positions)}"
            )
        mj = _get_mujoco()
        self._sync_ik_base_from_live(joint_positions)
        mj.mj_forward(self._ik_model, self._ik_data)
        pos = [float(v) for v in self._ik_data.site_xpos[self._ik_ee_site_id]]
        mat = self._ik_data.site_xmat[self._ik_ee_site_id].reshape(3, 3)
        rot = [[float(v) for v in row] for row in mat]
        return pos, rot

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Position-only IK (3-DoF), Jacobian DLS.

        Returns joint positions if converged within _IK_POS_TOL, else None.
        """
        return self._ik_impl(target_xyz, target_rot=None, current_joints=current_joints)

    def ik_top_down(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """6-DoF IK with gripper z-axis constrained to world -Z (top-down).

        Tries several canned "top-down ready" seed configurations and returns
        the first one that converges on both position and orientation within
        tolerance. Runs on an isolated MjModel+MjData (loaded at connect()),
        so it cannot race with the live physics thread.
        """
        self._require_connection()
        seeds: list[list[float] | None] = []
        if current_joints is not None:
            seeds.append(list(current_joints))
        seeds.extend(_IK_TOP_DOWN_SEEDS)

        # Sync IK data with current dog base pose (IK targets are world-frame;
        # the arm moves with the base, so FK through the isolated model needs
        # the same floating-base qpos as the live scene).
        self._sync_ik_base_from_live(arm_joints=None)
        for seed in seeds:
            sol = self._ik_iterate(target_xyz, target_rot=_R_TOP_DOWN,
                                   seed_joints=seed)
            if sol is not None:
                return sol
        return None

    # ------------------------------------------------------------------
    # Internal: IK implementation
    # ------------------------------------------------------------------

    def _sync_ik_base_from_live(self, arm_joints: list[float] | None) -> None:
        """Copy the live floating-base qpos into the isolated IK data.

        We pause the parent physics thread briefly during the read. Even
        though the IK model is separate, the snapshot of live qpos was
        observed (intermittently) to segfault under extended test runs —
        likely a MuJoCo-Python binding issue where Python numpy views share
        state with MuJoCo-allocated buffers. Pausing physics for ~1 ms makes
        the snapshot deterministic.
        """
        go2 = self._go2
        pause_fn = getattr(go2, "_pause_physics", None)
        resume_fn = getattr(go2, "_resume_physics", None)
        was_running = bool(getattr(go2, "_running", False))
        paused = False
        if callable(pause_fn) and was_running:
            pause_fn()
            paused = True
        try:
            live_qpos = np.asarray(self._go2._mj.data.qpos, dtype=np.float64).copy()
        finally:
            if paused and callable(resume_fn):
                resume_fn()
        self._ik_data.qpos[0:19] = live_qpos[0:19]
        if arm_joints is not None:
            for adr, q in zip(self._ik_arm_qpos_adr, arm_joints):
                self._ik_data.qpos[adr] = float(q)
        self._ik_data.qvel[:] = 0.0
        self._ik_data.qacc[:] = 0.0

    def _ik_impl(
        self,
        target_xyz: tuple[float, float, float],
        target_rot: np.ndarray | None,
        current_joints: list[float] | None,
    ) -> list[float] | None:
        """Single-seed IK — used by positional ik() (non-top-down callers)."""
        self._sync_ik_base_from_live(arm_joints=None)
        return self._ik_iterate(target_xyz, target_rot,
                                seed_joints=current_joints)

    def _ik_iterate(
        self,
        target_xyz: tuple[float, float, float],
        target_rot: np.ndarray | None,
        seed_joints: list[float] | None,
    ) -> list[float] | None:
        """Run iterative Jacobian DLS IK on the isolated (self._ik_data).

        Caller is responsible for syncing the base pose first via
        ``_sync_ik_base_from_live``. This method only writes the arm joints
        and iterates. Never touches the live MjData.
        """
        self._require_connection()
        mj = _get_mujoco()
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
            mj.mj_forward(m, d)

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

            mj.mj_jacSite(m, d, jacp, jacr, self._ik_ee_site_id)

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
