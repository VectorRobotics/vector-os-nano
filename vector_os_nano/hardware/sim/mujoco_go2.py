"""MuJoCo-based simulated Unitree Go2 quadruped.

Lifecycle: MuJoCoGo2(gui=False) -> connect() -> stand/sit/lie_down -> disconnect().

Dual-backend locomotion:
  - Backend A (sinusoidal): Pure numpy+mujoco, zero external deps. Always available.
  - Backend B (convex_mpc): Centroidal MPC + leg controller. Requires convex_mpc,
    casadi, pinocchio. Auto-detected on connect() when backend="auto".

Joint ordering (MuJoCo ctrl and qpos[7:19]):
    0-2:  FL  hip, thigh, calf
    3-5:  FR  hip, thigh, calf
    6-8:  RL  hip, thigh, calf
    9-11: RR  hip, thigh, calf

Quaternion convention: MuJoCo uses (w, x, y, z) in qpos[3:7].

Background physics thread:
    connect() starts a daemon thread (_physics_loop) at 1 kHz.
    set_velocity() writes (vx, vy, vyaw) under _cmd_lock (non-blocking).
    stand/sit/lie_down pause the thread, run PD synchronously, then resume.
    disconnect() stops the thread cleanly.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415
        _mujoco = mujoco
    return _mujoco


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MJCF_DIR: Path = Path(__file__).parent / "mjcf" / "go2"
_ROOM_XML: Path = Path(__file__).parent / "go2_room.xml"

# ---------------------------------------------------------------------------
# Constants — postures
# ---------------------------------------------------------------------------

_STAND_JOINTS: list[float] = [0.0, 0.9, -1.8] * 4
_SIT_JOINTS: list[float] = [0.0, 1.5, -2.5] * 4
_LIE_DOWN_JOINTS: list[float] = [0.0, 2.0, -2.7] * 4

# ---------------------------------------------------------------------------
# Constants — PD control
# ---------------------------------------------------------------------------

_KP: float = 120.0
_KD: float = 3.5

_TAU_HIP: float = 23.7 * 0.9
_TAU_KNEE: float = 45.43 * 0.9

_TAU_LIMITS: np.ndarray = np.array(
    [_TAU_HIP, _TAU_HIP, _TAU_KNEE] * 4, dtype=np.float64
)

# ---------------------------------------------------------------------------
# Constants — simulation timing
# ---------------------------------------------------------------------------

_SIM_HZ: int = 1000
_SIM_DT: float = 1.0 / _SIM_HZ
_CTRL_HZ: int = 200
_CTRL_DECIM: int = _SIM_HZ // _CTRL_HZ

_VIEWER_SYNC_EVERY: int = 8

# ---------------------------------------------------------------------------
# Constants — sinusoidal trotting gait
# ---------------------------------------------------------------------------

_GAIT_FREQ: float = 2.0          # steps per second (Hz)
_THIGH_AMP: float = 0.25         # thigh swing amplitude (rad)
_CALF_AMP: float = 0.25          # calf swing amplitude (rad)
_HIP_AMP: float = 0.10           # hip abduction amplitude for lateral motion (rad)
_CALF_PHASE: float = 0.0          # calf in-phase: foot down during forward sweep (propulsion)

# Trotting: diagonal legs in phase, adjacent legs in anti-phase
# FL+RR together, FR+RL together
_TROT_PHASES: tuple[float, ...] = (0.0, math.pi, math.pi, 0.0)

# ---------------------------------------------------------------------------
# Constants — velocity limits
# ---------------------------------------------------------------------------

_VX_MAX: float = 0.8
_VY_MAX: float = 0.4
_VYAW_MAX: float = 4.0

# ---------------------------------------------------------------------------
# Constants — MPC backend (convex_mpc)
# ---------------------------------------------------------------------------

_MPC_GAIT_HZ: int = 3
_MPC_GAIT_DUTY: float = 0.6
_MPC_DT_FACTOR: int = 16
_MPC_Z_DES: float = 0.27
_MPC_SAFETY: float = 0.9
_MPC_TAU_LIMITS: np.ndarray = _MPC_SAFETY * np.array(
    [23.7, 23.7, 45.43] * 4, dtype=np.float64
)
_MPC_LEG_NAMES: list[str] = ["FL", "FR", "RL", "RR"]

# ---------------------------------------------------------------------------
# Constants — lidar
# ---------------------------------------------------------------------------

_LIDAR_UPDATE_INTERVAL: int = 100  # physics steps between scans (~10 Hz)


# ---------------------------------------------------------------------------
# Minimal MuJoCo wrapper (replaces convex_mpc.MuJoCo_GO2_Model)
# ---------------------------------------------------------------------------

class _Go2Model:
    """Lightweight wrapper around MjModel/MjData for Go2.

    Caches actuator IDs so set_joint_torque() is fast.
    """

    __slots__ = ("model", "data", "base_bid", "_act_ids", "viewer")

    def __init__(self, model: Any, data: Any) -> None:
        mj = _get_mujoco()
        self.model = model
        self.data = data
        self.viewer = None
        self.base_bid: int = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_BODY, "base_link"
        )
        # Cache actuator IDs: FL_hip, FL_thigh, FL_calf, FR..., RL..., RR...
        self._act_ids: list[int] = []
        for leg in ("FL", "FR", "RL", "RR"):
            for joint in ("hip", "thigh", "calf"):
                self._act_ids.append(
                    mj.mj_name2id(
                        model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{joint}"
                    )
                )

    def set_joint_torque(self, torque: np.ndarray) -> None:
        """Apply 12 joint torques in canonical order."""
        for i, aid in enumerate(self._act_ids):
            self.data.ctrl[aid] = float(torque[i])


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def _build_flat_scene_xml() -> Path:
    """Generate a flat-ground scene XML in the local MJCF directory.

    The scene includes go2.xml via relative path so mesh assets resolve
    correctly from the same directory.
    """
    out = _MJCF_DIR / "scene_flat.xml"
    xml = """\
<mujoco model="go2_flat">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <include file="go2.xml"/>

  <option cone="elliptic" impratio="100"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="2d" name="grid" builtin="checker"
             rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="grid"/>
  </worldbody>

  <keyframe>
    <key name="stand"
         qpos="0 0 0.35 1 0 0 0  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8"/>
  </keyframe>
</mujoco>
"""
    out.write_text(xml)
    return out


def _build_room_scene_xml() -> Path:
    """Build composite room scene using local MJCF files.

    Resolves the go2_room.xml template with paths to the local go2.xml
    and assets directory (no convex_mpc dependency).
    """
    go2_xml = _MJCF_DIR / "go2.xml"
    assets_dir = _MJCF_DIR / "assets"

    template = _ROOM_XML.read_text()
    xml = template.replace("GO2_MODEL_PATH", str(go2_xml))
    xml = xml.replace("GO2_ASSETS_DIR", str(assets_dir))

    out = _MJCF_DIR / "scene_room.xml"
    out.write_text(xml)
    return out


# ---------------------------------------------------------------------------
# Sinusoidal gait generator
# ---------------------------------------------------------------------------

def _compute_gait_targets(
    t: float,
    vx: float,
    vy: float,
    vyaw: float,
) -> np.ndarray:
    """Compute 12 target joint positions for sinusoidal trotting gait.

    Args:
        t: Current simulation time (seconds).
        vx: Commanded forward velocity (m/s).
        vy: Commanded lateral velocity (m/s).
        vyaw: Commanded yaw rate (rad/s).

    Returns:
        Array of 12 target joint angles.
    """
    q_target = np.array(_STAND_JOINTS, dtype=np.float64)

    omega = 2.0 * math.pi * _GAIT_FREQ

    # Forward component: signed, maps vx to [-1, 1]
    fwd_amp = float(np.clip(vx / 0.5, -1.0, 1.0)) if abs(vx) > 0.01 else 0.0

    # Turn component: vyaw -> per-leg differential amplitude
    # Divisor of 1.0 ensures sufficient gait amplitude at low vyaw
    turn_amp = float(np.clip(vyaw / 1.0, -1.0, 1.0)) if abs(vyaw) > 0.01 else 0.0

    for leg_idx in range(4):
        base = leg_idx * 3
        phase = omega * t + _TROT_PHASES[leg_idx]

        # Turning torque: left legs push backward, right legs push forward → CCW
        # This is because torque = r × F: left(+Y) × backward(-X) = +Z = CCW
        is_left = leg_idx in (0, 2)
        leg_turn = -turn_amp if is_left else turn_amp

        # Combined per-leg amplitude (signed: positive=forward, negative=backward)
        total_amp = float(np.clip(fwd_amp + leg_turn, -1.5, 1.5))

        # Hip abduction — for lateral motion
        if abs(vy) > 0.01:
            q_target[base + 0] += _HIP_AMP * (vy / _VY_MAX) * math.sin(phase)

        # Per-leg calf phase: controls which direction the foot pushes
        # Positive amp → calf_phase=0 → foot down during forward sweep → forward push
        # Negative amp → calf_phase=pi → foot down during backward sweep → backward push
        if total_amp >= 0:
            leg_calf_phase = _CALF_PHASE
            amp = total_amp
        else:
            leg_calf_phase = _CALF_PHASE + math.pi
            amp = -total_amp  # use positive amplitude with flipped calf phase

        # Thigh swing
        q_target[base + 1] += _THIGH_AMP * amp * math.sin(phase)

        # Calf swing — phase determines foot contact timing
        q_target[base + 2] += _CALF_AMP * amp * math.sin(phase + leg_calf_phase)

    return q_target


# ---------------------------------------------------------------------------
# MuJoCoGo2
# ---------------------------------------------------------------------------

class MuJoCoGo2:
    """Unitree Go2 quadruped running in MuJoCo simulation.

    Dual-backend: sinusoidal gait (always available) or convex MPC
    (when convex_mpc package is installed).

    Args:
        gui: Open an interactive passive viewer on connect().
        room: Use indoor room scene instead of flat ground.
        backend: "auto" (try MPC, fall back to sinusoidal), "mpc", or "sinusoidal".
    """

    def __init__(
        self, gui: bool = False, room: bool = True, backend: str = "auto"
    ) -> None:
        self._gui: bool = gui
        self._room: bool = room
        self._backend_pref: str = backend
        self._mj: _Go2Model | None = None
        self._viewer: Any = None
        self._connected: bool = False

        # MPC stack (None when using sinusoidal backend)
        self._use_mpc: bool = False
        self._pin: Any = None
        self._gait: Any = None
        self._traj: Any = None
        self._mpc: Any = None
        self._leg_ctrl: Any = None

        # Background physics thread state
        self._cmd_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._cmd_lock: threading.Lock = threading.Lock()
        self._physics_thread: threading.Thread | None = None
        self._running: bool = False
        self._last_odom: Any = None
        self._last_scan: Any = None
        self._last_pointcloud: list = []
        self._scan_counter: int = 0

    # ------------------------------------------------------------------
    # Capability properties (BaseProtocol)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mujoco_go2"

    @property
    def supports_holonomic(self) -> bool:
        return True

    @property
    def supports_lidar(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Load MuJoCo model and optionally open viewer."""
        mj = _get_mujoco()

        if self._room:
            scene_path = _build_room_scene_xml()
            model = mj.MjModel.from_xml_path(str(scene_path))
            data = mj.MjData(model)
            self._mj = _Go2Model(model, data)

            # Place Go2 in the entry hall (center of house)
            data.qpos[0] = 10.0
            data.qpos[1] = 3.0
            data.qpos[2] = 0.35
            # Set standing joint angles
            data.qpos[7:19] = _STAND_JOINTS
        else:
            scene_path = _build_flat_scene_xml()
            model = mj.MjModel.from_xml_path(str(scene_path))
            data = mj.MjData(model)
            self._mj = _Go2Model(model, data)

            # Apply home keyframe (standing pose at origin)
            key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "stand")
            if key_id >= 0:
                mj.mj_resetDataKeyframe(model, data, key_id)

        # Set physics timestep to 1 kHz
        self._mj.model.opt.timestep = _SIM_DT

        mj.mj_forward(self._mj.model, self._mj.data)

        if self._gui:
            try:
                import mujoco.viewer  # noqa: PLC0415
                self._viewer = mujoco.viewer.launch_passive(
                    self._mj.model,
                    self._mj.data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
                if self._viewer is not None:
                    self._viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
                    if self._room:
                        self._viewer.cam.lookat[:] = [10.0, 7.0, 0.0]
                        self._viewer.cam.distance = 22.0
                        self._viewer.cam.elevation = -65
                        self._viewer.cam.azimuth = -90
                    else:
                        self._viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
                        self._viewer.cam.distance = 3.0
                        self._viewer.cam.elevation = -30
                        self._viewer.cam.azimuth = 120
            except Exception as exc:
                logger.warning("MuJoCoGo2 viewer failed to launch: %s", exc)
                self._viewer = None

        self._connected = True

        # Try to initialize MPC backend
        self._use_mpc = False
        if self._backend_pref in ("mpc", "auto"):
            try:
                self._init_mpc_stack()
                self._use_mpc = True
                logger.info("MuJoCoGo2: using convex_mpc backend")
            except Exception as exc:
                if self._backend_pref == "mpc":
                    raise RuntimeError(f"MPC backend requested but failed: {exc}") from exc
                logger.info("MuJoCoGo2: convex_mpc not available, using sinusoidal gait")

        backend_name = "mpc" if self._use_mpc else "sinusoidal"
        logger.info(
            "MuJoCoGo2 connected (gui=%s, room=%s, backend=%s)",
            self._gui, self._room, backend_name,
        )

        # Start background physics thread
        self._running = True
        self._physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="mujoco_go2_physics"
        )
        self._physics_thread.start()

    def disconnect(self) -> None:
        """Stop physics thread, close viewer and release model."""
        self._running = False
        if self._physics_thread is not None:
            self._physics_thread.join(timeout=2.0)
            self._physics_thread = None

        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001
                pass
            self._viewer = None
        self._mj = None
        self._pin = None
        self._gait = None
        self._traj = None
        self._mpc = None
        self._leg_ctrl = None
        self._use_mpc = False
        self._last_odom = None
        self._last_scan = None
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("MuJoCoGo2: not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Physics thread management
    # ------------------------------------------------------------------

    def _pause_physics(self) -> None:
        self._running = False
        if self._physics_thread is not None:
            self._physics_thread.join(timeout=2.0)
            self._physics_thread = None

    def _resume_physics(self) -> None:
        self._running = True
        self._physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="mujoco_go2_physics"
        )
        self._physics_thread.start()

    # ------------------------------------------------------------------
    # Background physics loop
    # ------------------------------------------------------------------

    def _init_mpc_stack(self) -> None:
        """Initialize convex_mpc control stack. Raises ImportError if unavailable."""
        from convex_mpc.go2_robot_data import PinGo2Model  # noqa: PLC0415
        from convex_mpc.gait import Gait                   # noqa: PLC0415
        from convex_mpc.com_trajectory import ComTraj       # noqa: PLC0415
        from convex_mpc.leg_controller import LegController # noqa: PLC0415

        self._pin = PinGo2Model()
        self._gait = Gait(_MPC_GAIT_HZ, _MPC_GAIT_DUTY)
        self._traj = ComTraj(self._pin)
        self._mpc = None  # lazy — first locomotion call
        self._leg_ctrl = LegController()

    def _physics_loop(self) -> None:
        """Background physics: read cmd_vel, compute gait, step MuJoCo.

        Runs at ~1 kHz. Controller updates at CTRL_HZ (200 Hz).
        Dispatches to MPC or sinusoidal backend based on self._use_mpc.
        """
        if self._use_mpc:
            self._physics_loop_mpc()
        else:
            self._physics_loop_sinusoidal()

    def _physics_loop_sinusoidal(self) -> None:
        """Physics loop using sinusoidal trotting gait (Backend A)."""
        mj = _get_mujoco()

        tau_hold: np.ndarray = np.zeros(12, dtype=float)
        sim_step: int = 0
        scan_counter: int = 0

        while self._running:
            loop_start = time.perf_counter()

            with self._cmd_lock:
                vx, vy, vyaw = self._cmd_vel

            time_now = float(self._mj.data.time)
            is_moving = (vx != 0.0 or vy != 0.0 or vyaw != 0.0)

            if sim_step % _CTRL_DECIM == 0:
                q_cur = np.array(self._mj.data.qpos[7:19], dtype=np.float64)
                dq_cur = np.array(self._mj.data.qvel[6:18], dtype=np.float64)

                if is_moving:
                    q_target = _compute_gait_targets(time_now, vx, vy, vyaw)
                else:
                    q_target = np.array(_STAND_JOINTS, dtype=np.float64)

                tau = _KP * (q_target - q_cur) - _KD * dq_cur
                tau = np.clip(tau, -_TAU_LIMITS, _TAU_LIMITS)
                tau_hold = tau.copy()

            mj.mj_step1(self._mj.model, self._mj.data)
            self._mj.set_joint_torque(tau_hold)
            mj.mj_step2(self._mj.model, self._mj.data)

            self._update_odometry()

            scan_counter += 1
            if scan_counter >= _LIDAR_UPDATE_INTERVAL:
                self._update_lidar()
                scan_counter = 0

            if self._viewer is not None and sim_step % _VIEWER_SYNC_EVERY == 0:
                self._viewer.sync()

            sim_step += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = _SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _physics_loop_mpc(self) -> None:
        """Physics loop using convex MPC locomotion (Backend B).

        Ported from the original convex_mpc-based implementation.
        MPC computes optimal contact forces, leg controller converts to torques.
        """
        mj = _get_mujoco()

        gait_period = self._gait.gait_period
        mpc_dt = gait_period / _MPC_DT_FACTOR
        mpc_hz = 1.0 / mpc_dt
        steps_per_mpc = max(1, int(_CTRL_HZ // mpc_hz))

        U_opt: Any = None
        ctrl_i: int = 0
        tau_hold: np.ndarray = np.zeros(12, dtype=float)
        sim_step: int = 0
        scan_counter: int = 0

        while self._running:
            loop_start = time.perf_counter()

            with self._cmd_lock:
                vx, vy, vyaw = self._cmd_vel

            time_now = float(self._mj.data.time)
            is_moving = (vx != 0.0 or vy != 0.0 or vyaw != 0.0)

            if sim_step % _CTRL_DECIM == 0:
                # Update Pinocchio model from MuJoCo state
                self._mj_update_pin()

                if is_moving:
                    # MPC locomotion — with solver failure protection
                    if ctrl_i % steps_per_mpc == 0:
                        try:
                            self._traj.generate_traj(
                                self._pin, self._gait, time_now,
                                vx, vy, _MPC_Z_DES, vyaw, time_step=mpc_dt,
                            )
                            if self._mpc is None:
                                from convex_mpc.centroidal_mpc import CentroidalMPC  # noqa: PLC0415
                                self._mpc = CentroidalMPC(self._pin, self._traj)

                            sol = self._mpc.solve_QP(self._pin, self._traj, False)
                            n = self._traj.N
                            w_opt = sol["x"].full().flatten()
                            U_opt = w_opt[12 * n:].reshape((12, n), order="F")
                        except Exception:
                            # QP solver failed — hold current torque (PD fallback)
                            pass

                    if U_opt is not None:
                        try:
                            mpc_force = U_opt[:, 0]
                            tau = np.zeros(12, dtype=float)
                            for i, leg in enumerate(_MPC_LEG_NAMES):
                                leg_out = self._leg_ctrl.compute_leg_torque(
                                    leg, self._pin, self._gait,
                                    mpc_force[i * 3:(i + 1) * 3], time_now,
                                )
                                tau[i * 3:(i + 1) * 3] = leg_out.tau
                            tau = np.clip(tau, -_MPC_TAU_LIMITS, _MPC_TAU_LIMITS)
                            tau_hold = tau.copy()
                        except Exception:
                            pass
                else:
                    # Idle: PD hold standing posture
                    q_cur = np.array(self._mj.data.qpos[7:19], dtype=np.float64)
                    dq_cur = np.array(self._mj.data.qvel[6:18], dtype=np.float64)
                    q_stand = np.array(_STAND_JOINTS, dtype=np.float64)
                    tau = _KP * (q_stand - q_cur) - _KD * dq_cur
                    tau = np.clip(tau, -_TAU_LIMITS, _TAU_LIMITS)
                    tau_hold = tau.copy()

                ctrl_i += 1

            mj.mj_step1(self._mj.model, self._mj.data)
            self._mj.set_joint_torque(tau_hold)
            mj.mj_step2(self._mj.model, self._mj.data)

            self._update_odometry()

            scan_counter += 1
            if scan_counter >= _LIDAR_UPDATE_INTERVAL:
                self._update_lidar()
                scan_counter = 0

            if self._viewer is not None and sim_step % _VIEWER_SYNC_EVERY == 0:
                self._viewer.sync()

            sim_step += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = _SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _mj_update_pin(self) -> None:
        """Sync Pinocchio model state from MuJoCo qpos/qvel.

        Converts MuJoCo (wxyz quaternion, world-frame linear vel) to
        Pinocchio (xyzw quaternion, body-frame linear vel) and runs
        the full set of Pinocchio computations the MPC solver needs.
        """
        mujoco_q = np.asarray(self._mj.data.qpos, dtype=float).reshape(-1)
        mujoco_dq = np.asarray(self._mj.data.qvel, dtype=float).reshape(-1)

        qw, qx, qy, qz = mujoco_q[3:7]

        import pinocchio as pin  # noqa: PLC0415
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        v_body = R.T @ mujoco_dq[0:3]
        w_body = mujoco_dq[3:6]

        q_pin = np.concatenate([mujoco_q[0:3], [qx, qy, qz, qw], mujoco_q[7:]])
        dq_pin = np.concatenate([v_body, w_body, mujoco_dq[6:]])

        self._pin.update_model(q_pin, dq_pin)

    # ------------------------------------------------------------------
    # Velocity command (non-blocking)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set target body velocity. Non-blocking."""
        self._require_connection()
        with self._cmd_lock:
            self._cmd_vel = (
                float(np.clip(vx, -_VX_MAX, _VX_MAX)),
                float(np.clip(vy, -_VY_MAX, _VY_MAX)),
                float(np.clip(vyaw, -_VYAW_MAX, _VYAW_MAX)),
            )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_position(self) -> list[float]:
        """Return base position [x, y, z] in world frame."""
        self._require_connection()
        return list(self._mj.data.qpos[0:3].astype(float))

    def get_velocity(self) -> list[float]:
        """Return base linear velocity [vx, vy, vz] in world frame."""
        self._require_connection()
        return list(self._mj.data.qvel[0:3].astype(float))

    def get_heading(self) -> float:
        """Return yaw angle (radians) from base quaternion."""
        self._require_connection()
        w, x, y, z = self._mj.data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(yaw)

    def get_joint_positions(self) -> list[float]:
        """Return all 12 joint positions (radians), ordered FL/FR/RL/RR."""
        self._require_connection()
        return list(self._mj.data.qpos[7:19].astype(float))

    def get_joint_velocities(self) -> list[float]:
        """Return all 12 joint velocities (rad/s), ordered FL/FR/RL/RR."""
        self._require_connection()
        return list(self._mj.data.qvel[6:18].astype(float))

    def get_odometry(self) -> Any:
        """Return full odometry snapshot as Odometry dataclass."""
        self._require_connection()
        if self._last_odom is None:
            self._update_odometry()
        return self._last_odom

    def get_lidar_scan(self) -> Any:
        """Return most recent 2D laser scan as LaserScan dataclass."""
        self._require_connection()
        if self._last_scan is None:
            self._update_lidar()
        return self._last_scan

    def get_3d_pointcloud(self) -> list[tuple[float, float, float, float]]:
        """Return most recent 3D point cloud as list of (x, y, z, intensity)."""
        self._require_connection()
        if not self._last_pointcloud:
            self._update_lidar()
        return self._last_pointcloud

    # ------------------------------------------------------------------
    # Sensor update helpers
    # ------------------------------------------------------------------

    def _update_odometry(self) -> None:
        from vector_os_nano.core.types import Odometry  # noqa: PLC0415
        q = self._mj.data.qpos
        v = self._mj.data.qvel
        self._last_odom = Odometry(
            timestamp=float(self._mj.data.time),
            x=float(q[0]),
            y=float(q[1]),
            z=float(q[2]),
            qx=float(q[4]),
            qy=float(q[5]),
            qz=float(q[6]),
            qw=float(q[3]),
            vx=float(v[0]),
            vy=float(v[1]),
            vz=float(v[2]),
            vyaw=float(v[5]),
        )

    def _update_lidar(self) -> None:
        """Cast rays in multiple elevation rings — Livox MID360-like 3D lidar."""
        from vector_os_nano.core.types import LaserScan  # noqa: PLC0415
        mj = _get_mujoco()

        pos = self._mj.data.qpos[0:3].copy().astype(np.float64)
        lidar_z = float(pos[2]) + 0.05
        pos_lidar = np.array([pos[0], pos[1], lidar_z], dtype=np.float64)

        heading = self.get_heading()
        robot_body_id = self._mj.base_bid

        n_azimuth = 360
        elevations = [-15, -10, -5, 0, 5, 10, 15]
        mid_ring_ranges: list[float] = []
        points_3d: list[tuple[float, float, float, float]] = []

        for elev_deg in elevations:
            elev_rad = math.radians(elev_deg)
            cos_elev = math.cos(elev_rad)
            sin_elev = math.sin(elev_rad)
            for i in range(n_azimuth):
                azimuth = heading + math.radians(i - 180)
                direction = np.array([
                    cos_elev * math.cos(azimuth),
                    cos_elev * math.sin(azimuth),
                    sin_elev,
                ], dtype=np.float64)
                geom_id = np.zeros(1, dtype=np.int32)
                dist = mj.mj_ray(
                    self._mj.model,
                    self._mj.data,
                    pos_lidar,
                    direction,
                    None,
                    1,
                    robot_body_id,
                    geom_id,
                )
                if dist > 0 and dist < 12.0:
                    px = pos_lidar[0] + dist * direction[0]
                    py = pos_lidar[1] + dist * direction[1]
                    pz = pos_lidar[2] + dist * direction[2]
                    points_3d.append((float(px), float(py), float(pz), 100.0))

                if elev_deg == 0:
                    mid_ring_ranges.append(
                        float(dist) if dist > 0 else float("inf")
                    )

        self._last_scan = LaserScan(
            timestamp=float(self._mj.data.time),
            angle_min=-math.pi,
            angle_max=math.pi,
            angle_increment=math.radians(1.0),
            range_min=0.1,
            range_max=12.0,
            ranges=tuple(mid_ring_ranges),
        )
        self._last_pointcloud = points_3d

    # ------------------------------------------------------------------
    # PD interpolation (runs synchronously — physics thread PAUSED)
    # ------------------------------------------------------------------

    def _pd_interpolate(
        self,
        target_joints: np.ndarray,
        duration: float = 2.0,
    ) -> None:
        """Drive joints to target using PD torque control with tanh ramp."""
        self._require_connection()

        was_running = self._running
        if was_running:
            self._pause_physics()

        mj = _get_mujoco()
        model = self._mj.model
        data = self._mj.data
        dt = model.opt.timestep
        total_steps = max(1, int(duration / dt))

        q_start = np.array(data.qpos[7:19], dtype=np.float64)
        q_target = np.asarray(target_joints, dtype=np.float64)

        hold_steps = max(0, int(0.5 / dt))
        total_steps_with_hold = total_steps + hold_steps

        for step in range(total_steps_with_hold):
            if step < total_steps:
                t_norm = (step + 1) * dt / (duration / 3.0)
                phase = float(np.tanh(t_norm))
                q_des = q_start + phase * (q_target - q_start)
            else:
                q_des = q_target

            q_cur = np.array(data.qpos[7:19], dtype=np.float64)
            dq_cur = np.array(data.qvel[6:18], dtype=np.float64)

            tau = _KP * (q_des - q_cur) - _KD * dq_cur
            tau = np.clip(tau, -_TAU_LIMITS, _TAU_LIMITS)

            self._mj.set_joint_torque(tau)
            mj.mj_step(model, data)

            if self._viewer is not None and (step % _VIEWER_SYNC_EVERY == 0):
                self._viewer.sync()

        if was_running:
            self._resume_physics()

    # ------------------------------------------------------------------
    # Posture commands
    # ------------------------------------------------------------------

    def stand(self, duration: float = 2.0) -> None:
        self._require_connection()
        self._pd_interpolate(
            np.array(_STAND_JOINTS, dtype=np.float64), duration=duration
        )

    def sit(self, duration: float = 2.0) -> None:
        self._require_connection()
        self._pd_interpolate(
            np.array(_SIT_JOINTS, dtype=np.float64), duration=duration
        )

    def lie_down(self, duration: float = 2.0) -> None:
        self._require_connection()
        self._pd_interpolate(
            np.array(_LIE_DOWN_JOINTS, dtype=np.float64), duration=duration
        )

    def stop(self) -> None:
        """Emergency stop: zero velocity command."""
        self._require_connection()
        with self._cmd_lock:
            self._cmd_vel = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Locomotion (blocking)
    # ------------------------------------------------------------------

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        """Walk at commanded velocity using sinusoidal trotting gait.

        The robot should be standing before calling (call stand() first).

        Returns:
            True if completed without falling over.
        """
        self._require_connection()
        self.set_velocity(vx, vy, vyaw)
        time.sleep(duration)
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(0.2)  # settle
        pos = self.get_position()
        return bool(pos[2] > 0.15)
