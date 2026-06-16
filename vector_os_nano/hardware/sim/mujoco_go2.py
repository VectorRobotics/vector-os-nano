# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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
import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# TEMP DIAGNOSTIC (off unless VECTOR_MPC_LOG set): count/log swallowed MPC solver
# failures so the explore "float" can be checked for silently-failing QP solves
# (the _physics_step_mpc except: pass would otherwise hide them → stale torque →
# float). Remove after debugging.
_MPC_DIAG: dict[str, int] = {"qp_ok": 0, "qp_fail": 0, "tau_fail": 0}

# TEMP DIAGNOSTIC (off unless VECTOR_PHYS_LOG set): every ~2s log sim-time/wall
# rate + count of live "mujoco_go2_physics" daemons — directly detects duplicate
# stepping (rate ~2x / daemons>1) or a stalled daemon (rate<<1). Remove after debug.
_PHYS_DIAG: dict[str, float] = {"last_wall": 0.0, "last_sim": 0.0}


def _phys_diag(sim_time: float) -> None:
    _p = os.environ.get("VECTOR_PHYS_LOG", "")
    if not _p:
        return
    now = time.perf_counter()
    lw = _PHYS_DIAG["last_wall"]
    if lw and (now - lw) < 2.0:
        return
    rate = ((sim_time - _PHYS_DIAG["last_sim"]) / (now - lw)) if lw else 0.0
    _PHYS_DIAG["last_wall"] = now
    _PHYS_DIAG["last_sim"] = sim_time
    n = sum(1 for t in threading.enumerate() if t.name == "mujoco_go2_physics")
    try:
        with open(_p, "a") as _f:
            _f.write(f"{now:.3f}\tsim_time={sim_time:.3f}\tsim/wall={rate:.2f}x\tphysics_daemons={n}\n")
    except Exception:  # noqa: BLE001
        pass


def _mpc_diag(kind: str, exc: BaseException | None = None) -> None:
    _MPC_DIAG[kind] = _MPC_DIAG.get(kind, 0) + 1
    _p = os.environ.get("VECTOR_MPC_LOG", "")
    if _p:
        try:
            with open(_p, "a") as _f:
                _f.write(
                    f"{time.time():.4f}\t{kind}\t"
                    f"{type(exc).__name__ if exc else ''}\t{str(exc)[:140] if exc else ''}\n"
                )
        except Exception:  # noqa: BLE001
            pass

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
_GO2_PIPER_XML: Path = Path(__file__).parent / "mjcf" / "go2_piper" / "go2_piper.xml"

# ---------------------------------------------------------------------------
# Constants — postures
# ---------------------------------------------------------------------------

_STAND_JOINTS: list[float] = [0.0, 0.9, -1.8] * 4
_SIT_JOINTS: list[float] = [0.0, 1.5, -2.5] * 4
_LIE_DOWN_JOINTS: list[float] = [0.0, 2.0, -2.7] * 4

# Piper stow pose — URDF zero configuration (all joints at 0).
# joint2 range=(0, 3.14) and joint3 range=(-2.697, 0) both sit at a limit,
# which is intentional: the URDF was designed so all-zeros is the canonical
# "initial" / "calibration" pose. Position actuators (kp=80, kv=5) with
# gravcomp=1 hold it to <0.3 deg drift per 2s — verified headless.
# Ordering: joint1..joint6 then finger joint7 (joint8 coupled via equality).
# Only applied when the loaded model has the Piper arm (nq >= 27).
_PIPER_STOW_QPOS: list[float] = [0.0] * 8
_PIPER_STOW_CTRL: list[float] = [0.0] * 7

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

_VIEWER_SYNC_EVERY: int = 30

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

_LIDAR_UPDATE_INTERVAL: int = 200  # physics steps between scans (~5 Hz, 5200 rays/scan)

# --- navigate_to closed-loop constants (campaign #11 M2) --------------------
# Mirror of g1.py's _NAV_* (the same face->walk->capture->settle controller +
# g1_vgraph routing), retuned for the Go2 quadruped. SEEDS pending a real-sim
# tuning pass (record actual stand-z / vyaw / stall before shipping as final).
# CRITICAL: the Go2 stands at qpos[2]=0.35 (connect()), NOT the G1 pelvis 0.77 —
# so _GO2_NAV_FALL_Z=0.20, NOT G1's 0.4 (0.4 would make `reached` NEVER true and
# silently report 'fell' on every successful arrival).
_GO2_NAV_FALL_Z = 0.20       # m: base height below this = fallen (Go2 stands ~0.35)
_GO2_NAV_TOL_FLOOR = 0.20    # gait COM oscillation floor — tol can't beat this
_GO2_NAV_SPEED = 0.5         # default forward cmd (Go2 _VX_MAX ~0.8)
_GO2_NAV_VYAW_MAX = 1.0      # cmd ceiling (Go2 _VYAW_MAX ~4.0; conservative)
_GO2_NAV_K_YAW = 2.0         # proportional heading gain
_GO2_NAV_FACE_TOL = 0.35     # rad: pivot-in-place until aligned within this
_GO2_NAV_YAW_DEADBAND = 0.12  # rad: stop steering when aligned (anti-hunt)
_GO2_NAV_CAPTURE_R = 0.5     # m: inside, freeze steering + drive straight
_GO2_NAV_TICK_S = 0.05       # 20 Hz drive ticks
_GO2_NAV_SETTLE_S = 0.5      # hold stance + re-sample before deciding arrival
_GO2_NAV_TIMEOUT_S = 60.0
_GO2_NAV_STALL_WINDOW_S = 6.0  # forward-walk no-progress window
_GO2_NAV_STALL_MIN_M = 0.10  # min net progress within the window
# Go2 is a cylinder ~0.34 m front / 0.19 m side (feedback) — inflate isotropic-
# conservative by the front radius so the planned path keeps the body clear.
_GO2_NAV_INFLATION = 0.34
_GO2_NAV_WAYPOINT_TOL = 0.45

# Campaign #11 R5: the shared world-agnostic controller's constant pack for Go2.
from vector_os_nano.hardware.sim._nav_controller import NavConsts  # noqa: E402
_GO2_NAV = NavConsts(
    fall_z=_GO2_NAV_FALL_Z, tol_floor=_GO2_NAV_TOL_FLOOR, speed=_GO2_NAV_SPEED,
    vyaw_max=_GO2_NAV_VYAW_MAX, k_yaw=_GO2_NAV_K_YAW, face_tol=_GO2_NAV_FACE_TOL,
    yaw_deadband=_GO2_NAV_YAW_DEADBAND, capture_r=_GO2_NAV_CAPTURE_R,
    tick_s=_GO2_NAV_TICK_S, settle_s=_GO2_NAV_SETTLE_S, timeout_s=_GO2_NAV_TIMEOUT_S,
    stall_window_s=_GO2_NAV_STALL_WINDOW_S, stall_min_m=_GO2_NAV_STALL_MIN_M,
    inflation=_GO2_NAV_INFLATION, waypoint_tol=_GO2_NAV_WAYPOINT_TOL)


# ---------------------------------------------------------------------------
# Minimal MuJoCo wrapper (replaces convex_mpc.MuJoCo_GO2_Model)
# ---------------------------------------------------------------------------

class _Go2Model:
    """Lightweight wrapper around MjModel/MjData for Go2.

    Caches actuator IDs so set_joint_torque() is fast.
    """

    __slots__ = ("model", "data", "base_bid", "_act_ids", "_robot_geom_ids", "viewer")

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
        # Collect ALL geom IDs belonging to the robot body tree.
        # mj_ray bodyexclude only filters ONE body. We need to filter all
        # robot geoms (trunk + 4 legs × 3 segments = 13+ bodies) to avoid
        # the lidar detecting Go2's own legs as obstacles.
        self._robot_geom_ids: set[int] = set()
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            # Walk up the body tree to check if this geom belongs to robot
            check_bid = bid
            while check_bid > 0:
                if check_bid == self.base_bid:
                    self._robot_geom_ids.add(gid)
                    break
                check_bid = model.body_parentid[check_bid]

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


def _build_room_scene_xml(with_arm: bool | None = None) -> Path:
    """Build composite room scene using local MJCF files.

    Resolves the go2_room.xml template with paths to the Go2 model (with
    optional Piper arm mounted on the back) and assets directory.

    Args:
        with_arm: True = Go2 + Piper arm. False = bare Go2 (no
            manipulation). Both modes run the MPC gait — _mj_update_pin
            slices legs out of the extended qpos so PinGo2Model stays
            happy with its fixed 12-DoF Pinocchio URDF.
            None (default) = read VECTOR_SIM_WITH_ARM env var ("1" → True,
            otherwise False). Lets sim_tool pass the user's choice into
            the MuJoCo subprocess without editing launch_explore.sh.
    """
    import os
    if with_arm is None:
        with_arm = os.environ.get("VECTOR_SIM_WITH_ARM", "0") == "1"
    if with_arm and _GO2_PIPER_XML.exists():
        go2_xml = _GO2_PIPER_XML
        scene_name = "scene_room_piper.xml"
    else:
        go2_xml = _MJCF_DIR / "go2.xml"
        scene_name = "scene_room.xml"
    assets_dir = _MJCF_DIR / "assets"

    template = _ROOM_XML.read_text()
    xml = template.replace("GO2_MODEL_PATH", str(go2_xml))
    xml = xml.replace("GO2_ASSETS_DIR", str(assets_dir))

    out = _MJCF_DIR / scene_name
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
        self, gui: bool = False, room: bool = True, backend: str = "auto",
        viewer_track: bool = True, furnished: bool = False,
        photoreal: bool = False, photoreal_bridge: "Any | None" = None,
    ) -> None:
        self._gui: bool = gui
        self._room: bool = room
        # Photoreal co-sim (campaign #10 R6): when on, get_camera_frame returns a
        # Blender Cycles/OptiX render from the SAME head-cam pose instead of
        # MuJoCo's basic render — the real VLM (and thus photoreal-driven Piper
        # picking + Go2 VLN) grounds a photoreal frame. Default OFF → byte-
        # identical. Shares the furnished-room co-sim factory with the G1 base
        # (rule 7). Bridge is a socket subprocess (safe off the control thread).
        self._photoreal: bool = photoreal
        self._photoreal_bridge = photoreal_bridge
        self._photoreal_renderer = None
        # Furnished VLM room (campaign #9 R2, track C): the SAME shared furnished
        # room g1 uses (g1_room.build_furnished_room_model) on the go2 flat scene
        # — collidable chair/sofa/plant targets for VLM recognition. Proves the
        # world-agnostic invariant (one room builder, two embodiments). Implies
        # room semantics (spawn at origin facing the targets, not the apartment).
        self._furnished: bool = furnished
        self._backend_pref: str = backend
        self._viewer_track: bool = viewer_track
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

        # Viewer drive mode (resolved in connect()): one of "main_thread_pump"
        # (macOS/mjpython), "background_daemon" (Linux/Windows window) or
        # "headless". Decides whether physics runs on the daemon thread or is
        # pumped by the caller's (viewer-owning) thread. See hardware/sim/
        # viewer_mode.py for the platform-aware seam.
        self._drive_mode: str = "headless"
        # Per-step physics state, lifted out of the loop locals so a single
        # step() can advance one increment statefully for the main-thread pump.
        self._tau_hold: np.ndarray = np.zeros(12, dtype=float)
        self._sim_step: int = 0
        self._mpc_U_opt: Any = None
        self._mpc_ctrl_i: int = 0
        self._mpc_steps_per_mpc: int = 1
        self._mpc_dt: float = 0.0

        # Skill-level exclusive control gate. walk()/turn() set this
        # to acquire control for the duration of a motion. During that
        # window, set_velocity() rejects writes from any thread OTHER
        # than the one holding the token — which blocks the 20 Hz bridge
        # path-follower loop (running on the rclpy spin thread) from
        # clobbering skill commands. The skill's own set_velocity() calls
        # pass through because they run on the same thread that acquired
        # the token (tid match).
        self._skill_ctrl_until: float = 0.0
        self._skill_ctrl_tid: int = 0
        # Obstacle polygons for navigate_to/geodesic (campaign #11 M2). Filled in
        # connect() for furnished/room scenes; [] on flat -> navigate_to degrades
        # to a straight-line drive (and the M1 switch path never AttributeErrors).
        self._obstacles: list = []

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

        if self._furnished:
            # Furnished VLM room: the shared world-agnostic builder on the go2
            # flat scene (walls + obstacles + collidable furniture targets). Go2
            # ships its own d435_rgb camera, so no pelvis head-cam is added.
            from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
            base_scene = _build_flat_scene_xml()
            # Mount a forward wide recognition camera on the base (Go2's stock
            # d435 is too low/narrow/down to frame furniture at range).
            model = g1_room.build_furnished_room_model(
                base_scene, recog_cam_body="base_link",
                recog_cam_pos=(0.28, 0.0, 0.06))
            data = mj.MjData(model)
            self._mj = _Go2Model(model, data)
            self._scene_xml_path = str(base_scene)
            # Spawn at the origin facing +x, toward the targets at x≈3.6 (same
            # layout as the g1 furnished room — world-agnostic parity).
            data.qpos[0] = 0.0
            data.qpos[1] = 0.0
            data.qpos[2] = 0.35
            data.qpos[7:19] = _STAND_JOINTS
            if model.nq >= 27:
                data.qpos[19:27] = _PIPER_STOW_QPOS
            if model.nu >= 19:
                data.ctrl[12:19] = _PIPER_STOW_CTRL
        elif self._room:
            scene_path = _build_room_scene_xml()
            model = mj.MjModel.from_xml_path(str(scene_path))
            data = mj.MjData(model)
            self._mj = _Go2Model(model, data)
            # Expose for downstream consumers that need to load an isolated
            # MjModel from the same MJCF (e.g. MuJoCoPiper's IK).
            self._scene_xml_path = str(scene_path)

            # Place Go2 in the entry hall (center of house)
            data.qpos[0] = 10.0
            data.qpos[1] = 3.0
            data.qpos[2] = 0.35
            # Set standing joint angles
            data.qpos[7:19] = _STAND_JOINTS
            # If Piper arm is mounted (nq=27 vs 19), stow it folded upright;
            # otherwise joint2 defaults to 0 and the arm extends horizontally,
            # shifting 1.2kg of link6+ mass forward and tipping the dog.
            if model.nq >= 27:
                data.qpos[19:27] = _PIPER_STOW_QPOS
            if model.nu >= 19:
                data.ctrl[12:19] = _PIPER_STOW_CTRL
        else:
            scene_path = _build_flat_scene_xml()
            model = mj.MjModel.from_xml_path(str(scene_path))
            data = mj.MjData(model)
            self._mj = _Go2Model(model, data)
            self._scene_xml_path = str(scene_path)

            # Apply home keyframe (standing pose at origin)
            key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "stand")
            if key_id >= 0:
                mj.mj_resetDataKeyframe(model, data, key_id)

        # Set physics timestep to 1 kHz
        self._mj.model.opt.timestep = _SIM_DT

        mj.mj_forward(self._mj.model, self._mj.data)

        # navigate_to / geodesic obstacle polygons (campaign #11 M2). The SAME
        # world-agnostic builder G1 uses (wall_*/obstacle_* geoms; target_*
        # furniture is intentionally excluded — the target IS the goal). Flat
        # scenes leave [] -> navigate_to degrades to a straight-line drive.
        if self._furnished or self._room:
            try:
                from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
                self._obstacles = g1_room.obstacles_from_model(
                    self._mj.model, self._mj.data)
            except Exception as _exc:  # noqa: BLE001
                logger.warning("obstacles_from_model failed (flat fallback): %s", _exc)
                self._obstacles = []

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
                    # Show ALL geom groups so the furnished room's group-3
                    # walls/furniture render in the window (else the owner sees
                    # an empty floor — owner-caught 2026-06-14).
                    if self._furnished:
                        self._viewer.opt.geomgroup[:] = 1
                    self._viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
                    if self._furnished:
                        # Frame the spawn + the targets ahead (x≈3.6).
                        self._viewer.cam.lookat[:] = [1.8, 0.0, 0.3]
                        self._viewer.cam.distance = 6.0
                        self._viewer.cam.elevation = -25
                        self._viewer.cam.azimuth = 180
                    elif self._room:
                        self._viewer.cam.lookat[:] = [10.0, 3.0, 0.3]
                        self._viewer.cam.distance = 5.5
                        self._viewer.cam.elevation = -20
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

        # Resolve how the viewer (if any) is driven, then start physics.
        # macOS/mjpython window -> NO background daemon: the caller pumps step()
        # on the viewer-owning (main) thread, because Apple GLFW is main-thread
        # only and a background viewer.sync() would segfault. Linux/Windows
        # window + headless -> background daemon, byte-identical to before.
        from vector_os_nano.hardware.sim.viewer_mode import (  # noqa: PLC0415
            resolve_viewer_drive_mode,
            uses_background_physics,
        )
        self._drive_mode = resolve_viewer_drive_mode(self._viewer is not None)
        self._reset_step_state()
        self._running = True
        if uses_background_physics(self._drive_mode):
            self._physics_thread = threading.Thread(
                target=self._physics_loop, daemon=True, name="mujoco_go2_physics"
            )
            self._physics_thread.start()
        else:
            logger.info(
                "MuJoCoGo2: viewer on the main thread (mjpython) — physics "
                "driven by the caller's step() pump, no background thread"
            )

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
        # Photoreal co-sim: terminate the Blender subprocess (idempotent).
        if self._photoreal_bridge is not None:
            try:
                self._photoreal_bridge.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._photoreal_bridge = None
        self._photoreal_renderer = None
        if getattr(self, "_pano", None) is not None:
            try:
                self._pano.close()      # release the pano GL renderers (M4)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._pano = None

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
        from vector_os_nano.hardware.sim.viewer_mode import (  # noqa: PLC0415
            uses_background_physics,
        )
        # In main-thread-pump mode there is no daemon to restart — the caller
        # keeps pumping step(). Restarting a thread here would reintroduce the
        # off-main-thread GLFW hazard the pump mode exists to avoid.
        if not uses_background_physics(self._drive_mode):
            self._running = True
            return
        self._reset_step_state()
        self._running = True
        self._physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="mujoco_go2_physics"
        )
        self._physics_thread.start()

    def _reset_step_state(self) -> None:
        """(Re)initialize per-step physics state before (re)starting physics.

        Matches the original fresh-locals semantics of the physics loops so the
        background-daemon path is byte-identical; the main-thread pump relies on
        the same state persisting across step() calls.
        """
        self._tau_hold = np.zeros(12, dtype=float)
        self._sim_step = 0
        self._scan_counter = 0
        self._mpc_U_opt = None
        self._mpc_ctrl_i = 0
        if self._use_mpc and self._gait is not None:
            gait_period = self._gait.gait_period
            self._mpc_dt = gait_period / _MPC_DT_FACTOR
            mpc_hz = 1.0 / self._mpc_dt
            self._mpc_steps_per_mpc = max(1, int(_CTRL_HZ // mpc_hz))
        else:
            self._mpc_dt = 0.0
            self._mpc_steps_per_mpc = 1

    def step(self, n: int = 1) -> None:
        """Advance physics by *n* increments on the CALLER's thread.

        This is the main-thread pump used on macOS/mjpython, where GLFW is
        main-thread-only and the background daemon must not touch the viewer.
        Mirrors :meth:`MuJoCoArm.step`. On Linux/Windows/headless the background
        daemon drives the same per-step bodies and callers never pump.
        """
        self._require_connection()
        for _ in range(n):
            if self._use_mpc:
                self._physics_step_mpc()
            else:
                self._physics_step_sinusoidal()

    def _drive_for(self, seconds: float) -> None:
        """Advance ~*seconds* of real time while a blocking motion runs.

        Daemon mode (Linux/Windows/headless): the background thread is already
        stepping the gait, so just sleep. Main-thread-pump mode (macOS/mjpython):
        drive :meth:`step` on THIS (the caller's = the viewer-owning main) thread
        so the gait animates and the viewer syncs on the main thread, instead of
        sleeping while a non-existent daemon would have stepped.
        """
        from vector_os_nano.hardware.sim.viewer_mode import (  # noqa: PLC0415
            uses_background_physics,
        )
        if uses_background_physics(self._drive_mode):
            time.sleep(seconds)
            return
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            self.step()
            sleep_t = _SIM_DT - (time.perf_counter() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Background physics loop
    # ------------------------------------------------------------------

    def _init_mpc_stack(self) -> None:
        """Initialize convex_mpc control stack. Raises ImportError if unavailable.

        Pinocchio is allowed to have *fewer* DoFs than MuJoCo — when an arm
        is mounted (Go2+Piper, MuJoCo nq=27 vs PinGo2 nq=19), _mj_update_pin
        slices the leg portion out of qpos/qvel. Only the reverse is an
        unrecoverable mismatch.
        """
        from convex_mpc.go2_robot_data import PinGo2Model  # noqa: PLC0415
        from convex_mpc.gait import Gait                   # noqa: PLC0415
        from convex_mpc.com_trajectory import ComTraj       # noqa: PLC0415
        from convex_mpc.leg_controller import LegController # noqa: PLC0415

        self._pin = PinGo2Model()
        if self._pin.model.nq > self._mj.model.nq:
            raise RuntimeError(
                f"MPC backend incompatible with scene: "
                f"Pinocchio nq={self._pin.model.nq} > MuJoCo nq={self._mj.model.nq}. "
                f"Loaded MJCF is missing DoFs the Pinocchio model requires."
            )
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
        """Background physics loop (sinusoidal trotting gait, Backend A).

        Drives one :meth:`_physics_step_sinusoidal` per iteration at ~1 kHz on
        the daemon thread (Linux/Windows window + headless).
        """
        while self._running:
            loop_start = time.perf_counter()
            self._physics_step_sinusoidal()
            _phys_diag(float(self._mj.data.time))
            elapsed = time.perf_counter() - loop_start
            sleep_time = _SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _physics_step_sinusoidal(self) -> None:
        """Advance ONE sinusoidal-gait physics increment (+ optional viewer sync).

        Stateful via ``self`` (``_tau_hold`` / ``_sim_step`` / ``_scan_counter``)
        so it can be driven either by the background daemon loop OR by the
        main-thread :meth:`step` pump.
        """
        mj = _get_mujoco()

        with self._cmd_lock:
            vx, vy, vyaw = self._cmd_vel

        time_now = float(self._mj.data.time)
        is_moving = (vx != 0.0 or vy != 0.0 or vyaw != 0.0)

        if self._sim_step % _CTRL_DECIM == 0:
            q_cur = np.array(self._mj.data.qpos[7:19], dtype=np.float64)
            dq_cur = np.array(self._mj.data.qvel[6:18], dtype=np.float64)

            if is_moving:
                q_target = _compute_gait_targets(time_now, vx, vy, vyaw)
            else:
                q_target = np.array(_STAND_JOINTS, dtype=np.float64)

            tau = _KP * (q_target - q_cur) - _KD * dq_cur
            tau = np.clip(tau, -_TAU_LIMITS, _TAU_LIMITS)
            self._tau_hold = tau.copy()

        mj.mj_step1(self._mj.model, self._mj.data)
        self._mj.set_joint_torque(self._tau_hold)
        mj.mj_step2(self._mj.model, self._mj.data)

        self._update_odometry()

        self._scan_counter += 1
        if self._scan_counter >= _LIDAR_UPDATE_INTERVAL:
            self._update_lidar()
            self._scan_counter = 0

        if self._viewer is not None and self._sim_step % _VIEWER_SYNC_EVERY == 0:
            if self._viewer_track:
                pos = self._mj.data.qpos[0:3]
                self._viewer.cam.lookat[:] = [float(pos[0]), float(pos[1]), 0.3]
            self._viewer.sync()

        self._sim_step += 1

    def _physics_loop_mpc(self) -> None:
        """Background physics loop (convex MPC locomotion, Backend B).

        Drives one :meth:`_physics_step_mpc` per iteration at ~1 kHz on the
        daemon thread (Linux/Windows window + headless).
        """
        while self._running:
            loop_start = time.perf_counter()
            self._physics_step_mpc()
            _phys_diag(float(self._mj.data.time))
            elapsed = time.perf_counter() - loop_start
            sleep_time = _SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _physics_step_mpc(self) -> None:
        """Advance ONE convex-MPC physics increment (+ optional viewer sync).

        Ported from the original convex_mpc-based loop: MPC computes optimal
        contact forces, the leg controller converts them to torques. Stateful
        via ``self`` so it can be driven by the daemon loop OR the :meth:`step`
        pump.
        """
        mj = _get_mujoco()

        with self._cmd_lock:
            vx, vy, vyaw = self._cmd_vel

        time_now = float(self._mj.data.time)
        is_moving = (vx != 0.0 or vy != 0.0 or vyaw != 0.0)

        if self._sim_step % _CTRL_DECIM == 0:
            # Update Pinocchio model from MuJoCo state
            self._mj_update_pin()

            if is_moving:
                # MPC locomotion — with solver failure protection
                if self._mpc_ctrl_i % self._mpc_steps_per_mpc == 0:
                    try:
                        self._traj.generate_traj(
                            self._pin, self._gait, time_now,
                            vx, vy, _MPC_Z_DES, vyaw, time_step=self._mpc_dt,
                        )
                        if self._mpc is None:
                            from convex_mpc.centroidal_mpc import CentroidalMPC  # noqa: PLC0415
                            self._mpc = CentroidalMPC(self._pin, self._traj)

                        sol = self._mpc.solve_QP(self._pin, self._traj, False)
                        n = self._traj.N
                        w_opt = sol["x"].full().flatten()
                        self._mpc_U_opt = w_opt[12 * n:].reshape((12, n), order="F")
                        _mpc_diag("qp_ok")
                    except Exception as _exc:  # noqa: BLE001
                        # QP solver failed — hold current torque (PD fallback)
                        _mpc_diag("qp_fail", _exc)

                if self._mpc_U_opt is not None:
                    try:
                        mpc_force = self._mpc_U_opt[:, 0]
                        tau = np.zeros(12, dtype=float)
                        for i, leg in enumerate(_MPC_LEG_NAMES):
                            leg_out = self._leg_ctrl.compute_leg_torque(
                                leg, self._pin, self._gait,
                                mpc_force[i * 3:(i + 1) * 3], time_now,
                            )
                            tau[i * 3:(i + 1) * 3] = leg_out.tau
                        tau = np.clip(tau, -_MPC_TAU_LIMITS, _MPC_TAU_LIMITS)
                        self._tau_hold = tau.copy()
                    except Exception as _exc:  # noqa: BLE001
                        _mpc_diag("tau_fail", _exc)
            else:
                # Idle: PD hold standing posture
                q_cur = np.array(self._mj.data.qpos[7:19], dtype=np.float64)
                dq_cur = np.array(self._mj.data.qvel[6:18], dtype=np.float64)
                q_stand = np.array(_STAND_JOINTS, dtype=np.float64)
                tau = _KP * (q_stand - q_cur) - _KD * dq_cur
                tau = np.clip(tau, -_TAU_LIMITS, _TAU_LIMITS)
                self._tau_hold = tau.copy()

            self._mpc_ctrl_i += 1

        mj.mj_step1(self._mj.model, self._mj.data)
        self._mj.set_joint_torque(self._tau_hold)
        mj.mj_step2(self._mj.model, self._mj.data)

        self._update_odometry()

        self._scan_counter += 1
        if self._scan_counter >= _LIDAR_UPDATE_INTERVAL:
            self._update_lidar()
            self._scan_counter = 0

        if self._viewer is not None and self._sim_step % _VIEWER_SYNC_EVERY == 0:
            if self._viewer_track:
                pos = self._mj.data.qpos[0:3]
                self._viewer.cam.lookat[:] = [float(pos[0]), float(pos[1]), 0.3]
            self._viewer.sync()

        self._sim_step += 1

    def _mj_update_pin(self) -> None:
        """Sync Pinocchio model state from MuJoCo qpos/qvel.

        Converts MuJoCo (wxyz quaternion, world-frame linear vel) to
        Pinocchio (xyzw quaternion, body-frame linear vel) and runs
        the full set of Pinocchio computations the MPC solver needs.

        When MuJoCo carries extra DoFs beyond the PinGo2 model (e.g. a
        mounted Piper arm adds 8 DoFs), the leg segment is sliced out —
        MuJoCo qpos layout is [base(7), legs(12), arm(...)], which matches
        the Pinocchio URDF exactly for the first 19 entries.
        """
        mujoco_q = np.asarray(self._mj.data.qpos, dtype=float).reshape(-1)
        mujoco_dq = np.asarray(self._mj.data.qvel, dtype=float).reshape(-1)

        qw, qx, qy, qz = mujoco_q[3:7]

        import pinocchio as pin  # noqa: PLC0415
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        v_body = R.T @ mujoco_dq[0:3]
        w_body = mujoco_dq[3:6]

        n_leg_q = self._pin.model.nq - 7   # legs-only qpos count (12 for Go2)
        n_leg_v = self._pin.model.nv - 6   # legs-only qvel count (12 for Go2)
        q_pin = np.concatenate([mujoco_q[0:3], [qx, qy, qz, qw], mujoco_q[7:7 + n_leg_q]])
        dq_pin = np.concatenate([v_body, w_body, mujoco_dq[6:6 + n_leg_v]])

        self._pin.update_model(q_pin, dq_pin)

    # ------------------------------------------------------------------
    # Velocity command (non-blocking)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set target body velocity. Non-blocking.

        Skill-exclusive gate: if a skill holds the control token
        (self._skill_ctrl_until in the future), calls from OTHER threads
        are silently ignored — this blocks the bridge path-follower /
        /cmd_vel_nav callback / safety-check from overriding a walk()
        or turn() in progress. The token holder (same thread) passes.
        """
        self._require_connection()
        _gated = (time.time() < self._skill_ctrl_until
                  and threading.get_ident() != self._skill_ctrl_tid)
        if _gated:
            return
        with self._cmd_lock:
            self._cmd_vel = (
                float(np.clip(vx, -_VX_MAX, _VX_MAX)),
                float(np.clip(vy, -_VY_MAX, _VY_MAX)),
                float(np.clip(vyaw, -_VYAW_MAX, _VYAW_MAX)),
            )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def reset_pose(self) -> None:
        """Reset robot to standing pose at current XY position.

        Fixes tip-overs without restarting the simulation. Keeps the robot
        at its current (x, y) but resets z, orientation, joint angles, and
        all velocities to the default standing state.
        """
        self._require_connection()
        import mujoco as mj
        data = self._mj.data
        # Keep current XY, reset everything else
        cur_x, cur_y = float(data.qpos[0]), float(data.qpos[1])
        data.qpos[0] = cur_x
        data.qpos[1] = cur_y
        data.qpos[2] = 0.35                    # standing height
        data.qpos[3:7] = [1, 0, 0, 0]          # upright quaternion (w,x,y,z)
        data.qpos[7:19] = _STAND_JOINTS         # standing joint angles
        # If Piper arm is mounted, set it to the stow pose too (nq=27 vs 19)
        if self._mj.model.nq >= 27:
            data.qpos[19:27] = _PIPER_STOW_QPOS
        data.qvel[:] = 0                         # zero all velocities
        data.ctrl[:] = 0                         # zero all actuators
        # Likewise drive Piper position actuators to stow (nu=19 vs 12)
        if self._mj.model.nu >= 19:
            data.ctrl[12:19] = _PIPER_STOW_CTRL
        mj.mj_forward(self._mj.model, data)

    def get_sim_time(self) -> float:
        """Return the MuJoCo simulation clock (seconds).

        The physics daemon advances this as it steps; it runs at whatever
        fraction of wall-clock the machine sustains (often <1x). Consumed by
        the ROS2 bridge's path follower to integrate its velocity ramps
        against sim-dt (wall-tick ramps slew too fast in gait time when
        sim/wall<1 and destabilize it); also usable as a ``/clock`` source if
        the nav stack ever moves to use_sim_time=true.
        """
        self._require_connection()
        return float(self._mj.data.time)

    def get_position(self) -> list[float]:
        """Return base position [x, y, z] in world frame."""
        self._require_connection()
        return list(self._mj.data.qpos[0:3].astype(float))

    # Seek step (campaign #9 R2): Go2's walk() is BLOCKING (drives the full
    # duration then stops), so — unlike G1's async deadman — there is no
    # mid-call stutter to bridge; a short step suffices and keeps the approach
    # well-sampled. VlmSeekSkill reads this hint (G1 leaves the 3.0 s default).
    seek_step_duration: float = 1.2

    def list_targets(self) -> "dict[str, tuple[float, float]]":
        """Labeled furniture targets in the furnished room ({name: (x, y)}),
        empty otherwise. The deterministic at_position verify anchor — the same
        world-agnostic furniture coordinates the g1 furnished room exposes."""
        if not self._furnished:
            return {}
        from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
        return g1_room.furnished_targets()

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

    def get_camera_frame(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Render first-person RGB from d435_rgb camera mounted on Go2 head.

        Returns an (H, W, 3) uint8 numpy array in RGB order.
        Uses the named 'd435_rgb' camera defined in the MJCF model, which is
        fixed to base_link. This gives the exact same view as a real D435
        mounted on the robot — no free-camera approximation.
        """
        self._require_connection()
        mj = _get_mujoco()

        if not hasattr(self, "_cam_renderer"):
            self._cam_renderer = mj.Renderer(self._mj.model, height, width)
            self._cam_renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
            self._cam_renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = True
            # Furnished VLM room: the shared room builder tags walls/furniture
            # with ENV_GEOM_GROUP=3, which the DEFAULT render option hides — the
            # VLM would see an empty floor (the g1 R9 / Case-from-R9 bug). Enable
            # all geom groups so the furniture renders. Scoped to furnished so
            # the apartment go2 view is byte-identical.
            if self._furnished:
                self._cam_opt = mj.MjvOption()
                self._cam_opt.geomgroup[:] = 1

        # Furnished VLM room uses the forward wide recognition camera; otherwise
        # the stock d435_rgb (apartment / depth pipeline unchanged).
        cam_name = "d435_rgb"
        if self._furnished:
            from vector_os_nano.hardware.sim.g1_room import RECOG_CAM  # noqa: PLC0415,E501
            if mj.mj_name2id(self._mj.model, mj.mjtObj.mjOBJ_CAMERA, RECOG_CAM) >= 0:
                cam_name = RECOG_CAM
        cam_id = self._mj.model.cam(cam_name).id
        if self._photoreal:
            return self._photoreal_frame(cam_id, cam_name)
        if getattr(self, "_cam_opt", None) is not None:
            self._cam_renderer.update_scene(
                self._mj.data, camera=cam_id, scene_option=self._cam_opt)
        else:
            self._cam_renderer.update_scene(self._mj.data, camera=cam_id)
        return self._cam_renderer.render().copy()

    def _ensure_photoreal_renderer(self, cam_name: str):
        if self._photoreal_renderer is None:
            if self._furnished:
                from vector_os_nano.playground.photoreal.cosim import (  # noqa: PLC0415,E501
                    furnished_room_renderer)
                renderer, bridge = furnished_room_renderer(
                    cam_name=cam_name, bridge=self._photoreal_bridge,
                    width=640, height=480, samples=48)
            else:
                # Manipulation scene: render the table + the live graspable
                # objects as photoreal primitives (campaign #10 R7).
                from vector_os_nano.playground.photoreal.cosim import (  # noqa: PLC0415,E501
                    build_pick_scene_spec, pick_object_texture, scene_renderer)
                objs = self._pick_objects()
                # R14: TEXTURE the physics cylinders (a product label) instead of
                # substituting a bottle MESH (R12/R13). The rendered object keeps
                # the cylinder's exact SHAPE, so the VLM bbox stays aligned with the
                # MuJoCo depth (R13 root cause: a mesh != the physics cylinder broke
                # depth-at-bbox). A textured cylinder is VLM-detectable as a 'can'
                # where a bare colour primitive garbled it (R11).
                tex = pick_object_texture()
                if tex:
                    for o in objs:
                        o["texture"] = tex
                # R15: render only ONE graspable object (the 3 cylinders sit 15 cm
                # apart and confuse the VLM — unstable bbox -> bad ray-to-plane
                # locate). A single object gives a clean, stable detection. The
                # other physics objects still exist; we just present one to grasp.
                if len(objs) > 1 and os.environ.get("VECTOR_PICK_SINGLE", "1") != "0":
                    # Optionally present a NAMED object (VECTOR_PICK_SINGLE_LABEL,
                    # substring of the body name) so the scene matches the query
                    # (e.g. 'can' -> pickable_can_red). This only chooses what to
                    # RENDER; the VLM still must find + locate it (rule 5).
                    want = os.environ.get("VECTOR_PICK_SINGLE_LABEL", "").strip().lower()
                    if want:
                        match = [o for o in objs if want in o.get("name", "").lower()]
                        objs = match[:1] or objs[:1]
                    else:
                        objs = objs[:1]
                spec = build_pick_scene_spec(objs, table=self._pick_table())
                renderer, bridge = scene_renderer(
                    spec, cam_name=cam_name, bridge=self._photoreal_bridge,
                    width=640, height=480, samples=48)
            self._photoreal_bridge = bridge
            self._photoreal_renderer = renderer
        return self._photoreal_renderer

    def _body_first_geom(self, bid: int):
        mj = _get_mujoco()
        for gid in range(self._mj.model.ngeom):
            if int(self._mj.model.geom_bodyid[gid]) == bid:
                return gid, mj
        return None, mj

    def _pick_objects(self) -> list:
        """Enumerate graspable ``pickable_*`` bodies as photoreal primitives at
        their LIVE world poses (type/size/colour read from the model — no
        hardcoding). The VLM still must FIND them in the image (rule 5)."""
        mj = _get_mujoco()
        m, d = self._mj.model, self._mj.data
        objs = []
        for bid in range(m.nbody):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid)
            if not name or not name.startswith("pickable_"):
                continue
            gid, _ = self._body_first_geom(bid)
            if gid is None:
                continue
            gtype = int(m.geom_type[gid])
            is_cyl = gtype == int(mj.mjtGeom.mjGEOM_CYLINDER)
            size = [float(v) for v in m.geom_size[gid]]
            objs.append({
                "name": name,
                "type": "cylinder" if is_cyl else "box",
                "pos": [float(v) for v in d.xpos[bid]],
                "size": size[:2] if is_cyl else size[:3],
                "color": [float(v) for v in m.geom_rgba[gid][:3]],
            })
        return objs

    def _pick_table(self) -> "dict | None":
        mj = _get_mujoco()
        m, d = self._mj.model, self._mj.data
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "pick_table")
        if bid < 0:
            return None
        gid, _ = self._body_first_geom(bid)
        if gid is None:
            return None
        # the table geom sits at body pos + its local geom offset (z lift)
        gpos = [float(v) for v in d.geom_xpos[gid]]
        return {"pos": gpos, "scale": [float(v) for v in m.geom_size[gid]],
                "color": [0.55, 0.40, 0.25]}

    def _photoreal_frame(self, cam_id: int, cam_name: str) -> "np.ndarray":
        """Photoreal RGB from the head cam's LIVE pose (hybrid co-sim — rgb from
        Blender, the rest of the pipeline unchanged). Off the control thread is
        fine: the Blender render is a socket subprocess, no GL in our process."""
        renderer = self._ensure_photoreal_renderer(cam_name)
        cam_pos = self._mj.data.cam_xpos[cam_id]
        cam_mat = self._mj.data.cam_xmat[cam_id]
        fovy = float(self._mj.model.cam_fovy[cam_id])
        return renderer.render_from_pose(cam_pos, cam_mat, fovy)

    def get_depth_frame(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Render depth from d435_depth camera mounted on Go2 head.

        Returns an (H, W) float32 numpy array in metres. Uses the named
        'd435_depth' camera — same mounting as RGB for pixel alignment.
        """
        self._require_connection()
        mj = _get_mujoco()

        if not hasattr(self, "_depth_renderer"):
            self._depth_renderer = mj.Renderer(self._mj.model, height, width)
            self._depth_renderer.enable_depth_rendering()
            self._depth_renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True

        cam_id = self._mj.model.cam("d435_depth").id
        self._depth_renderer.update_scene(self._mj.data, camera=cam_id)
        raw = self._depth_renderer.render().copy()

        import numpy as np
        depth = raw.astype(np.float32)
        depth[(depth < 0.1) | (depth > 10.0)] = 0.0
        return depth

    def get_camera_pose(self) -> tuple:
        """Return (cam_xpos, cam_xmat) for the d435_rgb camera.

        cam_xpos: (3,) world position
        cam_xmat: (9,) rotation matrix (row-major, reshape to 3x3)

        Used by depth_projection.camera_to_world for exact transforms.
        """
        self._require_connection()
        cam_id = self._mj.model.cam("d435_rgb").id
        return (
            self._mj.data.cam_xpos[cam_id].copy(),
            self._mj.data.cam_xmat[cam_id].copy(),
        )

    def get_rgbd_frame(
        self, width: int = 640, height: int = 480,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Render aligned RGB + depth from the same camera pose.

        Returns (rgb, depth) where:
            rgb: (H, W, 3) uint8 array
            depth: (H, W) float32 array in metres

        Simulates RealSense D435 aligned_depth_to_color output.
        """
        rgb = self.get_camera_frame(width, height)
        depth = self.get_depth_frame(width, height)
        return rgb, depth

    # ------------------------------------------------------------------
    # Eye-in-hand grasp perception (campaign #10 DQ-13)
    #
    # The forward d435 sees an in-reach object at a shallow grazing angle, so a
    # bbox-ray to the support plane overshoots by ~0.2-0.3 m (R15-R18). The Piper
    # carries a downward wrist camera (piper_wrist_rgb/_depth on link6, optical
    # axis = link6 +z = world -Z at a top-down pose). Observing from an overhead
    # SCAN POSE makes the ray near-vertical -> overshoot collapses. The scan point
    # and support height come ONLY from the table geometry (a scene constant), never
    # any pickable_* pose (rule 5). These are additive, optional, duck-typed methods
    # on the concrete world — kernel/BaseProtocol unchanged (rules 2, 7).
    # ------------------------------------------------------------------

    _WRIST_RGB_CAM = "piper_wrist_rgb"
    _WRIST_DEPTH_CAM = "piper_wrist_depth"

    def _has_wrist_cam(self) -> bool:
        import mujoco as mj  # noqa: PLC0415
        return mj.mj_name2id(
            self._mj.model, mj.mjtObj.mjOBJ_CAMERA, self._WRIST_RGB_CAM) >= 0

    def get_support_z(self) -> "float | None":
        """World z of the pick table's TOP surface — the grasp support plane.

        A SCENE constant (the table the robot picks from), read from the
        ``pick_table`` geometry, NOT any object's ground-truth pose (rule 5).
        Returned to the skill as ``support_z`` so ``locate_on_plane`` casts the
        wrist-cam bbox ray onto the correct height. None if no pick table.
        """
        self._require_connection()
        table = self._pick_table()
        if table is None:
            return None
        return float(table["pos"][2] + table["scale"][2])

    def get_scan_pose(self, scan_height: float = 0.25) -> "tuple | None":
        """Overhead observation point above the table centre: ``(cx, cy,
        table_top + scan_height)``. The arm is driven here (top-down) so the
        downward wrist cam frames the workspace from near-vertical. Derived
        ONLY from the table geometry — the object's xy is still found purely by
        the VLM + ``locate_on_plane`` (rule 5). None if no pick table.

        scan_height=0.25 was chosen from a real-sim reachability sweep (DQ-13
        STEP 0): reachable from the grasp standoff, ~10 deg off vertical (vs the
        forward cam's ~70 deg), and frames the full +-0.15 m object span at
        fovy=58.
        """
        self._require_connection()
        table = self._pick_table()
        if table is None:
            return None
        cx, cy = float(table["pos"][0]), float(table["pos"][1])
        top_z = float(table["pos"][2] + table["scale"][2])
        return (cx, cy, top_z + float(scan_height))

    def get_grasp_observation(
        self, width: int = 640, height: int = 480,
    ) -> "dict | None":
        """A ``{rgb, depth, cam_pos, cam_mat, fovy}`` observation from the
        downward wrist camera at its LIVE pose. RGB is photoreal (Blender
        co-sim via render_from_pose) when ``photoreal`` is on, else MuJoCo;
        depth is MuJoCo from the pixel-aligned ``piper_wrist_depth`` twin.

        None if the model has no wrist camera (bare-Go2 / no-arm scene) so
        callers fall back to the forward-cam path. cam_pos/cam_mat are read
        WITHOUT mj_forward (the pose is kept current by the physics thread —
        a main-thread mj_forward on live data races it: the R13 segfault).
        """
        self._require_connection()
        import mujoco as mj  # noqa: PLC0415
        if not self._has_wrist_cam():
            return None
        cam_id = self._mj.model.cam(self._WRIST_RGB_CAM).id

        if self._photoreal:
            rgb = self._photoreal_frame(cam_id, self._WRIST_RGB_CAM)
        else:
            if not hasattr(self, "_cam_renderer"):
                self._cam_renderer = mj.Renderer(self._mj.model, height, width)
                self._cam_renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
            self._cam_renderer.update_scene(self._mj.data, camera=cam_id)
            rgb = self._cam_renderer.render().copy()

        # Depth twin (MuJoCo) — pixel-aligned. Best-effort: the scan path drives
        # locate_on_plane (support_z), which needs no depth, but provide it for
        # the generic _observe contract / depth fallback.
        depth = None
        if mj.mj_name2id(self._mj.model, mj.mjtObj.mjOBJ_CAMERA,
                         self._WRIST_DEPTH_CAM) >= 0:
            if not hasattr(self, "_grasp_depth_renderer"):
                self._grasp_depth_renderer = mj.Renderer(self._mj.model, height, width)
                self._grasp_depth_renderer.enable_depth_rendering()
            ddid = self._mj.model.cam(self._WRIST_DEPTH_CAM).id
            self._grasp_depth_renderer.update_scene(self._mj.data, camera=ddid)
            import numpy as np  # noqa: PLC0415
            depth = self._grasp_depth_renderer.render().copy().astype(np.float32)
            depth[(depth < 0.05) | (depth > 5.0)] = 0.0

        return {
            "rgb": rgb,
            "depth": depth,
            "cam_pos": self._mj.data.cam_xpos[cam_id].copy(),
            "cam_mat": self._mj.data.cam_xmat[cam_id].copy(),
            "fovy": float(self._mj.model.cam_fovy[cam_id]),
        }

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
        """Cast rays in multiple elevation rings — Livox MID360-like 3D lidar.

        The MID360 is mounted tilted 30 degrees forward (pitch down).
        This means the lidar's "horizontal" plane is actually 30° below
        horizontal, so it sees the ground in front and walls ahead.
        """
        from vector_os_nano.core.types import LaserScan  # noqa: PLC0415
        mj = _get_mujoco()

        # Sensor mounting: on top of Go2 head — above all leg geoms.
        # 0.3m forward (head position) + 0.2m up (above trunk top).
        # At -20° tilt, nearest ground hit ≈ 0.9m ahead of lidar →
        # well past front legs (~0.1m ahead of lidar). No self-hits.
        # Must match bridge _SENSOR_X/_SENSOR_Z and nav stack sensorOffset.
        _LIDAR_OFFSET_X = 0.3
        _LIDAR_OFFSET_Z = 0.2

        pos = self._mj.data.qpos[0:3].copy().astype(np.float64)
        heading = self.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        # Lidar position in world frame = base + rotated offset
        pos_lidar = np.array([
            float(pos[0]) + cos_h * _LIDAR_OFFSET_X,
            float(pos[1]) + sin_h * _LIDAR_OFFSET_X,
            float(pos[2]) + _LIDAR_OFFSET_Z,
        ], dtype=np.float64)

        robot_body_id = self._mj.base_bid

        # Scan beam tilt: 20° downward from sensor horizontal plane
        # (sensor frame itself is NOT tilted — only the beams are)
        tilt_rad = math.radians(-20.0)
        cos_tilt = math.cos(tilt_rad)
        sin_tilt = math.sin(tilt_rad)

        # Livox MID360 FOV: -7° to +52° (asymmetric, 59° range)
        # With 20° downward tilt → world frame: -27° to +32°
        # This gives both ground hits (below horizontal) and wall hits (above)
        n_azimuth = 360
        elevations = list(range(-8, 53, 2))  # -8° to +52° in 2° steps, includes 0° for 2D scan
        mid_ring_ranges: list[float] = []
        points_3d: list[tuple[float, float, float, float]] = []

        for elev_deg in elevations:
            elev_rad = math.radians(elev_deg)
            cos_elev = math.cos(elev_rad)
            sin_elev = math.sin(elev_rad)
            azimuth_step = 360.0 / n_azimuth
            for i in range(n_azimuth):
                azimuth = heading + math.radians(i * azimuth_step - 180)

                # Ray direction in world frame (no tilt yet)
                dx_w = cos_elev * math.cos(azimuth)
                dy_w = cos_elev * math.sin(azimuth)
                dz_w = sin_elev

                # World → body frame (rotate by -heading around Z)
                dx_b = dx_w * cos_h + dy_w * sin_h    # forward
                dy_b = -dx_w * sin_h + dy_w * cos_h   # left
                dz_b = dz_w                            # up

                # Apply pitch tilt in body frame (rotate around body Y axis)
                # Forward points down, backward points up
                dx_bt = dx_b * cos_tilt - dz_b * sin_tilt
                dz_bt = dx_b * sin_tilt + dz_b * cos_tilt

                # Body → world frame (rotate by +heading)
                direction = np.array([
                    dx_bt * cos_h - dy_b * sin_h,
                    dx_bt * sin_h + dy_b * cos_h,
                    dz_bt,
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
                # Skip self-hits: mj_ray bodyexclude only filters the trunk
                # body. Leg geoms (hip/thigh/calf) are separate bodies and
                # can be hit by rays pointing downward/forward. Filter them
                # using the pre-built robot geom set.
                if dist > 0 and dist < 12.0 and int(geom_id[0]) not in self._mj._robot_geom_ids:
                    px = pos_lidar[0] + dist * direction[0]
                    py = pos_lidar[1] + dist * direction[1]
                    pz = pos_lidar[2] + dist * direction[2]
                    points_3d.append((float(px), float(py), float(pz), 0.0))

                if elev_deg == 0:
                    # Self-hit → treat as no hit (inf range) for LaserScan too
                    if dist > 0 and int(geom_id[0]) not in self._mj._robot_geom_ids:
                        mid_ring_ranges.append(float(dist))
                    else:
                        mid_ring_ranges.append(float("inf"))

        self._last_scan = LaserScan(
            timestamp=float(self._mj.data.time),
            angle_min=-math.pi,
            angle_max=math.pi,
            angle_increment=math.radians(azimuth_step),
            range_min=0.1,
            range_max=12.0,
            ranges=tuple(mid_ring_ranges),
            # World-frame return cloud (x,y,z,intensity) so recognize_navigate's
            # bearing+range locate has real points (campaign #11 M2). points_3d
            # is already world-frame (pos_lidar + dist*world_direction above).
            points=tuple(points_3d),
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

    def stand(self, duration: float = 2.0) -> bool:
        self._require_connection()
        self._pd_interpolate(
            np.array(_STAND_JOINTS, dtype=np.float64), duration=duration
        )
        return True

    def sit(self, duration: float = 2.0) -> bool:
        self._require_connection()
        self._pd_interpolate(
            np.array(_SIT_JOINTS, dtype=np.float64), duration=duration
        )
        return True

    def lie_down(self, duration: float = 2.0) -> bool:
        self._require_connection()
        self._pd_interpolate(
            np.array(_LIE_DOWN_JOINTS, dtype=np.float64), duration=duration
        )
        return True

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

        Acquires skill-level control authority for `duration + 0.3s` so a
        concurrently running bridge path-follower yields. Released on exit.

        Returns:
            True if completed without falling over.
        """
        self._require_connection()
        self._skill_ctrl_tid = threading.get_ident()
        self._skill_ctrl_until = time.time() + duration + 0.3
        try:
            self.set_velocity(vx, vy, vyaw)
            self._drive_for(duration)
            self.set_velocity(0.0, 0.0, 0.0)
            self._drive_for(0.2)  # settle
            pos = self.get_position()
            return bool(pos[2] > 0.15)
        finally:
            self._skill_ctrl_until = 0.0
            self._skill_ctrl_tid = 0

    def navigate_to(self, x: float, y: float, tol: float = 0.2,
                    speed: float = _GO2_NAV_SPEED) -> dict:
        """Drive to world (x, y), routing AROUND obstacles in room mode.

        Quadruped sibling of MuJoCoG1.navigate_to (campaign #11 M2): same
        g1_vgraph routing + face->walk->capture->settle controller, returning
        the IDENTICAL three-value dict so NavigateToPointSkill / recognize_navigate
        consume it unchanged. Flat scene (no obstacles) -> straight-line drive.
        An honest 'unreachable' (goal boxed in) returns reached=False, never a
        phantom straight line (rule 5)."""
        self._require_connection()
        # Shared world-agnostic controller (campaign #11 R5). Go2 passes its
        # skill-ctrl token factory (set_velocity is thread-gated) + _drive_for.
        from vector_os_nano.hardware.sim import _nav_controller  # noqa: PLC0415
        if not self._obstacles:
            return _nav_controller.drive_to_point(
                x=x, y=y, tol=tol, speed=speed,
                get_position=self.get_position, get_heading=self.get_heading,
                set_velocity=self.set_velocity, stop=self.stop,
                tick_fn=self._drive_for, consts=_GO2_NAV, ctrl_token=self._ctrl_token)
        return _nav_controller.route_and_drive(
            x=x, y=y, tol=tol, speed=speed, obstacles=self._obstacles,
            get_position=self.get_position, get_heading=self.get_heading,
            set_velocity=self.set_velocity, stop=self.stop,
            tick_fn=self._drive_for, consts=_GO2_NAV, ctrl_token=self._ctrl_token)

    def _go2_navigate_point(self, x: float, y: float, tol: float = 0.2,
                            speed: float = _GO2_NAV_SPEED) -> dict:
        """Single-point drive — delegates to the shared controller with Go2's
        ctrl-token factory (campaign #11 R5; kept for callers/tests)."""
        self._require_connection()
        from vector_os_nano.hardware.sim import _nav_controller  # noqa: PLC0415
        return _nav_controller.drive_to_point(
            x=x, y=y, tol=tol, speed=speed,
            get_position=self.get_position, get_heading=self.get_heading,
            set_velocity=self.set_velocity, stop=self.stop,
            tick_fn=self._drive_for, consts=_GO2_NAV, ctrl_token=self._ctrl_token)

    @contextlib.contextmanager
    def _ctrl_token(self):
        """Acquire skill control authority on THIS thread for the duration (else
        the bridge path-follower's set_velocity overrides the nav drive). Released
        on exit — never fences the bridge. Per-leg acquire/release (route_and_drive
        enters this each leg, matching the pre-R5 semantics)."""
        self._skill_ctrl_tid = threading.get_ident()
        self._skill_ctrl_until = (time.time() + _GO2_NAV_TIMEOUT_S
                                  + _GO2_NAV_SETTLE_S + 1.0)
        try:
            yield
        finally:
            self._skill_ctrl_until = 0.0
            self._skill_ctrl_tid = 0

    def geodesic_distance(self, a: "list[float]", b: "list[float]") -> float:
        """Planar distance a->b: visibility-graph path length around the room
        obstacles (the SAME planner navigate_to walks — rule 5), or straight-line
        on a flat scene. inf when boxed in."""
        from vector_os_nano.hardware.sim import _nav_controller  # noqa: PLC0415
        return _nav_controller.path_length_geodesic(
            a, b, getattr(self, "_obstacles", None) or [], _GO2_NAV.inflation)

    def get_pano(self) -> dict:
        """Pose-synced equirect panorama for SysNav (campaign #11 M4 feed source).

        Same contract as HabitatBase.get_pano / MuJoCoG1.get_pano — {rgb
        (640,1920,3) uint8, depth (640,1920) f32, pos[3], heading} — so the
        embodiment-agnostic SysNav feed drives Go2 too. Real MuJoCoPano360 render
        (rule 5: sim pixels, never GT)."""
        self._require_connection()
        if getattr(self, "_pano", None) is None:
            from vector_os_nano.hardware.sim.sensors.pano360 import MuJoCoPano360  # noqa: PLC0415,E501
            self._pano = MuJoCoPano360(
                self._mj.model, self._mj.data, body_name="base_link",
                offset=(0.0, 0.0, 0.1))   # just above the trunk (dog eye height)
        # Pause the 50Hz physics daemon during the cube-face render so update_scene
        # does NOT read mjData mid-mj_step (R8 review: off-thread render data race /
        # segfault risk). Resume in finally — never leave physics paused.
        self._pause_physics()
        try:
            rgb, depth = self._pano.render_rgbd()
        finally:
            self._resume_physics()
        pos = self.get_position()
        return {"rgb": rgb, "depth": depth,
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "heading": float(self.get_heading())}
