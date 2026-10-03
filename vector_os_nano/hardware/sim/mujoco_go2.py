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
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.embodiments.dof_layout import DofLayout
from vector_os_nano.hardware.sim.mujoco_g1 import obstacles_from_model

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

# ---------------------------------------------------------------------------
# Constants — obstacle-aware navigate_to
# ---------------------------------------------------------------------------

# Go2 is ~0.19m half-width / ~0.34m half-length; 0.28m is a safe planning
# radius that clears furniture corners by the gait's natural COM oscillation.
_GO2_BODY_RADIUS: float = 0.28

# Proportional heading gain and limits (mirrors g1 nav tuning, go2 is more agile)
_GO2_NAV_K_YAW: float = 2.0
_GO2_NAV_VYAW_MAX: float = 0.8
_GO2_NAV_FACE_TOL: float = 0.40       # rad: pivot until aligned within this
_GO2_NAV_YAW_DEADBAND: float = 0.12   # rad: stop correcting when aligned
_GO2_NAV_CAPTURE_R: float = 0.45      # m: inside → freeze steering
_GO2_NAV_WP_CAPTURE: float = 0.45     # m: intermediate waypoint capture tolerance
_GO2_NAV_TIMEOUT_S: float = 60.0      # default timeout (s) for the full path
_GO2_NAV_STEP_S: float = 0.5          # seconds per walk step


# ---------------------------------------------------------------------------
# Minimal MuJoCo wrapper (replaces convex_mpc.MuJoCo_GO2_Model)
# ---------------------------------------------------------------------------

class _Go2Model:
    """Lightweight wrapper around MjModel/MjData for Go2.

    Caches actuator IDs so set_joint_torque() is fast.
    """

    __slots__ = ("model", "data", "base_bid", "_act_ids", "_robot_geom_ids", "viewer", "layout")

    def __init__(self, model: Any, data: Any) -> None:
        mj = _get_mujoco()
        self.model = model
        self.data = data
        self.viewer = None
        # Generic DoF-layout introspection (Rule 11): the base freejoint's qpos/
        # qvel slice addresses come from DofLayout, not hardcoded literals. For
        # base_link (the first freejoint in the room scene) this yields
        # root_qpos_adr=0 / joint_qpos_start=7 — identical to the old literals.
        self.layout: DofLayout = DofLayout(model, "base_link", 12)
        self.base_bid: int = self.layout.root_bid
        # Cache actuator IDs: FL_hip, FL_thigh, FL_calf, FR..., RL..., RR...
        self._act_ids: list[int] = []
        for leg in ("FL", "FR", "RL", "RR"):
            for joint in ("hip", "thigh", "calf"):
                self._act_ids.append(
                    mj.mj_name2id(
                        model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{joint}"
                    )
                )
        # ALL geom IDs in the robot body tree (lidar self-filter): mj_ray's
        # bodyexclude filters only ONE body, so we need the whole subtree.
        # DofLayout.robot_geom_ids is the same subtree walk that used to be
        # inlined here.
        self._robot_geom_ids: set[int] = self.layout.robot_geom_ids

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


def _build_room_scene_xml(with_arm: bool = False) -> Path:
    """Build composite room scene using local MJCF files.

    Resolves the go2_room.xml template with paths to the Go2 model (with
    optional Piper arm mounted on the back) and assets directory.

    Args:
        with_arm: True = Go2 + Piper arm. False = bare Go2 (no
            manipulation). Both modes run the MPC gait — _mj_update_pin
            slices legs out of the extended qpos so PinGo2Model stays
            happy with its fixed 12-DoF Pinocchio URDF. The choice comes from
            the embodiment manifest (go2 vs go2_piper), never from the env.
    """
    from vector_os_nano.hardware.sim.scene_builder import build_room_scene
    assets_dir = _MJCF_DIR / "assets"
    if with_arm and not _GO2_PIPER_XML.exists():
        # Fail loud: a silent bare-dog fallback would hand the planner a body
        # with no arm while the user asked for one (Rule 8).
        raise FileNotFoundError(f"go2+arm scene model missing: {_GO2_PIPER_XML}")
    if with_arm:
        go2_xml = _GO2_PIPER_XML
        scene_name = "scene_room_piper.xml"
        # go2_piper.xml carries mesh paths relative to its own directory;
        # build_room_scene resolves them (meshes_dir=None = model-file dir).
        robot_meshes_dir = None
        # Grasp welds (name, body1, body2) — only with the arm (the no-arm model
        # has no piper_link6; a weld to a missing body fails to compile).
        # body1=piper_link6 (wrist carrying piper_ee_site + fingers), body2=each
        # pickable; created INACTIVE — the gripper's _try_grasp activates one on
        # close. This is what lets a grasp grade GROUNDED (the object physically
        # attaches + lifts; mirrors so101_mujoco.xml).
        welds: tuple[tuple[str, str, str], ...] = (
            ("grasp_pickable_bottle_blue", "piper_link6", "pickable_bottle_blue"),
            ("grasp_pickable_bottle_green", "piper_link6", "pickable_bottle_green"),
            ("grasp_pickable_can_red", "piper_link6", "pickable_can_red"),
        )
    else:
        go2_xml = _MJCF_DIR / "go2.xml"
        scene_name = "scene_room.xml"
        # bare go2.xml uses relative meshdir="assets" — make absolute on attach.
        robot_meshes_dir = assets_dir
        welds = ()

    # ONE scene-build path (Rule 11): MjSpec.attach + compile() (proven byte-
    # identical to the legacy <include>), then a reloadable file is written for
    # the cross-process scene_xml_path consumers (MuJoCoPiper IK, sim_tool
    # pickable-population). connect() loads that file via from_xml_path.
    out = _MJCF_DIR / scene_name
    _model, scene_path = build_room_scene(
        robot_model_path=go2_xml,
        room_template_path=_ROOM_XML,
        room_assets_dir=assets_dir,
        attach_prefix="",
        spawn_xy=(10.0, 3.0),
        welds=welds,
        out_path=out,
        robot_meshes_dir=robot_meshes_dir,
    )
    assert scene_path is not None
    return scene_path


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
        viewer_track: bool = True, with_arm: bool = False,
    ) -> None:
        self._gui: bool = gui
        self._room: bool = room
        # Mount the Piper arm (go2_piper scene). Set by the embodiment manifest.
        self._with_arm: bool = bool(with_arm)
        # Offscreen renderers keyed by (kind, width, height): a frame always has
        # the size it was asked for, so intrinsics computed from that size match.
        self._renderers: dict[tuple[str, int, int], Any] = {}
        self._backend_pref: str = backend
        self._viewer_track: bool = viewer_track
        self._mj: _Go2Model | None = None
        self._viewer: Any = None
        self._connected: bool = False

        # Embodiment manifest (Rule 11): spawn pose + nominal stance read from
        # embodiments/go2/robot.yaml instead of hardcoded literals. The stance
        # qpos vector is built in connect() (model-dependent joint order) and
        # equals _STAND_JOINTS (asserted in test_dof_layout).
        self._config = load_embodiment_config("go2")
        self._stance_qpos: np.ndarray | None = None

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

        # R2b actor-causation instrumentation. Bumped in set_velocity ONLY for
        # writes that pass the skill-exclusive ``_gated`` guard (an UNGATED command —
        # i.e. one issued by a walk()/turn() skill on the token thread, or by any
        # caller when no skill holds the token). ``_cmd_writes`` is the count;
        # ``_cmd_motion`` is the cumulative |vx|+|vy|+|vyaw| MAGNITUDE — the signal
        # the grader keys on, so a terminal set_velocity(0,0,0) stop (a write, zero
        # magnitude) never satisfies causation. A GATED write (the bridge/nav
        # cmd_vel path, rejected before reaching the actuators) is NOT counted, so
        # the live navigate route reads as zero commanded motion (scoped OUT — see
        # actor_causation docstring). Read via ``cmd_motion()`` (snapshot value).
        self._cmd_writes: int = 0
        self._cmd_motion: float = 0.0

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
            scene_path = _build_room_scene_xml(with_arm=self._with_arm)
            model = mj.MjModel.from_xml_path(str(scene_path))
            data = mj.MjData(model)
            self._mj = _Go2Model(model, data)
            # Expose for downstream consumers that need to load an isolated
            # MjModel from the same MJCF (e.g. MuJoCoPiper's IK).
            self._scene_xml_path = str(scene_path)

            # Build the nominal-stance qpos vector from the manifest, in the
            # model's leg-qpos order (byte-identical to _STAND_JOINTS).
            layout = self._mj.layout
            self._stance_qpos = layout.build_stance_vector(model, self._config.stance)

            # Place Go2 in the entry hall (center of house) — spawn from manifest.
            spawn = self._config.spawn
            rq = layout.root_qpos_adr
            jq = layout.joint_qpos_start
            data.qpos[rq + 0] = spawn.xy[0]
            data.qpos[rq + 1] = spawn.xy[1]
            data.qpos[rq + 2] = spawn.base_height
            # Set standing joint angles
            data.qpos[jq : jq + 12] = self._stance_qpos
            # If Piper arm is mounted (nu=19 vs 12 — the arm adds 7 actuators),
            # stow it folded upright; otherwise joint2 defaults to 0 and the arm
            # extends horizontally, shifting 1.2kg of link6+ mass forward and
            # tipping the dog. NOTE: gate on nu, NOT nq — the room's pickable_*
            # free joints make the BARE go2 nq=40 (not 19), so an nq threshold
            # mis-fires for the bare model under either scene-build path.
            if model.nu >= 19:
                arm_q0 = layout.joint_qpos_start + layout.num_actuated
                data.qpos[arm_q0 : arm_q0 + 8] = _PIPER_STOW_QPOS
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
                # LOUD (was info → silent): the dog walks far worse on the fallback,
                # so surface WHY + the fix instead of hiding it (tricky-bugs Case 0).
                logger.warning(
                    "MuJoCoGo2: convex-MPC gait unavailable (%s) — falling back to "
                    "the sinusoidal gait. Install the Go2 MPC deps for the real gait: "
                    "`uv pip install -e .[go2]` (casadi + pin).", exc)

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
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._renderers.clear()
        self._mj = None
        self._stance_qpos = None
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
        # EAGER dep check (tricky-bugs Case 0: "Go2 won't walk — casadi missing"):
        # casadi is imported LAZILY by convex_mpc.centroidal_mpc on the FIRST solve,
        # so without this line a missing casadi sets _use_mpc=True at connect and the
        # QP then fails EVERY tick silently (qp_fail, no torque, no walk). Importing
        # it here makes a missing casadi raise at connect → loud fallback below.
        import casadi  # noqa: F401,PLC0415
        from vector_os_nano.third_party.convex_mpc.go2_robot_data import PinGo2Model  # noqa: PLC0415
        from vector_os_nano.third_party.convex_mpc.gait import Gait                   # noqa: PLC0415
        from vector_os_nano.third_party.convex_mpc.com_trajectory import ComTraj       # noqa: PLC0415
        from vector_os_nano.third_party.convex_mpc.leg_controller import LegController # noqa: PLC0415

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
            jq = self._mj.layout.joint_qpos_start
            jd = self._mj.layout.joint_dof_start
            q_cur = np.array(self._mj.data.qpos[jq : jq + 12], dtype=np.float64)
            dq_cur = np.array(self._mj.data.qvel[jd : jd + 12], dtype=np.float64)

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
                rq = self._mj.layout.root_qpos_adr
                pos = self._mj.data.qpos[rq : rq + 3]
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
                            from vector_os_nano.third_party.convex_mpc.centroidal_mpc import CentroidalMPC  # noqa: PLC0415
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
                jq = self._mj.layout.joint_qpos_start
                jd = self._mj.layout.joint_dof_start
                q_cur = np.array(self._mj.data.qpos[jq : jq + 12], dtype=np.float64)
                dq_cur = np.array(self._mj.data.qvel[jd : jd + 12], dtype=np.float64)
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
                rq = self._mj.layout.root_qpos_adr
                pos = self._mj.data.qpos[rq : rq + 3]
                self._viewer.cam.lookat[:] = [float(pos[0]), float(pos[1]), 0.3]
            self._viewer.sync()

        self._sim_step += 1

    def _mj_update_pin(self) -> None:
        """Sync Pinocchio model state from MuJoCo qpos/qvel.

        Converts MuJoCo (wxyz quaternion, world-frame linear vel) to
        Pinocchio (xyzw quaternion, body-frame linear vel) and runs
        the full set of Pinocchio computations the MPC solver needs.

        When MuJoCo carries extra DoFs beyond the PinGo2 model (e.g. a
        mounted Piper arm adds 8 DoFs), the leg segment is sliced out via
        the DofLayout addresses (base freejoint -> legs -> arm), so the
        slice is correct whether the robot's freejoint sits at qpos[0]
        (legacy <include> scene) or qpos[21] (MjSpec.attach scene).
        """
        mujoco_q = np.asarray(self._mj.data.qpos, dtype=float).reshape(-1)
        mujoco_dq = np.asarray(self._mj.data.qvel, dtype=float).reshape(-1)

        L = self._mj.layout
        qw, qx, qy, qz = mujoco_q[L.quat_start : L.quat_start + 4]

        import pinocchio as pin  # noqa: PLC0415
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        v_body = R.T @ mujoco_dq[L.root_dof_adr : L.root_dof_adr + 3]
        w_body = mujoco_dq[L.angvel_start : L.angvel_start + 3]

        n_leg_q = self._pin.model.nq - 7   # legs-only qpos count (12 for Go2)
        n_leg_v = self._pin.model.nv - 6   # legs-only qvel count (12 for Go2)
        base_xyz = mujoco_q[L.root_qpos_adr : L.root_qpos_adr + 3]
        leg_q = mujoco_q[L.joint_qpos_start : L.joint_qpos_start + n_leg_q]
        leg_v = mujoco_dq[L.joint_dof_start : L.joint_dof_start + n_leg_v]
        q_pin = np.concatenate([base_xyz, [qx, qy, qz, qw], leg_q])
        dq_pin = np.concatenate([v_body, w_body, leg_v])

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
        # TEMP DIAGNOSTIC (off unless VECTOR_CMDVEL_LOG is set): record the full
        # command stream — source thread + values + skill-gate state — so the
        # explore "weird gait" can be checked for bursty/duplicate/conflicting
        # set_velocity traffic from the bridge/nav stack. Remove after debugging.
        _dbg = os.environ.get("VECTOR_CMDVEL_LOG", "")
        if _dbg:
            try:
                with open(_dbg, "a") as _f:
                    _f.write(
                        f"{time.time():.4f}\t{threading.current_thread().name}\t"
                        f"vx={vx:+.3f}\tvy={vy:+.3f}\tvyaw={vyaw:+.3f}\tgated={int(_gated)}\n"
                    )
            except Exception:  # noqa: BLE001
                pass
        if _gated:
            return
        with self._cmd_lock:
            cvx = float(np.clip(vx, -_VX_MAX, _VX_MAX))
            cvy = float(np.clip(vy, -_VY_MAX, _VY_MAX))
            cvyaw = float(np.clip(vyaw, -_VYAW_MAX, _VYAW_MAX))
            self._cmd_vel = (cvx, cvy, cvyaw)
            # R2b: count this UNGATED command and accumulate its magnitude. A
            # stop (0,0,0) adds a write but zero magnitude, so it never satisfies
            # the grader's MOTION_EPS threshold (which keys on _cmd_motion).
            self._cmd_writes += 1
            self._cmd_motion += abs(cvx) + abs(cvy) + abs(cvyaw)

    def cmd_motion(self) -> float:
        """Cumulative UNGATED commanded-velocity magnitude (R2b actor-causation).

        The sum of ``|vx|+|vy|+|vyaw|`` over every ``set_velocity`` call that passed
        the skill-exclusive gate (a real motor command — a walk()/turn() skill's
        own writes, or any write when no skill holds the token). The grader takes a
        baseline snapshot before a step and a post snapshot after, treating a
        delta >= ``MOTION_EPS`` as "the actor commanded motion". A terminal
        ``set_velocity(0,0,0)`` adds nothing to this sum, so a step that only
        stopped never reads as commanded motion. Monotonically non-decreasing.
        """
        return float(self._cmd_motion)

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
        layout = self._mj.layout
        rq = layout.root_qpos_adr
        jq = layout.joint_qpos_start
        # Keep current XY, reset everything else
        cur_x, cur_y = float(data.qpos[rq + 0]), float(data.qpos[rq + 1])
        data.qpos[rq + 0] = cur_x
        data.qpos[rq + 1] = cur_y
        data.qpos[rq + 2] = 0.35                    # standing height
        data.qpos[rq + 3 : rq + 7] = [1, 0, 0, 0]   # upright quaternion (w,x,y,z)
        data.qpos[jq : jq + 12] = _STAND_JOINTS      # standing joint angles
        # If Piper arm is mounted, set it to the stow pose too. Gate on nu (the
        # arm adds 7 actuators), NOT nq — the room's pickable_* free joints make
        # the bare go2 nq=40, so an nq threshold mis-fires for the bare model.
        if self._mj.model.nu >= 19:
            arm_q0 = jq + layout.num_actuated
            data.qpos[arm_q0 : arm_q0 + 8] = _PIPER_STOW_QPOS
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
        rq = self._mj.layout.root_qpos_adr
        return list(self._mj.data.qpos[rq : rq + 3].astype(float))

    def get_velocity(self) -> list[float]:
        """Return base linear velocity [vx, vy, vz] in world frame."""
        self._require_connection()
        rd = self._mj.layout.root_dof_adr
        return list(self._mj.data.qvel[rd : rd + 3].astype(float))

    def get_heading(self) -> float:
        """Return yaw angle (radians) from base quaternion."""
        self._require_connection()
        qs = self._mj.layout.quat_start
        w, x, y, z = self._mj.data.qpos[qs : qs + 4]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(yaw)

    def get_joint_positions(self) -> list[float]:
        """Return all 12 joint positions (radians), ordered FL/FR/RL/RR."""
        self._require_connection()
        jq = self._mj.layout.joint_qpos_start
        return list(self._mj.data.qpos[jq : jq + 12].astype(float))

    def get_joint_velocities(self) -> list[float]:
        """Return all 12 joint velocities (rad/s), ordered FL/FR/RL/RR."""
        self._require_connection()
        jd = self._mj.layout.joint_dof_start
        return list(self._mj.data.qvel[jd : jd + 12].astype(float))

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
        r = self._renderer("rgb", width, height)
        cam_id = self._mj.model.cam("d435_rgb").id
        r.update_scene(self._mj.data, camera=cam_id)
        return r.render().copy()

    def _renderer(self, kind: str, width: int, height: int) -> Any:
        """The cached offscreen renderer for (kind, width, height); closed on disconnect."""
        key = (kind, int(width), int(height))
        r = self._renderers.get(key)
        if r is None:
            mj = _get_mujoco()
            r = mj.Renderer(self._mj.model, key[2], key[1])
            if kind == "seg":
                r.enable_segmentation_rendering()
            else:
                r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
            if kind == "rgb":
                r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = True
            elif kind == "depth":
                r.enable_depth_rendering()
            self._renderers[key] = r
        return r

    def get_depth_frame(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Render depth from d435_depth camera mounted on Go2 head.

        Returns an (H, W) float32 numpy array in metres. Uses the named
        'd435_depth' camera — same mounting as RGB for pixel alignment.
        """
        self._require_connection()

        r = self._renderer("depth", width, height)
        cam_id = self._mj.model.cam("d435_depth").id
        r.update_scene(self._mj.data, camera=cam_id)
        raw = r.render().copy()

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

    def get_self_mask(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Bool (H, W) mask of pixels showing the robot's OWN Piper arm geoms.

        Segmentation render on the d435_rgb camera → per-pixel geom id → True where
        the geom's body name starts with 'piper'. The honest real-robot self-filter
        (by IDENTITY, pose/colour/group-agnostic): the arm at rest occludes the table
        in the head camera and would otherwise be picked as the grasp target (D30 —
        site/geom-group hides are no-ops after MjSpec.attach). Used by the perception
        backend to drop the arm's pixels before object detection.
        """
        self._require_connection()
        mj = _get_mujoco()
        import numpy as np
        model = self._mj.model
        seg_renderer = self._renderer("seg", width, height)
        if not hasattr(self, "_piper_geom_ids"):
            # Collect EVERY body in the arm subtree (descendants of piper_base_link),
            # not just "piper*"-named ones — the gripper/finger bodies are named
            # differently and a name-prefix filter misses them (R15: caught only 21px,
            # leaving the gripper as a distractor). Walk each body's parent chain to
            # the arm root; include its geoms. Pose/colour/group/name-agnostic.
            root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "piper_base_link")
            arm_bodies: set[int] = set()
            if root >= 0:
                for b in range(model.nbody):
                    x = b
                    while x > 0:
                        if x == root:
                            arm_bodies.add(b)
                            break
                        x = int(model.body_parentid[x])
            ids = [
                g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in arm_bodies
                or (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith("piper")
            ]
            self._piper_geom_ids = np.asarray(ids, dtype=np.int32)
        cam_id = model.cam("d435_rgb").id
        seg_renderer.update_scene(self._mj.data, camera=cam_id)
        seg = seg_renderer.render()  # (H, W, 2): [...,0] = geom id, -1 = none
        geom_ids = seg[:, :, 0] if seg.ndim == 3 else seg
        if self._piper_geom_ids.size == 0:
            return np.zeros(geom_ids.shape, dtype=bool)
        return np.isin(geom_ids, self._piper_geom_ids)

    # ------------------------------------------------------------------
    # Sensor update helpers
    # ------------------------------------------------------------------

    def _update_odometry(self) -> None:
        from vector_os_nano.core.types import Odometry  # noqa: PLC0415
        q = self._mj.data.qpos
        v = self._mj.data.qvel
        rq = self._mj.layout.root_qpos_adr
        rd = self._mj.layout.root_dof_adr
        self._last_odom = Odometry(
            timestamp=float(self._mj.data.time),
            x=float(q[rq + 0]),
            y=float(q[rq + 1]),
            z=float(q[rq + 2]),
            qx=float(q[rq + 4]),
            qy=float(q[rq + 5]),
            qz=float(q[rq + 6]),
            qw=float(q[rq + 3]),
            vx=float(v[rd + 0]),
            vy=float(v[rd + 1]),
            vz=float(v[rd + 2]),
            vyaw=float(v[rd + 5]),
        )

    def _update_lidar(self) -> None:
        """Cast rays in multiple elevation rings — Livox MID360-like 3D lidar.

        The MID360 is mounted tilted 30 degrees forward (pitch down).
        This means the lidar's "horizontal" plane is actually 30° below
        horizontal, so it sees the ground in front and walls ahead.
        """
        from vector_os_nano.core.types import LaserScan  # noqa: PLC0415

        # Sensor mounting: on top of Go2 head — above all leg geoms.
        # 0.3m forward (head position) + 0.2m up (above trunk top).
        # At -20° tilt, nearest ground hit ≈ 0.9m ahead of lidar →
        # well past front legs (~0.1m ahead of lidar). No self-hits.
        # Must match bridge _SENSOR_X/_SENSOR_Z and nav stack sensorOffset.
        _LIDAR_OFFSET_X = 0.3
        _LIDAR_OFFSET_Z = 0.2

        rq = self._mj.layout.root_qpos_adr
        pos = self._mj.data.qpos[rq : rq + 3].copy().astype(np.float64)
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

        # Livox MID360 FOV: -7° to +52° (asymmetric, 59° range)
        # With 20° downward tilt → world frame: -27° to +32°
        # This gives both ground hits (below horizontal) and wall hits (above).
        # Scan beam tilt is 20° downward from sensor horizontal plane (sensor
        # frame itself is NOT tilted — only the beams are).
        n_azimuth = 360
        elevations = list(range(-8, 53, 2))  # -8° to +52° in 2° steps, includes 0° for 2D scan
        azimuth_step = 360.0 / n_azimuth

        # Ray-casting loop (shared impl — byte-identical to the old inline body;
        # see sensors/lidar_raycast.py). go2: tilt -20, max_range 12, mid-ring
        # records any non-self hit regardless of range (mid_ring_apply_max_range
        # =False), and the near-zero self-hit diagnostic is unused here.
        # mj_ray bodyexclude only filters the trunk; leg geoms (hip/thigh/calf)
        # are separate bodies, so they are filtered via _robot_geom_ids.
        from vector_os_nano.hardware.sim.sensors.lidar_raycast import (  # noqa: PLC0415
            raycast_lidar,
        )

        raw = raycast_lidar(
            self._mj.model,
            self._mj.data,
            pos_lidar=pos_lidar,
            heading=heading,
            exclude_bid=robot_body_id,
            robot_geom_ids=self._mj._robot_geom_ids,
            tilt_deg=-20.0,
            elevations=elevations,
            n_azimuth=n_azimuth,
            max_range=12.0,
            mid_ring_apply_max_range=False,
        )
        mid_ring_ranges = raw.mid_ring_ranges
        points_3d = raw.points_3d

        self._last_scan = LaserScan(
            timestamp=float(self._mj.data.time),
            angle_min=-math.pi,
            angle_max=math.pi,
            angle_increment=math.radians(azimuth_step),
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

        jq = self._mj.layout.joint_qpos_start
        jd = self._mj.layout.joint_dof_start
        q_start = np.array(data.qpos[jq : jq + 12], dtype=np.float64)
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

            q_cur = np.array(data.qpos[jq : jq + 12], dtype=np.float64)
            dq_cur = np.array(data.qvel[jd : jd + 12], dtype=np.float64)

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

    def navigate_to(
        self,
        x: float,
        y: float,
        tol: float = 0.25,
        speed: float = 0.4,
        timeout: float | None = None,
        **_ignored: Any,
    ) -> bool:
        """Walk to world position (x, y) with obstacle avoidance.

        Uses the visibility-graph planner (g1_vgraph) to plan a collision-free
        path around furniture, walls, and the pick_table.  Returns True iff the
        dog ended within ``tol`` of (x, y).  ``timeout=None`` uses the module
        default (_GO2_NAV_TIMEOUT_S = 60 s).

        Extra kwargs are accepted and ignored for call-shape compatibility with
        the ROS2 proxy and the g1 driver.

        Threading: the 1 kHz physics daemon continues running throughout.
        Path planning (mj_forward + obstacle extraction) is done once at the
        start; afterwards only ``walk()`` (thread-safe) is called.  qpos is
        NEVER written directly.

        ``reason`` is logged at INFO level on exit:
          "arrived"     — reached goal within tol
          "timeout"     — overall timeout exhausted before arrival
          "unreachable" — planner found no collision-free path (goal inside an
                          inflated obstacle); does NOT fall back to open-loop.
        """
        from vector_os_nano.hardware.sim import g1_vgraph as vg  # noqa: PLC0415

        self._require_connection()

        # ---- Step 1: snapshot current pose and plan path ---------------------
        # THREAD SAFETY (R12 fix): the 1 kHz gait daemon steps MjData continuously.
        # Calling mj_forward HERE races that mj_step — mj_forward WRITES derived fields
        # (geom_xpos, etc.), not just reads — so concurrent with the daemon it segfaults /
        # hangs (observed on the go2+Piper grasp path). Read the start pose via the
        # thread-safe accessor instead, and read obstacle geometry WITHOUT mj_forward: the
        # daemon already keeps geom_xpos current, and the furniture obstacles are STATIC
        # (constant geom_xpos), so the snapshot is race-free for planning. NEVER call
        # mj_forward / write qpos on the live model while the daemon runs.
        pos = self.get_position()
        start = (float(pos[0]), float(pos[1]))
        goal = (float(x), float(y))

        obstacles = obstacles_from_model(
            self._mj.model,
            self._mj.data,
            robot_geom_ids=self._mj.layout.robot_geom_ids,
        )
        waypoints, plan_length = vg.plan_path(start, goal, obstacles, _GO2_BODY_RADIUS)

        if waypoints is None:
            logger.warning(
                "navigate_to(%.2f, %.2f): UNREACHABLE — planner returned None "
                "(goal inside inflated obstacle or boxed-in); NOT falling back "
                "to open-loop.  n_obstacles=%d",
                x, y, len(obstacles),
            )
            return False

        logger.info(
            "navigate_to(%.2f, %.2f): plan has %d waypoints, geodesic=%.2f m, "
            "n_obstacles=%d",
            x, y, len(waypoints), plan_length, len(obstacles),
        )

        # ---- Step 2: walk the waypoint chain ---------------------------------
        eff_timeout = float(timeout) if timeout is not None else _GO2_NAV_TIMEOUT_S
        deadline = time.monotonic() + eff_timeout

        # Walk intermediate waypoints (skip index 0 = start)
        for wp_idx in range(1, len(waypoints)):
            wp = waypoints[wp_idx]
            is_final = wp_idx == len(waypoints) - 1
            wp_tol = tol if is_final else _GO2_NAV_WP_CAPTURE

            reached = self._go2_walk_to_waypoint(
                wx=wp[0], wy=wp[1],
                tol=wp_tol,
                speed=speed,
                deadline=deadline,
            )
            if not reached:
                # Timeout or stall — report where we ended up
                pos = self.get_position()
                dist = math.hypot(pos[0] - x, pos[1] - y)
                logger.info(
                    "navigate_to(%.2f, %.2f): timeout/stall at waypoint %d/%d, "
                    "dist_to_goal=%.2f m, pos=(%.2f, %.2f)",
                    x, y, wp_idx, len(waypoints) - 1, dist, pos[0], pos[1],
                )
                return False

        # Final check: did we actually land within tolerance?
        self.stop()
        pos = self.get_position()
        dist = math.hypot(pos[0] - x, pos[1] - y)
        arrived = dist <= tol
        logger.info(
            "navigate_to(%.2f, %.2f): %s — final pos=(%.2f, %.2f), dist=%.2f m",
            x, y, "arrived" if arrived else "timeout", pos[0], pos[1], dist,
        )
        return arrived

    def _go2_walk_to_waypoint(
        self,
        wx: float,
        wy: float,
        tol: float,
        speed: float,
        deadline: float,
    ) -> bool:
        """Steer and walk toward waypoint (wx, wy) until within tol or deadline.

        Uses the same proportional heading + forward walk pattern as
        perception_grasp._approach_object: compute bearing, compute yaw error,
        turn-toward + creep forward, repeat until captured or timed out.

        Returns True on capture, False on timeout/deadline.
        """
        _MAX_WALK_STEPS = 200  # hard upper bound to prevent infinite loops

        for _ in range(_MAX_WALK_STEPS):
            if time.monotonic() >= deadline:
                return False

            pos = self.get_position()
            dx = wx - pos[0]
            dy = wy - pos[1]
            dist = math.hypot(dx, dy)

            if dist < tol:
                return True

            bearing = math.atan2(dy, dx)
            heading = self.get_heading()
            yaw_err = math.atan2(
                math.sin(bearing - heading), math.cos(bearing - heading)
            )

            # Inside capture radius — just creep forward, gentle yaw correction
            if dist < _GO2_NAV_CAPTURE_R:
                vyaw = float(
                    max(-_GO2_NAV_VYAW_MAX, min(_GO2_NAV_VYAW_MAX,
                                                _GO2_NAV_K_YAW * yaw_err))
                )
                self.walk(vx=speed, vy=0.0, vyaw=vyaw,
                          duration=_GO2_NAV_STEP_S)
                continue

            # Misaligned: pivot first, then creep
            if abs(yaw_err) > _GO2_NAV_FACE_TOL:
                vyaw = float(
                    max(-_GO2_NAV_VYAW_MAX, min(_GO2_NAV_VYAW_MAX,
                                                _GO2_NAV_K_YAW * yaw_err))
                )
                self.walk(vx=0.0, vy=0.0, vyaw=vyaw,
                          duration=_GO2_NAV_STEP_S)
                continue

            # Roughly aligned: walk forward with proportional yaw correction
            if abs(yaw_err) < _GO2_NAV_YAW_DEADBAND:
                vyaw = 0.0
            else:
                vyaw = float(
                    max(-_GO2_NAV_VYAW_MAX, min(_GO2_NAV_VYAW_MAX,
                                                _GO2_NAV_K_YAW * yaw_err))
                )
            self.walk(vx=speed, vy=0.0, vyaw=vyaw,
                      duration=_GO2_NAV_STEP_S)

        # Fell out of step budget without capturing the waypoint
        return False
