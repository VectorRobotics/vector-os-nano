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

Physics ownership (vector_os_nano/sim/runtime.py):
    connect() hands MjModel/MjData to ONE SimRuntime, whose single stepper runs
    the gait controller (_control) every 5 ms tick (200 Hz control, 1 kHz
    physics). Every other thread mutates through runtime.submit (applied at a
    step boundary) and reads runtime.snapshot() or a runtime.render() copy.
    set_velocity() writes (vx, vy, vyaw) under _cmd_lock (non-blocking).
    stand/sit/lie_down step a PD ramp on the caller's thread (stepper parked).
    Timed motions (walk, navigate_to) count SIM time, so they mean the same
    distance at any real-time ratio; VECTOR_SIM_LOCKSTEP=1 makes them advance
    the physics themselves (deterministic scripted runs).
    disconnect() stops the stepper and closes every renderer.
"""
from __future__ import annotations

import contextlib
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
from vector_os_nano.sim.nav import vgraph as vg
from vector_os_nano.sim.nav.follow import NavResult, follow
from vector_os_nano.sim.runtime import RGBDFrame, SimRuntime

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
# Consecutive failed MPC solves (or leg-torque computations) before the gait gives
# up on the plan and PD-holds the stance (tricky-bugs Case 2: a QP that fails every
# tick must never silently hold a stale force plan). One warning per interval.
_MPC_MAX_FAILS: int = 5
_MPC_WARN_EVERY_S: float = 5.0
# Every stop BRAKES (WP05 E5): dropping a trotting body straight onto the PD stance
# mid-stride kicks it (a 0.4 m/s walk read 0.35-0.53 m/s 0.2 sim-s after its zero
# command). Instead, when the command goes to zero, the MPC keeps trotting toward
# zero velocity for _BRAKE_S, hands over to the PD stance once all four feet are
# down (holding that pose), then ramps to the stand pose over _BRAKE_RAMP_S.
_BRAKE_S: float = 0.35
_BRAKE_RAMP_S: float = 1.0

# ---------------------------------------------------------------------------
# Constants — lidar
# ---------------------------------------------------------------------------

# A scan is recast (off the physics thread, on a copy) when the cached one is
# older than this much SIM time (5 Hz; 31 rings x 360 = 11,160 rays per scan).
_LIDAR_PERIOD_S: float = 0.2

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
    """Write a flat-ground scene (go2.xml included by absolute path) to the scene cache.

    The <compiler> comes AFTER the include so its absolute meshdir overrides
    go2.xml's relative one, and the file loads from the cache dir.
    """
    from vector_os_nano.hardware.sim.scene_builder import write_scene

    xml = """\
<mujoco model="go2_flat">
  <include file="MJCF_DIR/go2.xml"/>
  <compiler angle="radian" meshdir="MJCF_DIR/assets" autolimits="true"/>

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
    return write_scene("scene_flat", xml.replace("MJCF_DIR", str(_MJCF_DIR)))


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
        scene_name = "scene_room_piper"
        # go2_piper.xml carries mesh paths relative to its own directory;
        # build_room_scene resolves them (meshes_dir=None = model-file dir).
        robot_meshes_dir = None
        # Grasp welds (name, body1, body2) — only with the arm (the no-arm model
        # has no piper_link6; a weld to a missing body fails to compile).
        # body1=piper_link6 (wrist carrying piper_ee_site + fingers), body2=each
        # pickable; created INACTIVE — the gripper's close() activates one only
        # when both fingers grip that object, so a held object stays held under
        # the soft position-controlled jaws (mirrors so101_mujoco.xml).
        welds: tuple[tuple[str, str, str], ...] = (
            ("grasp_pickable_bottle_blue", "piper_link6", "pickable_bottle_blue"),
            ("grasp_pickable_bottle_green", "piper_link6", "pickable_bottle_green"),
            ("grasp_pickable_can_red", "piper_link6", "pickable_can_red"),
        )
    else:
        go2_xml = _MJCF_DIR / "go2.xml"
        scene_name = "scene_room"
        # bare go2.xml uses relative meshdir="assets" — make absolute on attach.
        robot_meshes_dir = assets_dir
        welds = ()

    # ONE scene-build path (Rule 11): MjSpec.attach + compile() (proven byte-
    # identical to the legacy <include>), then a reloadable file is written to
    # the scene cache, never the package dir (scene_xml_path consumers:
    # MuJoCoPiper IK). connect() loads that file via from_xml_path.
    _model, scene_path = build_room_scene(
        robot_model_path=go2_xml,
        room_template_path=_ROOM_XML,
        room_assets_dir=assets_dir,
        attach_prefix="",
        spawn_xy=(10.0, 3.0),
        welds=welds,
        cache_stem=scene_name,
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
        self._backend_pref: str = backend
        self._viewer_track: bool = viewer_track
        self._mj: _Go2Model | None = None
        self._viewer: Any = None
        self._connected: bool = False
        # The ONE owner of MjData (stepping, step-boundary mutations, snapshots,
        # renders). None while disconnected.
        self.runtime: SimRuntime | None = None

        # Embodiment manifest (Rule 11): spawn pose + nominal stance read from
        # embodiments/go2/robot.yaml instead of hardcoded literals. The stance
        # qpos vector is built in connect() (model-dependent joint order) and
        # equals _STAND_JOINTS (asserted in test_dof_layout).
        self._config = load_embodiment_config("go2")
        self._stance_qpos: np.ndarray | None = None
        self._stand_q: np.ndarray = np.array(_STAND_JOINTS, dtype=np.float64)

        # MPC stack (None when using sinusoidal backend)
        self._use_mpc: bool = False
        self._pin: Any = None
        self._gait: Any = None
        self._traj: Any = None
        self._mpc: Any = None
        self._leg_ctrl: Any = None

        self._cmd_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._cmd_lock: threading.Lock = threading.Lock()
        # Lidar: recast on demand from a private copy, at most every _LIDAR_PERIOD_S.
        self._lidar_lock = threading.Lock()
        self._lidar_data: Any = None
        self._lidar_cache: tuple[float, Any, list] | None = None

        # Viewer drive mode (resolved in connect()): one of "main_thread_pump"
        # (macOS/mjpython), "background_daemon" (Linux/Windows window) or
        # "headless". Decides whether the runtime steps on its own thread or is
        # pumped by the caller's (viewer-owning) thread. See hardware/sim/
        # viewer_mode.py for the platform-aware seam.
        self._drive_mode: str = "headless"
        # Controller state, reset by _reset_step_state (connect, reset_pose).
        self._tau_hold: np.ndarray = np.zeros(12, dtype=float)
        self._sim_step: int = 0
        self._mpc_U_opt: Any = None
        self._mpc_ctrl_i: int = 0
        self._mpc_steps_per_mpc: int = 1
        self._mpc_dt: float = 0.0
        self._mpc_fails: int = 0
        self._mpc_warned_at: float = -1e9
        # The stop brake (controller state, sim times; see _BRAKE_S).
        self._was_moving: bool = False
        self._brake_t: float | None = None     # when the command went to zero
        self._brake_q: np.ndarray | None = None  # the four-foot pose at the hand-over
        self._brake_ho_t: float = 0.0

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

    @property
    def gait_backend(self) -> str:
        """The locomotion backend actually running: 'mpc' or 'sinusoidal'."""
        return "mpc" if self._use_mpc else "sinusoidal"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Load MuJoCo model, optionally open the viewer, start the runtime."""
        if self.runtime is not None:
            raise RuntimeError("MuJoCoGo2: already connected (disconnect first)")
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
            self._stand_q = np.asarray(self._stance_qpos, dtype=np.float64)

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
        self.runtime = SimRuntime(
            self._mj.model, self._mj.data, controller=self._control,
            substeps=_CTRL_DECIM, name="mujoco_go2",
        )

        if self._gui:
            if os.environ.get("MUJOCO_GL", "").lower() == "egl":
                # EGL offscreen contexts must exist before the viewer's GLFW/GLX init.
                # ponytail: the head-camera sizes in use (go2 640x480, go2_piper and the
                # nav bridge 320x240); a new size with the window open fails loud.
                self.runtime.warm_renderers(("rgb", "depth", "seg"), ((320, 240), (640, 480)))
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
                    "MuJoCoGo2: convex-MPC gait unavailable (%s: %s) — falling back to "
                    "the sinusoidal gait. convex_mpc is vendored at "
                    "vector_os_nano/third_party/convex_mpc; its solver deps come from "
                    "`uv pip install -e .[go2]` (casadi + pin).",
                    type(exc).__name__, exc)

        logger.info(
            "MuJoCoGo2 connected (gui=%s, room=%s, backend=%s)",
            self._gui, self._room, self.gait_backend,
        )

        # Resolve how the viewer (if any) is driven, then start physics.
        # macOS/mjpython window -> NO stepper thread: the caller pumps step()
        # on the viewer-owning (main) thread, because Apple GLFW is main-thread
        # only and a background viewer.sync() would segfault. Linux/Windows
        # window + headless -> the runtime's stepper thread.
        from vector_os_nano.hardware.sim.viewer_mode import (  # noqa: PLC0415
            resolve_viewer_drive_mode,
            uses_background_physics,
        )
        self._drive_mode = resolve_viewer_drive_mode(self._viewer is not None)
        self._reset_step_state()
        self.runtime.threaded = self.runtime.threaded and uses_background_physics(self._drive_mode)
        self.runtime.start()
        if not self.runtime.threaded:
            logger.info(
                "MuJoCoGo2: physics advances on the caller's thread "
                "(step()/timed motions): no stepper thread"
            )

    def disconnect(self) -> None:
        """Stop the runtime (physics + renderers), close the viewer, release the model."""
        rt, self.runtime = self.runtime, None
        viewer, self._viewer = self._viewer, None
        if viewer is not None:  # window first (stepper parked: no sync mid-close), then GL
            try:
                if rt is not None:
                    with rt.paused():
                        viewer.close()
                else:
                    viewer.close()
            except Exception:  # noqa: BLE001
                pass
        if rt is not None:
            rt.close()
        self._mj = None
        self._stance_qpos = None
        self._pin = None
        self._gait = None
        self._traj = None
        self._mpc = None
        self._leg_ctrl = None
        self._use_mpc = False
        self._lidar_data = None
        self._lidar_cache = None
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected or self.runtime is None:
            raise RuntimeError("MuJoCoGo2: not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Physics (driven by self.runtime)
    # ------------------------------------------------------------------

    def _reset_step_state(self) -> None:
        """(Re)initialize the controller state: held torque and any MPC plan."""
        self._tau_hold = np.zeros(12, dtype=float)
        self._sim_step = 0
        self._mpc_U_opt = None
        self._mpc_ctrl_i = 0
        self._mpc_fails = 0
        self._was_moving, self._brake_t, self._brake_q = False, None, None
        if self._use_mpc and self._gait is not None:
            gait_period = self._gait.gait_period
            self._mpc_dt = gait_period / _MPC_DT_FACTOR
            mpc_hz = 1.0 / self._mpc_dt
            self._mpc_steps_per_mpc = max(1, int(_CTRL_HZ // mpc_hz))
        else:
            self._mpc_dt = 0.0
            self._mpc_steps_per_mpc = 1

    def step(self, n: int = 1) -> None:
        """Advance *n* control ticks (5 physics steps each) on the CALLER's thread.

        The main-thread pump used on macOS/mjpython, where GLFW is
        main-thread-only and no stepper thread may touch the viewer.
        Mirrors :meth:`MuJoCoArm.step`.
        """
        self._require_connection()
        self.runtime.step(n)

    def wait_sim(self, seconds: float, cancel: Any = None) -> bool:
        """Wait for *seconds* of SIM time (advancing it when caller-stepped)."""
        self._require_connection()
        return self.runtime.wait_sim(seconds, cancel=cancel)

    def wait_settled(self, timeout: float = 3.0) -> bool:
        """Wait (sim time) until a stop has braked out: the body stands in its stance.

        A zero command keeps the trunk moving ~1.4 sim-s (the brake's MPC trot, then
        the stance ramp: ~3 cm lower, ~1.5 deg less pitch), so anything planned
        against the live body pose (the arm's IK) waits for it. False on the timeout.
        """
        deadline = self.get_sim_time() + timeout
        # a zero command the controller has not seen yet is a brake about to start
        while self._brake_t is not None or (self._was_moving and self._cmd_vel == (0.0, 0.0, 0.0)):
            if self.get_sim_time() >= deadline or not self.wait_sim(0.05):
                return False
        return True

    def _drive_for(self, seconds: float) -> bool:
        """Let *seconds* of SIM time pass while a blocking motion runs.

        With the stepper thread this waits on the sim clock; caller-stepped
        (macOS pump, VECTOR_SIM_LOCKSTEP) it steps the physics here, so the gait
        animates and the viewer syncs on this thread. Sim time, not wall time:
        a walk of N seconds covers the same ground at any real-time ratio.
        A stop (the calling run's CancelToken) ends the motion now, not after it.
        """
        from vector_os_nano.vcli.cognitive.abort import current_token  # noqa: PLC0415

        return self.runtime.wait_sim(seconds, cancel=current_token())

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

    def _pd_torque(self, q_des: np.ndarray, data: Any) -> np.ndarray:
        jq = self._mj.layout.joint_qpos_start
        jd = self._mj.layout.joint_dof_start
        q_cur = np.asarray(data.qpos[jq : jq + 12], dtype=np.float64)
        dq_cur = np.asarray(data.qvel[jd : jd + 12], dtype=np.float64)
        return np.clip(_KP * (q_des - q_cur) - _KD * dq_cur, -_TAU_LIMITS, _TAU_LIMITS)

    def _control(self, model: Any, data: Any) -> None:
        """The runtime controller: one 200 Hz tick (runs before 5 physics steps).

        Moving: the MPC force plan (or the sinusoidal gait); idle: PD-hold the
        manifest stance. Then the held torque is written and the viewer synced.
        """
        with self._cmd_lock:
            vx, vy, vyaw = self._cmd_vel
        t = float(data.time)
        moving = vx != 0.0 or vy != 0.0 or vyaw != 0.0
        if moving:
            self._brake_t = self._brake_q = None
        elif self._was_moving and self._use_mpc:
            self._brake_t = t
        self._was_moving = moving
        # Braking: the MPC tracks zero velocity, until all four feet are down.
        braking = self._brake_t is not None and self._brake_q is None and (
            t - self._brake_t < _BRAKE_S or not self._gait.compute_current_mask(t).all())
        mpc = self._use_mpc and (moving or braking)
        tau = self._mpc_torque(vx, vy, vyaw, t) if mpc else None
        if self._use_mpc and not mpc:
            # A new motion must solve a fresh plan, never replay an old horizon.
            self._mpc_U_opt = None
            self._mpc_ctrl_i = 0
        if tau is None:
            if moving and not self._use_mpc:
                tau = self._pd_torque(_compute_gait_targets(t, vx, vy, vyaw), data)
            elif mpc and self._mpc_fails < _MPC_MAX_FAILS:
                tau = self._tau_hold  # one bad tick: keep the last good torque
            else:
                tau = self._pd_torque(self._stance_q(data, t), data)
        self._tau_hold = np.array(tau, dtype=float)
        self._mj.set_joint_torque(self._tau_hold)
        self._sync_viewer()
        self._sim_step += _CTRL_DECIM

    def _stance_q(self, data: Any, t: float) -> np.ndarray:
        """The PD stance target; after a brake, ramped from the four-foot hand-over pose."""
        if self._brake_t is None:
            return self._stand_q
        if self._brake_q is None:
            jq = self._mj.layout.joint_qpos_start
            self._brake_q = np.array(data.qpos[jq : jq + 12], dtype=np.float64)
            self._brake_ho_t = t
        a = min(1.0, (t - self._brake_ho_t) / _BRAKE_RAMP_S)
        q = self._brake_q + a * a * (3.0 - 2.0 * a) * (self._stand_q - self._brake_q)
        if a >= 1.0:
            self._brake_t = self._brake_q = None
        return q

    def _mpc_torque(self, vx: float, vy: float, vyaw: float, t: float) -> np.ndarray | None:
        """Leg torques from the convex MPC, or None when the plan failed.

        A failed solve drops the plan; after _MPC_MAX_FAILS consecutive failures
        the gait PD-holds the stance and warns (rate-limited) until a solve works.
        """
        self._mj_update_pin()
        try:
            if self._mpc_ctrl_i % self._mpc_steps_per_mpc == 0:
                self._traj.generate_traj(
                    self._pin, self._gait, t, vx, vy, _MPC_Z_DES, vyaw, time_step=self._mpc_dt,
                )
                if self._mpc is None:
                    from vector_os_nano.third_party.convex_mpc.centroidal_mpc import CentroidalMPC  # noqa: PLC0415
                    self._mpc = CentroidalMPC(self._pin, self._traj)
                sol = self._mpc.solve_QP(self._pin, self._traj, False)
                n = self._traj.N
                w_opt = sol["x"].full().flatten()
                self._mpc_U_opt = w_opt[12 * n:].reshape((12, n), order="F")
            self._mpc_ctrl_i += 1
            if self._mpc_U_opt is None:
                raise RuntimeError("no MPC force plan")
            mpc_force = self._mpc_U_opt[:, 0]
            tau = np.zeros(12, dtype=float)
            for i, leg in enumerate(_MPC_LEG_NAMES):
                leg_out = self._leg_ctrl.compute_leg_torque(
                    leg, self._pin, self._gait, mpc_force[i * 3:(i + 1) * 3], t,
                )
                tau[i * 3:(i + 1) * 3] = leg_out.tau
        except Exception as exc:  # noqa: BLE001 — counted, then PD fallback (fail loud)
            self._mpc_U_opt = None
            self._mpc_ctrl_i = 0
            self._mpc_fails += 1
            now = time.monotonic()
            if self._mpc_fails >= _MPC_MAX_FAILS and now - self._mpc_warned_at > _MPC_WARN_EVERY_S:
                self._mpc_warned_at = now
                logger.warning(
                    "MuJoCoGo2: convex-MPC failed %d ticks in a row (%s: %s) — holding the "
                    "stance with PD until a solve succeeds", self._mpc_fails,
                    type(exc).__name__, exc)
            return None
        self._mpc_fails = 0
        return np.clip(tau, -_MPC_TAU_LIMITS, _MPC_TAU_LIMITS)

    def _sync_viewer(self) -> None:
        """Every _VIEWER_SYNC_EVERY physics steps: track the dog and sync the window."""
        viewer = self._viewer
        if viewer is None or self._sim_step % _VIEWER_SYNC_EVERY >= _CTRL_DECIM:
            return
        if self._viewer_track:
            rq = self._mj.layout.root_qpos_adr
            pos = self._mj.data.qpos[rq : rq + 3]
            viewer.cam.lookat[:] = [float(pos[0]), float(pos[1]), 0.3]
        viewer.sync()

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

        Skill-exclusive gate: while a skill holds the control token (inside
        walk()), calls from OTHER threads are silently ignored — this blocks the
        bridge path-follower / /cmd_vel_nav callback / safety-check from
        overriding a walk() or turn() in progress. The token holder passes.
        """
        self._require_connection()
        _gated = (time.time() < self._skill_ctrl_until
                  and threading.get_ident() != self._skill_ctrl_tid)
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
        """Reset to the standing pose at the current XY and heading (tip-over recovery).

        Applied at a step boundary through the runtime: the velocity command is
        zeroed, the held torque and any MPC plan are dropped, and height + stance
        come from the manifest. The heading is kept (no odometry jump).
        """
        self._require_connection()
        mj = _get_mujoco()
        layout = self._mj.layout
        rq, jq = layout.root_qpos_adr, layout.joint_qpos_start
        half_yaw = 0.5 * self.get_heading()
        height = float(self._config.spawn.base_height)
        has_arm = self._mj.model.nu >= 19

        def _reset(model: Any, data: Any) -> None:
            with self._cmd_lock:
                self._cmd_vel = (0.0, 0.0, 0.0)
            data.qpos[rq + 2] = height
            data.qpos[rq + 3 : rq + 7] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
            data.qpos[jq : jq + 12] = self._stand_q
            # Gate on nu (the arm adds 7 actuators), NOT nq — the room's pickable_*
            # free joints make the bare go2 nq=40, so an nq threshold mis-fires.
            if has_arm:
                arm_q0 = jq + layout.num_actuated
                data.qpos[arm_q0 : arm_q0 + 8] = _PIPER_STOW_QPOS
            data.qvel[:] = 0
            data.ctrl[:] = 0
            if has_arm:
                data.ctrl[12:19] = _PIPER_STOW_CTRL
            self._reset_step_state()
            mj.mj_forward(model, data)

        self.runtime.call(_reset)

    def get_sim_time(self) -> float:
        """Return the MuJoCo simulation clock (seconds).

        The runtime advances this as it steps; it runs at whatever fraction of
        wall-clock the machine sustains (see runtime.rt_ratio()). Timed motions
        and the bridge's path follower integrate against it.
        """
        self._require_connection()
        return self.runtime.sim_time

    def _q(self) -> np.ndarray:
        self._require_connection()
        return self.runtime.snapshot().qpos

    def get_position(self) -> list[float]:
        """Return base position [x, y, z] in world frame."""
        rq = self._mj.layout.root_qpos_adr if self._mj is not None else 0
        return [float(v) for v in self._q()[rq : rq + 3]]

    def get_velocity(self) -> list[float]:
        """Return base linear velocity [vx, vy, vz] in world frame."""
        self._require_connection()
        rd = self._mj.layout.root_dof_adr
        return [float(v) for v in self.runtime.snapshot().qvel[rd : rd + 3]]

    def get_heading(self) -> float:
        """Return yaw angle (radians) from base quaternion."""
        q = self._q()
        qs = self._mj.layout.quat_start
        w, x, y, z = q[qs : qs + 4]
        return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def get_joint_positions(self) -> list[float]:
        """Return all 12 joint positions (radians), ordered FL/FR/RL/RR."""
        q = self._q()
        jq = self._mj.layout.joint_qpos_start
        return [float(v) for v in q[jq : jq + 12]]

    def get_joint_velocities(self) -> list[float]:
        """Return all 12 joint velocities (rad/s), ordered FL/FR/RL/RR."""
        self._require_connection()
        jd = self._mj.layout.joint_dof_start
        return [float(v) for v in self.runtime.snapshot().qvel[jd : jd + 12]]

    def get_odometry(self) -> Any:
        """Odometry of one instant (the latest runtime snapshot)."""
        from vector_os_nano.core.types import Odometry  # noqa: PLC0415

        self._require_connection()
        s = self.runtime.snapshot()
        q, v = s.qpos, s.qvel
        rq = self._mj.layout.root_qpos_adr
        rd = self._mj.layout.root_dof_adr
        return Odometry(
            timestamp=s.sim_time,
            x=float(q[rq + 0]), y=float(q[rq + 1]), z=float(q[rq + 2]),
            qx=float(q[rq + 4]), qy=float(q[rq + 5]), qz=float(q[rq + 6]), qw=float(q[rq + 3]),
            vx=float(v[rd + 0]), vy=float(v[rd + 1]), vz=float(v[rd + 2]),
            vyaw=float(v[rd + 5]),
        )

    def get_lidar_scan(self) -> Any:
        """The most recent 2D laser scan (at most _LIDAR_PERIOD_S of sim time old)."""
        return self._lidar()[1]

    def get_3d_pointcloud(self) -> list[tuple[float, float, float, float]]:
        """The most recent 3D point cloud as (x, y, z, intensity)."""
        return self._lidar()[2]

    def get_rgbd_frame(
        self, width: int = 640, height: int = 480,
        kinds: tuple[str, ...] = ("rgb", "depth"),
    ) -> RGBDFrame:
        """RGB / depth / segmentation + camera pose of the d435 head camera, ONE instant.

        Rendered from a step-boundary copy of the sim, so every field shares the
        frame's ``sim_time``; ``K`` comes from the camera fovy and the frame shape.
        Depth is metres with the D435 range (0.1-10 m) applied (0 = invalid).
        """
        self._require_connection()
        f = self.runtime.render("d435_rgb", width, height, kinds)
        if f.depth is not None:
            f.depth[(f.depth < 0.1) | (f.depth > 10.0)] = 0.0
        return f

    def get_camera_frame(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Render first-person RGB from the d435_rgb camera mounted on Go2 head.

        Returns an (H, W, 3) uint8 numpy array in RGB order.
        """
        return self.get_rgbd_frame(width, height, ("rgb",)).rgb

    def get_depth_frame(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Render depth (H, W) float32 metres from the head camera (rgb-aligned)."""
        return self.get_rgbd_frame(width, height, ("depth",)).depth

    def get_camera_pose(self) -> tuple:
        """Return (cam_xpos (3,), cam_xmat (9,) row-major) of the d435_rgb camera.

        Prefer :meth:`get_rgbd_frame`, whose pose belongs to its pixels.
        """
        self._require_connection()
        cam_id = self._mj.model.cam("d435_rgb").id
        return self.runtime.call(
            lambda _m, d: (d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].copy())
        )

    def self_mask(self, seg: "np.ndarray") -> "np.ndarray":
        """Bool mask of the pixels of *seg* (geom ids) showing the robot's OWN Piper arm.

        The honest real-robot self-filter (by IDENTITY, pose/colour/group-agnostic):
        the arm at rest occludes the table in the head camera and would otherwise be
        picked as the grasp target (D30 — site/geom-group hides are no-ops after
        MjSpec.attach). Every geom of every body in the arm subtree counts — not
        just "piper*"-named ones (the gripper/finger bodies are named differently;
        R15: a name-prefix filter left the gripper as a distractor).
        """
        self._require_connection()
        if not hasattr(self, "_piper_geom_ids"):
            mj = _get_mujoco()
            model = self._mj.model
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
        if self._piper_geom_ids.size == 0:
            return np.zeros(seg.shape, dtype=bool)
        return np.isin(seg, self._piper_geom_ids)

    def get_self_mask(
        self, width: int = 640, height: int = 480,
    ) -> "np.ndarray":
        """Bool (H, W) mask of the head-camera pixels showing the robot's own arm."""
        return self.self_mask(self.get_rgbd_frame(width, height, ("seg",)).seg)

    # ------------------------------------------------------------------
    # Sensor update helpers
    # ------------------------------------------------------------------

    def _lidar(self) -> tuple[float, Any, list]:
        """(sim_time, LaserScan, points) — recast when older than _LIDAR_PERIOD_S.

        Cast OFF the physics thread, on a private copy taken at a step boundary,
        with one vectorised mj_multiRay call: the scan costs the stepper nothing.
        """
        self._require_connection()
        with self._lidar_lock:
            c = self._lidar_cache
            if c is not None and 0.0 <= self.runtime.sim_time - c[0] < _LIDAR_PERIOD_S:
                return c
            mj = _get_mujoco()
            model = self._mj.model
            d = self._lidar_data
            if d is None:
                d = self._lidar_data = mj.MjData(model)
            self.runtime.call(lambda m, src: mj.mj_copyData(d, m, src))
            mj.mj_kinematics(model, d)  # geom poses of the copied qpos
            self._lidar_cache = self._cast_lidar(model, d)
            return self._lidar_cache

    def _cast_lidar(self, model: Any, data: Any) -> tuple[float, Any, list]:
        """Cast rays in multiple elevation rings — Livox MID360-like 3D lidar.

        The beams are pitched -20 deg in the body frame, so the "horizontal"
        ring sees the ground in front and walls ahead.
        """
        from vector_os_nano.core.types import LaserScan  # noqa: PLC0415

        # Sensor mounting: on top of Go2 head — above all leg geoms.
        # 0.3m forward (head position) + 0.2m up (above trunk top).
        # At -20° tilt, nearest ground hit ≈ 0.9m ahead of lidar →
        # well past front legs (~0.1m ahead of lidar). No self-hits.
        # Must match bridge _SENSOR_X/_SENSOR_Z and nav stack sensorOffset.
        _LIDAR_OFFSET_X = 0.3
        _LIDAR_OFFSET_Z = 0.2

        L = self._mj.layout
        rq = L.root_qpos_adr
        pos = np.asarray(data.qpos[rq : rq + 3], dtype=np.float64)
        w, x, y, z = data.qpos[L.quat_start : L.quat_start + 4]
        heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        # Lidar position in world frame = base + rotated offset
        pos_lidar = np.array([
            float(pos[0]) + cos_h * _LIDAR_OFFSET_X,
            float(pos[1]) + sin_h * _LIDAR_OFFSET_X,
            float(pos[2]) + _LIDAR_OFFSET_Z,
        ], dtype=np.float64)

        n_azimuth = 360
        elevations = list(range(-8, 53, 2))  # -8° to +52° in 2° steps, includes 0° for 2D scan
        azimuth_step = 360.0 / n_azimuth

        # Shared ray cast (sensors/lidar_raycast.py). go2: tilt -20, max_range 12,
        # mid-ring records any non-self hit regardless of range
        # (mid_ring_apply_max_range=False). Leg/arm geoms are separate bodies, so
        # self-hits are filtered via _robot_geom_ids.
        from vector_os_nano.hardware.sim.sensors.lidar_raycast import (  # noqa: PLC0415
            raycast_lidar,
        )

        raw = raycast_lidar(
            model,
            data,
            pos_lidar=pos_lidar,
            heading=heading,
            exclude_bid=self._mj.base_bid,
            robot_geom_ids=self._mj._robot_geom_ids,
            tilt_deg=-20.0,
            elevations=elevations,
            n_azimuth=n_azimuth,
            max_range=12.0,
            mid_ring_apply_max_range=False,
        )
        scan = LaserScan(
            timestamp=float(data.time),
            angle_min=-math.pi,
            angle_max=math.pi,
            angle_increment=math.radians(azimuth_step),
            range_min=0.1,
            range_max=12.0,
            ranges=tuple(raw.mid_ring_ranges),
        )
        return float(data.time), scan, raw.points_3d

    # ------------------------------------------------------------------
    # PD interpolation (stepped on the caller's thread, stepper parked)
    # ------------------------------------------------------------------

    def _pd_interpolate(
        self,
        target_joints: np.ndarray,
        duration: float = 2.0,
    ) -> None:
        """Drive joints to target using PD torque control with a tanh ramp, then hold 0.5 s."""
        self._require_connection()
        self._was_moving, self._brake_t, self._brake_q = False, None, None  # the posture owns the joints
        rt = self.runtime
        jq = self._mj.layout.joint_qpos_start
        q_start = np.array(rt.snapshot().qpos[jq : jq + 12], dtype=np.float64)
        q_target = np.asarray(target_joints, dtype=np.float64)
        dt = rt.tick_dt
        n_ramp = max(1, int(duration / dt))
        tick = [0]

        def ramp(_model: Any, data: Any) -> None:
            k = tick[0]
            tick[0] += 1
            if k < n_ramp:
                phase = float(np.tanh((k + 1) * dt / (duration / 3.0)))
                q_des = q_start + phase * (q_target - q_start)
            else:
                q_des = q_target
            self._mj.set_joint_torque(self._pd_torque(q_des, data))
            self._sync_viewer()
            self._sim_step += _CTRL_DECIM

        rt.step(n_ramp + max(0, int(0.5 / dt)), controller=ramp)

    # ------------------------------------------------------------------
    # Posture commands
    # ------------------------------------------------------------------

    def stand(self, duration: float = 2.0) -> bool:
        self._require_connection()
        self._pd_interpolate(self._stand_q, duration=duration)
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

    @contextlib.contextmanager
    def _skill_control(self):
        """Hold skill-level control for the block: other threads' set_velocity is
        ignored (the bridge path follower yields), and the command is zeroed on any
        exit, KeyboardInterrupt included (skills-nav:walk-runaway-on-exception)."""
        self._skill_ctrl_tid = threading.get_ident()
        self._skill_ctrl_until = math.inf
        try:
            yield
        finally:
            if self.runtime is not None:
                with self._cmd_lock:
                    self._cmd_vel = (0.0, 0.0, 0.0)
            self._skill_ctrl_until = 0.0
            self._skill_ctrl_tid = 0

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        """Walk at the commanded velocity for *duration* SIM seconds, then stop.

        The robot should be standing before calling (call stand() first).
        Holds skill-level control authority until it returns, so a concurrently
        running bridge path-follower yields. Always ends with a zero command,
        also when interrupted (KeyboardInterrupt).

        Returns:
            True if the motion ran its full sim time without falling over.
        """
        self._require_connection()
        with self._skill_control():
            self.set_velocity(vx, vy, vyaw)
            ran = self._drive_for(duration)
            self.set_velocity(0.0, 0.0, 0.0)
            self._drive_for(0.2)  # settle
            pos = self.get_position()
            return bool(ran and pos[2] > 0.15)

    def navigate_to(
        self,
        x: float,
        y: float,
        tol: float | None = None,
        speed: float | None = None,
        timeout: float | None = None,
        cancel: Any = None,
        on_progress: Any = None,
        **_ignored: Any,
    ) -> NavResult:
        """Walk to world (x, y) around the scene's obstacles (the shared follower).

        Plans on the vgraph over the static scene (read once at a step boundary),
        snapping a goal inside furniture to the nearest free point, and streams
        velocity in SIM time while holding skill control. ``timeout`` is sim
        seconds (None: a budget from the geodesic). ``bool(result)`` is True only
        on arrival; ``result.reason`` says why not. Per-body tuning comes from the
        manifest's ``nav`` block; the inflation grows by how far a mounted arm or a
        held object reaches past the body.
        """
        self._require_connection()
        layout = self._mj.layout
        robot = layout.robot_geom_ids
        extra_q = layout.joint_qpos_start + layout.num_actuated

        def read(m: Any, d: Any) -> tuple[list, float]:
            return (vg.obstacles_from_model(m, d, robot),
                    vg.overhang(m, d, layout.root_bid, extra_q))

        obstacles, reach = self.runtime.call(read)
        nav = self._config.nav
        with self._skill_control():
            res = follow(
                self, (x, y), obstacles, body=nav,
                inflation=float(nav["body_radius"]) + reach, tol=tol, speed=speed,
                timeout=timeout, cancel=cancel, on_progress=on_progress,
            )
        logger.info("navigate_to(%.2f, %.2f): %s", x, y, res)
        return res
