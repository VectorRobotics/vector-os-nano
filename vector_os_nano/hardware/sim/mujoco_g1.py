# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo-based simulated Unitree G1 humanoid — R8 floor.

navigate_to delegates to the ONE shared follower (vector_os_nano/sim/nav):
the vgraph planner routes AROUND pick_table, walls and furniture.  R2 scope retained: WALK via the 12-DOF RL policy (motion.pt) in
the go2 apartment room, plus sensors (lidar, head camera).

Architecture:
  _build_g1_room_scene_xml() — MjSpec attach of g1_12dof model into the go2
                               room scene (mirrors R1 approach, swaps robot).
  MuJoCoG1                  — loads scene, runs policy gait, exposes:
                                walk(), navigate_to(), set_velocity(), stop(),
                                get_base_height(), get_heading(),
                                get_lidar_scan(), get_camera_frame(),
                                get_camera_pose()

Policy control loop (50 Hz, caller-stepped):
  - _step_batch() advances 10 physics steps (0.002 s each = 0.02 s) through the
    SimRuntime on the caller's thread (no stepper thread), then runs one policy
    inference at that step boundary. Renders go through the runtime's ONE render
    thread from a step-boundary copy.
  - Obs layout (47): [3 ang_vel, 3 gravity, 3 cmd, 12 q, 12 dq, 12 prev_act, 2 gait_phase].
    Exact match to spike_12dof_walk.py (proven correct).

CRITICAL offset note (combined room scene):
  The g1 freejoint is NOT at qpos[0] in the combined scene (room has 3 static
  bodies before g1 is attached).  Empirically verified:
    pelvis_qpos_adr = 21   (qpos[21:28] = root pos x,y,z + quat w,x,y,z)
    pelvis_dof_adr  = 18   (qvel[18:24] = root lin_vel + ang_vel)
    leg qpos        = qpos[28:40]  (12 joints)
    leg qvel        = qvel[24:36]  (12 joints)
    ctrl indices    = ctrl[0:12]   (all 12 g1 actuators, only robot in scene)
    ang_vel obs     = qvel[21:24]  (pelvis_dof_adr + 3)
    gravity quat    = qpos[24:28]  (pelvis_qpos_adr + 3)

Camera: 'g1_head_rgb' on the pelvis body at pos=(0.04, 0, 0.42),
  xyaxes="0 -1 0 0 0 1" — forward-facing, matches go2 d435_rgb convention.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.embodiments.dof_layout import DofLayout
from vector_os_nano.hardware.sim.sensors.g1_lidar import g1_lidar_scan
from vector_os_nano.sim.nav import vgraph as vg
from vector_os_nano.sim.nav.follow import NavResult, follow
from vector_os_nano.sim.runtime import SimRuntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROOM_XML = _HERE / "go2_room.xml"
_GO2_ASSETS_DIR = _HERE / "mjcf" / "go2" / "assets"

# 12-DOF model + policy from assets/g1_gait/
_ASSET_DIR = _REPO_ROOT / "assets" / "g1_gait"
_G1_12DOF_XML = _ASSET_DIR / "g1_12dof.xml"
_G1_MESHES_DIR = _ASSET_DIR / "meshes"

# G1 spawn position in the room
_G1_SPAWN_X: float = 10.0
_G1_SPAWN_Y: float = 3.0
_G1_PELVIS_Z: float = 0.793  # from g1_12dof default stance height

# Camera name in the compiled scene (attach prefix "g1_" + camera name "head_rgb")
_SCENE_CAM_NAME: str = "g1_head_rgb"

# Lidar mount offsets from pelvis (z-offset brings sensor to ~head height)
_LIDAR_OFFSET_X: float = 0.0
_LIDAR_OFFSET_Z: float = 0.72  # pelvis z ~0.793 + 0.72 = ~1.51 m from floor

# ---------------------------------------------------------------------------
# Policy / gait constants (from spike_12dof_walk.py — proven correct)
# ---------------------------------------------------------------------------

_SIM_DT: float = 0.002          # MuJoCo physics timestep (s)
_DECIMATION: int = 10            # physics steps per policy step → 50 Hz policy
_KPS = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
_KDS = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
_DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float32,
)
_ANG_VEL_SCALE: float = 0.25
_DOF_POS_SCALE: float = 1.0
_DOF_VEL_SCALE: float = 0.05
_ACTION_SCALE: float = 0.25
_CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
_NUM_ACTIONS: int = 12
_NUM_OBS: int = 47
_GAIT_PERIOD: float = 0.8

# ---------------------------------------------------------------------------
# Lazy mujoco / torch imports
# ---------------------------------------------------------------------------

_mujoco: Any = None
_torch: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415
        _mujoco = mujoco
    return _mujoco


def _get_torch() -> Any:
    global _torch
    if _torch is None:
        import torch  # noqa: PLC0415
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Gravity orientation (matches spike & recovered code exactly)
# ---------------------------------------------------------------------------


def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Project gravity vector into body frame from quaternion (qw, qx, qy, qz)."""
    qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------


def _build_g1_room_scene_xml() -> Path:
    """Build composite G1-12dof-in-go2-room scene XML via MjSpec attach.

    Strategy:
    1. Load go2_room.xml with the go2 robot include removed (room geometry only).
    2. Load g1_12dof.xml via MjSpec, set meshdir to absolute path, add head camera.
    3. Attach g1 at spawn frame (10, 3, 0) with prefix "g1_".
    4. Compile and write to scene_g1_12dof_room.xml.

    Returns the path to the written scene XML.
    """
    from vector_os_nano.hardware.sim.scene_builder import build_room_scene

    if not _ROOM_XML.exists():
        raise FileNotFoundError(f"go2_room.xml not found at {_ROOM_XML}")
    if not _G1_12DOF_XML.exists():
        raise FileNotFoundError(
            f"g1_12dof.xml not found at {_G1_12DOF_XML}; "
            "run scripts/setup_g1_gait.sh first."
        )

    # ONE scene-build path (Rule 11), shared with go2: MjSpec.attach + compile().
    # g1 carries prefix "g1_" (keeps the room's pickable_* unprefixed), a head
    # camera on pelvis, and no grasp welds. connect() then sets standing z=0.793.
    _model, scene_path = build_room_scene(
        robot_model_path=_G1_12DOF_XML,
        room_template_path=_ROOM_XML,
        room_assets_dir=_GO2_ASSETS_DIR,
        attach_prefix="g1_",
        spawn_xy=(_G1_SPAWN_X, _G1_SPAWN_Y),
        welds=(),
        cache_stem="scene_g1_12dof_room",
        robot_meshes_dir=_G1_MESHES_DIR,
        camera={
            "mount_body": "pelvis",
            "name": "head_rgb",
            "pos": [0.04, 0.0, 0.42],
            "xyaxes": [0, -1, 0, 0, 0, 1],
        },
    )
    assert scene_path is not None
    logger.info("G1-12dof room scene written to %s", scene_path)
    return scene_path


# ---------------------------------------------------------------------------
# Robot geom set for lidar self-filtering
# ---------------------------------------------------------------------------


def _build_robot_geom_set(model: Any) -> set[int]:
    """Return set of geom ids belonging to any g1_* body (lidar self-filter)."""
    mj = _get_mujoco()
    robot_geom_ids: set[int] = set()
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or ""
        if body_name.startswith("g1_"):
            robot_geom_ids.add(gid)
    return robot_geom_ids


# ---------------------------------------------------------------------------
# Internal offset container
# ---------------------------------------------------------------------------


# g1 ctrl slice: all 12 actuators are the ONLY actuators in the room scene
# (verified: nu=12 and all are g1_ leg actuators). This is the ACTUATOR control
# array slice, not a DoF-layout (qpos/qvel) concept, so it stays a driver const.
_G1_CTRL_START: int = 0
_G1_CTRL_END: int = _NUM_ACTIONS


class _G1Offsets(DofLayout):
    """Backward-compatible adapter over the generic ``DofLayout`` (Rule 11).

    The qpos/qvel slice addresses are now introspected by the shared
    ``DofLayout`` (root_body="g1_pelvis", num_actuated=12) — there is no g1-only
    layout math any more. This subclass only re-exposes the legacy field names
    (``pelvis_*`` / ``leg_*`` / ``ctrl_*``) the existing g1 call sites read, so
    behavior is byte-identical with zero call-site churn. ``robot_geom_ids`` is
    kept as the original g1_-name-prefix set (equal to the subtree set, asserted
    in tests) for exactness.
    """

    def __init__(self, model: Any) -> None:
        super().__init__(model, "g1_pelvis", _NUM_ACTIONS)
        # Legacy aliases (identical values, old names).
        self.pelvis_qpos_adr: int = self.root_qpos_adr
        self.pelvis_dof_adr: int = self.root_dof_adr
        self.leg_qpos_start: int = self.joint_qpos_start
        self.leg_dof_start: int = self.joint_dof_start
        self.pelvis_bid: int = self.root_bid
        self.ctrl_start: int = _G1_CTRL_START
        self.ctrl_end: int = _G1_CTRL_END
        # Keep the original g1_-name-prefix geom set (not the subtree set).
        self.robot_geom_ids: set[int] = _build_robot_geom_set(model)

        logger.debug(
            "G1 offsets: qpos=%d dof=%d leg_qpos=%d:%d ctrl=%d:%d nu=%d",
            self.pelvis_qpos_adr, self.pelvis_dof_adr,
            self.leg_qpos_start, self.leg_qpos_start + _NUM_ACTIONS,
            self.ctrl_start, self.ctrl_end, model.nu,
        )


# ---------------------------------------------------------------------------
# MuJoCoG1
# ---------------------------------------------------------------------------


class MuJoCoG1:
    """Simulated Unitree G1 humanoid in the go2 apartment room — R2 walking.

    Driven by a 12-DOF RL policy (motion.pt, 50 Hz) in a single-threaded
    synchronous loop.  Exposes walk(), navigate_to(), set_velocity(), stop(),
    get_base_height(), get_heading(), get_lidar_scan(), get_camera_frame(),
    get_camera_pose(), build_scene().
    """

    def __init__(self, gui: bool = False, room: bool = True) -> None:
        self._gui = gui
        self._room = room
        self._model: Any = None
        self._data: Any = None
        self._policy: Any = None
        self._offsets: _G1Offsets | None = None
        self._viewer: Any = None
        # The ONE owner of MjData: steps on the caller's thread, renders on its
        # own render thread. None while disconnected.
        self.runtime: SimRuntime | None = None
        self._cam_name: str = _SCENE_CAM_NAME
        # Embodiment manifest (Rule 11): spawn pose + nominal stance read from
        # embodiments/g1/robot.yaml instead of the _G1_SPAWN_*/_DEFAULT_ANGLES
        # constants. Loaded once at connect(); stance vector built in the model's
        # leg-qpos order (== _DEFAULT_ANGLES, asserted in test_dof_layout).
        self._config = load_embodiment_config("g1")
        self._stance_qpos: np.ndarray | None = None
        # Policy per-batch state
        self._action = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        self._target = _DEFAULT_ANGLES.copy()
        self._obs = np.zeros(_NUM_OBS, dtype=np.float32)
        self._counter: int = 0
        # Velocity command [vx, vy, vyaw]
        self._cmd = np.zeros(3, dtype=np.float32)
        # Sensor cache
        self._last_scan: Any = None
        self._last_pointcloud: list[tuple[float, float, float, float]] = []

    # ------------------------------------------------------------------
    # Capability properties (BaseProtocol) — uniform with MuJoCoGo2
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """BaseProtocol identifier (mirrors MuJoCoGo2.name = 'mujoco_go2')."""
        return "mujoco_g1"

    @property
    def supports_holonomic(self) -> bool:
        """G1 is a biped — not omnidirectional (cannot true-strafe)."""
        return False

    @property
    def supports_lidar(self) -> bool:
        """get_lidar_scan() returns data (g1_lidar_scan), so True (matches go2)."""
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Build scene (if missing), load model + policy, reset to stance."""
        mj = _get_mujoco()
        torch = _get_torch()

        self._model = mj.MjModel.from_xml_path(str(_build_g1_room_scene_xml()))
        self._model.opt.timestep = _SIM_DT
        self._data = mj.MjData(self._model)
        self._offsets = _G1Offsets(self._model)
        # Build the nominal-stance qpos vector from the manifest, in the model's
        # leg-qpos order (byte-identical to _DEFAULT_ANGLES — see test_dof_layout).
        self._stance_qpos = self._offsets.build_stance_vector(
            self._model, self._config.stance
        )

        # Load policy
        policy_path = _ASSET_DIR / "motion.pt"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"motion.pt not found at {policy_path}; "
                "run scripts/setup_g1_gait.sh first."
            )
        self._policy = torch.jit.load(str(policy_path))

        # Reset control state
        self._action = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        self._target = _DEFAULT_ANGLES.copy()
        self._obs = np.zeros(_NUM_OBS, dtype=np.float32)
        self._counter = 0
        self._cmd = np.zeros(3, dtype=np.float32)

        self._reset_to_stance()
        self._cam_name = self._resolve_camera()
        self.runtime = SimRuntime(
            self._model, self._data, controller=self._pd_control, substeps=1,
            threaded=False, paced=False, name="mujoco_g1",
        )

        if self._gui:
            try:
                self._viewer = mj.viewer.launch_passive(self._model, self._data)
            except Exception as exc:
                logger.warning("Viewer unavailable (gui=True ignored): %s", exc)
                self._viewer = None

        off = self._offsets
        logger.info(
            "MuJoCoG1 R2 connected: nbody=%d nu=%d nq=%d qpos=%d ctrl=%d:%d",
            self._model.nbody, self._model.nu, self._model.nq,
            off.pelvis_qpos_adr, off.ctrl_start, off.ctrl_end,
        )

    def _reset_to_stance(self) -> None:
        """Reset physics to default 12-DOF stance at room spawn coordinates."""
        mj = _get_mujoco()
        d = self._data
        m = self._model
        off = self._offsets
        assert d is not None and m is not None and off is not None

        mj.mj_resetData(m, d)

        # Set leg joints to nominal stance angles (from the manifest; byte-
        # identical to _DEFAULT_ANGLES — built in connect()).
        stance = self._stance_qpos
        assert stance is not None
        lq = off.leg_qpos_start
        d.qpos[lq : lq + _NUM_ACTIONS] = stance

        # Place pelvis freejoint at world spawn (from the manifest spawn pose;
        # equals _G1_SPAWN_X/_G1_SPAWN_Y/_G1_PELVIS_Z).
        spawn = self._config.spawn
        qa = off.pelvis_qpos_adr
        d.qpos[qa + 0] = spawn.xy[0]
        d.qpos[qa + 1] = spawn.xy[1]
        d.qpos[qa + 2] = spawn.base_height
        d.qpos[qa + 3] = 1.0  # qw
        d.qpos[qa + 4] = 0.0  # qx
        d.qpos[qa + 5] = 0.0  # qy
        d.qpos[qa + 6] = 0.0  # qz

        mj.mj_forward(m, d)

    def close(self) -> None:
        """Release the runtime (and its renderers) and the MuJoCo resources."""
        rt, self.runtime = self.runtime, None
        if rt is not None:
            rt.close()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        self._model = None
        self._data = None
        self._policy = None
        self._offsets = None
        self._stance_qpos = None

    def disconnect(self) -> None:
        """BaseProtocol teardown — alias of :meth:`close` (idempotent).

        Uniform with MuJoCoGo2.disconnect(). sim_tool.SimTool.stop() calls
        ``base.disconnect()`` on the wrapped base; before this alias existed that
        call silently failed for G1 (only ``close()`` was defined), so the G1 sim
        was never actually torn down on stop. ``close()`` is retained — this is an
        additive alias, NOT a rename. Behavior of close() is unchanged.
        """
        self.close()

    def _require_connection(self) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError("Not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Policy step (core control loop — single-threaded)
    # ------------------------------------------------------------------

    def _pd_control(self, _model: Any, data: Any) -> None:
        """Runtime controller (every physics step): PD toward the policy target."""
        off = self._offsets
        lq, ld = off.leg_qpos_start, off.leg_dof_start
        tau = (self._target - data.qpos[lq : lq + _NUM_ACTIONS]) * _KPS - (
            data.qvel[ld : ld + _NUM_ACTIONS] * _KDS
        )
        data.ctrl[off.ctrl_start : off.ctrl_end] = tau
        self._counter += 1

    def _step_batch(self) -> None:
        """Advance one policy batch: _DECIMATION physics steps + one policy inference."""
        assert self.runtime is not None and self._policy is not None
        self.runtime.step(_DECIMATION)
        self.runtime.call(self._policy_tick)

    def _policy_tick(self, _model: Any, d: Any) -> None:
        """Observation -> policy -> next PD target (at a step boundary); viewer sync."""
        torch = _get_torch()
        off = self._offsets
        lq = off.leg_qpos_start
        ld = off.leg_dof_start
        cmd = self._cmd.copy()

        # Build observation vector (47 elements)
        obs = self._obs
        av = off.angvel_start
        obs[:3] = d.qvel[av : av + 3] * _ANG_VEL_SCALE
        obs[3:6] = _gravity_orientation(d.qpos[off.quat_start : off.quat_start + 4])
        obs[6:9] = cmd * _CMD_SCALE
        obs[9 : 9 + _NUM_ACTIONS] = (
            d.qpos[lq : lq + _NUM_ACTIONS] - _DEFAULT_ANGLES
        ) * _DOF_POS_SCALE
        obs[9 + _NUM_ACTIONS : 9 + 2 * _NUM_ACTIONS] = (
            d.qvel[ld : ld + _NUM_ACTIONS] * _DOF_VEL_SCALE
        )
        obs[9 + 2 * _NUM_ACTIONS : 9 + 3 * _NUM_ACTIONS] = self._action
        phase = (self._counter * _SIM_DT) % _GAIT_PERIOD / _GAIT_PERIOD
        obs[9 + 3 * _NUM_ACTIONS] = math.sin(2.0 * math.pi * phase)
        obs[9 + 3 * _NUM_ACTIONS + 1] = math.cos(2.0 * math.pi * phase)

        with torch.no_grad():
            self._action = (
                self._policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            )
        self._target = self._action * _ACTION_SCALE + _DEFAULT_ANGLES

        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    self._viewer.sync()
                else:
                    self._viewer = None
            except Exception:
                self._viewer = None

    def step(self, n: int = 1) -> None:
        """Step the policy n times (n policy batches = n * DECIMATION physics steps)."""
        self._require_connection()
        for _ in range(n):
            self._step_batch()

    # ------------------------------------------------------------------
    # Velocity / gait commands
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set the current velocity command [vx, vy, vyaw]."""
        self._cmd[0] = float(vx)
        self._cmd[1] = float(vy)
        self._cmd[2] = float(vyaw)

    def stop(self) -> None:
        """Zero the velocity command."""
        self._cmd[:] = 0.0

    def walk(
        self, vx: float, vy: float = 0.0, vyaw: float = 0.0, duration: float = 1.0
    ) -> None:
        """Walk with given velocity command for `duration` seconds, then stop."""
        self._require_connection()
        self.set_velocity(vx, vy, vyaw)
        n_batches = max(1, int(duration / (_SIM_DT * _DECIMATION)))
        for _ in range(n_batches):
            self._step_batch()
        self.stop()

    def get_sim_time(self) -> float:
        """The MuJoCo simulation clock (seconds)."""
        self._require_connection()
        return float(self._data.time)

    def wait_sim(self, seconds: float, cancel: Any = None) -> bool:
        """Advance *seconds* of sim time in policy batches (caller-stepped).

        False when *cancel* (``is_set()``) fires first.
        """
        self._require_connection()
        for _ in range(max(1, round(float(seconds) / (_SIM_DT * _DECIMATION)))):
            if cancel is not None and cancel.is_set():
                return False
            self._step_batch()
        return True

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

        ``bool(result)`` is True only on arrival; ``result.reason`` is one of
        ``NAV_REASONS``. ``timeout`` is sim seconds (None: a geodesic budget);
        per-body tuning is the manifest's ``nav`` block.
        """
        self._require_connection()
        mj = _get_mujoco()
        self.runtime.call(mj.mj_forward)
        obstacles = vg.obstacles_from_model(self._model, self._data, self._offsets.robot_geom_ids)
        res = follow(self, (x, y), obstacles, body=self._config.nav, tol=tol, speed=speed,
                     timeout=timeout, cancel=cancel, on_progress=on_progress)
        logger.info("navigate_to(%.2f, %.2f): %s", x, y, res)
        return res

    # ------------------------------------------------------------------
    # Pose queries
    # ------------------------------------------------------------------

    def get_base_height(self) -> float:
        """Return pelvis z coordinate in world frame."""
        self._require_connection()
        qa = self._offsets.pelvis_qpos_adr  # type: ignore[union-attr]
        return float(self._data.qpos[qa + 2])

    def get_position(self) -> list[float]:
        """Return pelvis [x, y, z] in world frame (BaseProtocol accessor)."""
        self._require_connection()
        qa = self._offsets.pelvis_qpos_adr  # type: ignore[union-attr]
        q = self._data.qpos
        return [float(q[qa]), float(q[qa + 1]), float(q[qa + 2])]

    def get_heading(self) -> float:
        """Return yaw (radians) from pelvis quaternion (qw, qx, qy, qz)."""
        self._require_connection()
        off = self._offsets
        assert off is not None
        q = self._data.qpos[off.quat_start : off.quat_start + 4]
        qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_velocity(self) -> list[float]:
        """Return pelvis linear velocity [vx, vy, vz] in world frame.

        BaseProtocol accessor, uniform with MuJoCoGo2.get_velocity() (which reads
        qvel[0:3]). The pelvis freejoint linear DoFs sit at qvel[pelvis_dof_adr:+3]
        in the combined scene.
        """
        self._require_connection()
        off = self._offsets
        assert off is not None
        dof = off.pelvis_dof_adr
        v = self._data.qvel
        return [float(v[dof]), float(v[dof + 1]), float(v[dof + 2])]

    def get_odometry(self) -> Any:
        """Return full odometry snapshot as Odometry dataclass.

        BaseProtocol accessor, uniform with MuJoCoGo2.get_odometry(). Pose comes
        from the pelvis freejoint qpos (x,y,z + wxyz quat), velocity from its qvel
        (linear xyz + angular; vyaw is the world-z angular rate). Quaternion is
        MuJoCo (w,x,y,z); Odometry stores qx/qy/qz/qw like the go2 path.
        """
        self._require_connection()
        from vector_os_nano.core.types import Odometry  # noqa: PLC0415
        off = self._offsets
        assert off is not None
        qa = off.pelvis_qpos_adr
        dof = off.pelvis_dof_adr
        q = self._data.qpos
        v = self._data.qvel
        return Odometry(
            timestamp=float(self._data.time),
            x=float(q[qa]),
            y=float(q[qa + 1]),
            z=float(q[qa + 2]),
            qw=float(q[qa + 3]),
            qx=float(q[qa + 4]),
            qy=float(q[qa + 5]),
            qz=float(q[qa + 6]),
            vx=float(v[dof]),
            vy=float(v[dof + 1]),
            vz=float(v[dof + 2]),
            vyaw=float(v[dof + 5]),
        )

    # ------------------------------------------------------------------
    # Lidar
    # ------------------------------------------------------------------

    def get_lidar_scan(self) -> Any:
        """Cast lidar rays from head height, return LaserScan + 3D points.

        Thin wrapper around :func:`~vector_os_nano.hardware.sim.sensors.g1_lidar.g1_lidar_scan`.
        Lidar mounted at ~1.51 m (pelvis_z + LIDAR_OFFSET_Z) — above all g1 leg
        geoms.  Uses the live pelvis pose so the scan is valid while walking.
        """
        self._require_connection()
        d = self._data
        m = self._model
        off = self._offsets
        assert d is not None and m is not None and off is not None

        qa = off.pelvis_qpos_adr
        pos_base = d.qpos[qa : qa + 3].copy().astype(np.float64)
        heading = self.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        pos_lidar = np.array(
            [
                float(pos_base[0]) + cos_h * _LIDAR_OFFSET_X,
                float(pos_base[1]) + sin_h * _LIDAR_OFFSET_X,
                float(pos_base[2]) + _LIDAR_OFFSET_Z,
            ],
            dtype=np.float64,
        )

        result = g1_lidar_scan(
            m,
            d,
            pelvis_bid=off.pelvis_bid,
            robot_geom_ids=off.robot_geom_ids,
            pos_lidar=pos_lidar,
            heading=heading,
        )
        self._last_scan = result
        self._last_pointcloud = result.points_3d
        return result

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _resolve_camera(self) -> str:
        """The head camera's name in the loaded scene ('g1_head_rgb' or any '*head_rgb')."""
        mj = _get_mujoco()
        m = self._model
        names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(m.ncam)]
        if _SCENE_CAM_NAME in names:
            return _SCENE_CAM_NAME
        for nm in names:
            if "head_rgb" in nm:
                return nm
        raise RuntimeError(f"Camera '{_SCENE_CAM_NAME}' not found. Available: {names}")

    def get_camera_frame(self, width: int = 640, height: int = 480) -> "np.ndarray":
        """Render RGB from the g1_head_rgb camera (runtime render thread, any caller).

        Returns (H, W, 3) uint8 numpy array in RGB order, at the requested size.
        """
        self._require_connection()
        return self.runtime.render(self._cam_name, width, height, ("rgb",)).rgb

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (cam_xpos, cam_xmat) for the g1_head_rgb camera.

        cam_xpos: (3,) world position.
        cam_xmat: (9,) row-major rotation matrix; -Z column is optical axis.
        """
        self._require_connection()
        cam_id = self._model.cam(self._cam_name).id
        return self.runtime.call(
            lambda _m, d: (d.cam_xpos[cam_id].copy(), d.cam_xmat[cam_id].copy())
        )

    # ------------------------------------------------------------------
    # Scene builder (external callers / tests)
    # ------------------------------------------------------------------

    @staticmethod
    def build_scene() -> Path:
        """(Re)build the g1-12dof-in-room scene XML and return its path."""
        return _build_g1_room_scene_xml()
