# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""G1MuJoCoBase — real humanoid gait via the unitree_rl_gym pretrained policy.

Campaign #5 (DQ-9 approved): the owner saw the habitat G1 GLIDE (rigid-body
kinematics); this base makes it WALK — a 12-DOF pretrained TorchScript policy
(motion.pt, BSD-3, runs at 50 Hz over 500 Hz MuJoCo physics) drives real
stepping, and every odometry/motion-evidence number comes from the PHYSICS
state, never from command echo.

Control architecture (the proven streaming shape from go2/habitat):
- a background thread steps physics at 1× wall time (sim dt 0.002 s);
- every 10 physics steps the policy turns (vx, vy, vyaw) + proprioception
  into 12 joint targets (PD constants from the upstream g1.yaml);
- ``set_velocity`` is non-blocking with a 0.6 s deadman (a dead client
  never leaves the robot marching); ``walk``/``stop`` build on it.

Assets are NOT vendored: ``scripts/setup_g1_gait.sh`` fetches the policy +
MJCF (~57 MB) into ``assets/g1_gait/`` (gitignored) at a pinned commit.
Obs layout (47) reverse-engineered from upstream deploy_mujoco.py:
[3 ang_vel, 3 gravity, 3 cmd, 12 q, 12 dq, 12 prev_action, 2 gait_phase].
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "g1_gait"

# Upstream deploy constants (deploy_mujoco/configs/g1.yaml @ pinned commit).
_SIM_DT = 0.002
_DECIMATION = 10                       # 50 Hz policy
_KPS = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
_KDS = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
_DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float32)
_ANG_VEL_SCALE = 0.25
_DOF_POS_SCALE = 1.0
_DOF_VEL_SCALE = 0.05
_ACTION_SCALE = 0.25
_CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
_NUM_ACTIONS = 12
_NUM_OBS = 47
_GAIT_PERIOD = 0.8
_DEADMAN_S = 0.6                       # cmd_vel keep-alive (go2/habitat lesson)
_VIEWER_FPS = 30                       # live-window render cap (decoupled from
                                       # 500 Hz physics — see _last_sync)

# --- navigate_to closed-loop constants (campaign #6) ------------------------
# Tuned from a sandbox characterization spike of THIS policy's command->motion
# map (workflow wqknh9p8u finding #1: the policy tracks a SCALED joystick cmd,
# not a rad/s — gains must come from measurement, not theory). Measured:
# vyaw cmd 0.6 -> ~0.29 rad/s achieved; vx=0 turn-in-place is STABLE (does NOT
# stagger — the review's fear was wrong for this policy); base z ~0.77 standing.
_NAV_TOL_FLOOR = 0.30        # gait COM oscillation floor — tol can't beat this
_NAV_VYAW_MAX = 0.6          # cmd ceiling (~0.29 rad/s achieved)
_NAV_K_YAW = 2.0             # proportional heading gain (saturates ~0.3 rad err)
_NAV_FACE_TOL = 0.35         # rad: pivot-in-place until aligned within this
_NAV_YAW_DEADBAND = 0.12     # rad: stop steering when aligned (anti-hunt)
_NAV_SPEED = 0.5             # default forward cmd
_NAV_CAPTURE_R = 0.5         # m: inside, freeze steering + drive straight
_NAV_FALL_Z = 0.4            # m: base height below this = fallen
_NAV_TICK_S = 0.05           # 20 Hz: 12x deadman margin
_NAV_SETTLE_S = 1.0          # hold stance + re-sample before deciding arrival
_NAV_TIMEOUT_S = 60.0
_NAV_STALL_WINDOW_S = 6.0    # forward-walk no-progress window
# Obstacle routing (campaign #8 R3, room mode). Inflate obstacles by the G1
# body radius + clearance so the planned path keeps the torso off the geometry;
# intermediate waypoints only need to be rounded loosely (the final goal uses
# the real tol). Body half-width ~0.2 m + margin.
_NAV_INFLATION = 0.40
_NAV_WAYPOINT_TOL = 0.45
_NAV_STALL_MIN_M = 0.10      # min net progress within the window


def g1_assets_ready() -> bool:
    """True when scripts/setup_g1_gait.sh has installed the assets."""
    return (_ASSET_DIR / "motion.pt").exists() and (
        _ASSET_DIR / "scene.xml").exists()


def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ], dtype=np.float32)


class G1MuJoCoBase:
    """BaseProtocol over a policy-driven walking G1 in MuJoCo.

    ``supports_holonomic`` is True — the policy takes a real vy (unlike the
    habitat kinematic base), so lateral walks are honest motion here.
    """

    supports_holonomic = True
    supports_lidar = False           # BaseProtocol conformance (no lidar)

    def __init__(
        self,
        asset_dir: "Path | str | None" = None,
        gui: bool = False,
        room: bool = False,
    ) -> None:
        self._asset_dir = Path(asset_dir) if asset_dir else _ASSET_DIR
        if not (self._asset_dir / "motion.pt").exists():
            raise FileNotFoundError(
                f"G1 gait assets missing at {self._asset_dir} — run "
                f"scripts/setup_g1_gait.sh first (downloads the pretrained "
                f"policy + MJCF; never vendored into git)")
        self._lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._cmd_stamp = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._model = None
        self._data = None
        self._policy = None
        # Thread-safe pose SNAPSHOT (campaign #5 R3, workflow-confirmed
        # critical): mj_step writes the 37-float qpos in ONE native C call the
        # GIL does NOT interrupt, so a reader touching self._data.qpos mid-step
        # gets a torn vector that never existed in any physics state. The
        # control thread publishes a consistent (qpos[:7], qvel[:6]) copy under
        # _snap_lock once per policy batch; readers consume ONLY the snapshot,
        # never self._data. Lower contention than locking the whole mj_step.
        self._snap_lock = threading.Lock()
        self._snap = None  # (qpos7: np.ndarray, qvel6: np.ndarray)
        # On-demand viewer-frame render (campaign #5 batch 3). ALL mujoco
        # access — including the Renderer's GL context + update_scene reading
        # mjData — stays on the control thread (Case 12 thread discipline; a
        # cross-thread render would tear the pose or crash the GL context).
        # The caller requests a frame via an Event and the control loop renders
        # it between policy batches. One render in flight (serialized).
        self._render_lock = threading.Lock()
        self._render_req = threading.Event()
        self._render_done = threading.Event()
        self._render_png: "bytes | None" = None
        self._renderer = None  # lazily created on the control thread
        # Live passive viewer window (campaign #8 R0). The owner controls the
        # G1 entirely through vector-cli and must SEE the gait, not just an
        # offscreen PNG. The window is launched in connect() and synced by the
        # control thread that already owns mjData — same Case 12 discipline as
        # the offscreen render (a cross-thread viewer.sync() would tear the
        # pose / crash GL). Only safe in Linux/Windows background-daemon mode;
        # under mjpython (macOS) GLFW is main-thread-only, so we degrade to
        # headless there (the gait still runs; the owner is on Linux).
        self._gui: bool = gui
        self._viewer = None
        # When a live window is open the control loop runs in PUMP mode on the
        # caller thread, NOT a daemon (a passive viewer's render thread starves
        # a background control thread to ~0.4x real time — the owner's "卡";
        # caller-thread pumping holds 1.0x). Resolved in connect().
        self._pump_mode: bool = False
        # Viewer render is rate-capped to _VIEWER_FPS (wall-clock gated) so the
        # ~5-8 ms sync cost never dominates the 20 ms physics batch — smooth
        # window without touching the 500 Hz fidelity the policy needs.
        self._sync_interval: float = 1.0 / _VIEWER_FPS
        self._last_sync: float = 0.0
        # Per-batch control state (instance-scoped so _step_batch can be driven
        # by EITHER the daemon thread or the caller-thread pump).
        self._action = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        self._target = _DEFAULT_ANGLES.copy()
        self._obs = np.zeros(_NUM_OBS, dtype=np.float32)
        self._counter = 0
        self._mujoco = None
        self._torch = None
        # Room mode (campaign #8 R3): a closed MJCF room with walls + obstacle
        # boxes (REAL collision) + labeled targets, instead of the flat scene.
        # Brings real obstacle avoidance + a virtual lidar into the G1 world,
        # all substrate-agnostic (MuJoCo physics is reused under DQ-10 A or D).
        self._room: bool = room
        # Routing polygons enumerated from the compiled model at connect()
        # (g1_vgraph.obstacles_from_model) — the single source of truth for both
        # the path navigate_to walks AND the geodesic verify reads (rule 5).
        self._obstacles: list = []
        # Virtual Livox-360 lidar (campaign #8 R3). Stepped ONLY on the control
        # thread (Case 12/13 — it holds a mjData ref); the latest scan is
        # published under a lock and cross-thread readers consume the snapshot.
        self._lidar = None
        self._lidar_snap_lock = threading.Lock()
        self._lidar_snap = None     # latest LidarSample, or None
        self._occ = None            # OccupancyGrid (room mode), filled by observe()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        import mujoco
        import torch

        self._mujoco = mujoco
        self._torch = torch
        if self._room:
            # Room scene built programmatically (walls + obstacles + targets)
            # from the flat gait scene — keeps asset paths intact (R1 spike).
            from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
            self._model = g1_room.build_room_model(self._asset_dir)
        else:
            self._model = mujoco.MjModel.from_xml_path(
                str(self._asset_dir / "scene.xml"))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = _SIM_DT
        self._policy = torch.jit.load(str(self._asset_dir / "motion.pt"))
        # Publish the initial pose so readers before the first policy batch
        # still get a real snapshot (not None).
        self._snap = (self._data.qpos[:7].copy(), self._data.qvel[:6].copy())
        if self._room:
            # Enumerate routing geometry + spin up the lidar once the model is
            # compiled. mj_forward populates geom_xpos for the obstacle scan.
            mujoco.mj_forward(self._model, self._data)
            from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
            from vector_os_nano.hardware.sim.sensors.lidar360 import (  # noqa: PLC0415,E501
                MuJoCoLivox360,
            )
            self._obstacles = g1_room.obstacles_from_model(
                self._model, self._data)
            try:
                # max_range 3 m + include_misses: a limited range leaves
                # UNKNOWN beyond the sensing disc → real frontier exploration
                # (the robot must MOVE to grow the map). include_misses frees
                # the full ray span even into open space, so OPEN directions are
                # mapped (not just where a wall was struck) — without it a
                # short-range lidar leaves a blank map and explore cannot
                # progress. geom_group masks the env so rays ignore the robot.
                self._lidar = MuJoCoLivox360(
                    self._model, self._data, body_name="pelvis",
                    max_range=3.0, geom_group=g1_room.ENV_GEOM_GROUP,
                    include_misses=True)
            except Exception as exc:  # noqa: BLE001 — lidar is non-fatal
                logger.warning("G1 lidar init failed: %s", exc)
                self._lidar = None
            # Occupancy grid for autonomous exploration (campaign #8 R6). It is
            # filled by observe() on the CALLER thread (snapshot reads, NOT the
            # control thread) — ray-marching 5760 rays at 10 Hz would starve the
            # gait (the R0/Case-13 lesson), so mapping stays off the hot path.
            from vector_os_nano.hardware.sim.occupancy import OccupancyGrid  # noqa: PLC0415,E501
            self._occ = OccupancyGrid(*g1_room.room_bounds(), resolution=0.25)
        # Live viewer window (campaign #8 R0): open it on the caller thread.
        # Skip under mjpython (macOS main-thread-only GLFW); launch failure
        # (no display / GL) degrades to headless without breaking the boot.
        if self._gui:
            self._launch_viewer(mujoco)
        self._running = True
        # Mode resolution: a live window → PUMP (caller thread drives the loop
        # via _advance, no daemon, so the gait runs 1x against the viewer's
        # render thread). No window → DAEMON (the proven headless path).
        self._pump_mode = self._viewer is not None
        if self._pump_mode:
            self._target = _DEFAULT_ANGLES.copy()   # fresh stance reference
            self._advance(0.3)                       # settle into stance (pump)
        else:
            self._thread = threading.Thread(
                target=self._control_loop, daemon=True, name="g1-gait-control")
            self._thread.start()
            time.sleep(0.3)   # let the daemon settle the policy into stance

    def _launch_viewer(self, mujoco) -> None:
        """Open a passive viewer window (caller thread), Linux/Windows only.

        Skipped under mjpython (macOS main-thread-only GLFW); any launch
        failure (no display, GL unavailable) degrades to headless with a
        warning so the boot never breaks. The control thread drives sync().
        """
        from vector_os_nano.hardware.sim.viewer_mode import running_under_mjpython

        if running_under_mjpython():
            logger.warning(
                "G1MuJoCoBase: gui requested but running under mjpython — "
                "the gait control thread owns mjData and a main-thread-only "
                "viewer cannot sync from it; staying headless.")
            return
        try:
            import mujoco.viewer  # noqa: PLC0415
            self._viewer = mujoco.viewer.launch_passive(
                self._model, self._data,
                show_left_ui=False, show_right_ui=False)
            cam = self._viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 3.5
            cam.elevation = -20
            cam.azimuth = 120
            cam.lookat[:] = self._data.qpos[:3]
        except Exception as exc:  # noqa: BLE001 — never break the boot
            logger.warning("G1MuJoCoBase viewer failed to launch: %s", exc)
            self._viewer = None

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001 — best-effort window teardown
                pass
            self._viewer = None
        with self._snap_lock:
            self._snap = None     # post-disconnect readers raise, not crash
        with self._lidar_snap_lock:
            self._lidar_snap = None
        self._lidar = None
        self._occ = None
        self._model = self._data = self._policy = None

    close = disconnect

    def _require_connected(self) -> None:
        if self._data is None or not self._running:
            raise RuntimeError("G1MuJoCoBase not connected (call connect())")

    # -- on-demand viewer frame (control-thread render) --------------------
    def _render_tracking_frame(self, m, d, h: int = 480, w: int = 640) -> bytes:
        """Render a chase-camera frame of the robot to PNG (control thread)."""
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(m, height=h, width=w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = d.qpos[:3]
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
        self._renderer.update_scene(d, camera=cam)
        rgb = self._renderer.render()
        import io

        import PIL.Image
        buf = io.BytesIO()
        PIL.Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def viewer_frame_png(self, timeout: float = 5.0) -> bytes:
        """A chase-camera PNG of the current pose (acceptance/owner artifact).

        Rendered ON the control thread (mujoco thread discipline); the caller
        blocks until it lands. Serialized — one render in flight."""
        self._require_connected()
        with self._render_lock:
            self._render_done.clear()
            self._render_req.set()
            if not self._render_done.wait(timeout=timeout):
                raise RuntimeError("g1 viewer frame render timed out")
            png = self._render_png
        if not png:
            raise RuntimeError("g1 viewer frame render produced no image")
        return png

    # -- control loop (one thread owns ALL mujoco/torch state) -------------
    # The control loop runs in ONE of two modes (resolved in connect()):
    #   • DAEMON  (headless / no window): a background thread runs _control_loop
    #     — the proven campaign #5-#7 path, byte-identical, all tests use it.
    #   • PUMP    (live window open): NO daemon. The caller thread (the REPL /
    #     skill) drives _step_batch() via _advance() during walk/turn/navigate.
    #     Measured necessity: a passive viewer spawns an internal render thread
    #     that STARVES a background control thread to ~0.4x real time (the
    #     owner's "卡"); the SAME loop on the caller thread holds 1.0x. Both
    #     modes call the identical _step_batch — only WHO drives it differs.
    def _step_batch(self) -> None:
        """Advance one policy batch (10 physics steps), update obs/policy/
        snapshot, and sync the live viewer. Owns mjData — Case 12 discipline:
        only the single thread driving this (daemon OR caller) touches it."""
        mujoco, torch = self._mujoco, self._torch
        m, d = self._model, self._data
        with self._lock:
            cmd = self._cmd.copy()
            stale = (time.monotonic() - self._cmd_stamp) > _DEADMAN_S
        if stale:
            cmd[:] = 0.0               # deadman: dead client never marches
        target = self._target
        for _ in range(_DECIMATION):
            tau = ((target - d.qpos[7:]) * _KPS - d.qvel[6:] * _KDS)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            self._counter += 1
        obs = self._obs
        obs[:3] = d.qvel[3:6] * _ANG_VEL_SCALE
        obs[3:6] = _gravity_orientation(d.qpos[3:7])
        obs[6:9] = cmd * _CMD_SCALE
        obs[9:9 + _NUM_ACTIONS] = (
            (d.qpos[7:] - _DEFAULT_ANGLES) * _DOF_POS_SCALE)
        obs[9 + _NUM_ACTIONS:9 + 2 * _NUM_ACTIONS] = (
            d.qvel[6:] * _DOF_VEL_SCALE)
        obs[9 + 2 * _NUM_ACTIONS:9 + 3 * _NUM_ACTIONS] = self._action
        phase = (self._counter * _SIM_DT) % _GAIT_PERIOD / _GAIT_PERIOD
        obs[9 + 3 * _NUM_ACTIONS] = math.sin(2 * math.pi * phase)
        obs[9 + 3 * _NUM_ACTIONS + 1] = math.cos(2 * math.pi * phase)
        with torch.no_grad():
            self._action = self._policy(
                torch.from_numpy(obs).unsqueeze(0)
            ).numpy().squeeze()
        self._target = self._action * _ACTION_SCALE + _DEFAULT_ANGLES
        # Publish a consistent pose snapshot for cross-thread readers.
        with self._snap_lock:
            self._snap = (d.qpos[:7].copy(), d.qvel[:6].copy())
        # Virtual lidar: step ON THIS thread (owns mjData — Case 12/13), rate
        # self-gated by the sensor (due()), and publish the scan as a snapshot
        # so cross-thread readers never touch mjData. Non-fatal on error.
        if self._lidar is not None and self._lidar.due():
            try:
                scan = self._lidar.step()
                with self._lidar_snap_lock:
                    self._lidar_snap = scan
            except Exception as exc:  # noqa: BLE001 — never break the gait
                logger.warning("g1 lidar step failed: %s", exc)
        # Live viewer window: sync on THIS thread (owns mjData — Case 12), at
        # most _VIEWER_FPS/sec (wall-clock gated). Chase the base so the owner
        # always sees the walking robot. If the owner closed the window, drop
        # the handle and keep stepping (a closed window must never stall gait).
        if self._viewer is not None:
            now_s = time.monotonic()
            if now_s - self._last_sync >= self._sync_interval:
                try:
                    if self._viewer.is_running():
                        self._viewer.cam.lookat[:] = d.qpos[:3]
                        self._viewer.sync()
                        self._last_sync = now_s
                    else:
                        self._viewer = None
                except Exception as exc:  # noqa: BLE001 — never break gait
                    logger.warning("g1 viewer sync failed: %s", exc)
                    self._viewer = None
        # On-demand viewer frame: render on THIS thread (owns mjData + GL).
        if self._render_req.is_set():
            try:
                self._render_png = self._render_tracking_frame(m, d)
            except Exception as exc:  # noqa: BLE001 — never break the gait
                logger.warning("g1 render failed: %s", exc)
                self._render_png = None
            self._render_req.clear()
            self._render_done.set()

    def _control_loop(self) -> None:
        # DAEMON mode only. Pace in POLICY-PERIOD batches (10 physics steps =
        # 20 ms), one sleep per batch against an absolute deadline. Per-step
        # sleeping ran the sim at ~0.5x real time: time.sleep()'s ~1-2 ms
        # granularity error, paid 500x/s, dwarfed the 2 ms step budget. 50
        # sleeps/s amortizes it; the absolute next_tick self-corrects drift.
        batch_dt = _SIM_DT * _DECIMATION
        next_tick = time.monotonic()
        while self._running:
            self._step_batch()
            # pace to 1x wall time: coarse sleep, then spin the last ~3 ms —
            # time.sleep() overshoots multiple ms on a desktop kernel and at
            # 50 batches/s that alone dragged the sim to ~0.67x wall.
            next_tick += batch_dt
            remaining = next_tick - time.monotonic()
            if remaining > 0.004:
                time.sleep(remaining - 0.003)
            if remaining < -0.2:
                next_tick = time.monotonic()  # fell far behind — resync
            else:
                while time.monotonic() < next_tick:
                    pass

    def _advance(self, seconds: float) -> None:
        """Let ``seconds`` of wall-time pass with physics advancing.

        DAEMON mode: the background thread is already stepping, so just sleep.
        PUMP mode: there is no daemon — drive _step_batch() on THIS (the
        caller's) thread at 1x wall time for the duration, so the gait both
        animates and renders on the viewer-friendly main thread.
        """
        seconds = max(0.0, float(seconds))
        if not self._pump_mode:
            time.sleep(seconds)
            return
        batch_dt = _SIM_DT * _DECIMATION
        deadline = time.monotonic() + seconds
        next_tick = time.monotonic()
        while self._running and time.monotonic() < deadline:
            self._step_batch()
            next_tick += batch_dt
            remaining = next_tick - time.monotonic()
            if remaining > 0.004:
                # clamp >= 0: past the deadline, (deadline - now) goes negative
                # and time.sleep() would raise 'sleep length must be negative'.
                time.sleep(max(0.0, min(remaining - 0.003,
                                        deadline - time.monotonic())))
            if remaining < -0.2:
                next_tick = time.monotonic()
            else:
                while time.monotonic() < next_tick < deadline + batch_dt:
                    pass

    def _snapshot(self):
        """A consistent (qpos7, qvel6) copy, or raise if disconnected.

        Collapses the connected-check and the read into ONE locked region so
        a concurrent disconnect() can't null the snapshot between guard and
        use (TOCTOU, workflow-confirmed). Readers NEVER touch self._data.
        """
        with self._snap_lock:
            snap = self._snap
        if snap is None:
            raise RuntimeError("G1MuJoCoBase not connected (call connect())")
        return snap

    def get_lidar_scan(self):
        """Latest virtual-lidar :class:`LidarSample` (room mode), or None.

        Reads the snapshot the control thread publishes — never touches mjData
        (Case 12/13). None when no lidar is attached (flat scene) or before the
        first scan."""
        with self._lidar_snap_lock:
            return self._lidar_snap

    def list_targets(self) -> "dict[str, tuple[float, float]]":
        """Labeled target objects in the room ({name: (x, y)}), empty if flat."""
        if not self._room:
            return {}
        from vector_os_nano.hardware.sim import g1_room  # noqa: PLC0415
        return {t.name: (t.cx, t.cy) for t in g1_room.TARGETS}

    def observe(self) -> float:
        """Integrate the current lidar scan into the occupancy grid and return
        the new coverage fraction (campaign #8 R6 — autonomous exploration).

        Runs on the CALLER thread: it only consumes the lidar SNAPSHOT and the
        pose snapshot (never mjData), so it never contends the control thread /
        the gait (the R0 lesson). No-op (returns 0.0) outside room mode."""
        if self._occ is None:
            return 0.0
        scan = self.get_lidar_scan()
        if scan is not None and getattr(scan, "num_points", 0) > 0:
            px, py, _pz = self.get_position()
            self._occ.update_from_scan((px, py), scan.points)
        return self._occ.coverage()

    def get_occupancy(self):
        """The live OccupancyGrid (room mode), or None on the flat scene."""
        return self._occ

    # -- BaseProtocol -------------------------------------------------------
    def set_velocity(self, vx: float, vy: float = 0.0,
                     vyaw: float = 0.0) -> None:
        """Non-blocking streaming command; 0.6 s deadman keep-alive."""
        self._require_connected()
        with self._lock:
            self._cmd[:] = (float(vx), float(vy), float(vyaw))
            self._cmd_stamp = time.monotonic()

    def walk(self, vx: float, vy: float = 0.0, vyaw: float = 0.0,
             duration: float = 1.0) -> bool:
        """Timed walk: stream the command for ``duration`` wall-seconds.

        Motion evidence comes from the caller reading get_position() before/
        after (WalkSkill contract) — REAL physics displacement, not echo.
        """
        self._require_connected()
        deadline = time.monotonic() + max(0.0, float(duration))
        while time.monotonic() < deadline:
            self.set_velocity(vx, vy, vyaw)   # re-arm the deadman
            self._advance(0.1)                # daemon: sleep; pump: step gait
        self.stop()
        return True

    def navigate_to(self, x: float, y: float, tol: float = 0.2,
                    speed: float = _NAV_SPEED) -> dict:
        """Drive to world (x, y), routing AROUND obstacles when in room mode.

        Flat scene (no obstacles): delegates straight to the single-point
        controller — campaign #6 behaviour unchanged. Room mode: plans an
        obstacle-avoiding waypoint chain (g1_vgraph.plan_path over the geometry
        enumerated at connect) and drives each leg with the same controller;
        intermediate legs use a loose tol, the final leg the real tol. The
        three-value contract is preserved (moved_m accrues across legs; net_m
        is the start->end displacement; geodesic = the planned path length).
        An honest 'unreachable' (goal inside an inflated obstacle / boxed in)
        returns reached=False reason=unreachable — never a phantom straight
        line (rule 5)."""
        import math
        self._require_connected()
        if not self._obstacles:
            return self._navigate_point(x, y, tol, speed)
        if not all(math.isfinite(v) for v in (x, y, tol, speed)):
            return {"reached": False, "already_there": False, "moved_m": 0.0,
                    "elapsed_s": 0.0, "remaining": float("inf"),
                    "reason": "bad_params_nan", "transport": "sim_oracle"}
        from vector_os_nano.hardware.sim import g1_vgraph  # noqa: PLC0415
        sx, sy, _sz = self.get_position()
        waypoints, _length = g1_vgraph.plan_path(
            (sx, sy), (float(x), float(y)), self._obstacles, _NAV_INFLATION)
        if waypoints is None:
            return {"reached": False, "already_there": False, "moved_m": 0.0,
                    "elapsed_s": 0.0, "remaining": float("inf"),
                    "reason": "unreachable", "transport": "sim_oracle"}
        # Drive each leg; waypoints[0] is the start, so skip it. Intermediate
        # waypoints use a loose tol (just round the corner), the goal the real.
        t0 = time.monotonic()
        total_moved = 0.0
        legs = waypoints[1:]
        last = None
        for i, (wx, wy) in enumerate(legs):
            is_final = (i == len(legs) - 1)
            leg_tol = tol if is_final else _NAV_WAYPOINT_TOL
            last = self._navigate_point(wx, wy, leg_tol, speed)
            total_moved += float(last.get("moved_m", 0.0))
            if last.get("reason") in ("fell", "stalled_no_progress"):
                break
        fx, fy, fz = self.get_position()
        remaining = math.hypot(float(x) - fx, float(y) - fy)
        eff_tol = max(float(tol), _NAV_TOL_FLOOR)
        reached = bool(remaining <= eff_tol and fz >= _NAV_FALL_Z)
        return {
            "reached": reached, "already_there": False,
            "moved_m": round(total_moved, 3),
            "net_m": round(math.hypot(fx - sx, fy - sy), 3),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "remaining": round(remaining, 3),
            "pos": [fx, fy, fz], "effective_tol": eff_tol,
            "waypoints": [[round(p[0], 2), round(p[1], 2)] for p in waypoints],
            "reason": "ok" if reached else (last or {}).get("reason", "timeout"),
            "transport": "sim_oracle",
        }

    def _navigate_point(self, x: float, y: float, tol: float = 0.2,
                        speed: float = _NAV_SPEED) -> dict:
        """Closed-loop face->walk->arrive drive to a SINGLE world (x, y).
        The campaign #6 controller; navigate_to chains it over planned
        waypoints in room mode. Returns the three-value contract
        {reached, already_there, moved_m, elapsed_s, remaining, pos, reason,
        transport}.

        Runs ENTIRELY on the caller thread — only the public primitives
        (set_velocity / get_position / get_heading / stop), NEVER mjData (the
        50 Hz control thread owns it; reads go through the pose snapshot).
        Constants are measured (see _NAV_* and the campaign #6 spike); the
        adversarial review's gait traps are handled explicitly: a tol FLOOR at
        the gait's COM-oscillation amplitude, a heading dead-band (anti-hunt),
        a capture mode near the goal (freeze steering, drive straight — no
        pivot-orbit), a SETTLE phase before deciding arrival (the biped coasts
        past the break), fall detection, a mode-gated stall detector, and a NaN
        guard. ``moved_m`` is real path length; ``net_m`` the start->end
        displacement (an orbit inflates the former, not the latter).
        """
        import math

        self._require_connected()
        if not all(math.isfinite(v) for v in (x, y, tol, speed)):
            return {"reached": False, "already_there": False, "moved_m": 0.0,
                    "elapsed_s": 0.0, "remaining": float("inf"),
                    "reason": "bad_params_nan", "transport": "sim_oracle"}
        tx, ty = float(x), float(y)
        eff_tol = max(float(tol), _NAV_TOL_FLOOR)   # gait floor, reported below
        speed = max(0.1, float(speed))

        sx, sy, sz = self.get_position()
        start_dist = math.hypot(tx - sx, ty - sy)
        if start_dist <= eff_tol:
            # already there — never command the gait (no spurious steps)
            return {"reached": True, "already_there": True, "moved_m": 0.0,
                    "net_m": 0.0, "elapsed_s": 0.0, "remaining": start_dist,
                    "pos": [sx, sy, sz], "effective_tol": eff_tol,
                    "reason": "already_within_tol", "transport": "sim_oracle"}

        t0 = time.monotonic()
        px, py = sx, sy
        moved = 0.0
        best_dist = start_dist
        best_t = t0
        reason = "timeout"

        def _wrap(a: float) -> float:
            return math.atan2(math.sin(a), math.cos(a))

        try:
            while time.monotonic() - t0 < _NAV_TIMEOUT_S:
                cx, cy, cz = self.get_position()
                yaw = self.get_heading()
                moved += math.hypot(cx - px, cy - py)
                px, py = cx, cy
                dist = math.hypot(tx - cx, ty - cy)

                if cz < _NAV_FALL_Z:           # fallen — stop trying
                    reason = "fell"
                    break
                if dist <= eff_tol:
                    reason = "arrived"
                    break

                err = _wrap(math.atan2(ty - cy, tx - cx) - yaw)
                if dist < _NAV_CAPTURE_R:
                    # capture: stop steering (yaw error near goal is noise),
                    # creep straight at the last heading — no pivot-orbit.
                    vx, vyaw = max(0.15, 0.4 * speed), 0.0
                elif abs(err) > _NAV_FACE_TOL:
                    # face-first: pivot in place (measured stable for this
                    # policy), don't walk wide of a badly-aimed target.
                    vx = 0.0
                    vyaw = max(-_NAV_VYAW_MAX, min(_NAV_VYAW_MAX,
                                                   _NAV_K_YAW * err))
                else:
                    vx = speed
                    vyaw = (0.0 if abs(err) < _NAV_YAW_DEADBAND
                            else max(-_NAV_VYAW_MAX, min(_NAV_VYAW_MAX,
                                                         _NAV_K_YAW * err)))
                self.set_velocity(vx, 0.0, vyaw)

                # stall: only judged while actually walking forward (a long
                # legitimate pivot is not a stall). Net progress toward goal.
                if vx > 0.0:
                    if dist < best_dist - _NAV_STALL_MIN_M:
                        best_dist, best_t = dist, time.monotonic()
                    elif time.monotonic() - best_t > _NAV_STALL_WINDOW_S:
                        reason = "stalled_no_progress"
                        break
                else:
                    best_t = time.monotonic()   # pause stall clock while pivoting
                self._advance(_NAV_TICK_S)
        finally:
            self.stop()

        # Settle: hold commanded stance ~1 gait period and re-sample the
        # AUTHORITATIVE arrival pose — the biped coasts past the in-flight
        # break sample (review high-finding). moved_m accrues across settle.
        settle_end = time.monotonic() + _NAV_SETTLE_S
        while time.monotonic() < settle_end:
            self.set_velocity(0.0, 0.0, 0.0)   # re-arm deadman at zero
            cx, cy, _cz = self.get_position()
            moved += math.hypot(cx - px, cy - py)
            px, py = cx, cy
            self._advance(_NAV_TICK_S)

        fx, fy, fz = self.get_position()
        remaining = math.hypot(tx - fx, ty - fy)
        reached = bool(remaining <= eff_tol and fz >= _NAV_FALL_Z)
        if reached:
            reason = "ok"
        return {
            "reached": reached, "already_there": False,
            "moved_m": round(moved, 3),
            "net_m": round(math.hypot(fx - sx, fy - sy), 3),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "remaining": round(remaining, 3),
            "pos": [fx, fy, fz], "effective_tol": eff_tol,
            "reason": reason, "transport": "sim_oracle",
        }

    def stop(self) -> None:
        self._require_connected()
        with self._lock:
            self._cmd[:] = 0.0
            self._cmd_stamp = time.monotonic()

    def get_position(self) -> "list[float]":
        q, _ = self._snapshot()
        return [float(q[0]), float(q[1]), float(q[2])]

    def get_heading(self) -> float:
        q, _ = self._snapshot()
        qw, qx, qy, qz = q[3], q[4], q[5], q[6]
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

    def get_velocity(self) -> "list[float]":
        """Body linear velocity [vx, vy, vz] from physics state."""
        _, v = self._snapshot()
        return [float(v[0]), float(v[1]), float(v[2])]

    def geodesic_distance(self, a: "list[float]", b: "list[float]") -> float:
        """Planar distance between world points a and b.

        On the FLAT, obstacle-free g1_flat scene the geodesic distance is
        EXACTLY the straight-line distance (no obstacles to route around), so
        this returns the euclidean planar distance — the honest geodesic for
        THIS world, not a loosening. In ROOM mode it returns the visibility-
        graph PATH LENGTH around the real obstacles (inf when unreachable) —
        the SAME planner navigate_to walks, so execution and verify never
        diverge (rule 5). Declaring it lets the kernel's ``geodesic_dist(x, y)``
        verify predicate bind for coordinate goals.
        """
        import math
        if getattr(self, "_obstacles", None):
            from vector_os_nano.hardware.sim import g1_vgraph  # noqa: PLC0415
            return g1_vgraph.path_length(
                (float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                self._obstacles, _NAV_INFLATION)
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def get_odometry(self):
        from vector_os_nano.core.types import Odometry

        q, v = self._snapshot()
        return Odometry(
            timestamp=time.time(),
            x=float(q[0]), y=float(q[1]), z=float(q[2]),
            qw=float(q[3]), qx=float(q[4]), qy=float(q[5]), qz=float(q[6]),
            vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
            vyaw=float(v[5]),
        )
