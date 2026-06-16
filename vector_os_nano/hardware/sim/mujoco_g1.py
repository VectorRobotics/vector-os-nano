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
import os
import threading
import time
from pathlib import Path
from typing import Any

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

# Campaign #11 R5: the shared world-agnostic controller's constant pack for G1.
# EVERY value is the existing _NAV_* above — extraction must not change behavior.
from vector_os_nano.hardware.sim._nav_controller import NavConsts  # noqa: E402
_G1_NAV = NavConsts(
    fall_z=_NAV_FALL_Z, tol_floor=_NAV_TOL_FLOOR, speed=_NAV_SPEED,
    vyaw_max=_NAV_VYAW_MAX, k_yaw=_NAV_K_YAW, face_tol=_NAV_FACE_TOL,
    yaw_deadband=_NAV_YAW_DEADBAND, capture_r=_NAV_CAPTURE_R, tick_s=_NAV_TICK_S,
    settle_s=_NAV_SETTLE_S, timeout_s=_NAV_TIMEOUT_S,
    stall_window_s=_NAV_STALL_WINDOW_S, stall_min_m=_NAV_STALL_MIN_M,
    inflation=_NAV_INFLATION, waypoint_tol=_NAV_WAYPOINT_TOL)
_OCC_RESOLUTION = 0.25       # m/cell — occupancy grid resolution (room mode)


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
        furnished: bool = False,
        prefer_daemon: bool = False,
        photoreal: bool = False,
        photoreal_bridge: "Any | None" = None,
    ) -> None:
        # prefer_daemon (campaign #9 R7): when a viewer is open, normally PUMP
        # mode drives the gait on the caller thread (1.0x). But when the base is
        # booted MID-REPL (start_simulation tool, on a worker thread), nothing
        # pumps it → frozen gait. prefer_daemon forces a DAEMON control thread
        # that drives the gait AND syncs the viewer itself (thread-agnostic,
        # ~0.4x under the passive viewer's render thread — Case 13 — but it
        # walks and renders). The --scenario startup path leaves this False (the
        # main REPL thread pumps, 1.0x).
        self._prefer_daemon: bool = prefer_daemon
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
        # On-demand FORWARD-camera RGB frame (campaign #8 R9 — visual object
        # recognition). Same control-thread discipline as the chase-cam render
        # (mujoco.Renderer owns a GL context — Case 12, never the caller
        # thread). The caller requests via _cam_req; the control loop renders a
        # first-person forward view and publishes the (H,W,3) rgb array.
        self._cam_lock = threading.Lock()
        self._cam_req = threading.Event()
        self._cam_done = threading.Event()
        self._cam_rgb = None     # holds the latest camera OBSERVATION dict (R5)
        self._cam_renderer = None
        self._cam_depth_renderer = None   # depth (R5 depth-at-bbox); lazily built
        self._cam_opt = None     # MjvOption (all geom groups) — lazily built
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
        # Furnished room (campaign #9 R1, track A): real Kenney furniture meshes
        # (chair/sofa/potted plant) as semantic targets instead of colour boxes,
        # so a real VLM can recognise an object CLASS. Walls + obstacles + lidar
        # + camera are identical; only the target geometry differs.
        self._furnished: bool = furnished
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
        # Photoreal co-sim (campaign #10): when on, get_camera_observation's RGB
        # is rendered by Blender Cycles/OptiX from the SAME camera pose MuJoCo
        # rendered the depth at (hybrid — rgb=Blender, depth/pose=MuJoCo, one
        # camera frame). Default OFF → the furnished-room behaviour is byte-
        # identical (rule 2/9). The Blender bridge is a subprocess over a socket
        # (no GL in our process → safe off the control thread). A bridge may be
        # injected for tests; otherwise it is spawned lazily on first camera obs.
        self._photoreal: bool = photoreal
        self._photoreal_bridge = photoreal_bridge   # injected (tests) or lazy
        self._photoreal_renderer = None

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
            self._model = g1_room.build_room_model(
                self._asset_dir, furnished=self._furnished)
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
                # Colour room: 3 m forces frontier exploration (explore must
                # MOVE to grow the map). Furnished VLM room (campaign #9 R3): its
                # purpose is recognise→navigate, which ranges a recognised object
                # to a world point — give it room-spanning range so the lidar
                # locates a target the VLM sees (the chair sits at 3.6 m) without
                # a fragile close-in phase.
                _lidar_range = 8.0 if self._furnished else 3.0
                self._lidar = MuJoCoLivox360(
                    self._model, self._data, body_name="pelvis",
                    max_range=_lidar_range, geom_group=g1_room.ENV_GEOM_GROUP,
                    include_misses=True)
            except Exception as exc:  # noqa: BLE001 — lidar is non-fatal
                logger.warning("G1 lidar init failed: %s", exc)
                self._lidar = None
            # Occupancy grid for autonomous exploration (campaign #8 R6). It is
            # filled by observe() on the CALLER thread (snapshot reads, NOT the
            # control thread) — ray-marching 5760 rays at 10 Hz would starve the
            # gait (the R0/Case-13 lesson), so mapping stays off the hot path.
            from vector_os_nano.hardware.sim.occupancy import OccupancyGrid  # noqa: PLC0415,E501
            self._occ = OccupancyGrid(*g1_room.room_bounds(), resolution=_OCC_RESOLUTION)
        # Live viewer window (campaign #8 R0): open it on the caller thread.
        # Skip under mjpython (macOS main-thread-only GLFW); launch failure
        # (no display / GL) degrades to headless without breaking the boot.
        if self._gui:
            self._launch_viewer(mujoco)
        self._running = True
        # Mode resolution: a live window → PUMP (caller thread drives the loop
        # via _advance, no daemon, so the gait runs 1x against the viewer's
        # render thread). No window → DAEMON (the proven headless path).
        # PUMP only when a viewer is open AND we are NOT forced to daemon. A
        # mid-REPL (tool) boot sets prefer_daemon so the gait runs on its own
        # thread (and _step_batch syncs the viewer) regardless of which thread
        # booted it — the pump-mode caller-thread driver would never run there.
        self._pump_mode = self._viewer is not None and not self._prefer_daemon
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
            # Show ALL geom groups: the room geometry (walls / obstacles / colour
            # targets / furniture) is tagged ENV_GEOM_GROUP=3 for the lidar mask,
            # which the viewer HIDES by default — so the owner would see only the
            # robot on an empty floor (owner-caught 2026-06-14). The robot's own
            # first-person camera already enables all groups; the live window must
            # too, or the GUI acceptance view is misleading.
            self._viewer.opt.geomgroup[:] = 1
            cam = self._viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 5.5            # pulled back to frame the room + targets
            cam.elevation = -25
            cam.azimuth = 130
            cam.lookat[:] = [1.6, 0.0, 0.4]   # room centre, not just the robot
        except Exception as exc:  # noqa: BLE001 — never break the boot
            logger.warning("G1MuJoCoBase viewer failed to launch: %s", exc)
            self._viewer = None

    def disconnect(self) -> None:
        self._running = False
        # Wake any in-flight render/camera waiter so it sees the disconnect and
        # raises promptly instead of blocking the full timeout (review R-final).
        self._render_done.set()
        self._cam_done.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001 — best-effort window teardown
                pass
            self._viewer = None
        # Release the offscreen GL renderers (chase-cam + first-person camera)
        # — leaking the EGL context across reconnect/multi-instance runs is the
        # historical crash class (tricky-bugs Case 12 discipline).
        for _r in (self._renderer, self._cam_renderer, self._cam_depth_renderer):
            if _r is not None:
                try:
                    _r.close()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass
        self._renderer = None
        with self._snap_lock:
            self._snap = None     # post-disconnect readers raise, not crash
        with self._lidar_snap_lock:
            self._lidar_snap = None
        self._lidar = None
        self._occ = None
        self._cam_renderer = self._cam_depth_renderer = None
        if getattr(self, "_pano", None) is not None:
            try:
                self._pano.close()      # release the pano GL renderers (M4)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._pano = None
        self._cam_rgb = self._cam_opt = None
        self._model = self._data = self._policy = None
        # Photoreal co-sim: terminate the Blender subprocess (idempotent).
        if self._photoreal_bridge is not None:
            try:
                self._photoreal_bridge.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._photoreal_bridge = None
        self._photoreal_renderer = None

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
        blocks until it lands. Serialized — one render in flight. In PUMP mode
        the caller thread IS the control thread, so render directly — the
        _render_req hand-off would deadlock (same as get_camera_frame)."""
        self._require_connected()
        if self._pump_mode:
            return self._render_tracking_frame(self._model, self._data)
        with self._render_lock:
            self._render_done.clear()
            self._render_req.set()
            if not self._render_done.wait(timeout=timeout):
                raise RuntimeError("g1 viewer frame render timed out")
            png = self._render_png
        if not png:
            raise RuntimeError("g1 viewer frame render produced no image")
        return png

    def _render_camera_obs(self, m, d, h: int = 240, w: int = 320):
        """Render the pelvis FIRST-PERSON camera (control thread) and capture an
        atomic observation: ``{rgb (h,w,3 uint8), depth (h,w float32 m), cam_pos
        (3,), cam_mat (9,), fovy}``. RGB + depth + camera pose come from ONE
        scene update so they are mutually consistent (the robot does not move
        between them) — the recognise→navigate pivot (R5) back-projects the depth
        at the recognised bbox to a world point. Depth needs its own Renderer
        (MuJoCo renders either colour OR depth per renderer)."""
        import mujoco
        if self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(m, height=h, width=w)
            # The env geoms (walls/obstacles/targets) are in group
            # ENV_GEOM_GROUP for the lidar mask; the offscreen Renderer hides
            # non-default groups by default, so enable ALL groups or the camera
            # sees an empty floor.
            self._cam_opt = mujoco.MjvOption()
            self._cam_opt.geomgroup[:] = 1
        if self._cam_depth_renderer is None:
            self._cam_depth_renderer = mujoco.Renderer(m, height=h, width=w)
            self._cam_depth_renderer.enable_depth_rendering()
        from vector_os_nano.hardware.sim.g1_room import HEAD_CAM  # noqa: PLC0415
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAM)
        self._cam_renderer.update_scene(d, camera=HEAD_CAM, scene_option=self._cam_opt)
        rgb = self._cam_renderer.render().copy()
        self._cam_depth_renderer.update_scene(d, camera=HEAD_CAM, scene_option=self._cam_opt)
        depth = self._cam_depth_renderer.render().copy()
        return {
            "rgb": rgb,
            "depth": depth,
            "cam_pos": np.array(d.cam_xpos[cam_id], dtype=np.float64).copy(),
            "cam_mat": np.array(d.cam_xmat[cam_id], dtype=np.float64).copy(),
            "fovy": float(m.cam_fovy[cam_id]),
        }

    def _render_camera_frame(self, m, d, h: int = 240, w: int = 320):
        """RGB-only convenience (back-compat): the ``rgb`` of the full obs."""
        return self._render_camera_obs(m, d, h, w)["rgb"]

    def get_camera_frame(self, timeout: float = 5.0):
        """A first-person forward RGB frame (H,W,3 uint8) for recognition.

        Rendered ON the control thread (Case 12 — mujoco.Renderer owns GL); the
        caller blocks until it lands. In PUMP mode the CALLER thread IS the
        control thread (it drives _step_batch), so the _cam_req hand-off would
        DEADLOCK (the caller would block waiting for a batch it must itself
        pump) — render directly instead. In DAEMON mode the daemon serves the
        request. Requires room mode (the forward camera is mounted by
        g1_room) — fails loud on the flat scene rather than rendering blank."""
        self._require_connected()
        if not self._room:
            raise RuntimeError(
                "no forward camera — get_camera_frame needs room mode "
                "(--scenario g1_room)")
        return self.get_camera_observation(timeout=timeout)["rgb"]

    def get_camera_observation(self, timeout: float = 5.0) -> dict:
        """Atomic first-person observation ``{rgb, depth, cam_pos, cam_mat,
        fovy}`` (R5). Same control-thread render discipline as get_camera_frame
        (pump → direct; daemon → _cam_req hand-off). The recognise→navigate skill
        back-projects the depth at the recognised bbox to a world point."""
        self._require_connected()
        if not self._room:
            raise RuntimeError(
                "no forward camera — needs room mode (--scenario g1_room)")
        if self._pump_mode:
            obs = self._render_camera_obs(self._model, self._data)
            return self._photoreal_swap(obs)
        with self._cam_lock:
            self._cam_done.clear()
            self._cam_req.set()
            if not self._cam_done.wait(timeout=timeout):
                raise RuntimeError("g1 camera obs render timed out")
            obs = self._cam_rgb
        if obs is None:
            raise RuntimeError("g1 camera obs render produced no image")
        return self._photoreal_swap(obs)

    # -- photoreal co-sim (campaign #10) -----------------------------------
    def _ensure_photoreal_renderer(self):
        """Lazily build the PhotorealRenderer (spawns the Blender bridge unless one
        was injected). Fails loud if photoreal was requested with no Blender."""
        if self._photoreal_renderer is not None:
            return self._photoreal_renderer
        from vector_os_nano.hardware.sim.g1_room import HEAD_CAM  # noqa: PLC0415
        from vector_os_nano.playground.photoreal.cosim import furnished_room_renderer

        # Shared furnished-room co-sim factory (rule 7 — same wiring the Go2 base
        # uses; only the head camera differs). Reuses an injected bridge in tests.
        renderer, bridge = furnished_room_renderer(
            cam_name=HEAD_CAM, bridge=self._photoreal_bridge,
            width=640, height=480, samples=48)
        self._photoreal_bridge = bridge
        self._photoreal_renderer = renderer
        return self._photoreal_renderer

    def _photoreal_swap(self, obs: dict) -> dict:
        """Replace the MuJoCo RGB with a photoreal Blender frame from the SAME
        pose (no-op when photoreal is off — keeps existing behaviour identical)."""
        if not self._photoreal:
            return obs
        renderer = self._ensure_photoreal_renderer()
        rgb = renderer.render_from_pose(obs["cam_pos"], obs["cam_mat"], obs["fovy"])
        out = dict(obs)
        out["rgb_mujoco"] = obs["rgb"]   # keep the physics render for debugging
        out["rgb"] = rgb
        return out

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
        # On-demand forward-camera OBSERVATION (R9 rgb + R5 depth/pose): render
        # on THIS thread (owns GL). _cam_rgb holds the full obs dict.
        if self._cam_req.is_set():
            try:
                self._cam_rgb = self._render_camera_obs(m, d)
            except Exception as exc:  # noqa: BLE001 — never break the gait
                logger.warning("g1 camera render failed: %s", exc)
                self._cam_rgb = None
            self._cam_req.clear()
            self._cam_done.set()

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
                time.sleep(max(0.0, remaining - 0.003))
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
        if self._furnished:
            return g1_room.furnished_targets()
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
        self._require_connected()
        if not self._obstacles:
            return self._navigate_point(x, y, tol, speed)
        # Shared world-agnostic controller (campaign #11 R5); G1 uses a no-op
        # ctrl_token so its control path is byte-identical to campaign #6.
        from vector_os_nano.hardware.sim import _nav_controller  # noqa: PLC0415
        return _nav_controller.route_and_drive(
            x=x, y=y, tol=tol, speed=speed, obstacles=self._obstacles,
            get_position=self.get_position, get_heading=self.get_heading,
            set_velocity=self.set_velocity, stop=self.stop,
            tick_fn=self._advance, consts=_G1_NAV)

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

        self._require_connected()
        # Shared world-agnostic controller (campaign #11 R5). G1 omits ctrl_token
        # (-> no-op nullcontext), so the control path is byte-identical to the
        # campaign #6 loop this was extracted from.
        from vector_os_nano.hardware.sim import _nav_controller  # noqa: PLC0415
        return _nav_controller.drive_to_point(
            x=x, y=y, tol=tol, speed=speed,
            get_position=self.get_position, get_heading=self.get_heading,
            set_velocity=self.set_velocity, stop=self.stop,
            tick_fn=self._advance, consts=_G1_NAV)

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
        from vector_os_nano.hardware.sim import _nav_controller  # noqa: PLC0415
        return _nav_controller.path_length_geodesic(
            a, b, getattr(self, "_obstacles", None) or [], _G1_NAV.inflation)

    def get_pano(self) -> dict:
        """Pose-synced equirect panorama for SysNav (campaign #11 M4 feed source).

        Returns the SAME contract as HabitatBase.get_pano — {rgb (640,1920,3)
        uint8, depth (640,1920) f32, pos[3], heading} — so the embodiment-agnostic
        SysNav feed (wire_sysnav_feed) drives G1 with no habitat dependency. The
        pano is a real MuJoCoPano360 render (rule 5: pixels from the sim, never GT)."""
        self._require_connected()
        if getattr(self, "_pano", None) is None:
            from vector_os_nano.hardware.sim.sensors.pano360 import MuJoCoPano360  # noqa: PLC0415,E501
            self._pano = MuJoCoPano360(
                self._model, self._data, body_name="pelvis",
                offset=(0.0, 0.0, 0.5))   # ~eye height above the pelvis
        rgb, depth = self._pano.render_rgbd()
        pos = self.get_position()
        return {"rgb": rgb, "depth": depth,
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "heading": float(self.get_heading())}

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
