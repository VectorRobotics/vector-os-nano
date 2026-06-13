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

    def __init__(self, asset_dir: "Path | str | None" = None) -> None:
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

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        import mujoco
        import torch

        self._model = mujoco.MjModel.from_xml_path(
            str(self._asset_dir / "scene.xml"))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = _SIM_DT
        self._policy = torch.jit.load(str(self._asset_dir / "motion.pt"))
        # Publish the initial pose so readers before the first policy batch
        # still get a real snapshot (not None).
        self._snap = (self._data.qpos[:7].copy(), self._data.qvel[:6].copy())
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, daemon=True, name="g1-gait-control")
        self._thread.start()
        # let the policy settle into stance before the first command
        time.sleep(0.3)

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._snap_lock:
            self._snap = None     # post-disconnect readers raise, not crash
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

    # -- control loop (sim thread owns ALL mujoco/torch state) -------------
    def _control_loop(self) -> None:
        import mujoco
        import torch

        m, d = self._model, self._data
        action = np.zeros(_NUM_ACTIONS, dtype=np.float32)
        target = _DEFAULT_ANGLES.copy()
        obs = np.zeros(_NUM_OBS, dtype=np.float32)
        counter = 0
        # Pace in POLICY-PERIOD batches (10 physics steps = 20 ms), one sleep
        # per batch against an absolute deadline. Per-step sleeping ran the
        # sim at ~0.5x real time: time.sleep()'s ~1-2 ms granularity error,
        # paid 500x/s, dwarfed the 2 ms step budget. 50 sleeps/s amortizes
        # it; the absolute next_tick self-corrects residual drift (the sim
        # is ~3.5x RT capable, so it always catches back up).
        batch_dt = _SIM_DT * _DECIMATION
        next_tick = time.monotonic()
        while self._running:
            with self._lock:
                cmd = self._cmd.copy()
                stale = (time.monotonic() - self._cmd_stamp) > _DEADMAN_S
            if stale:
                cmd[:] = 0.0           # deadman: dead client never marches
            for _ in range(_DECIMATION):
                tau = ((target - d.qpos[7:]) * _KPS - d.qvel[6:] * _KDS)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                counter += 1
            obs[:3] = d.qvel[3:6] * _ANG_VEL_SCALE
            obs[3:6] = _gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * _CMD_SCALE
            obs[9:9 + _NUM_ACTIONS] = (
                (d.qpos[7:] - _DEFAULT_ANGLES) * _DOF_POS_SCALE)
            obs[9 + _NUM_ACTIONS:9 + 2 * _NUM_ACTIONS] = (
                d.qvel[6:] * _DOF_VEL_SCALE)
            obs[9 + 2 * _NUM_ACTIONS:9 + 3 * _NUM_ACTIONS] = action
            phase = (counter * _SIM_DT) % _GAIT_PERIOD / _GAIT_PERIOD
            obs[9 + 3 * _NUM_ACTIONS] = math.sin(2 * math.pi * phase)
            obs[9 + 3 * _NUM_ACTIONS + 1] = math.cos(2 * math.pi * phase)
            with torch.no_grad():
                action = self._policy(
                    torch.from_numpy(obs).unsqueeze(0)
                ).numpy().squeeze()
            target = action * _ACTION_SCALE + _DEFAULT_ANGLES
            # Publish a consistent pose snapshot for cross-thread readers.
            with self._snap_lock:
                self._snap = (d.qpos[:7].copy(), d.qvel[:6].copy())
            # On-demand viewer frame: render on THIS thread (owns mjData + GL).
            if self._render_req.is_set():
                try:
                    self._render_png = self._render_tracking_frame(m, d)
                except Exception as exc:  # noqa: BLE001 — never break the gait
                    logger.warning("g1 render failed: %s", exc)
                    self._render_png = None
                self._render_req.clear()
                self._render_done.set()
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
            time.sleep(0.1)
        self.stop()
        return True

    def navigate_to(self, x: float, y: float, tol: float = 0.2,
                    speed: float = _NAV_SPEED) -> dict:
        """Closed-loop face->walk->arrive drive to world (x, y) on the flat
        scene. The sim-oracle ``base.navigate_to`` NavigateToPointSkill calls;
        returns the three-value contract {reached, already_there, moved_m,
        elapsed_s, remaining, pos, reason, transport}.

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
                time.sleep(_NAV_TICK_S)
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
            time.sleep(_NAV_TICK_S)

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
        THIS world, not a loosening (a future G1 scene with obstacles would
        need a real navmesh oracle). Declaring it lets the kernel's
        ``geodesic_dist(x, y)`` verify predicate bind for coordinate goals,
        which would otherwise fail-safe to inf (no oracle) and never pass.
        """
        import math
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
