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

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        import mujoco
        import torch

        self._model = mujoco.MjModel.from_xml_path(
            str(self._asset_dir / "scene.xml"))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = _SIM_DT
        self._policy = torch.jit.load(str(self._asset_dir / "motion.pt"))
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
        self._model = self._data = self._policy = None

    close = disconnect

    def _require_connected(self) -> None:
        if self._data is None or not self._running:
            raise RuntimeError("G1MuJoCoBase not connected (call connect())")

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

    # -- BaseProtocol -------------------------------------------------------
    def set_velocity(self, vx: float, vy: float = 0.0,
                     vyaw: float = 0.0) -> dict:
        """Non-blocking streaming command; 0.6 s deadman keep-alive."""
        self._require_connected()
        with self._lock:
            self._cmd[:] = (float(vx), float(vy), float(vyaw))
            self._cmd_stamp = time.monotonic()
        return {"ok": True}

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

    def stop(self) -> dict:
        self._require_connected()
        with self._lock:
            self._cmd[:] = 0.0
            self._cmd_stamp = time.monotonic()
        return {"ok": True}

    def get_position(self) -> "list[float]":
        self._require_connected()
        return [float(self._data.qpos[0]), float(self._data.qpos[1]),
                float(self._data.qpos[2])]

    def get_heading(self) -> float:
        self._require_connected()
        qw, qx, qy, qz = self._data.qpos[3:7]
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

    def get_odometry(self):
        self._require_connected()
        from vector_os_nano.core.types import Odometry

        q = self._data.qpos
        v = self._data.qvel
        return Odometry(
            timestamp=time.time(),
            x=float(q[0]), y=float(q[1]), z=float(q[2]),
            qw=float(q[3]), qx=float(q[4]), qy=float(q[5]), qz=float(q[6]),
            vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
            vyaw=float(v[5]),
        )
