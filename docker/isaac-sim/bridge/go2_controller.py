#!/usr/bin/env python3
"""Go2 RL locomotion controller for Isaac Sim 5.1.

Uses a pre-trained JIT policy (Glowing-Torch, legged_gym trained).
45-dim observation, 12-dim joint position output.

Joint reordering between Isaac Sim USD (per-type) and policy (per-leg).
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("go2_ctrl")

# Policy defaults (per-leg: FL_h,FL_t,FL_c, FR_h,FR_t,FR_c, RL, RR)
_DEF_LAB = np.array([
    0.1, 0.8, -1.5,    # FL
    -0.1, 0.8, -1.5,   # FR
    0.1, 1.0, -1.5,    # RL
    -0.1, 1.0, -1.5,   # RR
], dtype=np.float32)

# Joint reorder: SIM_FROM_LAB[sim_i] = lab_i
# Sim: FL_h FR_h RL_h RR_h FL_t FR_t RL_t RR_t FL_c FR_c RL_c RR_c
# Lab: FL_h FL_t FL_c FR_h FR_t FR_c RL_h RL_t RL_c RR_h RR_t RR_c
SIM_FROM_LAB = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
LAB_FROM_SIM = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

# Default positions in Isaac Sim joint order
DEFAULT_JOINT_POS = _DEF_LAB[SIM_FROM_LAB].copy()

ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.3], dtype=np.float32)

_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", "go2_jit.pt")


class Go2RLController:
    """Go2 locomotion using pre-trained JIT RL policy."""

    def __init__(self, policy_path: str | None = None) -> None:
        self._cmd_vel = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._policy = None
        self._torch = None

        path = policy_path or _POLICY_PATH
        if os.path.exists(path):
            self._load_policy(path)
        else:
            logger.error("Policy not found: %s", path)

    def _load_policy(self, path: str) -> None:
        try:
            import torch
            self._torch = torch
            self._policy = torch.jit.load(path, map_location="cpu")
            self._policy.eval()
            logger.info("Go2 RL policy loaded: %s", os.path.basename(path))
        except Exception as exc:
            logger.error("Failed to load policy: %s", exc)

    def set_command(self, vx: float, vy: float, vyaw: float) -> None:
        self._cmd_vel = np.array([
            np.clip(vx, -2.0, 2.0),
            np.clip(vy, -2.0, 2.0),
            np.clip(vyaw, -2.0, 2.0),
        ], dtype=np.float32)

    def compute_action(
        self,
        joint_pos_sim: np.ndarray,
        joint_vel_sim: np.ndarray,
        ang_vel: np.ndarray,
        quat: np.ndarray,
    ) -> np.ndarray:
        """Run RL policy. Returns joint targets in Isaac Sim order."""
        if self._policy is None:
            return DEFAULT_JOINT_POS.copy()

        # Sim → Lab reorder
        jp_lab = joint_pos_sim.astype(np.float32)[LAB_FROM_SIM]
        jv_lab = joint_vel_sim.astype(np.float32)[LAB_FROM_SIM]

        # Body-frame quantities
        proj_grav = _quat_rotate_inv(quat, np.array([0, 0, -1], dtype=np.float32))
        ang_vel_b = _quat_rotate_inv(quat, ang_vel.astype(np.float32))

        # 45-dim observation
        obs = np.zeros(45, dtype=np.float32)
        obs[0:3] = self._cmd_vel * CMD_SCALE
        obs[3:6] = proj_grav
        obs[6:9] = ang_vel_b * 0.3
        obs[9:21] = jp_lab - _DEF_LAB
        obs[21:33] = jv_lab * 0.05
        obs[33:45] = self._last_action

        with self._torch.no_grad():
            action = self._policy(
                self._torch.tensor(obs).unsqueeze(0)
            ).squeeze(0).numpy()

        self._last_action = action.astype(np.float32)

        # Lab → Sim reorder
        targets_lab = _DEF_LAB + ACTION_SCALE * action
        return targets_lab[SIM_FROM_LAB]


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by inverse of quaternion q (w,x,y,z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    u = np.array([-x, -y, -z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def setup_keyboard_control(controller: Go2RLController) -> None:
    """Keyboard control for Isaac Sim GUI."""
    try:
        import carb.input
        import omni.appwindow

        appwindow = omni.appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()

        key_map = {
            "UP": [1.5, 0.0, 0.0],
            "DOWN": [-1.5, 0.0, 0.0],
            "LEFT": [0.0, 0.8, 0.0],
            "RIGHT": [0.0, -0.8, 0.0],
            "N": [0.0, 0.0, 1.5],
            "M": [0.0, 0.0, -1.5],
        }
        active_keys: set[str] = set()

        def _on_key(event, *args, **kwargs):
            inp = event.input
            name = inp.name if hasattr(inp, "name") else str(inp)
            if "." in name:
                name = name.split(".")[-1]

            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if name == "SPACE":
                    active_keys.clear()
                    controller.set_command(0, 0, 0)
                    return True
                if name in key_map:
                    active_keys.add(name)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                active_keys.discard(name)

            cmd = np.zeros(3)
            for k in active_keys:
                if k in key_map:
                    cmd += np.array(key_map[k])
            controller.set_command(float(cmd[0]), float(cmd[1]), float(cmd[2]))
            return True

        input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
        logger.info("Keyboard: arrows=move, N/M=yaw, Space=stop")
    except Exception as exc:
        logger.warning("Keyboard unavailable: %s", exc)
