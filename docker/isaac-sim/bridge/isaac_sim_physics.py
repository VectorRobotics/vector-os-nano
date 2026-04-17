#!/usr/bin/env python3
"""Isaac Sim physics process — runs under Isaac Sim's Python 3.11.

Creates the simulation scene, steps physics, reads sensor data,
and writes state to shared files for the ROS2 publisher process.

State exchange via /tmp/isaac_state/:
  ready       — flag file, created when sim is initialized
  odom.bin    — 13 floats: x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz
  joints.bin  — 12 floats: joint positions
  cmd_vel.bin — 3 floats: vx, vy, vyaw (written by ROS2 process)
"""
import os
import sys
import math
import time
import logging
import struct
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("isaac_physics")

# Isaac Sim init
from isaacsim import SimulationApp

_HEADLESS = os.environ.get("ISAAC_HEADLESS", "true").lower() == "true"
_SCENE = os.environ.get("ISAAC_SCENE", "flat")
_PHYSICS_HZ = int(os.environ.get("ISAAC_PHYSICS_HZ", "200"))
_STATE_DIR = os.environ.get("ISAAC_STATE_DIR", "/tmp/isaac_state")

logger.info("Starting Isaac Sim (headless=%s, scene=%s, hz=%d)", _HEADLESS, _SCENE, _PHYSICS_HZ)
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot

# Scene builders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_scene import (
    create_flat_scene, create_room_scene, create_apartment_scene,
    create_navigation_scene, create_hospital_scene,
)
from go2_sensors import attach_all_sensors

SCENE_BUILDERS = {
    "flat": create_flat_scene,
    "room": create_room_scene,
    "apartment": create_apartment_scene,
    "navigation": create_navigation_scene,
    "hospital": create_hospital_scene,
}


def _write_state(filename: str, data: bytes) -> None:
    """Atomic write to state file."""
    tmp = os.path.join(_STATE_DIR, filename + ".tmp")
    target = os.path.join(_STATE_DIR, filename)
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, target)


def _read_cmd_vel() -> tuple[float, float, float]:
    """Read velocity command from ROS2 process."""
    path = os.path.join(_STATE_DIR, "cmd_vel.bin")
    try:
        with open(path, "rb") as f:
            data = f.read(12)
        if len(data) == 12:
            return struct.unpack("fff", data)
    except (FileNotFoundError, OSError):
        pass
    return (0.0, 0.0, 0.0)


def _compute_gait(t: float, vx: float, vy: float, vyaw: float) -> np.ndarray:
    """Sinusoidal trotting gait — same algorithm as MuJoCo mujoco_go2.py.

    Isaac Sim joint order: FL_hip, FR_hip, RL_hip, RR_hip,
                           FL_thigh, FR_thigh, RL_thigh, RR_thigh,
                           FL_calf, FR_calf, RL_calf, RR_calf
    """
    import math
    # Standing pose
    stand = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8],
                     dtype=np.float32)

    if abs(vx) < 0.01 and abs(vy) < 0.01 and abs(vyaw) < 0.01:
        return stand

    targets = stand.copy()
    freq = 2.0
    omega = 2.0 * math.pi * freq
    thigh_amp = 0.20
    calf_amp = 0.20
    hip_amp = 0.08

    fwd_amp = float(np.clip(vx / 0.5, -1.0, 1.0))
    turn_amp = float(np.clip(vyaw / 1.0, -1.0, 1.0))

    # Trot phases: FL+RR together (0), FR+RL together (pi)
    # Leg indices in Isaac order: 0=FL, 1=FR, 2=RL, 3=RR
    phases = [0.0, math.pi, math.pi, 0.0]

    for leg in range(4):
        phase = omega * t + phases[leg]
        is_left = leg in (0, 2)
        leg_turn = -turn_amp if is_left else turn_amp
        total = float(np.clip(fwd_amp + leg_turn, -1.5, 1.5))
        amp = abs(total)

        # Hip (index 0-3)
        if abs(vy) > 0.01:
            targets[leg] += hip_amp * (vy / 0.4) * math.sin(phase)

        # Thigh (index 4-7)
        targets[4 + leg] += thigh_amp * amp * math.sin(phase)

        # Calf (index 8-11)
        calf_ph = 0.0 if total >= 0 else math.pi
        targets[8 + leg] += calf_amp * amp * math.sin(phase + calf_ph)

    return targets


def main() -> None:
    from go2_controller import Go2RLController, setup_keyboard_control, DEFAULT_JOINT_POS
    from isaacsim.core.utils.types import ArticulationAction

    # Create world (50 Hz policy, 200 Hz physics decimation=4, 30 Hz render)
    world = World(physics_dt=1.0 / 200.0, rendering_dt=1.0 / 30.0)

    # Build scene
    builder = SCENE_BUILDERS.get(_SCENE, create_flat_scene)
    logger.info("Building scene: %s", _SCENE)
    robot = builder(world)

    # Attach sensors
    sensors = attach_all_sensors(robot.prim_path)
    logger.info("Sensors: %s", list(sensors.keys()))

    # Reset world (initializes articulations)
    world.reset()
    logger.info("World reset complete")

    # Initialize articulation controller
    art_controller = None
    is_articulated = False
    try:
        robot.initialize()
        if robot.num_dof is not None and robot.num_dof >= 12:
            is_articulated = True
            art_controller = robot.get_articulation_controller()
    except Exception:
        pass

    # Initialize Go2 RL controller
    controller = Go2RLController()

    if is_articulated:
        logger.info("Go2 articulated: %d DOF", robot.num_dof)
        try:
            # Set PD gains (tested: kp=120 kd=6 → z=0.276 STAND)
            kp = np.full(12, 120.0)
            kd = np.full(12, 6.0)
            art_controller.set_gains(kp, kd)

            stand = np.array([0,0,0,0, 0.9,0.9,0.9,0.9, -1.8,-1.8,-1.8,-1.8], dtype=np.float32)
            art_controller.apply_action(ArticulationAction(joint_positions=stand))
            for _ in range(400):
                world.step(render=False)
            z = robot.get_world_pose()[0][2]
            logger.info("Go2 standing (z=%.3f)", z)
        except Exception as e:
            logger.warning("Initial pose failed: %s", e)
    else:
        logger.info("Go2 placeholder (no joints)")

    # Set up keyboard control (GUI mode only)
    if not _HEADLESS:
        setup_keyboard_control(controller)

    # Signal ready
    with open(os.path.join(_STATE_DIR, "ready"), "w") as f:
        f.write("1")
    logger.info("Isaac Sim ready — physics loop starting (keyboard: arrows=move, N/M=yaw, Space=stop)")

    # Physics loop
    step = 0
    sim_time = 0.0
    physics_dt = 1.0 / 200.0
    policy_decimation = 4
    try:
        while simulation_app.is_running():
            # Run RL policy at 50 Hz
            if step % policy_decimation == 0:
                # Read velocity command from ROS2 or keyboard
                vx, vy, vyaw = _read_cmd_vel()
                kbd_cmd = controller._cmd_vel
                if np.any(np.abs(kbd_cmd) > 0.01):
                    vx, vy, vyaw = float(kbd_cmd[0]), float(kbd_cmd[1]), float(kbd_cmd[2])
                else:
                    controller.set_command(vx, vy, vyaw)

                if step % 200 == 0 and (abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01):
                    logger.info("cmd: vx=%.2f vy=%.2f vyaw=%.2f", vx, vy, vyaw)

                # Compute action from RL policy
                if is_articulated and art_controller is not None:
                    try:
                        jp = robot.get_joint_positions()
                        jv = robot.get_joint_velocities()
                        av = robot.get_angular_velocity()
                        _, qt = robot.get_world_pose()
                        if jp is not None and jv is not None:
                            targets = controller.compute_action(jp, jv, av, qt)
                            art_controller.apply_action(
                                ArticulationAction(joint_positions=targets)
                            )
                    except Exception as e:
                        if step % 1000 == 0:
                            logger.warning("RL error: %s", e)
                else:
                    try:
                        robot.set_linear_velocity(np.array([vx, vy, 0.0]))
                        robot.set_angular_velocity(np.array([0.0, 0.0, vyaw]))
                    except Exception:
                        pass

            # Step physics
            world.step(render=not _HEADLESS)
            sim_time += physics_dt

            # Write state at ~50 Hz
            if step % policy_decimation == 0:
                try:
                    pos, quat = robot.get_world_pose()
                    lin_vel = robot.get_linear_velocity()
                    ang_vel = robot.get_angular_velocity()

                    odom_data = struct.pack("13f",
                        float(pos[0]), float(pos[1]), float(pos[2]),
                        float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]),
                        float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2]),
                        float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2]),
                    )
                    _write_state("odom.bin", odom_data)

                    if is_articulated:
                        joint_pos = robot.get_joint_positions()
                        if joint_pos is not None:
                            joints_data = struct.pack(f"{len(joint_pos)}f",
                                                      *[float(j) for j in joint_pos])
                            _write_state("joints.bin", joints_data)
                except Exception as e:
                    if step % 5000 == 0:
                        logger.warning("State write: %s", e)

            step += 1

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
