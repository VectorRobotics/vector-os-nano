"""vector_os_nano.hardware.sim — simulation backends.

Provides drop-in replacements for hardware arm and gripper,
allowing skills and the Agent to run in headless simulation
without physical hardware.

Three backends available:
    MuJoCo (recommended) — higher fidelity contact physics, real grasping
    PyBullet (legacy)    — lighter weight, basic simulation
    Isaac Sim            — photorealistic, requires Docker + ROS2

MuJoCo and PyBullet are imported eagerly. Isaac Sim proxies are imported
lazily inside try/except blocks because they depend on rclpy which is only
available in ROS2 environments.
"""
from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
from vector_os_nano.hardware.sim.pybullet_arm import SimulatedArm
from vector_os_nano.hardware.sim.pybullet_gripper import SimulatedGripper

__all__ = [
    "MuJoCoArm",
    "MuJoCoGo2",
    "MuJoCoGripper",
    "MuJoCoPerception",
    "SimulatedArm",
    "SimulatedGripper",
]

# Isaac Sim proxies — optional, require rclpy (ROS2 must be sourced).
try:
    from vector_os_nano.hardware.sim.isaac_sim_proxy import IsaacSimProxy

    __all__ += ["IsaacSimProxy"]
except Exception:
    pass

try:
    from vector_os_nano.hardware.sim.isaac_sim_arm_proxy import IsaacSimArmProxy

    __all__ += ["IsaacSimArmProxy"]
except Exception:
    pass

try:
    from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy

    __all__ += ["GazeboGo2Proxy"]
except Exception:
    pass
