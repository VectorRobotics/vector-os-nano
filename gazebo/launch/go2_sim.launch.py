"""ROS2 launch file for Go2 Gazebo Harmonic simulation.

Starts the full simulation stack:
  1. Gz Sim server (physics) — loads the world SDF
  2. Gz Sim GUI (optional) — graphical window
  3. robot_state_publisher — publishes TF from SDF joint states
  4. Spawn Go2 — loads model.sdf into the running Gz world
  5. ros_gz_bridge — bridges Gz-transport topics to ROS2
  6. RViz2 (optional) — ROS2 visualizer

Usage:
    ros2 launch gazebo/launch/go2_sim.launch.py
    ros2 launch gazebo/launch/go2_sim.launch.py world:=empty_room gui:=false
    ros2 launch gazebo/launch/go2_sim.launch.py world:=apartment gui:=true use_rviz:=true
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ---------------------------------------------------------------------------
# Directory resolution — launch file lives at:
#   <repo>/gazebo/launch/go2_sim.launch.py
# So gazebo/ is one level up from __file__.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GAZEBO_DIR = os.path.join(_THIS_DIR, "..")  # gazebo/
_MODELS_DIR = os.path.join(_GAZEBO_DIR, "models")
_CONFIG_DIR = os.path.join(_GAZEBO_DIR, "config")
_WORLDS_DIR = os.path.join(_GAZEBO_DIR, "worlds")

_GO2_MODEL_SDF = os.path.join(_MODELS_DIR, "go2", "model.sdf")
_BRIDGE_YAML = os.path.join(_CONFIG_DIR, "bridge.yaml")

# ros_gz_sim package share directory — contains upstream launch files
_ROS_GZ_SIM_SHARE = get_package_share_directory("ros_gz_sim")


def generate_launch_description() -> LaunchDescription:
    """Build and return the full Gz Sim + Go2 launch description."""

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------

    # Which world SDF file to load (default: apartment)
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="apartment",
        description="World name to load (SDF file in gazebo/worlds/)",
    )

    # Show the Gz Sim GUI window (set false for headless/CI)
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="true",
        description="Launch Gz Sim GUI window (true/false)",
    )

    # Start RViz2 alongside the simulation
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Launch RViz2 for ROS2 visualization (true/false)",
    )

    # Shorthand LaunchConfiguration references
    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")
    use_rviz = LaunchConfiguration("use_rviz")

    # ------------------------------------------------------------------
    # 1. Gz Sim server — loads the world file
    #    gz_args: path to world SDF + headless flag when gui=false
    #    We always pass -r (run on start) for automation friendliness.
    # ------------------------------------------------------------------

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(_ROS_GZ_SIM_SHARE, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            # Pass world file path as gz_args; -r auto-runs simulation
            "gz_args": [
                os.path.join(_WORLDS_DIR, ""),
                world,
                ".sdf -r",
            ],
            "gz_version": "8",
            "on_exit_shutdown": "true",
        }.items(),
    )

    # ------------------------------------------------------------------
    # 2. robot_state_publisher — reads SDF joint definitions, publishes TF
    #    Reads model.sdf directly; ros_gz_bridge provides /joint_states.
    # ------------------------------------------------------------------

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                # Feed the SDF file to robot_state_publisher
                "robot_description": open(_GO2_MODEL_SDF).read()
                if os.path.exists(_GO2_MODEL_SDF)
                else "",
                "use_sim_time": True,
            }
        ],
    )

    # ------------------------------------------------------------------
    # 3. Spawn Go2 model into running Gz world
    #    Delay 2s to let Gz Sim server start before spawning.
    # ------------------------------------------------------------------

    spawn_go2 = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_sim",
                executable="create",
                name="spawn_go2",
                output="screen",
                arguments=[
                    "-file", _GO2_MODEL_SDF,
                    "-name", "go2",
                    "-x", "0",
                    "-y", "0",
                    "-z", "0.35",  # spawn above ground (Go2 standing height)
                ],
            )
        ],
    )

    # ------------------------------------------------------------------
    # 4. ros_gz_bridge — bridges Gz-transport <-> ROS2 topics
    #    Config file: gazebo/config/bridge.yaml
    # ------------------------------------------------------------------

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[
            {
                "config_file": _BRIDGE_YAML,
                "use_sim_time": True,
            }
        ],
    )

    # ------------------------------------------------------------------
    # 5. RViz2 (optional, only when use_rviz:=true)
    # ------------------------------------------------------------------

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
    )

    # ------------------------------------------------------------------
    # Assemble LaunchDescription
    # ------------------------------------------------------------------

    return LaunchDescription(
        [
            # Declare arguments first
            world_arg,
            gui_arg,
            use_rviz_arg,
            # Start simulation server
            gz_sim,
            # Start TF publisher
            robot_state_publisher,
            # Spawn robot model (delayed to let server start)
            spawn_go2,
            # Start topic bridge
            ros_gz_bridge,
            # Optional visualizer
            rviz2,
        ]
    )
