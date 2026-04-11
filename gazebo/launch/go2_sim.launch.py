"""ROS2 launch file for Go2 Gazebo Harmonic simulation.

Simplified: velocity-controlled Go2 (no joint controller).
Sensors: MID-360 lidar + D435 RGB-D camera + IMU.
Purpose: navigation testing, VLN development.

Usage:
    ros2 launch gazebo/launch/go2_sim.launch.py
    ros2 launch gazebo/launch/go2_sim.launch.py world:=empty_room gui:=false
"""
from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GAZEBO_DIR = os.path.join(_THIS_DIR, "..")
_WORLDS_DIR = os.path.join(_GAZEBO_DIR, "worlds")
_GO2_NAV_SDF = os.path.join(_GAZEBO_DIR, "models", "go2", "go2_nav.sdf")


def _launch_setup(context, *args, **kwargs):
    world = context.launch_configurations["world"]
    gui = context.launch_configurations["gui"]
    world_sdf = os.path.join(_WORLDS_DIR, f"{world}.sdf")

    gz_args = f"{world_sdf} -r -v 3"
    if gui.lower() != "true":
        gz_args += " -s"

    # 1. Gz Sim
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ros_gz_sim"),
                                   "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments=[("gz_args", gz_args)],
    )

    # 2. Spawn Go2 from SDF file (velocity-controlled, no URDF needed)
    gz_spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file", _GO2_NAV_SDF,
            "-name", "go2",
            "-x", "0", "-y", "0", "-z", "0",
        ],
    )

    # 3. ros_gz_bridge — sensor + command topics
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/model/go2/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/depth@sensor_msgs/msg/Image[gz.msgs.Image",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        remappings=[
            ("/model/go2/odometry", "/state_estimation"),
            ("/cmd_vel", "/cmd_vel_nav"),
            ("/scan/points", "/registered_scan"),
            ("/imu", "/imu/data"),
        ],
        output="screen",
    )

    # 4. RViz2 (optional)
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(
            context.launch_configurations.get("use_rviz", "false")
        ),
    )

    return [gz_sim, gz_spawn, ros_gz_bridge, rviz2]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="apartment"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        OpaqueFunction(function=_launch_setup),
    ])
