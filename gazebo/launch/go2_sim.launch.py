"""ROS2 launch for Go2 in Gazebo Harmonic.

Uses standard gz_ros2_control (effort interface) — no custom hardware plugin.
Sensors: MID-360 lidar + D435 RGB-D + IMU + odometry.
"""
from __future__ import annotations

import os

import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GAZEBO_DIR = os.path.join(_THIS_DIR, "..")
_WORLDS_DIR = os.path.join(_GAZEBO_DIR, "worlds")


def _launch_setup(context, *args, **kwargs):
    world = context.launch_configurations["world"]
    gui = context.launch_configurations["gui"]
    world_sdf = os.path.join(_WORLDS_DIR, f"{world}.sdf")

    # Process URDF xacro (our wrapper: standard gz_ros2_control + sensors)
    wrapper_xacro = os.path.join(_GAZEBO_DIR, "models", "go2", "robot_with_sensors.xacro")
    robot_description = xacro.process_file(wrapper_xacro).toxml()

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

    # 2. Robot state publisher (URDF → /robot_description)
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "publish_frequency": 20.0,
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    # 3. Spawn from /robot_description topic
    gz_spawn = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "go2",
            "-allow_renaming", "true",
            "-z", "0.5",
        ],
    )

    # 4. ros_gz_bridge
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

    # 5. Controller chain: spawn → joint_state → imu → unitree_guide
    joint_state_broadcaster = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    imu_sensor_broadcaster = Node(
        package="controller_manager", executable="spawner",
        arguments=["imu_sensor_broadcaster", "--controller-manager", "/controller_manager"],
    )
    unitree_guide_controller = Node(
        package="controller_manager", executable="spawner",
        arguments=["unitree_guide_controller", "--controller-manager", "/controller_manager"],
    )

    return [
        gz_sim,
        robot_state_publisher,
        ros_gz_bridge,
        gz_spawn,
        RegisterEventHandler(OnProcessExit(
            target_action=gz_spawn,
            on_exit=[joint_state_broadcaster, imu_sensor_broadcaster],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[unitree_guide_controller],
        )),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="apartment"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        OpaqueFunction(function=_launch_setup),
    ])
