"""ROS2 launch file for Go2 Gazebo Harmonic simulation.

Starts the full simulation stack:
  1. Gz Sim server — loads the world SDF
  2. robot_state_publisher — publishes URDF with ros2_control tags
  3. Spawn Go2 — from /robot_description topic (URDF→SDF conversion)
  4. ros_gz_bridge — bridges Gz-transport topics to ROS2
  5. Controller spawners — imu, joint_state, unitree_guide_controller
  6. RViz2 (optional)

Uses URDF xacro from quadruped_ros2_control (go2_description) which includes
<ros2_control> hardware interface tags required by gz_quadruped_hardware.
Our custom world SDF provides the environment (apartment, empty_room, etc.).

Usage:
    ros2 launch gazebo/launch/go2_sim.launch.py
    ros2 launch gazebo/launch/go2_sim.launch.py world:=empty_room gui:=false
"""
from __future__ import annotations

import os

import xacro
from ament_index_python.packages import get_package_share_directory
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

# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GAZEBO_DIR = os.path.join(_THIS_DIR, "..")
_CONFIG_DIR = os.path.join(_GAZEBO_DIR, "config")
_WORLDS_DIR = os.path.join(_GAZEBO_DIR, "worlds")
_BRIDGE_YAML = os.path.join(_CONFIG_DIR, "bridge.yaml")


def _launch_setup(context, *args, **kwargs):
    """Deferred launch setup — resolves LaunchConfiguration values."""

    world = context.launch_configurations["world"]
    gui = context.launch_configurations["gui"]

    world_sdf = os.path.join(_WORLDS_DIR, f"{world}.sdf")

    # ------------------------------------------------------------------
    # Process Go2 URDF xacro from quadruped_ros2_control
    # Includes <ros2_control> tags needed by gz_quadruped_hardware
    # Then inject our sensor xacro (MID-360 lidar + D435 camera)
    # ------------------------------------------------------------------
    pkg_path = get_package_share_directory("go2_description")
    xacro_file = os.path.join(pkg_path, "xacro", "robot.xacro")
    sensors_xacro = os.path.join(_GAZEBO_DIR, "models", "go2", "sensors.xacro")

    # Process base robot URDF
    robot_xml = xacro.process_file(
        xacro_file, mappings={"GAZEBO": "true"}
    ).toxml()

    # Inject sensor links/joints/gazebo tags into URDF
    # Parse sensors.xacro separately and merge into robot XML
    if os.path.exists(sensors_xacro):
        sensors_xml = xacro.process_file(sensors_xacro).toxml()
        # Extract content between <robot> tags from sensors
        import re
        sensor_content = re.search(
            r'<robot[^>]*>(.*)</robot>', sensors_xml, re.DOTALL
        )
        if sensor_content:
            # Insert before closing </robot> tag of base URDF
            robot_description = robot_xml.replace(
                '</robot>',
                sensor_content.group(1) + '\n</robot>'
            )
        else:
            robot_description = robot_xml
    else:
        robot_description = robot_xml

    # ------------------------------------------------------------------
    # 1. Gz Sim — load world SDF
    # ------------------------------------------------------------------
    gz_args = f"{world_sdf} -r -v 3"
    if gui.lower() != "true":
        gz_args += " -s"  # server-only (headless)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ros_gz_sim"),
                                   "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments=[("gz_args", gz_args)],
    )

    # ------------------------------------------------------------------
    # 2. robot_state_publisher — publishes URDF to /robot_description
    # ------------------------------------------------------------------
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "publish_frequency": 20.0,
            "use_tf_static": True,
            "robot_description": robot_description,
            "ignore_timestamp": True,
        }],
    )

    # ------------------------------------------------------------------
    # 3. Spawn Go2 from /robot_description topic (URDF→SDF at spawn)
    # ------------------------------------------------------------------
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "go2",
            "-allow_renaming", "true",
            "-x", "0", "-y", "0", "-z", "0.5",
        ],
    )

    # ------------------------------------------------------------------
    # 4. ros_gz_bridge — topic relay
    #    Use command-line argument syntax: topic@ROS_type[gz_type (GZ→ROS)
    #    or topic@ROS_type]gz_type (ROS→GZ)
    # ------------------------------------------------------------------
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/depth@sensor_msgs/msg/Image[gz.msgs.Image",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        remappings=[
            ("/scan/points", "/registered_scan"),
            ("/imu", "/imu/data"),
        ],
        output="screen",
    )

    # ------------------------------------------------------------------
    # 5. Controller spawners — chained via OnProcessExit
    #    Order: spawn_entity → imu_broadcaster + joint_state_broadcaster
    #           → unitree_guide_controller
    # ------------------------------------------------------------------
    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    unitree_guide_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["unitree_guide_controller",
                   "--controller-manager", "/controller_manager"],
    )

    # ------------------------------------------------------------------
    # 6. RViz2 (optional)
    # ------------------------------------------------------------------
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

    return [
        gz_sim,
        robot_state_publisher,
        ros_gz_bridge,
        gz_spawn_entity,
        # Chain: after spawn → imu + joint_state broadcasters
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[imu_sensor_broadcaster, joint_state_broadcaster],
            )
        ),
        # Chain: after joint_state → guide controller
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[unitree_guide_controller],
            )
        ),
        rviz2,
    ]


def generate_launch_description() -> LaunchDescription:
    """Build and return the full Gz Sim + Go2 launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "world", default_value="apartment",
            description="World name (SDF file in gazebo/worlds/)",
        ),
        DeclareLaunchArgument(
            "gui", default_value="true",
            description="Launch Gz Sim GUI window (true/false)",
        ),
        DeclareLaunchArgument(
            "use_rviz", default_value="false",
            description="Launch RViz2 (true/false)",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
