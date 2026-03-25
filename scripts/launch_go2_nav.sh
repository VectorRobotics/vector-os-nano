#!/bin/bash
# Launch Go2 MuJoCo Bridge + Navigation Stack
#
# Terminal 1: ./scripts/launch_go2_nav.sh bridge    — MuJoCo + ROS2 bridge
# Terminal 2: ./scripts/launch_go2_nav.sh nav        — Navigation stack
# Terminal 3: ./scripts/launch_go2_nav.sh goal 10 5  — Send goal (x, y)
#
# Prerequisites:
#   - Vector navigation stack built: cd ~/Desktop/vector_navigation_stack && colcon build ...
#   - System python3.10 (NOT conda): deactivate conda or use a fresh terminal

set -e

# Force system python (not conda)
export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONPATH=""

# Source ROS2
source /opt/ros/humble/setup.bash

# Project paths
NANO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAV_ROOT="$HOME/Desktop/vector_navigation_stack"
MPC_ROOT="$HOME/Desktop/go2-convex-mpc"

case "${1:-help}" in
  bridge)
    echo "=== Go2 MuJoCo Bridge ==="
    echo "Publishing: /state_estimation, /registered_scan, /tf"
    echo "Subscribing: /cmd_vel"
    echo ""

    # Add python paths for vector_os_nano + convex_mpc
    export PYTHONPATH="$NANO_ROOT:$NANO_ROOT/.venv/lib/python3.10/site-packages:$MPC_ROOT/src:$PYTHONPATH"

    python3 "$NANO_ROOT/vector_os_nano/ros2/nodes/go2_bridge.py"
    ;;

  nav)
    echo "=== Navigation Stack (Go2 config) ==="
    export ROBOT_CONFIG_PATH="unitree/unitree_go2"

    if [ -f "$NAV_ROOT/install/setup.bash" ]; then
      source "$NAV_ROOT/install/setup.bash"
    else
      echo "ERROR: Navigation stack not built. Run:"
      echo "  cd $NAV_ROOT"
      echo "  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \\"
      echo "    --packages-skip arise_slam_mid360 arise_slam_mid360_msgs livox_ros_driver2"
      exit 1
    fi

    # Launch navigation without Unity (no endpoint, no sim_image_repub)
    # Just: local_planner + terrain_analysis + FAR planner + visualization
    ros2 launch local_planner local_planner.launch.py realRobot:=false &
    sleep 1
    ros2 launch terrain_analysis terrain_analysis.launch.py &
    sleep 1
    ros2 launch terrain_analysis_ext terrain_analysis_ext.launch &
    sleep 1
    ros2 launch sensor_scan_generation sensor_scan_generation.launch &
    sleep 1
    ros2 launch far_planner far_planner.launch config:=indoor &
    sleep 1

    echo ""
    echo "Navigation stack running. Send goals with:"
    echo "  ./scripts/launch_go2_nav.sh goal <x> <y>"
    echo ""
    wait
    ;;

  goal)
    X="${2:-10.0}"
    Y="${3:-5.0}"
    echo "Sending goal: ($X, $Y)"
    ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
      "{header: {frame_id: 'map'}, point: {x: $X, y: $Y, z: 0.0}}"
    ;;

  topics)
    echo "=== Active topics ==="
    ros2 topic list
    echo ""
    echo "=== Key topic Hz ==="
    timeout 3 ros2 topic hz /state_estimation 2>/dev/null || true
    ;;

  help|*)
    echo "Usage: $0 {bridge|nav|goal|topics|help}"
    echo ""
    echo "  bridge        Start MuJoCo Go2 + ROS2 bridge (Terminal 1)"
    echo "  nav           Start navigation stack (Terminal 2)"
    echo "  goal <x> <y>  Send navigation goal (Terminal 3)"
    echo "  topics        List active ROS2 topics"
    echo ""
    echo "Quick start:"
    echo "  Terminal 1: $0 bridge"
    echo "  Terminal 2: $0 nav"
    echo "  Terminal 3: $0 goal 5 8    # navigate to (5, 8)"
    ;;
esac
