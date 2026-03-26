#!/bin/bash
# Launch Go2 MuJoCo Bridge + Navigation Stack
#
# Terminal 1: ./scripts/launch_go2_nav.sh bridge     — MuJoCo + ROS2 bridge
# Terminal 2: ./scripts/launch_go2_nav.sh nav         — Navigation stack
# Terminal 3: ./scripts/launch_go2_nav.sh goal 10 5   — Send goal (x, y)
#
# Prerequisites:
#   - conda deactivate (system python3.10 required)
#   - Navigation stack built at ~/Desktop/vector_navigation_stack/

set -e

# Force system python
export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONPATH=""

source /opt/ros/humble/setup.bash

NANO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAV_ROOT="$HOME/Desktop/vector_navigation_stack"
MPC_ROOT="$HOME/Desktop/go2-convex-mpc"

case "${1:-help}" in
  bridge)
    echo "=== Go2 MuJoCo Bridge ==="
    export PYTHONPATH="$NANO_ROOT:$NANO_ROOT/.venv/lib/python3.10/site-packages:$MPC_ROOT/src:$PYTHONPATH"

    python3 "$NANO_ROOT/vector_os_nano/ros2/nodes/go2_bridge.py"
    ;;

  nav)
    echo "=== Navigation Stack (Go2 config, MuJoCo mode) ==="
    export ROBOT_CONFIG_PATH="unitree/unitree_go2"

    if [ ! -f "$NAV_ROOT/install/setup.bash" ]; then
      echo "ERROR: Navigation stack not built."
      exit 1
    fi
    source "$NAV_ROOT/install/setup.bash"

    # Launch nav stack WITHOUT vehicleSimulator/Unity (our bridge provides those)
    ros2 launch local_planner local_planner.launch.py &
    sleep 2
    ros2 launch terrain_analysis terrain_analysis.launch.py &
    sleep 1
    ros2 launch terrain_analysis_ext terrain_analysis_ext.launch &
    sleep 1
    ros2 launch sensor_scan_generation sensor_scan_generation.launch &
    sleep 1
    ros2 launch far_planner far_planner.launch config:=indoor &
    sleep 1
    ros2 launch cmd_vel_mux cmd_vel_mux-launch.py &
    sleep 1

    echo ""
    echo "Navigation stack running (MuJoCo mode)."
    echo "Send goals: ./scripts/launch_go2_nav.sh goal <x> <y>"
    wait
    ;;

  goal)
    X="${2:-5.0}"
    Y="${3:-5.0}"
    echo "Sending goal: ($X, $Y)"
    source "$NAV_ROOT/install/setup.bash" 2>/dev/null
    ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
      "{header: {frame_id: 'map'}, point: {x: $X, y: $Y, z: 0.0}}"
    ;;

  kill)
    echo "Killing all..."
    pkill -9 -f "ros2|rviz|Model" 2>/dev/null
    sleep 1
    echo "Done."
    ;;

  topics)
    source "$NAV_ROOT/install/setup.bash" 2>/dev/null
    echo "=== Topic Hz ==="
    for t in /state_estimation /odom_base_link /registered_scan /terrain_map /cmd_vel /navigation_cmd_vel /global_path; do
      hz=$(timeout 3 ros2 topic hz $t 2>&1 | grep "average" | head -1)
      echo "  $t: ${hz:-NO DATA}"
    done
    ;;

  *)
    echo "Usage: $0 {bridge|nav|goal|kill|topics}"
    echo ""
    echo "  bridge         MuJoCo Go2 + ROS2 bridge (Terminal 1)"
    echo "  nav            Navigation stack (Terminal 2)"
    echo "  goal <x> <y>   Send navigation goal (Terminal 3)"
    echo "  kill           Kill everything"
    echo "  topics         Check data flow"
    ;;
esac
