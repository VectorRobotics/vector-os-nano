#!/bin/bash
# Go2 Nav2 Navigation — one-command launch
#
# Usage:
#   cd ~/Desktop/vector_os_nano
#   ./scripts/launch_nav2.sh              # with MuJoCo viewer
#   ./scripts/launch_nav2.sh --no-gui     # headless
#   ./scripts/launch_nav2.sh --rviz       # with RViz visualization
#
# Architecture:
#   Terminal 1: go2_nav_bridge.py (MuJoCoGo2 + ROS2 publishers)
#   Terminal 2: Nav2 stack (AMCL + MPPI + SmacPlanner2D)
#   Terminal 3: (optional) RViz
#
# After launch, send a Nav2 goal:
#   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
#     "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 12.0, y: 3.0}, orientation: {w: 1.0}}}}" \
#     --feedback

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_WS="/home/yusen/Desktop/vector_go2_sim"

# Parse args
NO_GUI=""
RVIZ=""
SINUSOIDAL=""
for arg in "$@"; do
    case $arg in
        --no-gui) NO_GUI="--no-gui" ;;
        --rviz) RVIZ="1" ;;
        --sinusoidal) SINUSOIDAL="--sinusoidal" ;;
    esac
done

# PYTHONPATH: venv packages + cmeel (pinocchio) + convex_mpc source + repo
VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

# Source ROS2
source /opt/ros/jazzy/setup.bash
# Source Nav2 workspace if built
if [ -f "$NAV_WS/install/setup.bash" ]; then
    source "$NAV_WS/install/setup.bash"
fi

echo "======================================"
echo "  Go2 Nav2 Navigation"
echo "======================================"
echo "  Bridge: MuJoCoGo2 → /odom, /scan, /cmd_vel"
echo "  Nav2:   AMCL + MPPI + SmacPlanner2D"
echo "  Map:    house (20x14m, Go2 starts at 10,3)"
echo "======================================"

# Terminal 1: Go2 bridge
echo "[1/2] Starting Go2 Nav Bridge..."
python3 "$SCRIPT_DIR/go2_nav_bridge.py" $NO_GUI $SINUSOIDAL &
BRIDGE_PID=$!
sleep 6  # wait for MuJoCo to load + stand

# Terminal 2: Nav2 AMCL stack
echo "[2/2] Starting Nav2 stack..."
MAP_FILE="$NAV_WS/maps/house.yaml"
NAV2_PARAMS="$NAV_WS/src/vector_go2_navigation/config/nav2_params.yaml"

if [ ! -f "$MAP_FILE" ]; then
    echo "ERROR: Map file not found: $MAP_FILE"
    kill $BRIDGE_PID 2>/dev/null
    exit 1
fi

ros2 launch nav2_bringup bringup_launch.py \
    map:="$MAP_FILE" \
    params_file:="$NAV2_PARAMS" \
    use_sim_time:=false \
    autostart:=true &
NAV2_PID=$!

# Optional: RViz
if [ -n "$RVIZ" ]; then
    sleep 3
    echo "Starting RViz..."
    RVIZ_CFG="$NAV_WS/src/vector_go2_navigation/config/nav2.rviz"
    if [ -f "$RVIZ_CFG" ]; then
        rviz2 -d "$RVIZ_CFG" &
    else
        rviz2 &
    fi
fi

echo ""
echo "Ready! Send a navigation goal:"
echo "  ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \\"
echo "    \"{pose: {header: {frame_id: 'map'}, pose: {position: {x: 12.0, y: 3.0}, orientation: {w: 1.0}}}}\" \\"
echo "    --feedback"
echo ""
echo "Press Ctrl+C to stop all."

# Wait for bridge to exit
wait $BRIDGE_PID
kill $NAV2_PID 2>/dev/null
