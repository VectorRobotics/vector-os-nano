#!/bin/bash
# Go2 + Vector Navigation Stack — one-command launch
#
# Usage:
#   cd ~/Desktop/vector_os_nano
#   ./scripts/launch_vnav.sh              # MuJoCo viewer + RViz
#   ./scripts/launch_vnav.sh --no-gui     # headless MuJoCo
#
# Sends a goal:
#   ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
#     "{header: {frame_id: 'map'}, point: {x: 5.0, y: 3.0, z: 0.0}}"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"

NO_GUI=""
for arg in "$@"; do
    case $arg in --no-gui) NO_GUI="--no-gui" ;; esac
done

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

cleanup() {
    echo ""
    echo "Stopping all..."
    kill $BRIDGE_PID $SSG_PID $TA_PID $TAE_PID $LP_PID $PF_PID $FAR_PID $MUX_PID $ODOM_PID $RVIZ_PID 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT

RVIZ_CFG="$REPO_DIR/config/nav2_go2.rviz"
LP_CONFIG="$NAV_STACK/src/base_autonomy/local_planner/config/unitree/unitree_go2.yaml"

echo "======================================"
echo "  Go2 + Vector Navigation Stack"
echo "======================================"
echo "  MuJoCo: Go2 MPC in house scene"
echo "  Nav: FAR Planner + terrain analysis"
echo "  Control: localPlanner + pathFollower"
echo "======================================"

# 1. Bridge (MuJoCoGo2 → ROS2)
echo "[1/6] Starting bridge..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" $NO_GUI &
BRIDGE_PID=$!
sleep 6

# 2. Sensor scan generation (syncs /registered_scan + /state_estimation)
echo "[2/6] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
SSG_PID=$!
sleep 2

# 3. Terrain analysis (local + extended)
echo "[3/6] Starting terrain analysis..."
ros2 run terrain_analysis terrainAnalysis &
TA_PID=$!
ros2 run terrain_analysis_ext terrainAnalysisExt &
TAE_PID=$!
sleep 2

# 4. Local planner + path follower
echo "[4/6] Starting local planner..."
ros2 run local_planner localPlanner &
LP_PID=$!
ros2 run local_planner pathFollower &
PF_PID=$!
sleep 2

# 5. FAR planner (global route)
echo "[5/6] Starting FAR planner..."
ros2 launch far_planner far_planner.launch.py &
FAR_PID=$!
sleep 2

# 6. Utilities
echo "[6/6] Starting utilities..."
ros2 run odom_transformer odomTransformer &
ODOM_PID=$!
ros2 run cmd_vel_mux cmd_vel_mux &
MUX_PID=$!

# RViz
rviz2 -d "$RVIZ_CFG" &
RVIZ_PID=$!

echo ""
echo "Ready! Send a navigation goal:"
echo "  ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \\"
echo "    \"{header: {frame_id: 'map'}, point: {x: 5.0, y: 3.0, z: 0.0}}\""
echo ""
echo "Press Ctrl+C to stop."

wait $BRIDGE_PID
