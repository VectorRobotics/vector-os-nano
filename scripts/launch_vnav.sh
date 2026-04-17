#!/bin/bash
# Go2 + Vector Navigation Stack (Full) — one-command launch
#
# Usage:
#   cd ~/Desktop/vector_os_nano
#   ./scripts/launch_vnav.sh              # MuJoCo viewer + RViz
#   ./scripts/launch_vnav.sh --no-gui     # headless MuJoCo
#
# Send goal:
#   ros2 topic pub --once /goal_point geometry_msgs/msg/PointStamped \
#     "{header: {frame_id: 'map'}, point: {x: 3.0, y: 2.5, z: 0.0}}"

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
export ROBOT_CONFIG_PATH="unitree/unitree_go2"

source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

# Run in own process group so cleanup can kill entire tree
set -m

PIDS=()
cleanup() {
    echo ""
    echo "Stopping all processes..."
    # Kill tracked PIDs and their children
    for p in "${PIDS[@]}"; do
        kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    done
    sleep 1
    # Force kill any stragglers by name
    for proc in pathFollower terrainAnalysis terrainAnalysisExt sensorScanGeneration \
                localPlanner far_planner graphDecoder visualizationTools \
                odom_transformer vehicleTransPublisher sensorTransPublisher; do
        pkill -9 -f "$proc" 2>/dev/null || true
    done
    # Clean stale SHM and nav flag
    rm -f /dev/shm/fastrtps_* 2>/dev/null
    rm -f /tmp/vector_nav_active 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# Clean stale nav flag from previous session
rm -f /tmp/vector_nav_active 2>/dev/null

RVIZ_CFG="$REPO_DIR/config/vnav.rviz"

echo "======================================"
echo "  Go2 + Vector Navigation Stack"
echo "======================================"
echo "  MuJoCo: Go2 MPC in house scene"
echo "  Local:  localPlanner + pathFollower"
echo "  Global: FAR Planner (visibility graph)"
echo "  Terrain: terrainAnalysis + ext"
echo "======================================"

# 1. Bridge (MuJoCoGo2 → ROS2 topics)
echo "[1/6] Starting bridge..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" $NO_GUI &
PIDS+=($!)
sleep 7

# 2. Local planner stack (localPlanner, pathFollower, odomTransformer, static TFs)
echo "[2/6] Starting local planner..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true &
PIDS+=($!)
sleep 4

# 3. Sensor scan generation (syncs /registered_scan + /state_estimation)
echo "[3/6] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
sleep 1

# 4. Terrain analysis (local + extended)
# Use ros2 run (same as working launch_explore.sh — the launch.py version
# loads different params that may break sensorScanGeneration sync)
echo "[4/6] Starting terrain analysis..."
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p clearDyObs:=true \
  -p minDyObsDis:=0.14 \
  -p minOutOfFovPointNum:=20 \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=0.3 \
  -p limitGroundLift:=true \
  -p maxGroundLift:=0.05 \
  -p minDyObsVFOV:=-30.0 \
  -p maxDyObsVFOV:=35.0 &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=0.3 &
PIDS+=($!)
sleep 3

# 5. FAR Planner (global route) + graph decoder
echo "[5/6] Starting FAR planner..."
ros2 launch far_planner far_planner.launch config:=indoor &
PIDS+=($!)
sleep 3

# 6. Visualization + RViz
echo "[6/6] Starting visualization + RViz..."
ros2 run visualization_tools visualizationTools 2>/dev/null &
PIDS+=($!)
rviz2 -d "$RVIZ_CFG" 2>/dev/null &
PIDS+=($!)

# Seed FAR planner with initial scan data.
# Path follower is DISABLED (no /tmp/vector_nav_active flag yet),
# so these velocity commands move the robot directly via _cmd_vel_cb
# without triggering autonomous path following.
echo ""
echo "Seeding FAR planner..."
sleep 2
for i in $(seq 1 4); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{linear: {x: 0.2}}" 2>/dev/null
    sleep 1
done
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{}" 2>/dev/null
sleep 2

echo ""
echo "Ready! Controls:"
echo "  Teleop:  Use RViz teleop panel (drag to drive)"
echo "  Goal:    ros2 topic pub --once /goal_point geometry_msgs/msg/PointStamped \\"
echo "           \"{header: {frame_id: 'map'}, point: {x: 3.0, y: 2.5, z: 0.0}}\""
echo "  Manual:  ros2 topic pub /cmd_vel_nav geometry_msgs/msg/Twist '{linear: {x: 0.3}}' -r 10"
echo ""
echo "Press Ctrl+C to stop."

wait ${PIDS[0]}
