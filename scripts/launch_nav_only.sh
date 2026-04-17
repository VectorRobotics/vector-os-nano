#!/bin/bash
# Vector Nav Stack nodes ONLY — no MuJoCo bridge.
# Used when the bridge is managed by another process (e.g., run.py agent).
#
# Usage:
#   ./scripts/launch_nav_only.sh              # VNav + TARE + RViz
#   ./scripts/launch_nav_only.sh --no-tare    # VNav only (manual goals)
#   ./scripts/launch_nav_only.sh --no-rviz    # headless (no RViz)

set -e
set -m

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"

NO_TARE=""
NO_RVIZ=""
for arg in "$@"; do
    case $arg in
        --no-tare) NO_TARE="1" ;;
        --no-rviz) NO_RVIZ="1" ;;
    esac
done

export ROBOT_CONFIG_PATH="unitree/unitree_go2"
source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

PIDS=()
cleanup() {
    echo ""
    echo "Stopping nav stack..."
    for p in "${PIDS[@]}"; do
        kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    done
    sleep 1
    for proc in pathFollower terrainAnalysis terrainAnalysisExt sensorScanGeneration \
                localPlanner far_planner graphDecoder visualizationTools \
                odom_transformer vehicleTransPublisher sensorTransPublisher \
                tare_planner_node; do
        pkill -9 -f "$proc" 2>/dev/null || true
    done
    rm -f /dev/shm/fastrtps_* 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

RVIZ_CFG="$REPO_DIR/config/vnav.rviz"

echo "[1/5] Starting local planner..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true &
PIDS+=($!)
sleep 4

echo "[2/5] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
sleep 1

echo "[3/5] Starting terrain analysis..."
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p clearDyObs:=true \
  -p minDyObsDis:=0.14 \
  -p minOutOfFovPointNum:=20 \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 \
  -p limitGroundLift:=true \
  -p maxGroundLift:=0.05 \
  -p minDyObsVFOV:=-30.0 \
  -p maxDyObsVFOV:=35.0 &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 &
PIDS+=($!)
sleep 3

echo "[4/5] Starting FAR planner..."
ros2 launch far_planner far_planner.launch config:=indoor &
PIDS+=($!)
sleep 3

if [ -z "$NO_TARE" ]; then
    echo "[4.5/5] Starting TARE planner..."
    ros2 launch tare_planner explore.launch scenario:=indoor_small &
    PIDS+=($!)
    sleep 2
fi

echo "[5/5] Starting visualization..."
ros2 run visualization_tools visualizationTools 2>/dev/null &
PIDS+=($!)

if [ -z "$NO_RVIZ" ]; then
    rviz2 -d "$RVIZ_CFG" 2>/dev/null &
    PIDS+=($!)
fi

echo ""
echo "Nav stack ready. Waiting for bridge to provide /state_estimation + /registered_scan."
echo "Press Ctrl+C to stop."

# Keep alive
wait
