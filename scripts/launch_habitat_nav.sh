#!/bin/bash
# Habitat house world + Vector Nav Stack (local planner chain) — N2.
#
# Launches ONLY the nav stack nodes (sibling workspace, never vendored):
# local_planner.launch.py (localPlanner + pathFollower + odom_transformer +
# static TFs) + sensor_scan_generation + terrain_analysis. The habitat feed
# (/state_estimation 50Hz + TF map->sensor + /registered_scan + /speed) is
# provided by HabitatSysnavBridge in the caller's process (vector-cli or a
# harness). FAR/TARE are out of N2 scope.
#
# Robot config: unitree_g1 — head-mounted sensor (offsets 0/0, matching the
# habitat eye-height camera) and a 0.5 m square footprint.
#
# Usage:
#   ./scripts/launch_habitat_nav.sh            # headless
#   ./scripts/launch_habitat_nav.sh --rviz     # + RViz
#
# Send a goal:
#   ros2 topic pub --once /way_point geometry_msgs/msg/PointStamped \
#     "{header: {frame_id: 'map'}, point: {x: -7.4, y: -2.4, z: 0.0}}"

set -e
set -m

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="${VECTOR_NAV_STACK:-/home/yusen/Desktop/vector_navigation_stack}"

if [ ! -f "$NAV_STACK/install/setup.bash" ]; then
    echo "ERROR: nav stack workspace not built: $NAV_STACK (set VECTOR_NAV_STACK)" >&2
    exit 1
fi

RVIZ=""
for arg in "$@"; do
    case $arg in
        --rviz) RVIZ="1" ;;
    esac
done

export ROBOT_CONFIG_PATH="unitree/unitree_g1"
source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

PIDS=()
cleanup() {
    echo ""
    echo "Stopping habitat nav stack..."
    for p in "${PIDS[@]}"; do
        kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    done
    sleep 1
    for proc in pathFollower terrainAnalysis sensorScanGeneration \
                localPlanner odom_transformer vehicleTransPublisher \
                sensorTransPublisher; do
        pkill -9 -f "$proc" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "[1/3] Starting local planner (G1 profile)..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_g1 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true &
PIDS+=($!)
sleep 4

echo "[2/3] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
sleep 1

echo "[3/3] Starting terrain analysis..."
# Defaults (minRelZ -1.5 / maxRelZ 0.2 relative to the 1.32 m sensor) fit
# the habitat eye-height feed: ground at rel -1.2 is in band, the ceiling is
# auto-cut. obstacleHeightThre matches the G1 flat-terrain profile.
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p obstacleHeightThre:=0.1 &
PIDS+=($!)

if [ -n "$RVIZ" ]; then
    rviz2 -d "$REPO_DIR/config/vnav.rviz" 2>/dev/null &
    PIDS+=($!)
fi

echo ""
echo "Habitat nav stack ready — waiting for the feed (/state_estimation +"
echo "/registered_scan from HabitatSysnavBridge) and a /way_point goal."
wait
