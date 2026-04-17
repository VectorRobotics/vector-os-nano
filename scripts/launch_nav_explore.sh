#!/bin/bash
# Nav stack + TARE exploration — assumes bridge is already running.
# Starts localPlanner, terrainAnalysis, FAR, TARE, seed, all in ONE process group.
#
# Usage: ./scripts/launch_nav_explore.sh
# Requires: bridge already publishing /state_estimation and /registered_scan

set -e
set -m

NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export ROBOT_CONFIG_PATH="unitree/unitree_go2"
source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

# Deploy Go2-tuned TARE config
cp "$REPO_DIR/config/tare_go2_indoor.yaml" \
   "$NAV_STACK/install/tare_planner/share/tare_planner/indoor_small.yaml" 2>/dev/null || true

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
    rm -f /tmp/vector_nav_active 2>/dev/null
    echo "Nav stack stopped."
}
trap cleanup EXIT INT TERM

RVIZ_CFG="$REPO_DIR/config/vnav.rviz"

echo "[1/7] Starting local planner..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 \
    autonomyMode:=true \
    joyToSpeedDelay:=0.0 \
    twoWayDrive:=true &
PIDS+=($!)
sleep 4

echo "[2/7] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
sleep 1

echo "[3/7] Starting terrain analysis..."
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

echo "[4/7] Starting FAR planner..."
ros2 launch far_planner far_planner.launch config:=indoor &
PIDS+=($!)
sleep 3

echo "[5/7] Starting TARE planner..."
ros2 launch tare_planner explore.launch scenario:=indoor_small &
PIDS+=($!)
sleep 2

echo "[6/7] Starting RViz..."
ros2 run visualization_tools visualizationTools 2>/dev/null &
PIDS+=($!)
rviz2 -d "$RVIZ_CFG" 2>/dev/null &
PIDS+=($!)

# Enable bridge path follower
touch /tmp/vector_nav_active

echo "[7/7] Seeding planners..."
sleep 2
for i in $(seq 1 4); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{linear: {x: 0.2}}" 2>/dev/null
    sleep 1
done
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{}" 2>/dev/null

echo ""
echo "Exploration active. Ctrl+C to stop."
wait
