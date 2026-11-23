#!/usr/bin/env bash
set -e
set -m
folder=${1:?navigation configuration directory required}
source "$(dirname -- "${BASH_SOURCE[0]}")/nav_env.sh"
source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"
log_dir=${VECTOR_DEMO_RUN:-$VECTOR_RUN_DIR}
timeout 60 ros2 topic echo /clock rosgraph_msgs/msg/Clock --once > "$log_dir/nav-clock.txt"
pids=()
cleanup() { nav_stop_groups "${pids[@]}"; wait 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ros2 run local_planner localPlanner --ros-args --params-file "$folder/local.yaml" -p pathFolder:="$NAV_STACK/install/local_planner/share/local_planner/paths" & pids+=($!)
ros2 run local_planner pathFollower --ros-args --params-file "$folder/local.yaml" & pids+=($!)
ros2 run sensor_scan_generation sensorScanGeneration --ros-args -p use_sim_time:=true & pids+=($!)
ros2 run terrain_analysis terrainAnalysis --ros-args -p use_sim_time:=true -p vehicleHeight:=0.25505 -p maxRelZ:=1.5 -p obstacleHeightThre:=0.1 -p limitGroundLift:=true -p maxGroundLift:=0.05 & pids+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args -p use_sim_time:=true -p vehicleHeight:=0.25505 -p maxRelZ:=1.5 -p obstacleHeightThre:=0.1 & pids+=($!)
ros2 run far_planner far_planner --ros-args -r __node:=far_planner --params-file "$folder/far.yaml" -r /odom_world:=/odom_base_link -r /terrain_cloud:=/terrain_map_ext -r /scan_cloud:=/registered_scan -r /terrain_local_cloud:=/terrain_map -r /way_point:=/far/way_point & pids+=($!)
ros2 run tare_planner tare_planner_node --ros-args --params-file "$folder/tare.yaml" -r /way_point:=/tare/way_point & pids+=($!)

# TARE starts only on a CLI explore request; the controller selects one waypoint source.
# Any missing/crashed node ends this owned stack.
wait -n
exit 1
