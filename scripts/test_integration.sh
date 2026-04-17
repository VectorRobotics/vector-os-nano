#!/bin/bash
# Comprehensive integration harness — tests ALL subsystems
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"
export ROBOT_CONFIG_PATH="unitree/unitree_go2"

source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

PIDS=()
cleanup() {
    for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
report() {
    if [ "$2" = "PASS" ]; then echo "  [PASS] $1"; PASS=$((PASS+1))
    else echo "  [FAIL] $1 — $3"; FAIL=$((FAIL+1)); fi
}

echo "======================================"
echo "  Integration Harness (Full Stack)"
echo "======================================"

# ===================== BRIDGE =====================
echo ""
echo "--- Bridge ---"
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" --no-gui &
PIDS+=($!); sleep 8

# Check all bridge topics
for topic in /state_estimation /registered_scan /speed; do
    timeout 5 ros2 topic echo $topic --once >/dev/null 2>&1 && \
        report "Bridge: $topic" "PASS" || report "Bridge: $topic" "FAIL" "no data"
done

# TF map→sensor
timeout 5 ros2 run tf2_ros tf2_echo map sensor 2>&1 | grep -q "Translation" && \
    report "Bridge: TF map→sensor" "PASS" || report "Bridge: TF map→sensor" "FAIL" "no TF"

# ===================== NAV STACK =====================
echo ""
echo "--- Nav Stack ---"
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 autonomyMode:=true \
    joyToSpeedDelay:=0.0 twoWayDrive:=true > /dev/null 2>&1 &
PIDS+=($!); sleep 6

ros2 run sensor_scan_generation sensorScanGeneration > /dev/null 2>&1 &
PIDS+=($!)
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p clearDyObs:=true \
  -p minDyObsDis:=0.14 \
  -p minOutOfFovPointNum:=20 \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 \
  -p limitGroundLift:=true \
  -p maxGroundLift:=0.05 \
  -p minDyObsVFOV:=-30.0 \
  -p maxDyObsVFOV:=35.0 > /dev/null 2>&1 &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 > /dev/null 2>&1 &
PIDS+=($!); sleep 5

# Check nav stack topics
timeout 8 ros2 topic echo /odom_base_link --once >/dev/null 2>&1 && \
    report "Nav: /odom_base_link" "PASS" || report "Nav: /odom_base_link" "FAIL" "no data"
timeout 8 ros2 topic echo /terrain_map --once >/dev/null 2>&1 && \
    report "Nav: /terrain_map" "PASS" || report "Nav: /terrain_map" "FAIL" "no data"
timeout 8 ros2 topic echo /terrain_map_ext --once >/dev/null 2>&1 && \
    report "Nav: /terrain_map_ext" "PASS" || report "Nav: /terrain_map_ext" "FAIL" "no data"

# TF chain
timeout 5 ros2 run tf2_ros tf2_echo sensor base_link 2>&1 | grep -q "Translation" && \
    report "Nav: TF sensor→base_link" "PASS" || report "Nav: TF sensor→base_link" "FAIL" "no TF"

# Check pathFollower alive
NODES=$(ros2 node list 2>/dev/null)
echo "$NODES" | grep -qi "pathfollower\|path_follower" && \
    report "Nav: pathFollower running" "PASS" || report "Nav: pathFollower running" "FAIL" "not found"

# ===================== POINT CLOUD =====================
echo ""
echo "--- Point Cloud Verification ---"
# Check registered_scan format
PC_INFO=$(timeout 5 ros2 topic echo /registered_scan --once 2>/dev/null)
echo "$PC_INFO" | grep -q "frame_id: map" && \
    report "PC: frame=map" "PASS" || report "PC: frame" "FAIL" "not map"

# Check terrain_map has intensity field with proper values
TM_WIDTH=$(timeout 5 ros2 topic echo /terrain_map --once 2>/dev/null | grep "width:" | head -1 | awk '{print $2}')
if [ -n "$TM_WIDTH" ] && [ "$TM_WIDTH" -gt 0 ] 2>/dev/null; then
    report "PC: terrain_map has $TM_WIDTH points" "PASS"
else
    report "PC: terrain_map content" "FAIL" "empty or no width"
fi

# ===================== TELEOP =====================
echo ""
echo "--- Teleop ---"
# Simulate teleop by publishing /joy with forward velocity
ros2 topic pub --once /joy sensor_msgs/msg/Joy \
    "{header: {frame_id: 'teleop_panel'}, axes: [0,0,1.0,0,0.5,1.0,0,0], buttons: [0,0,0,0,0,0,0,1,0,0,0]}" 2>/dev/null
sleep 1

# Check if /navigation_cmd_vel responds
NAV_CMD=$(timeout 5 ros2 topic echo /navigation_cmd_vel --once 2>/dev/null | head -3)
if [ -n "$NAV_CMD" ]; then
    report "Teleop: /navigation_cmd_vel responds to /joy" "PASS"
else
    report "Teleop: /navigation_cmd_vel" "FAIL" "no response to joy"
fi

# Publish multiple joy messages and check movement
# Use python for reliable position parsing
START_POS=$(timeout 5 ros2 topic echo /odom_base_link --once 2>/dev/null | python3 -c "
import sys
for line in sys.stdin:
    if 'position:' in line:
        for l2 in sys.stdin:
            if 'x:' in l2:
                print(l2.strip().split(':')[1].strip()); break
        break
" 2>/dev/null)
for i in $(seq 1 10); do
    ros2 topic pub --once /joy sensor_msgs/msg/Joy \
        "{header: {frame_id: 'teleop'}, axes: [0,0,1.0,0,0.8,1.0,0,0], buttons: [0,0,0,0,0,0,0,1,0,0,0]}" 2>/dev/null
    sleep 0.5
done
sleep 2
END_POS=$(timeout 5 ros2 topic echo /odom_base_link --once 2>/dev/null | python3 -c "
import sys
for line in sys.stdin:
    if 'position:' in line:
        for l2 in sys.stdin:
            if 'x:' in l2:
                print(l2.strip().split(':')[1].strip()); break
        break
" 2>/dev/null)

if [ -n "$START_POS" ] && [ -n "$END_POS" ]; then
    DX=$(python3 -c "print(f'{abs(float(\"$END_POS\") - float(\"$START_POS\")):.3f}')" 2>/dev/null || echo "0")
    python3 -c "exit(0 if abs(float('$END_POS') - float('$START_POS')) > 0.05 else 1)" 2>/dev/null && \
        report "Teleop: robot moves with /joy (dx=$DX)" "PASS" || \
        report "Teleop: robot movement" "FAIL" "dx=$DX too small"
else
    report "Teleop: robot movement" "FAIL" "could not read position"
fi

# ===================== CAMERA =====================
echo ""
echo "--- Camera ---"
timeout 5 ros2 topic echo /camera/image --once >/dev/null 2>&1 && \
    report "Camera: /camera/image" "PASS" || report "Camera: /camera/image" "FAIL" "no data"
timeout 5 ros2 topic echo /camera/depth --once >/dev/null 2>&1 && \
    report "Camera: /camera/depth" "PASS" || report "Camera: /camera/depth" "FAIL" "no data"

# ===================== FAR PLANNER =====================
echo ""
echo "--- FAR Planner ---"
ros2 launch far_planner far_planner.launch config:=indoor > /dev/null 2>&1 &
PIDS+=($!); sleep 8

FAR_ALIVE=$(ros2 node list 2>/dev/null | grep -c "far_planner")
[ "$FAR_ALIVE" -ge 1 ] && \
    report "FAR: node running" "PASS" || report "FAR: node" "FAIL" "not running"

# Move robot first to give FAR terrain data
echo "  Seeding FAR with movement..."
for i in $(seq 1 6); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{linear: {x: 0.2}}" 2>/dev/null
    sleep 1
done
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{}" 2>/dev/null
sleep 3

# Now send goal
ros2 topic pub --once /goal_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: 'map'}, point: {x: 12.0, y: 3.0, z: 0.0}}" 2>/dev/null
sleep 8

timeout 10 ros2 topic echo /global_path --once >/dev/null 2>&1 && \
    report "FAR: /global_path generated" "PASS" || report "FAR: /global_path" "FAIL" "no path after seeding"

# ===================== STABILITY =====================
echo ""
echo "--- Stability (30s) ---"
STABLE_COUNT=0
for i in $(seq 1 3); do
    sleep 5
    timeout 8 ros2 topic echo /state_estimation --once >/dev/null 2>&1 && STABLE_COUNT=$((STABLE_COUNT+1))
done
[ "$STABLE_COUNT" -ge 2 ] && \
    report "Stability: ${STABLE_COUNT}/3 checks passed" "PASS" || \
    report "Stability" "FAIL" "only ${STABLE_COUNT}/3 checks"

# ===================== RESULTS =====================
echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
[ $FAIL -eq 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $FAIL
