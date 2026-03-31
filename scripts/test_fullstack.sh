#!/bin/bash
# Full stack harness: bridge + terrain + localPlanner + FAR planner
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
    echo "[cleanup]"
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
echo "  Full Stack Harness"
echo "======================================"

# 1. Bridge
echo "[1/5] Bridge..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" --no-gui &
PIDS+=($!); sleep 8

timeout 3 ros2 topic echo /state_estimation --once >/dev/null 2>&1 && \
    report "F0: Bridge /state_estimation" "PASS" || report "F0: Bridge" "FAIL" "no data"
timeout 3 ros2 topic echo /registered_scan --once >/dev/null 2>&1 && \
    report "F0: Bridge /registered_scan" "PASS" || report "F0: Bridge" "FAIL" "no scan"

# 2. Local planner
echo "[2/5] Local planner..."
ros2 launch local_planner local_planner.launch.py \
    robot_config:=unitree/unitree_go2 autonomyMode:=true \
    joyToSpeedDelay:=0.0 twoWayDrive:=true > /dev/null 2>&1 &
PIDS+=($!); sleep 6

timeout 8 ros2 topic echo /odom_base_link --once >/dev/null 2>&1 && \
    report "F0: odomTransformer" "PASS" || report "F0: odomTransformer" "FAIL" "no data"

# 3. Terrain
echo "[3/5] Terrain analysis..."
ros2 run sensor_scan_generation sensorScanGeneration &
PIDS+=($!)
ros2 run terrain_analysis terrainAnalysis &
PIDS+=($!)
ros2 run terrain_analysis_ext terrainAnalysisExt &
PIDS+=($!); sleep 5

timeout 5 ros2 topic echo /terrain_map --once >/dev/null 2>&1 && \
    report "F1: terrain_map" "PASS" || report "F1: terrain_map" "FAIL" "no data"
timeout 5 ros2 topic echo /terrain_map_ext --once >/dev/null 2>&1 && \
    report "F1: terrain_map_ext" "PASS" || report "F1: terrain_map_ext" "FAIL" "no data"

# 4. FAR planner
echo "[4/5] FAR planner..."
ros2 launch far_planner far_planner.launch config:=indoor > /dev/null 2>&1 &
PIDS+=($!); sleep 8

FAR_NODES=$(ros2 node list 2>/dev/null | grep -c "far_planner\|graph_decoder")
[ "$FAR_NODES" -ge 1 ] && \
    report "F2: FAR planner running ($FAR_NODES nodes)" "PASS" || \
    report "F2: FAR planner" "FAIL" "not running"

# 5. Send distant goal and check movement
echo "[5/5] Navigation test..."

# Get start position via odom_base_link
START_POS=$(timeout 3 ros2 topic echo /odom_base_link --once 2>/dev/null)
START_X=$(echo "$START_POS" | grep -A5 "position:" | grep "x:" | head -1 | awk '{print $2}')
echo "  Start x=$START_X"

# Send goal to distant point (living room at 3, 2.5)
ros2 topic pub --once /goal_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: 'map'}, point: {x: 3.0, y: 2.5, z: 0.0}}" 2>/dev/null

# Check if FAR planner generates global path
sleep 3
timeout 5 ros2 topic echo /global_path --once >/dev/null 2>&1 && \
    report "F2: FAR /global_path published" "PASS" || \
    report "F2: FAR /global_path" "FAIL" "no path"

# Wait 30s for movement
echo "  Waiting 30s for movement..."
sleep 30

END_POS=$(timeout 3 ros2 topic echo /odom_base_link --once 2>/dev/null)
END_X=$(echo "$END_POS" | grep -A5 "position:" | grep "x:" | head -1 | awk '{print $2}')
echo "  End x=$END_X"

if [ -n "$START_X" ] && [ -n "$END_X" ]; then
    DX=$(python3 -c "print(f'{abs(float(\"$END_X\") - float(\"$START_X\")):.3f}')" 2>/dev/null)
    if python3 -c "exit(0 if abs(float('$END_X') - float('$START_X')) > 0.5 else 1)" 2>/dev/null; then
        report "F3: Robot moved toward goal (dx=$DX)" "PASS"
    else
        report "F3: Robot movement" "FAIL" "dx=$DX (need >0.5)"
    fi
else
    report "F3: Robot movement" "FAIL" "could not read position"
fi

echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
[ $FAIL -eq 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $FAIL
