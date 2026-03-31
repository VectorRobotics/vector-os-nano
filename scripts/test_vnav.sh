#!/bin/bash
# Vector Nav Stack integration harness
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash
if [ -f "$NAV_STACK/install/setup.bash" ]; then
    source "$NAV_STACK/install/setup.bash"
fi

cleanup() {
    kill $BRIDGE_PID $NAV_PID 2>/dev/null
    wait $BRIDGE_PID $NAV_PID 2>/dev/null
}
trap cleanup EXIT

PASS=0; FAIL=0
report() {
    if [ "$2" = "PASS" ]; then echo "  [PASS] $1"; PASS=$((PASS+1))
    else echo "  [FAIL] $1 — $3"; FAIL=$((FAIL+1)); fi
}

echo "======================================"
echo "  Vector Nav Stack Integration Test"
echo "======================================"

# --- N0: Bridge topics ---
echo ""
echo "--- N0-N2: Bridge topics ---"
echo "[1] Starting bridge (headless)..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" --no-gui &
BRIDGE_PID=$!
sleep 8

TOPICS=$(ros2 topic list 2>/dev/null)
echo "$TOPICS" | grep -q "/state_estimation" && report "N0: /state_estimation published" "PASS" || report "N0: /state_estimation" "FAIL" "not found"
echo "$TOPICS" | grep -q "/registered_scan" && report "N0: /registered_scan published" "PASS" || report "N0: /registered_scan" "FAIL" "not found"

# N1: TF map→sensor
TF_CHECK=$(timeout 5 ros2 run tf2_ros tf2_echo map sensor 2>&1 | head -5)
if echo "$TF_CHECK" | grep -q "Translation"; then
    report "N1: TF map→sensor" "PASS"
else
    report "N1: TF map→sensor" "FAIL" "not found"
fi

# N2: joy + speed
JOY=$(timeout 3 ros2 topic echo /joy --once 2>/dev/null | grep "axes:" | head -1)
[ -n "$JOY" ] && report "N2: /joy published" "PASS" || report "N2: /joy" "FAIL" "no data"

SPEED=$(timeout 3 ros2 topic echo /speed --once 2>/dev/null | grep "data:" | head -1)
[ -n "$SPEED" ] && report "N2: /speed published" "PASS" || report "N2: /speed" "FAIL" "no data"

# Check /state_estimation has correct frame
ODOM_FRAME=$(timeout 3 ros2 topic echo /state_estimation --once 2>/dev/null | grep "frame_id:" | head -1)
if echo "$ODOM_FRAME" | grep -q "map"; then
    report "N0: /state_estimation frame=map" "PASS"
else
    report "N0: /state_estimation frame" "FAIL" "expected 'map', got: $ODOM_FRAME"
fi

# --- N3: Nav Stack launch ---
echo ""
echo "--- N3: Vector Nav Stack ---"

if [ ! -f "$NAV_STACK/install/setup.bash" ]; then
    report "N3: Nav stack workspace" "FAIL" "not built — run: cd $NAV_STACK && colcon build"
    echo ""
    echo "======================================"
    echo "  RESULTS: $PASS passed, $FAIL failed"
    echo "======================================"
    exit $FAIL
fi

echo "[2] Starting nav stack (localPlanner + terrainAnalysis)..."

# Start a minimal set: sensor_scan_generation + terrain_analysis + local_planner
# We skip FAR planner for initial test — just verify terrain_map is built
ros2 launch sensor_scan_generation sensor_scan_generation.launch.py 2>/dev/null &
NAV_PID=$!
sleep 3

# Also start terrain_analysis
ros2 run terrain_analysis terrainAnalysis 2>/dev/null &
TA_PID=$!
sleep 5

TOPICS2=$(ros2 topic list 2>/dev/null)
echo "$TOPICS2" | grep -q "/terrain_map" && report "N3: /terrain_map published" "PASS" || report "N3: /terrain_map" "FAIL" "not found"

# N4: Manual cmd_vel → robot moves
echo ""
echo "--- N4: Movement via cmd_vel ---"
START_X=$(timeout 2 ros2 topic echo /state_estimation --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')

for i in $(seq 1 6); do
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/TwistStamped \
        "{header: {frame_id: 'base_link'}, twist: {linear: {x: 0.3}, angular: {z: 0.0}}}" 2>/dev/null
    sleep 0.5
done
sleep 2

END_X=$(timeout 2 ros2 topic echo /state_estimation --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')

if [ -n "$START_X" ] && [ -n "$END_X" ]; then
    DX=$(python3 -c "print(f'{abs(float(\"$END_X\") - float(\"$START_X\")):.3f}')" 2>/dev/null)
    if python3 -c "exit(0 if abs(float('$END_X') - float('$START_X')) > 0.1 else 1)" 2>/dev/null; then
        report "N4: TwistStamped cmd_vel moves robot (dx=$DX)" "PASS"
    else
        report "N4: robot movement" "FAIL" "dx=$DX too small"
    fi
else
    report "N4: robot movement" "FAIL" "could not read position"
fi

kill $TA_PID 2>/dev/null

echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
[ $FAIL -eq 0 ] && echo "  ALL TESTS PASSED" || echo "  SOME TESTS FAILED"
exit $FAIL
