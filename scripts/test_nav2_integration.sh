#!/bin/bash
# Automated Nav2 integration test — runs all layers, reports pass/fail
# Usage: cd ~/Desktop/vector_os_nano && ./scripts/test_nav2_integration.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
NAV_WS="/home/yusen/Desktop/vector_go2_sim"

VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash
if [ -f "$NAV_WS/install/setup.bash" ]; then
    source "$NAV_WS/install/setup.bash"
fi

cleanup() {
    echo "[cleanup] Stopping all processes..."
    kill $BRIDGE_PID $NAV2_PID 2>/dev/null
    wait $BRIDGE_PID $NAV2_PID 2>/dev/null
    echo "[cleanup] Done."
}
trap cleanup EXIT

PASS=0
FAIL=0
report() {
    if [ "$2" = "PASS" ]; then
        echo "  [PASS] $1"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $1 — $3"
        FAIL=$((FAIL+1))
    fi
}

echo "======================================"
echo "  Nav2 Integration Test Suite"
echo "======================================"

# ==========================================================================
echo ""
echo "--- Layer 1: Bridge standalone ---"
echo "[1/6] Starting bridge (headless, MPC backend)..."
python3 "$SCRIPT_DIR/go2_nav_bridge.py" --no-gui &
BRIDGE_PID=$!
sleep 8

# Check topics exist
echo "[2/6] Checking ROS2 topics..."
TOPICS=$(ros2 topic list 2>/dev/null)
echo "$TOPICS" | grep -q "/odom" && report "/odom published" "PASS" || report "/odom published" "FAIL" "topic not found"
echo "$TOPICS" | grep -q "/scan" && report "/scan published" "PASS" || report "/scan published" "FAIL" "topic not found"
echo "$TOPICS" | grep -q "/tf" && report "/tf published" "PASS" || report "/tf published" "FAIL" "topic not found"

# Check odom has data
ODOM_POS=$(timeout 3 ros2 topic echo /odom --once 2>/dev/null | grep -A3 "position:" | head -4)
if echo "$ODOM_POS" | grep -q "x:"; then
    report "/odom has position data" "PASS"
else
    report "/odom has position data" "FAIL" "no data received"
fi

# Check scan has data
SCAN_RANGES=$(timeout 3 ros2 topic echo /scan --once 2>/dev/null | grep "ranges:" | head -1)
if [ -n "$SCAN_RANGES" ]; then
    report "/scan has range data" "PASS"
else
    report "/scan has range data" "FAIL" "no scan data"
fi

# Check TF odom→base
TF_CHECK=$(timeout 3 ros2 run tf2_ros tf2_echo odom base 2>&1 | head -5)
if echo "$TF_CHECK" | grep -q "Translation"; then
    report "TF odom→base" "PASS"
else
    report "TF odom→base" "FAIL" "transform not found"
fi

# ==========================================================================
echo ""
echo "--- Layer 2: cmd_vel → robot movement ---"
echo "[3/6] Publishing cmd_vel_nav (vx=0.3 for 3s)..."

# Record start position
START_X=$(timeout 2 ros2 topic echo /odom --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')

# Publish velocity for 3 seconds
for i in $(seq 1 6); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist \
        "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" 2>/dev/null
    sleep 0.5
done

sleep 1

# Record end position
END_X=$(timeout 2 ros2 topic echo /odom --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')

if [ -n "$START_X" ] && [ -n "$END_X" ]; then
    DX=$(python3 -c "print(f'{float($END_X) - float($START_X):.3f}')")
    if python3 -c "exit(0 if abs(float($END_X) - float($START_X)) > 0.1 else 1)"; then
        report "cmd_vel_nav moves robot (dx=$DX)" "PASS"
    else
        report "cmd_vel_nav moves robot (dx=$DX)" "FAIL" "displacement too small"
    fi
else
    report "cmd_vel_nav moves robot" "FAIL" "could not read position"
fi

# Test turning
echo "[4/6] Publishing cmd_vel_nav (vyaw=1.0 for 3s)..."
for i in $(seq 1 6); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist \
        "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.0}}" 2>/dev/null
    sleep 0.5
done
sleep 1

# Check robot is still upright
Z_POS=$(timeout 2 ros2 topic echo /odom --once 2>/dev/null | grep "z:" | head -1 | awk '{print $2}')
if [ -n "$Z_POS" ] && python3 -c "exit(0 if float($Z_POS) > 0.15 else 1)"; then
    report "Robot upright after turning (z=$Z_POS)" "PASS"
else
    report "Robot upright after turning" "FAIL" "z=$Z_POS (fell over?)"
fi

# Stop robot
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" 2>/dev/null

# ==========================================================================
echo ""
echo "--- Layer 3: Nav2 stack ---"
echo "[5/6] Starting Nav2 (AMCL + MPPI + SmacPlanner2D)..."

MAP_FILE="$NAV_WS/maps/house.yaml"
NAV2_PARAMS="$NAV_WS/src/vector_go2_navigation/config/nav2_params.yaml"

if [ ! -f "$MAP_FILE" ]; then
    report "Nav2 map file" "FAIL" "$MAP_FILE not found"
else
    ros2 launch nav2_bringup bringup_launch.py \
        map:="$MAP_FILE" \
        params_file:="$NAV2_PARAMS" \
        use_sim_time:=false \
        autostart:=true > /tmp/nav2_test.log 2>&1 &
    NAV2_PID=$!

    echo "    Waiting for Nav2 activation (30s)..."
    ACTIVATED=0
    for i in $(seq 1 30); do
        if grep -q "Managed nodes are active" /tmp/nav2_test.log 2>/dev/null; then
            ACTIVATED=1
            break
        fi
        sleep 1
    done

    if [ $ACTIVATED -eq 1 ]; then
        report "Nav2 all nodes active" "PASS"
    else
        report "Nav2 all nodes active" "FAIL" "timeout waiting for activation"
    fi

    # Check AMCL TF
    sleep 2
    MAP_ODOM_TF=$(timeout 3 ros2 run tf2_ros tf2_echo map odom 2>&1 | head -3)
    if echo "$MAP_ODOM_TF" | grep -q "Translation"; then
        report "TF map→odom (AMCL localization)" "PASS"
    else
        report "TF map→odom (AMCL localization)" "FAIL" "AMCL not publishing TF"
    fi

    # ==========================================================================
    echo ""
    echo "--- Layer 4: Nav2 goal → robot walks ---"
    echo "[6/6] Sending NavigateToPose goal (2m forward)..."

    # Get current position
    CUR_X=$(timeout 2 ros2 topic echo /odom --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')
    if [ -z "$CUR_X" ]; then CUR_X="10.0"; fi
    GOAL_X=$(python3 -c "print(f'{float($CUR_X) + 2.0:.1f}')")

    # Send goal
    ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
        "{pose: {header: {frame_id: 'map'}, pose: {position: {x: $GOAL_X, y: 3.0, z: 0.0}, orientation: {w: 1.0}}}}" \
        2>/dev/null &
    GOAL_PID=$!

    # Monitor for 30 seconds
    echo "    Monitoring position for 30s (goal_x=$GOAL_X)..."
    MOVED=0
    for i in $(seq 1 30); do
        POS_X=$(timeout 2 ros2 topic echo /odom --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')
        if [ -n "$POS_X" ] && [ -n "$CUR_X" ]; then
            DISP=$(python3 -c "print(f'{abs(float($POS_X) - float($CUR_X)):.2f}')")
            if python3 -c "exit(0 if abs(float($POS_X) - float($CUR_X)) > 0.5 else 1)" 2>/dev/null; then
                MOVED=1
                echo "    Robot moved! displacement=$DISP m at t=${i}s"
                break
            fi
        fi
        sleep 1
    done

    kill $GOAL_PID 2>/dev/null

    if [ $MOVED -eq 1 ]; then
        # Wait more and check final position
        sleep 10
        FINAL_X=$(timeout 2 ros2 topic echo /odom --once 2>/dev/null | grep "x:" | head -1 | awk '{print $2}')
        FINAL_DISP=$(python3 -c "print(f'{abs(float($FINAL_X) - float($CUR_X)):.2f}')" 2>/dev/null)
        report "Nav2 goal: robot moved ($FINAL_DISP m)" "PASS"
    else
        report "Nav2 goal: robot moved" "FAIL" "no significant movement in 30s"
    fi
fi

# ==========================================================================
echo ""
echo "======================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
    exit 0
else
    echo "  SOME TESTS FAILED"
    exit 1
fi
