#!/bin/bash
# Test: verify TARE receives data and publishes close waypoints.
# Run AFTER 'launch go2 sim' starts the nav stack.
#
# Usage:
#   # Terminal 1: source .venv-nano/bin/activate && vector-cli → launch go2 sim
#   # Terminal 2: ./scripts/test_tare_dataflow.sh
#
# This script:
# 1. Checks all required topics
# 2. Starts TARE manually (same way as launch_explore.sh)
# 3. Seeds with movement
# 4. Monitors /way_point for 30 seconds
# 5. Reports waypoint distances

set -e

NAV_STACK="/home/yusen/Desktop/vector_navigation_stack"
REPO="/home/yusen/Desktop/vector_os_nano"

source /opt/ros/jazzy/setup.bash
source "$NAV_STACK/install/setup.bash"

echo "=== TARE Data Flow Test ==="
echo ""

# Step 1: Check critical topics
echo "Step 1: Topic check"
for topic in /state_estimation /registered_scan /terrain_map /path /state_estimation_at_scan; do
    count=$(ros2 topic info "$topic" 2>/dev/null | grep -c "Publisher count:" || echo "0")
    pubs=$(ros2 topic info "$topic" 2>/dev/null | grep "Publisher count:" | awk '{print $3}' || echo "0")
    if [ "$pubs" -gt 0 ] 2>/dev/null; then
        echo "  OK    $topic ($pubs publishers)"
    else
        echo "  FAIL  $topic (no publishers!)"
    fi
done

echo ""

# Step 2: Check /state_estimation_at_scan frequency
echo "Step 2: /state_estimation_at_scan frequency (5s sample)"
hz_output=$(timeout 6 ros2 topic hz /state_estimation_at_scan --window 50 2>&1 | tail -1 || echo "no data")
echo "  $hz_output"

echo ""

# Step 3: Deploy TARE config and start TARE
echo "Step 3: Starting TARE (same as launch_explore.sh)"
# Deploy our config
cp "$REPO/config/tare_go2_indoor.yaml" \
   "$NAV_STACK/install/tare_planner/share/tare_planner/indoor_small.yaml" 2>/dev/null

# Kill any existing TARE
pkill -f tare_planner_node 2>/dev/null || true
sleep 1

# Start TARE
ros2 launch tare_planner explore.launch scenario:=indoor_small &
TARE_PID=$!
echo "  TARE started (PID=$TARE_PID)"
sleep 3

echo ""

# Step 4: Seed movement
echo "Step 4: Seeding planners..."
# Create nav flag so bridge follows paths
touch ${VECTOR_RUN_DIR:-/tmp}/vector_nav_active
for i in $(seq 1 4); do
    ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{linear: {x: 0.2}}" 2>/dev/null
    sleep 1
done
ros2 topic pub --once /cmd_vel_nav geometry_msgs/msg/Twist "{}" 2>/dev/null
sleep 2

echo ""

# Step 5: Monitor /way_point
echo "Step 5: Monitoring /way_point for 20s..."
echo "  (Expecting waypoints within 5m of robot, updating every few seconds)"
echo ""
timeout 20 ros2 topic echo /way_point --no-arr --field header.stamp --field point 2>/dev/null &
ECHO_PID=$!

# Also check /way_point frequency
sleep 20
kill $ECHO_PID 2>/dev/null || true

echo ""
echo "Step 6: TARE /way_point frequency"
hz_wp=$(timeout 6 ros2 topic hz /way_point --window 10 2>&1 | tail -1 || echo "no data")
echo "  $hz_wp"

echo ""

# Cleanup
kill $TARE_PID 2>/dev/null || true
rm -f ${VECTOR_RUN_DIR:-/tmp}/vector_nav_active

echo "=== Test Complete ==="
