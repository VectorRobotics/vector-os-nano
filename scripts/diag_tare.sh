#!/bin/bash
# TARE data flow diagnostic — run while launch_explore.sh is active
# Shows whether each layer of the planner stack is alive and healthy

source /opt/ros/jazzy/setup.bash
source ~/Desktop/vector_navigation_stack/install/setup.bash

echo "=========================================="
echo "  TARE Planner Diagnostic"
echo "=========================================="
echo ""

echo "--- Layer 0: Bridge → Nav Stack ---"
echo -n "  /state_estimation:      "
timeout 3 ros2 topic hz /state_estimation --window 10 2>&1 | grep "average" | head -1 || echo "NO DATA"
echo -n "  /registered_scan:       "
timeout 3 ros2 topic hz /registered_scan --window 5 2>&1 | grep "average" | head -1 || echo "NO DATA"

echo ""
echo "--- Layer 1: sensorScanGen → TARE ---"
echo -n "  /state_estimation_at_scan: "
timeout 3 ros2 topic hz /state_estimation_at_scan --window 5 2>&1 | grep "average" | head -1 || echo "NO DATA (CRITICAL!)"
echo -n "  /terrain_map:           "
timeout 3 ros2 topic hz /terrain_map --window 5 2>&1 | grep "average" | head -1 || echo "NO DATA"
echo -n "  /terrain_map_ext:       "
timeout 3 ros2 topic hz /terrain_map_ext --window 5 2>&1 | grep "average" | head -1 || echo "NO DATA"

echo ""
echo "--- Layer 2: FAR Planner ---"
echo -n "  /navigation_boundary:   "
timeout 3 ros2 topic hz /navigation_boundary --window 3 2>&1 | grep "average" | head -1 || echo "NO DATA"

echo ""
echo "--- Layer 3: TARE → Movement ---"
echo -n "  /way_point:             "
timeout 3 ros2 topic hz /way_point --window 5 2>&1 | grep "average" | head -1 || echo "NO DATA (TARE not publishing!)"
echo ""
echo "  Latest waypoint:"
timeout 2 ros2 topic echo /way_point --once 2>&1 | grep -A3 "point:" || echo "  (none)"

echo ""
echo "--- Layer 1b: localPlanner → Bridge ---"
echo -n "  /path:                  "
timeout 3 ros2 topic hz /path --window 5 2>&1 | grep "average" | head -1 || echo "NO DATA"

echo ""
echo "--- Nav Flag ---"
if [ -f ${VECTOR_RUN_DIR:-/tmp}/vector_nav_active ]; then
    echo "  ${VECTOR_RUN_DIR:-/tmp}/vector_nav_active: EXISTS (nav ON)"
else
    echo "  ${VECTOR_RUN_DIR:-/tmp}/vector_nav_active: MISSING (nav OFF)"
fi

echo ""
echo "--- Robot Position ---"
timeout 2 ros2 topic echo /state_estimation --once 2>&1 | grep -A4 "position:" | head -5

echo ""
echo "=========================================="
