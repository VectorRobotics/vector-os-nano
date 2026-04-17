#!/bin/bash
# Stop all Gazebo Harmonic processes for Vector OS Nano.
#
# Usage:
#   ./scripts/stop_gazebo.sh

set -euo pipefail

echo "[stop_gazebo] Stopping Gazebo processes..."

# gz sim (main simulator)
if pkill -f "gz sim" 2>/dev/null; then
    echo "  [ok] gz sim stopped"
else
    echo "  [--] gz sim was not running"
fi

# ros_gz_bridge
if pkill -f "ros_gz_bridge" 2>/dev/null; then
    echo "  [ok] ros_gz_bridge stopped"
else
    echo "  [--] ros_gz_bridge was not running"
fi

# controller_manager
if pkill -f "controller_manager" 2>/dev/null; then
    echo "  [ok] controller_manager stopped"
else
    echo "  [--] controller_manager was not running"
fi

# robot_state_publisher
if pkill -f "robot_state_publisher" 2>/dev/null; then
    echo "  [ok] robot_state_publisher stopped"
else
    echo "  [--] robot_state_publisher was not running"
fi

# Cleanup shared state marker
if [[ -f /tmp/vector_nav_active ]]; then
    rm -f /tmp/vector_nav_active
    echo "  [ok] removed /tmp/vector_nav_active"
fi

echo "[stop_gazebo] Done."
