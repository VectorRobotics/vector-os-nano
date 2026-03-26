#!/bin/bash
# Run Vector OS Nano brain test against Nav2 in Gazebo
#
# Prerequisites:
#   Terminal 1: ros2 launch vector_go2_gazebo full_stack.launch.py mode:=nav
#   Terminal 2: ./scripts/run_nav2_brain.sh [--rooms|--single X Y]

set -e
export PATH=/usr/bin:/usr/local/bin:$PATH
export PYTHONPATH=""

source /opt/ros/humble/setup.bash
source ~/Desktop/vector_go2_sim/install/setup.bash 2>/dev/null || true

NANO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$NANO_ROOT:$NANO_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"

python3 "$NANO_ROOT/scripts/test_nav2_brain.py" "$@"
