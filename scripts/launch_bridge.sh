#!/bin/bash
# Go2 MuJoCo Bridge ONLY — no nav stack, no TARE, no exploration.
# Dog stands up and publishes ROS2 topics. That's it.
#
# Usage:
#   ./scripts/launch_bridge.sh              # with MuJoCo viewer
#   ./scripts/launch_bridge.sh --no-gui     # headless

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

NO_GUI=""
for arg in "$@"; do
    case $arg in --no-gui) NO_GUI="--no-gui" ;; esac
done

# Repo venv site-packages (uv-managed .venv; falls back to legacy .venv-nano)
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
[ -d "$VENV_SP" ] || VENV_SP="$REPO_DIR/.venv-nano/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"
CONVEX_SRC="/home/yusen/Desktop/go2-convex-mpc/src"
export PYTHONPATH="$VENV_SP:$CMEEL_SP:$CONVEX_SRC:$REPO_DIR:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash

# Clean stale state
rm -f ${VECTOR_RUN_DIR:-/tmp}/vector_nav_active 2>/dev/null

echo "Starting Go2 MuJoCo bridge..."
exec python3 "$SCRIPT_DIR/go2_vnav_bridge.py" $NO_GUI
