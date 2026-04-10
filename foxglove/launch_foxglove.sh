#!/usr/bin/env bash
# Vector OS — Foxglove Bridge 启动脚本
# 启动 foxglove_bridge，然后用浏览器连接 Foxglove Studio
#
# Usage: ./foxglove/launch_foxglove.sh
# 需要: ros-jazzy-foxglove-bridge, ROS2 topics 已在发布

set -e

PORT="${FOXGLOVE_PORT:-8765}"

# Colors
TEAL='\033[36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}${TEAL}Vector OS — Foxglove Bridge${RESET}"
echo ""

# Check foxglove_bridge
if ! ros2 pkg list 2>/dev/null | grep -q foxglove_bridge; then
    echo "ERROR: foxglove_bridge not found."
    echo "Install: sudo apt install ros-jazzy-foxglove-bridge"
    exit 1
fi

# Source ROS2 if needed
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Cleanup on exit
cleanup() {
    echo -e "\n${DIM}Shutting down foxglove_bridge...${RESET}"
    kill "$BRIDGE_PID" 2>/dev/null
    wait "$BRIDGE_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# Start foxglove_bridge
echo -e "${DIM}Starting foxglove_bridge on port ${PORT}...${RESET}"
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="$PORT" &
BRIDGE_PID=$!

sleep 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${TEAL}Foxglove Bridge ready${RESET} on ws://localhost:${PORT}"
echo ""
echo -e "  ${BOLD}连接方式:${RESET}"
echo -e "  1. 打开 ${TEAL}https://app.foxglove.dev${RESET}"
echo -e "  2. Open connection → Foxglove WebSocket → ${TEAL}ws://localhost:${PORT}${RESET}"
echo -e "  3. 导入 dashboard: Layout menu → Import → ${DIM}${SCRIPT_DIR}/vector-os-dashboard.json${RESET}"
echo ""
echo -e "${DIM}Press Ctrl+C to stop${RESET}"

wait "$BRIDGE_PID"
