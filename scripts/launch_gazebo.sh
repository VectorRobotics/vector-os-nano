#!/bin/bash
# Launch Gazebo Harmonic with Go2 quadruped.
#
# Usage:
#   ./scripts/launch_gazebo.sh [--world WORLD] [--headless] [--controller CTRL]
#
# Options:
#   --world WORLD       World to load: apartment (default), flat, outdoor
#   --headless          Run without GUI (no viewport window)
#   --controller CTRL   Controller to use: guide (default) or rl
#
# Environment overrides:
#   VECTOR_LOG_DIR      Host path for logs (default: /tmp/vector_gazebo_logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLDS_DIR="${SCRIPT_DIR}/gazebo/worlds"
LAUNCH_FILE="${SCRIPT_DIR}/gazebo/launch/go2_sim.launch.py"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WORLD="apartment"
GUI="true"
CONTROLLER="guide"

# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------
usage() {
    sed -n '2,14p' "$0"
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --world)
            if [[ -z "${2-}" ]]; then
                echo "ERROR: --world requires an argument (apartment|flat|outdoor)" >&2
                exit 1
            fi
            WORLD="$2"
            shift 2
            ;;
        --headless)
            GUI="false"
            shift
            ;;
        --controller)
            if [[ -z "${2-}" ]]; then
                echo "ERROR: --controller requires an argument (guide|rl)" >&2
                exit 1
            fi
            CONTROLLER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "       Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# Validate controller
case "${CONTROLLER}" in
    guide|rl) ;;
    *)
        echo "ERROR: Invalid controller '${CONTROLLER}'. Choose guide or rl." >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Source quadruped_ros2_control (Go2 URDF + controllers)
# Temporarily disable nounset — colcon setup.bash uses unbound variables
# ---------------------------------------------------------------------------
QUADRUPED_DIR="${HOME}/Desktop/quadruped_ros2_control"
if [[ -f "${QUADRUPED_DIR}/install/setup.bash" ]]; then
    set +u
    source "${QUADRUPED_DIR}/install/setup.bash"
    set -u
else
    echo "ERROR: quadruped_ros2_control not built." >&2
    echo "       cd ~/Desktop/quadruped_ros2_control && colcon build --packages-up-to unitree_guide_controller --symlink-install" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[launch_gazebo] Pre-flight checks..."

# 1. gz sim available
if ! gz sim --version &>/dev/null; then
    echo "ERROR: gz sim not found. Install Gazebo Harmonic:" >&2
    echo "       https://gazebosim.org/docs/harmonic/install_ubuntu" >&2
    exit 1
fi
echo "  [ok] gz sim: $(gz sim --version 2>&1 | head -1)"

# 2. ROS2 is sourced
if ! command -v ros2 &>/dev/null; then
    echo "ERROR: ros2 not found. Source your ROS2 installation:" >&2
    echo "       source /opt/ros/jazzy/setup.bash" >&2
    exit 1
fi
if ! ros2 topic list &>/dev/null 2>&1; then
    echo "WARNING: ros2 topic list failed — ROS2 daemon may not be running yet." >&2
fi
echo "  [ok] ros2 found: $(command -v ros2)"

# 3. World SDF file exists
WORLD_SDF="${WORLDS_DIR}/${WORLD}.sdf"
if [[ ! -f "${WORLD_SDF}" ]]; then
    echo "ERROR: World file not found: ${WORLD_SDF}" >&2
    echo "       Available worlds: $(ls "${WORLDS_DIR}"/*.sdf 2>/dev/null | xargs -I{} basename {} .sdf | tr '\n' ' ' || echo 'none')" >&2
    exit 1
fi
echo "  [ok] world SDF: ${WORLD_SDF}"

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
VECTOR_LOG_DIR="${VECTOR_LOG_DIR:-/tmp/vector_gazebo_logs}"
mkdir -p "${VECTOR_LOG_DIR}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo ""
echo "[launch_gazebo] Starting Gazebo Harmonic..."
echo "  World     : ${WORLD}"
echo "  GUI       : ${GUI}"
echo "  Controller: ${CONTROLLER}"
echo "  Logs      : ${VECTOR_LOG_DIR}"
echo ""

ros2 launch "${LAUNCH_FILE}" \
    world:="${WORLD}" \
    gui:="${GUI}" \
    controller:="${CONTROLLER}" \
    &

LAUNCH_PID=$!

# ---------------------------------------------------------------------------
# Wait for /state_estimation topic (max 30s)
# ---------------------------------------------------------------------------
MAX_WAIT=30
INTERVAL=2
elapsed=0

echo "[launch_gazebo] Waiting for /state_estimation topic (max ${MAX_WAIT}s)..."

while true; do
    if ros2 topic list 2>/dev/null | grep -q "/state_estimation"; then
        echo "[launch_gazebo] /state_estimation topic is active."
        break
    fi

    if [[ ${elapsed} -ge ${MAX_WAIT} ]]; then
        echo "WARNING: /state_estimation not detected after ${MAX_WAIT}s." >&2
        echo "         Simulation may still be starting up." >&2
        break
    fi

    sleep ${INTERVAL}
    elapsed=$((elapsed + INTERVAL))
    echo "  [${elapsed}s/${MAX_WAIT}s] waiting for topics..."
done

# ---------------------------------------------------------------------------
# Print active topics and connection info
# ---------------------------------------------------------------------------
echo ""
echo "[launch_gazebo] Active ROS2 topics:"
ros2 topic list 2>/dev/null | head -20 || echo "  (ros2 topic list unavailable)"
echo ""
echo "  Monitor topics:"
echo "    ros2 topic hz /state_estimation"
echo "    ros2 topic hz /scan"
echo ""
echo "  Stop the simulation:"
echo "    ./scripts/stop_gazebo.sh"
echo ""

wait ${LAUNCH_PID}
