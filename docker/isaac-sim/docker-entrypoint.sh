#!/bin/bash
# Docker entrypoint for Vector Isaac Sim 5.1 bridge.
#
# Two-process architecture:
#   1. Isaac Sim (Python 3.11) — physics + rendering, writes state to shared mem
#   2. ROS2 node (Python 3.12) — reads shared state, publishes ROS2 topics
#
# Why: Isaac Sim 5.1 uses Python 3.11 internally, but ROS2 Jazzy needs 3.12.
# The rclpy C extensions are version-locked and cannot cross Python versions.

set -eo pipefail

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

# State exchange directory (tmpfs for speed)
export ISAAC_STATE_DIR="/tmp/isaac_state"
mkdir -p "${ISAAC_STATE_DIR}"

# Start Isaac Sim physics process (Python 3.11) in background
echo "[entrypoint] Starting Isaac Sim physics..."
/isaac-sim/python.sh /vector/bridge/isaac_sim_physics.py "$@" &
ISAAC_PID=$!

# Wait for Isaac Sim to be ready (writes a ready flag)
echo "[entrypoint] Waiting for Isaac Sim to initialize..."
for i in $(seq 1 120); do
    if [ -f "${ISAAC_STATE_DIR}/ready" ]; then
        echo "[entrypoint] Isaac Sim ready after ${i}s"
        break
    fi
    sleep 1
done

if [ ! -f "${ISAAC_STATE_DIR}/ready" ]; then
    echo "[entrypoint] ERROR: Isaac Sim failed to start within 120s"
    kill $ISAAC_PID 2>/dev/null
    exit 1
fi

# Start ROS2 publisher (system Python 3.12)
echo "[entrypoint] Starting ROS2 publisher..."
source /opt/ros/jazzy/setup.bash
python3 /vector/bridge/ros2_publisher.py &
ROS2_PID=$!

echo "[entrypoint] Both processes running (isaac=$ISAAC_PID, ros2=$ROS2_PID)"

# Wait for either to exit
wait -n $ISAAC_PID $ROS2_PID
EXIT_CODE=$?

# Cleanup
kill $ISAAC_PID $ROS2_PID 2>/dev/null
wait
exit $EXIT_CODE
