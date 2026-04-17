#!/bin/bash
# Launch Isaac Sim Docker container for Vector OS Nano.
#
# Usage:
#   ./scripts/launch_isaac.sh [--gui] [--scene flat|room|apartment]
#
# Options:
#   --gui               Enable the Isaac Sim viewport window (requires X11)
#   --scene <name>      Scene to load: flat (default), room, or apartment
#
# Environment overrides (set before calling this script):
#   ISAAC_PHYSICS_HZ    Physics step rate in Hz (default: 200)
#   OMNI_SERVER         Nucleus server URL (e.g. omniverse://localhost)
#   VECTOR_LOG_DIR      Host path for Isaac Sim logs (default: /tmp/vector_isaac_logs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker/isaac-sim/docker-compose.yaml"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ISAAC_HEADLESS="true"
ISAAC_SCENE="flat"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gui)
            ISAAC_HEADLESS="false"
            shift
            ;;
        --scene)
            if [[ -z "${2-}" ]]; then
                echo "ERROR: --scene requires an argument (flat|room|apartment)" >&2
                exit 1
            fi
            ISAAC_SCENE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Validate scene name
case "${ISAAC_SCENE}" in
    flat|room|apartment) ;;
    *)
        echo "ERROR: Invalid scene '${ISAAC_SCENE}'. Choose flat, room, or apartment." >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "[launch_isaac] Pre-flight checks..."

# Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker Engine first." >&2
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running. Start it with: sudo systemctl start docker" >&2
    exit 1
fi

# docker compose (v2 plugin)
if ! docker compose version &>/dev/null; then
    echo "ERROR: docker compose v2 not found. Install the Docker Compose plugin." >&2
    exit 1
fi

# NVIDIA GPU + container toolkit
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers." >&2
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-container-toolkit not configured or GPU not accessible." >&2
    echo "       Install with: sudo apt install nvidia-container-toolkit" >&2
    echo "       Then restart Docker: sudo systemctl restart docker" >&2
    exit 1
fi

# X11 display check (only needed for GUI mode)
if [[ "${ISAAC_HEADLESS}" == "false" ]]; then
    if [[ -z "${DISPLAY-}" ]]; then
        echo "ERROR: --gui mode requires DISPLAY to be set (X11 session)." >&2
        exit 1
    fi
    if ! xhost &>/dev/null; then
        echo "WARNING: xhost command not found; X11 forwarding may not work." >&2
    else
        # Allow Docker root user to connect to the local X display
        xhost +local:root &>/dev/null || true
    fi
fi

# Log directory
VECTOR_LOG_DIR="${VECTOR_LOG_DIR:-/tmp/vector_isaac_logs}"
mkdir -p "${VECTOR_LOG_DIR}"

# ---------------------------------------------------------------------------
# Build image if not present or stale
# ---------------------------------------------------------------------------
IMAGE_TAG="vector-isaac-sim:latest"

if ! docker image inspect "${IMAGE_TAG}" &>/dev/null; then
    echo "[launch_isaac] Image '${IMAGE_TAG}' not found. Building..."
    docker build \
        -t "${IMAGE_TAG}" \
        "${SCRIPT_DIR}/docker/isaac-sim/" \
        --progress=plain
else
    echo "[launch_isaac] Using existing image '${IMAGE_TAG}'. (Re-build with: docker build -t ${IMAGE_TAG} docker/isaac-sim/)"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
export ISAAC_HEADLESS
export ISAAC_SCENE
export VECTOR_LOG_DIR
export DISPLAY="${DISPLAY:-:0}"

echo ""
echo "[launch_isaac] Starting Isaac Sim container..."
echo "  Scene   : ${ISAAC_SCENE}"
echo "  Headless: ${ISAAC_HEADLESS}"
echo "  Logs    : ${VECTOR_LOG_DIR}"
echo ""

docker compose -f "${COMPOSE_FILE}" up -d

# ---------------------------------------------------------------------------
# Wait for health check (max 3 minutes — shader compilation is slow)
# ---------------------------------------------------------------------------
CONTAINER="vector-isaac-sim"
MAX_WAIT=180
INTERVAL=10
elapsed=0

echo "[launch_isaac] Waiting for bridge to become healthy (max ${MAX_WAIT}s)..."
echo "  Isaac Sim shader compilation can take 2-3 minutes on first launch."

while true; do
    status="$(docker inspect --format='{{.State.Health.Status}}' "${CONTAINER}" 2>/dev/null || echo "missing")"

    case "${status}" in
        healthy)
            echo "[launch_isaac] Container is healthy."
            break
            ;;
        unhealthy)
            echo "ERROR: Container reported unhealthy. Check logs:" >&2
            docker logs --tail 50 "${CONTAINER}" >&2
            exit 1
            ;;
        missing)
            echo "ERROR: Container '${CONTAINER}' not found." >&2
            exit 1
            ;;
        *)
            # starting or none (no health check result yet)
            if [[ ${elapsed} -ge ${MAX_WAIT} ]]; then
                echo "ERROR: Timed out waiting for health check after ${MAX_WAIT}s." >&2
                echo "       Container logs:" >&2
                docker logs --tail 80 "${CONTAINER}" >&2
                exit 1
            fi
            echo "  [${elapsed}s/${MAX_WAIT}s] Status: ${status} — waiting..."
            sleep ${INTERVAL}
            elapsed=$((elapsed + INTERVAL))
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Print connection info
# ---------------------------------------------------------------------------
echo ""
echo "[launch_isaac] Isaac Sim bridge is running."
echo ""
echo "  ROS2 topics (on host):"
echo "    ros2 topic list"
echo "    ros2 topic hz /state_estimation"
echo "    ros2 topic hz /registered_scan"
echo ""
echo "  Container logs:"
echo "    docker logs -f ${CONTAINER}"
echo ""
echo "  Stop the simulation:"
echo "    ./scripts/stop_isaac.sh"
echo ""
