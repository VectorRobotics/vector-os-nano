#!/bin/bash
# Stop the Isaac Sim Docker container for Vector OS Nano.
#
# Usage:
#   ./scripts/stop_isaac.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker/isaac-sim/docker-compose.yaml"

echo "[stop_isaac] Stopping Isaac Sim container..."
docker compose -f "${COMPOSE_FILE}" down

echo "[stop_isaac] Done."
