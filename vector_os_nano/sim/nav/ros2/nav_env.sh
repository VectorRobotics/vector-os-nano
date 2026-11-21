# Shared environment for the nav launch scripts — SOURCED, not executed.
# The CMU nav stack (TARE/FAR/localPlanner/terrainAnalysis) is an external ROS2
# workspace; point VECTOR_NAV_STACK at it (the dir holding install/setup.bash),
# e.g. in the repo-root .env. It is treated read-only: configs are passed with
# --params-file, never copied into its install tree (it may be shared).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -z "${VECTOR_NAV_STACK:-}" ] || [ ! -f "$VECTOR_NAV_STACK/install/setup.bash" ]; then
    echo "[nav_env] VECTOR_NAV_STACK='${VECTOR_NAV_STACK:-}' is unset or has no install/setup.bash" >&2
    echo "[nav_env] set it to the built CMU nav stack workspace (see docs/getting-started.md)" >&2
    exit 2
fi
NAV_STACK="$VECTOR_NAV_STACK"
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
CMEEL_SP="$VENV_SP/cmeel.prefix/lib/python3.12/site-packages"

# Stop ONLY the process groups this script started (each background job is its
# own group under `set -m`). Never pkill by name or wipe /dev/shm: other ROS
# stacks on this machine share those names and FastDDS segments.
nav_stop_groups() {
    for p in "$@"; do kill -TERM -- -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done
    sleep 2
    for p in "$@"; do kill -KILL -- -"$p" 2>/dev/null || true; done
}
