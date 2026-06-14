#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Launch the SysNav perception pair (detection + semantic mapping) against
# whatever publishes /camera/image + /registered_scan + /state_estimation
# (e.g. vector-cli --scenario apartment with VECTOR_HABITAT_SYSNAV=1).
#
# Prereqs (one-time, already provisioned on this box):
#   - ~/Desktop/SysNav built (tare_planner + semantic_mapping, --symlink-install)
#   - ~/Desktop/SysNav/.venv-sysnav (numpy 1.26 stack, SAM2/YOLOE weights + engines)
#
# Usage:  source /opt/ros/jazzy/setup.bash && source ~/Desktop/SysNav/install/setup.bash
#         ./scripts/launch_sysnav_nodes.sh        # Ctrl-C stops both nodes
set -u
SYSNAV="${VECTOR_SYSNAV_WS:-$HOME/Desktop/SysNav}"
PY="$SYSNAV/.venv-sysnav/bin/python"
PARAMS="$SYSNAV/install/semantic_mapping/share/semantic_mapping/mapping_mecanum_sim.yaml"
export PYTHONPATH="$SYSNAV/src/semantic_mapping:$SYSNAV/src/semantic_mapping/semantic_mapping/external/sam2:${PYTHONPATH:-}"

if [ ! -x "$PY" ]; then
    echo "FATAL: $PY missing — the SysNav venv is not provisioned" >&2
    exit 1
fi

cd "$SYSNAV"   # nodes use workspace-relative checkpoint/config paths
echo "[sysnav] starting detection_node + semantic_mapping_node (Ctrl-C stops both)"
"$PY" -c 'from semantic_mapping.detection_node import main; main()' \
      --ros-args --params-file "$PARAMS" &
DET=$!
"$PY" -c 'from semantic_mapping.semantic_mapping_node import main; main()' \
      --ros-args --params-file "$PARAMS" &
MAP=$!
trap 'kill $DET $MAP 2>/dev/null; wait' INT TERM
wait