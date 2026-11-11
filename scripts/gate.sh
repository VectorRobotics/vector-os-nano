#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
#
# The regression gate (internal; the owner's acceptance is bare vector-cli + NL).
#
#   scripts/gate.sh <hermetic|sim|e2e|all> [extra pytest args...]
#
#   hermetic  everything not marked sim/live_llm; no LLM keys
#   sim       tests marked sim (auto-marked by file/fixture in tests/conftest.py); the
#             session sim_preflight skips the tier with ONE reason if this box
#             cannot run MuJoCo (no EGL, missing weights, scene does not compile)
#   e2e       tests/e2e_repl: bare vector-cli on a PTY, one REPL per scenario
#   all       hermetic, then sim, then e2e
#
# Env: GATE_PER_FILE=1  (default) one capped pytest process per test file, so a
#                       crash/abort/OOM in one file cannot take down the run
#                       (GATE_JOBS=2 files at a time for hermetic, 2 x 6G inside the
#                       box's 16G rule; sim/e2e serially). A file that fails to
#                       collect is red, never dropped from the list.
#                       GATE_PER_FILE=0 runs one process: today that aborts in
#                       rclpy (a leaked ROS2-proxy spin thread) and crawls behind
#                       CodeExecutor's runaway `while True` thread (WP03/WP04/WP15).
#      GATE_PATHS="a b" test paths instead of the tier default
#      GATE_OUT=<dir>   junit + logs (default: a fresh temp dir, printed)
#      MEMMAX=16G (FILE_MEMMAX=6G per file in per-file mode)  TMO=<sec>
#      ROS_DOMAIN_ID=<n> (unset or 0 -> random 60-89; 0 is where other stacks live)
#      VECTOR_LIVE_LLM=1 keeps the LLM keys and also runs live_llm tests (billed)
#
# Safe on a shared box: memory-capped cgroup, own ROS domain (random 60-89), a
# throwaway HOME for every tier (model caches read from the real one), and the
# suite fails any test that runs rosm/pkill/killall or leaves a child process.
set -o pipefail
TIER=${1:?usage: scripts/gate.sh <hermetic|sim|e2e|all> [pytest args...]}
shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 2

if [ "$TIER" = all ]; then
  rc=0
  for t in hermetic sim e2e; do "$0" "$t" "$@" || rc=1; done
  exit $rc
fi
LIVE=${VECTOR_LIVE_LLM:-}
case "$TIER" in
  hermetic) PATHS=(tests); MARK="not sim"; IGNORE=(--ignore=tests/e2e_repl) ;;
  sim)      PATHS=(tests); MARK="sim";     IGNORE=(--ignore=tests/e2e_repl) ;;
  e2e)      PATHS=(tests/e2e_repl); MARK=""; IGNORE=() ;;
  *) echo "unknown tier: $TIER (hermetic|sim|e2e|all)" >&2; exit 2 ;;
esac
[ "$LIVE" = 1 ] || MARK="${MARK:+$MARK and }not live_llm"
[ -n "${GATE_PATHS:-}" ] && read -r -a PATHS <<< "$GATE_PATHS"

# The repo venv (a git worktree uses the main checkout's).
VENV=${VECTOR_VENV:-$REPO/.venv}
[ -x "$VENV/bin/python" ] || VENV=$(git -C "$REPO" worktree list --porcelain | awk 'NR==1{print $2}')/.venv
PY=$VENV/bin/python
[ -x "$PY" ] || { echo "[gate] no venv python at $PY" >&2; exit 2; }

# ROS is optional: source it, then put the venv (and its cmeel pinocchio) AHEAD of
# it, else ROS's own NumPy-1 pinocchio is imported and segfaults collection.
for d in /opt/ros/*/setup.bash; do [ -f "$d" ] && { source "$d"; break; }; done
read -r VSP PYV < <("$PY" -c 'import sys, sysconfig; print(sysconfig.get_paths()["purelib"], "python%d.%d" % sys.version_info[:2])')
export PYTHONPATH=$REPO:$VSP/cmeel.prefix/lib/$PYV/site-packages:$VSP${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$VSP/cmeel.prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ "${ROS_DOMAIN_ID:-0}" = 0 ] && ROS_DOMAIN_ID=$((60 + RANDOM % 30))
export ROS_DOMAIN_ID
export MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONDONTWRITEBYTECODE=1
if [ "$LIVE" != 1 ]; then
  for k in $(env | grep -oE '^(DEEPSEEK_|OPENROUTER_|ANTHROPIC_|OPENAI_|VECTOR_LLM_)[A-Za-z0-9_]*'); do unset "$k"; done
  unset VECTOR_PROVIDER
fi
if [ -z "${GATE_HOME_SET:-}" ]; then
  # Every tier: a throwaway HOME, so nothing reads or writes the real ~/.vector (or
  # ~/.vector_os_nano, ~/.cache/vector_os_nano). The model weights stay found, read
  # only: the real HF cache, offline. So do the robot assets scripts/fetch_assets.py
  # provisioned (~/.vector/assets, linked; no test writes it).
  export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 GATE_HOME_SET=1
  GATE_HOME=$(mktemp -d -t vos_gate_home_XXXX) || exit 2
  trap 'rm -rf "$GATE_HOME"' EXIT
  if [ -d "$HOME/.vector/assets" ]; then
    mkdir -p "$GATE_HOME/.vector" && ln -s "$HOME/.vector/assets" "$GATE_HOME/.vector/assets"
  fi
  HOME=$GATE_HOME; export HOME
fi
# Always the repo's pytest config (addopts disable ROS's broken pytest plugins),
# whatever paths are passed: a path outside the repo would otherwise run config-less.
CFG=(-c "$REPO/pyproject.toml" --rootdir "$REPO")
OUT=${GATE_OUT:-$(mktemp -d -t "vos_gate_${TIER}_XXXX")}
mkdir -p "$OUT"

# Run "$@" in its own memory-capped, named scope (MEMMAX), then stop the scope: if
# pytest dies (timeout, OOM), every process the run left behind dies with it.
SCOPES=; command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null && SCOPES=1
capped() {
  [ -n "$SCOPES" ] || { "$@"; return; }
  local unit=vos-gate-$$-$RANDOM rc
  systemd-run --user --scope -q --unit="$unit" -p MemoryMax="${MEMMAX:-16G}" -p MemorySwapMax=0 "$@"
  rc=$?
  systemctl --user stop "$unit.scope" 2>/dev/null
  return $rc
}

if [ "${GATE_PER_FILE-1}" = 1 ]; then
  : > "$OUT/rc.txt"
  capped "$PY" -m pytest "${CFG[@]}" -p no:cacheprovider --co -q "${PATHS[@]}" ${MARK:+-m "$MARK"} "${IGNORE[@]}" "$@" \
    > "$OUT/collect.log" 2>&1
  crc=$?
  grep -E '^[^ ]+\.py::' "$OUT/collect.log" | cut -d: -f1 | sort -u > "$OUT/files.txt"
  # A file that fails to import yields no item line: it is red here, never dropped.
  if [ $crc != 0 ] && [ $crc != 5 ]; then
    grep -E '^ERROR ' "$OUT/collect.log" | sed 's/^/[gate] RED collect: /'
    echo "$crc (collection) $OUT/collect.log" >> "$OUT/rc.txt"
    echo "[gate] RED rc=$crc collection ($OUT/collect.log)"
  fi
  if [ ! -s "$OUT/files.txt" ]; then
    # Nothing to run is never green. A path outside the repo yields no file lines
    # (run those with GATE_PER_FILE=0); a path whose every test is deselected by the
    # tier marker proves nothing either.
    echo "1 (no test files collected) $OUT/collect.log" >> "$OUT/rc.txt"
    echo "[gate] RED: no test files collected for ${PATHS[*]} in tier $TIER (outside the repo? use GATE_PER_FILE=0) ($OUT/collect.log)"
  fi
  JOBS=${GATE_JOBS:-2}; [ "$TIER" = hermetic ] || JOBS=1
  echo "[gate] $TIER per-file: $(wc -l < "$OUT/files.txt") files, $JOBS at a time -> $OUT"
  export TIER OUT MEMMAX=${FILE_MEMMAX:-6G} TMO=${TMO:-1500}
  # shellcheck disable=SC2016
  xargs -P "$JOBS" -I{} bash -c '
    k=$(echo "$1" | tr / _)
    GATE_PER_FILE= GATE_PATHS="$1" GATE_OUT="$OUT/$k" "$0" "$TIER" "${@:2}" > "$OUT/$k.log" 2>&1
    rc=$?; echo "$rc $1" >> "$OUT/rc.txt"
    [ $rc = 0 ] || [ $rc = 5 ] || echo "[gate] RED rc=$rc $1 ($OUT/$k.log)"
  ' "$0" {} "$@" < "$OUT/files.txt"
  red=$(awk '$1 != 0 && $1 != 5' "$OUT/rc.txt" | wc -l)
  echo "[gate] $TIER per-file: $red of $(wc -l < "$OUT/rc.txt") runs red (one per file, plus collection if it failed) ($OUT/rc.txt)"
  [ "$red" = 0 ]
  exit
fi

echo "[gate] $TIER: ${PATHS[*]} -m '$MARK' -> $OUT/junit.xml (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
capped timeout "${TMO:-7200}" "$PY" -m pytest "${CFG[@]}" -p no:cacheprovider --timeout=300 -q -rfE \
  -o junit_family=xunit2 --junitxml="$OUT/junit.xml" \
  "${PATHS[@]}" ${MARK:+-m "$MARK"} "${IGNORE[@]}" "$@" 2>&1 | tee "$OUT/pytest.log"
exit "${PIPESTATUS[0]}"
