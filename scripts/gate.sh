#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
#
# The regression gate (internal; the owner's acceptance is bare vector-cli + NL).
#
#   scripts/gate.sh <hermetic|sim|e2e|all> [extra pytest args...]
#
#   hermetic  everything not marked sim/live_llm; throwaway HOME, no LLM keys
#   sim       tests marked sim (auto-marked by file/fixture in tests/conftest.py); the
#             session sim_preflight skips the tier with ONE reason if this box
#             cannot run MuJoCo (no EGL, missing weights, scene does not compile)
#   e2e       tests/e2e_repl: bare vector-cli on a PTY, one REPL per scenario
#   all       hermetic, then sim, then e2e
#
# Env: GATE_PER_FILE=1  one capped pytest process per test file, so a crash/abort/
#                       OOM in one file cannot take down the run (GATE_JOBS=4 files
#                       at a time for hermetic; sim/e2e run serially)
#      GATE_PATHS="a b" test paths instead of the tier default
#      GATE_OUT=<dir>   junit + logs (default: a fresh temp dir, printed)
#      MEMMAX=16G (6G per file in per-file mode)  TMO=<sec>  ROS_DOMAIN_ID=<n>
#      VECTOR_LIVE_LLM=1 keeps the LLM keys and also runs live_llm tests (billed)
#
# Safe on a shared box: memory-capped cgroup, own ROS domain (random 60-89), and the
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
VSP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPATH=$REPO:$VSP/cmeel.prefix/lib/python3.12/site-packages:$VSP${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$VSP/cmeel.prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-$((60 + RANDOM % 30))}
export MUJOCO_GL=${MUJOCO_GL:-egl} PYTHONDONTWRITEBYTECODE=1
if [ "$LIVE" != 1 ]; then
  for k in $(env | grep -oE '^(DEEPSEEK_|OPENROUTER_|ANTHROPIC_|OPENAI_|VECTOR_LLM_)[A-Za-z0-9_]*'); do unset "$k"; done
  unset VECTOR_PROVIDER
fi
if [ "$TIER" = hermetic ] && [ -z "${GATE_HOME_SET:-}" ]; then
  # A throwaway HOME: nothing reads or writes the real ~/.vector; HF cache kept.
  export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} GATE_HOME_SET=1
  HOME=$(mktemp -d -t vos_gate_home_XXXX); export HOME
fi
OUT=${GATE_OUT:-$(mktemp -d -t "vos_gate_${TIER}_XXXX")}
mkdir -p "$OUT"

if [ "${GATE_PER_FILE:-}" = 1 ]; then
  "$PY" -m pytest -p no:cacheprovider --co -q "${PATHS[@]}" ${MARK:+-m "$MARK"} "${IGNORE[@]}" "$@" 2>/dev/null \
    | grep '::' | cut -d: -f1 | sort -u > "$OUT/files.txt"
  JOBS=${GATE_JOBS:-4}; [ "$TIER" = hermetic ] || JOBS=1
  echo "[gate] $TIER per-file: $(wc -l < "$OUT/files.txt") files, $JOBS at a time -> $OUT"
  : > "$OUT/rc.txt"
  export TIER OUT MEMMAX=${FILE_MEMMAX:-6G} TMO=${TMO:-1500}
  # shellcheck disable=SC2016
  xargs -P "$JOBS" -I{} bash -c '
    k=$(echo "$1" | tr / _)
    GATE_PER_FILE= GATE_PATHS="$1" GATE_OUT="$OUT/$k" "$0" "$TIER" "${@:2}" > "$OUT/$k.log" 2>&1
    rc=$?; echo "$rc $1" >> "$OUT/rc.txt"
    [ $rc = 0 ] || [ $rc = 5 ] || echo "[gate] RED rc=$rc $1 ($OUT/$k.log)"
  ' "$0" {} "$@" < "$OUT/files.txt"
  red=$(awk '$1 != 0 && $1 != 5' "$OUT/rc.txt" | wc -l)
  echo "[gate] $TIER per-file: $red of $(wc -l < "$OUT/rc.txt") files red ($OUT/rc.txt)"
  [ "$red" = 0 ]
  exit
fi

CAP=(); UNIT=
if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
  # A named scope: if pytest dies (timeout, OOM), stopping the scope still kills
  # every process the run left behind, wherever it re-parented.
  UNIT=vos-gate-$$-$RANDOM
  CAP=(systemd-run --user --scope -q --unit="$UNIT" -p MemoryMax="${MEMMAX:-16G}" -p MemorySwapMax=0)
fi
echo "[gate] $TIER: ${PATHS[*]} -m '$MARK' -> $OUT/junit.xml (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
"${CAP[@]}" timeout "${TMO:-7200}" "$PY" -m pytest -p no:cacheprovider --timeout=300 -q -rfE \
  -o junit_family=xunit2 --junitxml="$OUT/junit.xml" \
  "${PATHS[@]}" ${MARK:+-m "$MARK"} "${IGNORE[@]}" "$@" 2>&1 | tee "$OUT/pytest.log"
rc=${PIPESTATUS[0]}
[ -n "$UNIT" ] && systemctl --user stop "$UNIT.scope" 2>/dev/null
exit "$rc"
