#!/bin/bash
cd "$(dirname "$0")"
# Resolve the repo venv (.venv preferred, legacy .venv-nano fallback).
# Never fall back to system python silently — that is exactly the
# wrong-interpreter trap documented in docs/tricky-bugs.md Case 3.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv-nano/bin/activate ]; then
    source .venv-nano/bin/activate
else
    echo "sim.sh: no .venv or .venv-nano found — refusing to run on system python." >&2
    echo "Create one with: uv venv .venv && uv pip install -e '.[all]'" >&2
    exit 1
fi
exec python3 -m vector_os_nano.vcli.cli "$@"
