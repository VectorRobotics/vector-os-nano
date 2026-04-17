#!/bin/bash
cd "$(dirname "$0")"
# Use .venv if it exists, otherwise system python3
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
exec python3 -m vector_os_nano.vcli.cli "$@"
