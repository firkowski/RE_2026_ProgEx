#!/usr/bin/env bash
# Recursive Estimation 2026: Linux launcher (local virtual environment).
#
# Usage:
#     bash run.sh            run simulation, open plots as native windows
#     bash run.sh headless   run simulation, save plots as PNG files
#
# Requires Python 3.10 (the obfuscated simulator is built for it), either as
# `python3.10` on the PATH or via uv (https://docs.astral.sh/uv/).

set -euo pipefail

VENV=".venv"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Setting up Python 3.10 environment (one time setup)..."
    if command -v python3.10 >/dev/null 2>&1; then
        python3.10 -m venv "$VENV"
    elif command -v uv >/dev/null 2>&1; then
        uv venv --quiet --seed -p 3.10 "$VENV"
    else
        echo "ERROR: Python 3.10 not found. Install python3.10 or uv."
        exit 1
    fi
fi

if ! "$VENV/bin/python" -c "import matplotlib, numpy" 2>/dev/null; then
    echo "Installing dependencies..."
    "$VENV/bin/python" -m pip install --quiet -r requirements.txt
fi

if [[ "${1:-}" == "headless" ]]; then
    echo ">>> Running headless..."
    exec "$VENV/bin/python" run.py --headless
fi

echo "Running simulation..."
"$VENV/bin/python" run.py

echo "Opening plots..."
exec "$VENV/bin/python" plot_results.py
