#!/usr/bin/env bash
# Recursive Estimation 2026: Linux / macOS launcher.
#
# Usage (from inside the exercise/ folder):
#     bash run.sh         run simulation, open plots as native windows
#
# Requires:
#   - Docker  : https://docs.docker.com/get-docker/

set -euo pipefail

IMAGE=re-student
VENV=".venv"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed. Get it at https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo ">>> Building Docker image (one time setup)..."
    docker build -t "$IMAGE" .
fi

COMMON=(--rm --cpus=4 --memory=2g --memory-swap=2g -v "$DIR":/work -w /work)

if [[ "${1:-}" == "headless" ]]; then
    echo ">>> Running headless..."
    docker run "${COMMON[@]}" "$IMAGE" python run.py --headless
    exit 0
fi

echo "Starting simulation environment (4 CPU cores, 2 GB RAM)..."
docker run "${COMMON[@]}" "$IMAGE"

if [[ ! -f "$VENV/bin/python" ]]; then
    echo "Setting up Python environment (one time setup)..."
    python3 -m venv "$VENV" 2>/dev/null || python -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import matplotlib, numpy" 2>/dev/null; then
    echo "Installing dependencies..."
    "$VENV/bin/pip" install --quiet --upgrade pip matplotlib numpy
fi

echo "Opening plots..."
"$VENV/bin/python" "$DIR/plot_results.py"
