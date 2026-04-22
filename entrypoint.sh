#!/bin/bash
# Runs run.py once and exits. Results are saved to results.npz in /work
# (volume-mounted from the host) for host-side plotting by plot_results.py.
set -e
cd /work
exec python run.py "$@"
