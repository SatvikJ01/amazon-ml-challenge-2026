#!/usr/bin/env bash
# Run ON the instance: the whole v4 job graph inside tmux (survives ssh disconnects).
# Usage: scripts/aws/run_v4.sh "--channels name,caddr --entities 1000000 --stage4"
set -eu
cd "$(dirname "$0")/../.."
ARGS=${1:-"--channels name,caddr --entities 1000000 --stage4"}
MEM=$(free -g | awk '/Mem:/{print int($2*0.9)}'); CPUS=$(nproc)
tmux new-session -d -s v4 ".venv/bin/python -u -m src.v4_dag --run v4 --mem $MEM --cpus $CPUS $ARGS 2>&1 | tee -a logs/v4.log"
echo "started: tmux attach -t v4   |   tail -f logs/dag/v4/_dag.log"
