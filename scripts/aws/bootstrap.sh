#!/usr/bin/env bash
# Run ON the instance (Ubuntu 24.04, Python 3.12) from the project directory.
# EMBED=1 also installs the dense-channel stack (CPU torch + sentence-transformers + faiss).
set -eu
cd "$(dirname "$0")/../.."
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-dev build-essential zstd tmux htop >/dev/null
sudo loginctl enable-linger "$USER" || true          # user systemd survives logout (memory-capped scopes)
python3 -m venv .venv
.venv/bin/pip install -q -U pip wheel
.venv/bin/pip install -q -r requirements.txt
if [ "${EMBED:-0}" = 1 ]; then
  .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
  .venv/bin/pip install -q sentence-transformers==3.3.1 faiss-cpu==1.9.0
fi
nproc; free -g | head -2
systemd-run --user --scope -q -p MemoryMax=1G true && echo "memory-capped scopes: OK" || echo "no user systemd: run with DAG_NO_SYSTEMD=1"
.venv/bin/python -m pytest -q tests/ 2>&1 | tail -2
