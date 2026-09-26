#!/usr/bin/env bash
# E031 self-training (LOCO US -> India), queued into a memory-safe slot.
set -u
cd "$(dirname "$0")/.."
until grep -q PASS1_DONE logs/test_v3c.log 2>/dev/null && grep -q ALL_DONE logs/E030b.log 2>/dev/null; do sleep 30; done
systemd-run --user --scope -q -p MemoryMax=6500M -p MemorySwapMax=0 .venv/bin/python -u -m src.selftrain --tag v3c --source US --target India --rounds 2
echo "EXIT $?"
