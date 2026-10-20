#!/usr/bin/env bash
# Usage: 11_run_matrix.sh [-f ids.txt] [run_id ...] [-- extra args for 10_run_one.py, e.g. --override harmony.alpha=1.0]
# Skips ids already ok in results/runs.parquet; one fresh process per run under the GPU lock.
set -u
cd "$(dirname "$0")"
ids=(); extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    -f) mapfile -t -O "${#ids[@]}" ids < <(grep -v '^\s*\(#\|$\)' "$2"); shift 2 ;;
    --) shift; extra=("$@"); break ;;
    *) ids+=("$1"); shift ;;
  esac
done
for id in "${ids[@]}"; do
  if pixi run python run_status.py "$id" 2>/dev/null; then echo "[skip] $id (ok)"; continue; fi
  echo "[run] $id"
  flock /mnt/scvi_pca/gpu.lock pixi run python 10_run_one.py "$id" "${extra[@]}" || echo "[fail] $id"
done
