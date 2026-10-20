#!/usr/bin/env bash
# Usage: 13_run_dataset.sh <ds> — run 2: setup reps (full models), the run list, metrics, then plots in the background.
# Resumable: setup reps with an ok row + saved latent are skipped, as are ok runs. Launch detached:
#   setsid nohup bash scripts/13_run_dataset.sh <ds> > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
ds=$1
log=/mnt/scvi_pca/run2/logs/campaign_${ds}.log
mkdir -p /mnt/scvi_pca/run2/logs
exec > >(tee -a "$log") 2>&1
echo "=== $ds start $(date -u +%FT%TZ)"
for rep in 1 2; do
  if pixi run python run_status.py "$ds-retrain-all-r$rep" 2>/dev/null && [ -f "/mnt/scvi_pca/run2/models/$ds/full_r$rep/latent.npy" ]; then
    echo "[skip] setup $ds r$rep"; continue
  fi
  echo "[setup] $ds r$rep $(date -u +%T)"
  flock /mnt/scvi_pca/gpu.lock pixi run python 02_train_full.py "$ds" "$rep" || echo "[fail] setup $ds r$rep"
done
ids=/mnt/scvi_pca/run2/logs/ids_${ds}.txt
pixi run python 12_run_list.py "$ds" > "$ids"
bash 11_run_matrix.sh -f "$ids"
echo "[metrics] $ds $(date -u +%T)"
pixi run python 20_metrics.py "$ds" || echo "[fail] metrics $ds"
echo "[plots] $ds $(date -u +%T)"
(nice -n 19 env OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 pixi run python 30_plots.py "$ds" > "/mnt/scvi_pca/run2/logs/plots_${ds}.log" 2>&1 && echo "[plots done] $ds" >> "$log" || echo "[fail] plots $ds" >> "$log") &
pixi run python - "$ds" <<'PY'
import sys, pandas as pd
r = pd.read_parquet("../results/runs.parquet"); ids = open(f"/mnt/scvi_pca/run2/logs/ids_{sys.argv[1]}.txt").read().split()
ids += [f"{sys.argv[1]}-retrain-all-r1", f"{sys.argv[1]}-retrain-all-r2"]
sub = r[r.run_id.isin(ids)]
print(f"[summary] {len(sub)}/{len(ids)} rows, ok={int((sub.status == 'ok').sum())}, missing={sorted(set(ids) - set(sub.run_id))}")
PY
echo "=== $ds done $(date -u +%FT%TZ)"
