#!/usr/bin/env bash
# Sweep runner for rapids_benchmark.py — single-GPU vs multi-GPU × zarr read config.
#
# Runs the grid one SUBPROCESS PER CONFIG (fresh CUDA context + dask cluster + clean
# VRAM each time), each writing its own results/<label>.json and appending a table to
# results/sweep_Run_results.txt. A final pass aggregates every JSON into one
# results/sweep_summary.csv + a printed comparison table.
#
#   Grid:  GPUs {single, multi} × zarr concurrency {4,16} × zarr max-workers {4,8}
#   Neighbors algo DERIVED: multi-GPU -> mg_ivfflat, single-GPU -> ivfflat
#   threads_per_worker = TOTAL_THREADS / n_gpus  -> uses the full 64-core box every run
#     (single=64, quad=16 each = 64 total; keeps host CPU pressure equal across the
#      GPU-count variable — killer #1). Held constant with chunk_rows, preset, harmony off.
#
# Usage (on the GPU node's pixi env):
#   DRY_RUN=1 bash sweep_gpu_zarr.sh          # print planned commands, run nothing
#   bash sweep_gpu_zarr.sh                    # run the full grid
#   DATA=/other/store.zarr bash sweep_gpu_zarr.sh
#
# CACHE: first run reads the store cold, later runs warm (can't drop page cache w/o
# root) — load_zarr mixes cold(1st)/warm; reported, not hidden.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/rapids_benchmark.py"

# ── config (edit here) ────────────────────────────────────────────────────────
DATA="${DATA:-/home/workspace/zarrs/5M_sparse_sorted_9.zarr}"
OUTDIR="${OUTDIR:-$HERE/results}"
PY="${PY:-pixi run python}"                 # runner python; override if not using pixi

GPU_SETUPS=("0" "0,1,2,3")                  # single vs multi (add "0,1" for a midpoint)
CONCURRENCY=(4 16)
MAX_WORKERS=(4 8)

# held constant across every run (isolate the swept variables)
TOTAL_THREADS=64                            # machine cores; threads/worker = TOTAL/n_gpus
CHUNK_ROWS=24000
PRESET=capacity                             # tcp + managed memory
DATA_TAG=5Msorted                           # label prefix
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTDIR"
TXT="$OUTDIR/sweep_Run_results.txt"

echo "== sweep on $DATA =="
echo "   held constant: TOTAL_THREADS=$TOTAL_THREADS (=> threads/worker=TOTAL/n_gpus), "\
"chunk_rows=$CHUNK_ROWS, preset=$PRESET, harmony=off"

# ── run the grid ──────────────────────────────────────────────────────────────
i=0
for gpus in "${GPU_SETUPS[@]}"; do
  commas="${gpus//[^,]/}"                    # count commas without a failing grep (set -e safe)
  n_gpus=$(( ${#commas} + 1 ))
  tpw=$(( TOTAL_THREADS / n_gpus )); (( tpw < 1 )) && tpw=1
  if (( n_gpus > 1 )); then algo=mg_ivfflat; else algo=ivfflat; fi
  gtag="${gpus//,/-}"
  for c in "${CONCURRENCY[@]}"; do
    for w in "${MAX_WORKERS[@]}"; do
      i=$((i+1))
      label="${DATA_TAG}_g${gtag}_c${c}_w${w}"
      args=(--data-path "$DATA" --gpus "$gpus"
        --zarr-concurrency "$c" --zarr-max-workers "$w"
        --neighbors-algorithm "$algo" --threads-per-worker "$tpw"
        --chunk-rows "$CHUNK_ROWS" --preset "$PRESET" --batch-key ""
        --label "$label" --results-json "$OUTDIR/$label.json" --results-txt "$TXT")
      # $PY may be "pixi run python" (multiple words) — split it, then the script + args
      read -r -a py_words <<< "$PY"
      full=("${py_words[@]}" "$BENCH" "${args[@]}")
      echo
      echo "############################################################################"
      echo "# [$i] $label  (gpus=$gpus n_gpus=$n_gpus threads/worker=$tpw algo=$algo c=$c w=$w)"
      echo "############################################################################"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf '  %q' "${full[@]}"; echo
      else
        # don't let one bad config abort the whole sweep
        "${full[@]}" || echo "# -> FAILED ($label), continuing"
      fi
    done
  done
done

# ── aggregate every per-run JSON into one clean table + CSV ───────────────────
if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

# JSON parsing in bash is a trap; use the pixi python for a compact, robust pass.
read -r -a py_words <<< "$PY"
OUTDIR="$OUTDIR" TAG="$DATA_TAG" DATA="$DATA" "${py_words[@]}" - <<'PY'
import os, glob, json, csv
outdir, tag, data = os.environ["OUTDIR"], os.environ["TAG"], os.environ["DATA"]
files = sorted(glob.glob(os.path.join(outdir, f"{tag}_g*_c*_w*.json")))
step_order, rows = [], []
for p in files:
    try:
        j = json.loads(open(p).read())
    except Exception:
        continue
    cfg = j.get("provenance", {}).get("config", {})
    row = {"label": j.get("provenance", {}).get("label", os.path.basename(p)[:-5]),
           "gpus": cfg.get("gpus", "?"),
           "n_gpus": len(str(cfg.get("gpus", "")).split(",")),
           "threads_per_worker": cfg.get("threads_per_worker"),
           "zarr_concurrency": cfg.get("zarr_concurrency"),
           "zarr_max_workers": cfg.get("zarr_max_workers"),
           "neighbors_algorithm": cfg.get("neighbors_algorithm")}
    for s in j.get("results", []):
        if s["step"] not in step_order:
            step_order.append(s["step"])
        row[f"{s['step']}_s"] = round(s["wall_s"], 2)
    t = j.get("totals", {})
    row["total_wall_s"] = round(t.get("wall_s", 0.0), 2)
    row["peak_host_mb"] = round(t.get("peak_host_mb", 0.0), 0)
    row["peak_gpu_mb"] = round(t.get("peak_gpu_mb", 0.0), 0)
    rows.append(row)

step_cols = [f"{s}_s" for s in step_order]
fields = (["label", "n_gpus", "gpus", "threads_per_worker", "zarr_concurrency",
           "zarr_max_workers", "neighbors_algorithm"] + step_cols +
          ["total_wall_s", "peak_host_mb", "peak_gpu_mb"])
csv_path = os.path.join(outdir, "sweep_summary.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

print("\n" + "=" * 92)
print(f"SWEEP SUMMARY  ({len(rows)} runs)  —  {data}")
print("=" * 92)
hdr = (f"{'label':28s} {'ng':>2s} {'tpw':>4s} {'c':>3s} {'w':>3s} {'algo':>11s} "
       f"{'load_s':>8s} {'h2d_s':>7s} {'total_s':>9s} {'GPU_GB':>7s} {'host_GB':>7s}")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['label']:28s} {r['n_gpus']:>2d} {str(r.get('threads_per_worker')):>4s} "
          f"{str(r.get('zarr_concurrency')):>3s} {str(r.get('zarr_max_workers')):>3s} "
          f"{str(r.get('neighbors_algorithm')):>11s} {str(r.get('load_zarr_s','-')):>8s} "
          f"{str(r.get('h2d_transfer_s','-')):>7s} {str(r.get('total_wall_s','-')):>9s} "
          f"{r.get('peak_gpu_mb',0)/1024:>7.1f} {r.get('peak_host_mb',0)/1024:>7.1f}")
print("-" * len(hdr))
print(f"\nRaw per-run JSON:  {outdir}/{tag}_*.json")
print(f"Summary CSV:       {csv_path}")
PY
