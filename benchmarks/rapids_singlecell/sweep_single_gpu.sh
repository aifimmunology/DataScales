#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/rapids_benchmark.py"


##-------CONFIG--------
OUTDIR="${OUTDIR:-$HERE/results}"

DATASETS=(
  "/home/workspace/zarrs/datasetA.zarr"
  "/home/workspace/zarrs/datasetB.zarr"
)

RMM=(managed pool)
CHUNK_ROWS=(6000 24000 48000)
POOL_SIZES=(50% 90%)

THREAD_SPLITS=(1:12 3:4 6:2 12:1)
ZARR_CONCURRENCY=32



#------------RUN LOOP
for data in "${DATASETS[@]}"; do
  dtag="$(basename "$data" .zarr)"
  for rmm in "${RMM[@]}"; do
    if [[ "$rmm" == "pool" ]]; then szlist=("${POOL_SIZES[@]}"); else szlist=("-"); fi
    for ck in "${CHUNK_ROWS[@]}"; do
      for psz in "${szlist[@]}"; do
        for split in "${THREAD_SPLITS[@]}"; do
          tpw="${split%%:*}"; zmw="${split##*:}"   # threads_per_worker : zarr_max_workers
          label="${dtag}_g0_${rmm}_ck${ck}"
          extra=()
          if [[ "$psz" != "-" ]]; then extra=(--rmm-pool-size "$psz"); label="${label}_p${psz%\%}"; fi
          label="${label}_t${tpw}x${zmw}"
          args=(--data-path "$data" --gpus 0 --protocol tcp --rmm-mode "$rmm"
                --chunk-rows "$ck" --neighbors-algorithm ivfflat --batch-key ""
                --threads-per-worker "$tpw" --zarr-max-workers "$zmw"
                --zarr-concurrency "$ZARR_CONCURRENCY"
                --label "$label" --results-json "$OUTDIR/$label.json" "${extra[@]}")
          echo "== $label =="
          pixi run python "$BENCH" "${args[@]}" \
            || echo "# -> FAILED ($label), continuing"
        done
      done
    done
  done
done

OUTDIR="$OUTDIR" pixi run python - <<'PY'
import os, glob, json, csv
outdir = os.environ["OUTDIR"]
rows, step_order = [], []
for path in sorted(glob.glob(os.path.join(outdir, "*_g0_*_ck*.json"))):
    try:
        j = json.load(open(path))
    except Exception:
        continue
    cfg = j.get("provenance", {}).get("config", {})
    row = {"label": cfg.get("label", os.path.basename(path)[:-5]),
           "dataset": os.path.basename(cfg.get("data_path", "")),
           "rmm_mode": cfg.get("rmm_mode"), "chunk_rows": cfg.get("chunk_rows"),
           "rmm_pool_size": cfg.get("rmm_pool_size"),
           "threads_per_worker": cfg.get("threads_per_worker"),
           "zarr_max_workers": cfg.get("zarr_max_workers")}
    for s in j.get("results", []):
        if s["step"] not in step_order:
            step_order.append(s["step"])
        row[f"{s['step']}_s"] = round(s["wall_s"], 2)
    t = j.get("totals", {})
    row["total_wall_s"] = round(t.get("wall_s", 0.0), 2)
    row["peak_host_mb"] = round(t.get("peak_host_mb", 0.0))
    row["peak_gpu_mb"] = round(t.get("peak_gpu_mb", 0.0))
    rows.append(row)

fields = (["label", "dataset", "rmm_mode", "chunk_rows", "rmm_pool_size",
           "threads_per_worker", "zarr_max_workers"] +
          [f"{s}_s" for s in step_order] +
          ["total_wall_s", "peak_host_mb", "peak_gpu_mb"])
csv_path = os.path.join(outdir, "sweep_single_gpu.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"wrote {csv_path}  ({len(rows)} runs)")
PY
