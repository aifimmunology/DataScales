#!/usr/bin/env bash
# Sweep an explicit list of stores × parameters (edit the CONFIG block, run).
# Appends one JSON object per run to OUT — readable by
# `pixi run python -m zarr_query_bench.compare`.
# For "every .zarr under a directory/prefix" use run_query_sweep.sh instead.
set -euo pipefail

##-------CONFIG--------
STORES=(
  "zarr_dbs/2M_csr.zarr"
  "zarr_dbs/2M_dense_5x5.zarr"
  #"gs://my-bucket/atlas_csr.zarr"
)
AXIS=row
FORMAT=csr                 # final output format; use --native in the loop to skip conversion
COUNTS=(20000 100000)
THREADS=(4 16 64)          # sets BOTH --concurrency and --max-workers
OUT="${OUT:-sweep_results.jsonl}"

#------------RUN LOOP
: > "$OUT"
for store in "${STORES[@]}"; do
  for count in "${COUNTS[@]}"; do
    for t in "${THREADS[@]}"; do
      echo ">> $store  axis=$AXIS  count=$count  threads=$t  format=$FORMAT" >&2
      pixi run -q zarr-bench --store "$store" --axis "$AXIS" --count "$count" \
        --format "$FORMAT" --concurrency "$t" --max-workers "$t" --json >> "$OUT" \
        || echo "# -> FAILED ($store count=$count t=$t), continuing" >&2
    done
  done
done
echo "Done. Compare: pixi run python -m zarr_query_bench.compare $OUT" >&2
