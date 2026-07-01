#!/usr/bin/env bash
# Benchmark row + col query time across every .zarr store in a directory.
# Each (store, axis) run appends one JSON line to the output file.
#
# STORE_DIR may be a local directory OR a remote fsspec prefix (e.g.
# gs://bucket/prefix). A local dir is shell-globbed for *.zarr; a gs:// (or any
# scheme://) prefix is listed via gcsfs, since a bucket can't be shell-globbed.
# Remote reads use zarr's FsspecStore and gcloud Application Default Credentials.
#
# Usage:  dev/run_query_sweep.sh [STORE_DIR] [COUNT] [FORMAT] [ASYNC] [MODE] [OUT]
# Example (local):  dev/run_query_sweep.sh zarr_dbs 1000 csr 10 sequential bench_results.jsonl
# Example (gcs):    dev/run_query_sweep.sh gs://my-bucket/stores 1000 csr 10 sequential bench_results.jsonl
set -euo pipefail

STORES=(
  gs://rapid-zarr_storage/2M_csc_9.zarr
  gs://rapid-zarr_storage/2M_csr_9.zarr
  gs://rapid-zarr_storage/2M_dense_1x1.zarr
  gs://rapid-zarr_storage/2M_dense_1x1_10S.zarr
  gs://rapid-zarr_storage/2M_dense_5x5.zarr
  gs://rapid-zarr_storage/2M_dense_5xA.zarr
)

COUNT="${1:-1000}"
FORMAT="${2:-csr}"
ASYNC="${3:-10}"
MODE="${4:-sequential}"
OUT="${5:-bench_results.jsonl}"


echo "[" > "$OUT"
first=1
for store in "${STORES[@]}"; do
# for store in "$STORE_DIR"/*.zarr; do
  echo ">> $store  axis=row  count=$COUNT  ASYNC=$ASYNC format=$FORMAT  mode=$MODE" >&2
  if [[ "$first" -eq 0 ]]; then echo "," >> "$OUT"; fi
  pixi run -q zarr-bench --store "$store" --axis row --concurrency "$ASYNC" --max-workers "$ASYNC" \
    --count "$COUNT" --mode "$MODE" --format "$FORMAT" --json >> "$OUT"
  first=0
done
echo "]" >> "$OUT"
echo "Wrote $(grep -c '"store"' "$OUT") result(s) to $OUT" >&2