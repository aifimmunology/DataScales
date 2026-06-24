#!/usr/bin/env bash
# Benchmark row + col query time across every .zarr store in a directory.
# Each (store, axis) run appends one JSON line to the output file.
#
# Usage:  dev/run_query_sweep.sh [STORE_DIR] [COUNT] [FORMAT] [MODE] [OUT]
# Example: dev/run_query_sweep.sh zarr_dbs 1000 csr sequential bench_results.jsonl
set -euo pipefail

STORE_DIR="${1:-zarr_dbs}"
COUNT="${2:-1000}"
FORMAT="${3:-csr}"      # csr | dense  (final format; conversion is included in the timing)
ASYNC="${4:-10}"
MODE="${5:-sequential}" # sequential | random
OUT="${6:-bench_results.jsonl}"

: > "$OUT"
for store in "$STORE_DIR"/*.zarr; do
  for axis in row; do
    echo ">> $store  axis=$axis  count=$COUNT  ASYNC=$ASYNC format=$FORMAT  mode=$MODE" >&2
    pixi run -q zarr-bench --store "$store" --axis "$axis" --concurrency "$ASYNC" --count "$COUNT" \
      --mode "$MODE" --format "$FORMAT" --json 2>/dev/null >> "$OUT"
  done
done
echo "Wrote $(wc -l < "$OUT") results to $OUT" >&2
