#!/usr/bin/env bash
# Sweep query time (row + col) across every .zarr store in a directory and append
# one JSON object per (store, axis) run to a JSON Lines file that
# `python -m zarr_query_benchmarking.compare` can read directly.
#
# STORE_DIR may be a local directory OR a remote fsspec prefix (e.g.
# gs://bucket/prefix). A local dir is shell-globbed for *.zarr; a scheme:// prefix
# (gs://, s3://, ...) is listed via fsspec, since a bucket can't be shell-globbed.
# Remote reads use zarr's FsspecStore and gcloud Application Default Credentials.
#
# Usage:  run_query_sweep.sh [STORE_DIR] [COUNT] [FORMAT] [THREAD_CONCURRENCY] [MODE] [OUT]
# Local:  zarr_query_benchmarking/examples/run_query_sweep.sh zarr_dbs 1000 csr 32 sequential bench_results.jsonl
# GCS:    zarr_query_benchmarking/examples/run_query_sweep.sh gs://my-bucket/stores 1000 csr 32 sequential bench_results.jsonl
set -euo pipefail

STORE_DIR="${1:-zarr_dbs}"
COUNT="${2:-1000}"
FORMAT="${3:-csr}"
CONC="${4:-32}"           # sets BOTH --concurrency and --max-workers
MODE="${5:-sequential}"
OUT="${6:-bench_results.jsonl}"

# Resolve the list of stores: local glob, or remote listing via fsspec.
# (Built with a while-read loop, not `mapfile`, to stay bash 3.2 compatible for macOS.)
STORES=()
if [[ "$STORE_DIR" == *://* ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && STORES+=("$line")
  done < <(pixi run -q python - "$STORE_DIR" <<'PY'
import sys, fsspec
prefix = sys.argv[1].rstrip("/")
fs, _, _ = fsspec.get_fs_token_paths(prefix)
proto = prefix.split("://", 1)[0]
for p in fs.ls(prefix, detail=False):
    if p.rstrip("/").endswith(".zarr"):
        print(p if "://" in p else f"{proto}://{p}")
PY
  )
else
  for s in "$STORE_DIR"/*.zarr; do
    [[ -e "$s" ]] && STORES+=("$s")
  done
fi

if [[ "${#STORES[@]}" -eq 0 ]]; then
  echo "No .zarr stores found under '$STORE_DIR'." >&2
  exit 1
fi

: > "$OUT"   # truncate; one JSON object per line (JSON Lines)
for store in "${STORES[@]}"; do
  for axis in row col; do
    echo ">> $store  axis=$axis  count=$COUNT  conc=$CONC  format=$FORMAT  mode=$MODE" >&2
    pixi run -q zarr-bench --store "$store" --axis "$axis" \
      --concurrency "$CONC" --max-workers "$CONC" \
      --count "$COUNT" --mode "$MODE" --format "$FORMAT" --json >> "$OUT"
  done
done
echo "Wrote $(grep -c '"store"' "$OUT") result(s) to $OUT" >&2
