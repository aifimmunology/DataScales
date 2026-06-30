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

STORE_DIR="${1:-zarr_dbs}"
COUNT="${2:-1000}"
FORMAT="${3:-csr}"      # csr | dense  (final format; conversion is included in the timing)
ASYNC="${4:-10}"
MODE="${5:-sequential}" # sequential | random
OUT="${6:-bench_results.jsonl}"

# Build the list of stores. Remote prefixes (scheme://) are listed with gcsfs;
# local dirs are globbed. Each line is a full store path/URL passed to --store.
list_stores() {
  if [[ "$STORE_DIR" == *"://"* ]]; then
    pixi run -q python - "$STORE_DIR" <<'PY'
import sys, fsspec
base = sys.argv[1]
proto = base.split("://", 1)[0]
fs = fsspec.filesystem(proto)
for p in fs.ls(base, detail=False):
    if p.rstrip("/").endswith(".zarr"):
        print(f"{proto}://{p.lstrip('/')}" if "://" not in p else p)
PY
  else
    for s in "$STORE_DIR"/*.zarr; do echo "$s"; done
  fi
}

mapfile -t STORES < <(list_stores)
if [[ "${#STORES[@]}" -eq 0 ]]; then
  echo "ERROR: no .zarr stores found under $STORE_DIR" >&2
  exit 1
fi
echo ">> found ${#STORES[@]} store(s) under $STORE_DIR" >&2

: > "$OUT"
for store in "${STORES[@]}"; do
  for axis in row; do
    echo ">> $store  axis=$axis  count=$COUNT  ASYNC=$ASYNC format=$FORMAT  mode=$MODE" >&2
    pixi run -q zarr-bench --store "$store" --axis "$axis" --concurrency "$ASYNC" --count "$COUNT" \
      --mode "$MODE" --format "$FORMAT" --json 2>/dev/null >> "$OUT"
  done
done
echo "Wrote $(wc -l < "$OUT") results to $OUT" >&2
