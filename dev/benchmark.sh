#!/usr/bin/env bash
# benchmark.sh — wall-clock time and peak RSS for datascale conversions
#
# Works on macOS (uses /usr/bin/time -l) and Linux (uses /usr/bin/time -v).
# Results are printed as a table and optionally saved to a file.
#
# Usage:
#   ./benchmark.sh                  # print to terminal
#   ./benchmark.sh results.txt      # also save to file

set -euo pipefail

OS="$(uname -s)"
OUT_FILE="${1:-}"         # optional first arg: path to save results
TMP_ZARR="/tmp/ds_bench.zarr"

# ── Benchmark jobs ────────────────────────────────────────────────────────────
# Format: "Label|command"
# Output zarr is intentionally overwritten each run — we're testing write perf.
JOBS=(
    "pbmc3k_raw     sparse-csr|pixi run datascale convert-h5ad --input ./data/pbmc3k_raw.h5ad --output $TMP_ZARR --overwrite --x-storage sparse-csr"
    "pbmc3k_raw     sparse-csc|pixi run datascale convert-h5ad --input ./data/pbmc3k_raw.h5ad --output $TMP_ZARR --overwrite --x-storage sparse-csc"
    "pbmc3k_raw     dense(eager)|pixi run datascale convert-h5ad --input ./data/pbmc3k_raw.h5ad --output $TMP_ZARR --overwrite --x-storage dense"
    "pbmc3k_raw     dense(naive)|pixi run python my_scripts/naive_dense_convert.py --input ./data/pbmc3k_raw.h5ad --output $TMP_ZARR --overwrite"
    "scanpy_pbmc3k  sparse-csr|pixi run datascale convert-h5ad --input ./data/scanpy-pbmc3k.h5ad --output $TMP_ZARR --overwrite --x-storage sparse-csr"
    "scanpy_pbmc3k  dense(eager)|pixi run datascale convert-h5ad --input ./data/scanpy-pbmc3k.h5ad --output $TMP_ZARR --overwrite --x-storage dense"
    "scanpy_pbmc3k  dense(naive)|pixi run python my_scripts/naive_dense_convert.py --input ./data/scanpy-pbmc3k.h5ad --output $TMP_ZARR --overwrite"
    "health_atlas   sparse-csr|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage sparse-csr"
    "health_atlas   sparse-csc|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage sparse-csc"
    "health_atlas   dense(w=1)|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage dense --n-dense-workers 1"
    "health_atlas   dense(w=2)|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage dense --n-dense-workers 2"
    "health_atlas   dense(w=4)|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage dense --n-dense-workers 4"
    "health_atlas   dense(w=8)|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage dense --n-dense-workers 8"
    "health_atlas   dense(backed)|pixi run datascale convert-h5ad --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite --x-storage dense --backed"
    "health_atlas   dense(naive)|pixi run python my_scripts/naive_dense_convert.py --input ./data/human_immune_health_atlas_other.h5ad --output $TMP_ZARR --overwrite"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# macOS: parse "  1.23 real" from /usr/bin/time -l stderr
_mac_wall() { grep " real" "$1" | awk '{print $1}'; }

# macOS: parse "maximum resident set size" (bytes) → MB
_mac_mem_mb() { grep "maximum resident set size" "$1" | awk '{printf "%.0f", $1/1048576}'; }

# Linux: convert "h:mm:ss.ss" or "m:ss.ss" wall-clock string to seconds
_linux_wall_sec() {
    local t="$1" h=0 m=0 s=0
    IFS=: read -r a b c <<< "$t"
    if [[ -n "${c:-}" ]]; then h=$a; m=$b; s=$c; else m=$a; s=$b; fi
    echo "$h $m $s" | awk '{printf "%.2f", $1*3600 + $2*60 + $3}'
}

# Linux: parse "Maximum resident set size (kbytes)" → MB
_linux_mem_mb() { grep "Maximum resident set size" "$1" | awk '{printf "%.0f", $NF/1024}'; }

# ── Runner ────────────────────────────────────────────────────────────────────

declare -a RESULTS=()

run_job() {
    local label="$1" cmd="$2"
    local tmp; tmp=$(mktemp)

    printf "  %-30s  ... " "$label"

    if [[ "$OS" == "Darwin" ]]; then
        if ! /usr/bin/time -l bash -c "$cmd" > /dev/null 2>"$tmp"; then
            printf "FAILED\n"; rm -f "$tmp"; RESULTS+=("$label|FAILED|--"); return
        fi
        wall=$(_mac_wall "$tmp")
        mem=$(_mac_mem_mb "$tmp")
    else
        if ! /usr/bin/time -v bash -c "$cmd" > /dev/null 2>"$tmp"; then
            printf "FAILED\n"; rm -f "$tmp"; RESULTS+=("$label|FAILED|--"); return
        fi
        raw_wall=$(grep "Elapsed (wall clock)" "$tmp" | awk '{print $NF}')
        wall=$(_linux_wall_sec "$raw_wall")
        mem=$(_linux_mem_mb "$tmp")
    fi

    rm -f "$tmp"
    printf "%6ss   %6s MB\n" "$wall" "$mem"
    RESULTS+=("$label|${wall}s|${mem} MB")
}

# ── Main ──────────────────────────────────────────────────────────────────────

HEADER="DataScale Benchmark  —  $(date '+%Y-%m-%d %H:%M')  —  OS: $OS"
DIVIDER="$(printf '=%.0s' {1..60})"

{
    echo ""
    echo "$HEADER"
    echo "$DIVIDER"
    echo ""
    printf "  %-30s  %8s  %10s\n" "Job" "Wall" "Peak RSS"
    printf "  %-30s  %8s  %10s\n" "---" "----" "--------"
} | tee -a "${OUT_FILE:-/dev/null}"

for job in "${JOBS[@]}"; do
    label="${job%%|*}"
    cmd="${job##*|}"
    run_job "$label" "$cmd"
done

{
    echo ""
    echo "$DIVIDER"
    echo ""
    printf "  %-30s  %8s  %10s\n" "Job" "Wall" "Peak RSS"
    printf "  %-30s  %8s  %10s\n" "---" "----" "--------"
    for r in "${RESULTS[@]}"; do
        IFS='|' read -r lbl wall mem <<< "$r"
        printf "  %-30s  %8s  %10s\n" "$lbl" "$wall" "$mem"
    done
    echo ""
    echo "Temp zarr: $TMP_ZARR  (overwritten each run, safe to delete)"
    echo ""
} | tee -a "${OUT_FILE:-/dev/null}"

if [[ -n "$OUT_FILE" ]]; then
    echo "Results saved to: $OUT_FILE"
fi
