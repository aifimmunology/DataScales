#!/usr/bin/env bash
# One combined run: BRI (BR1+BR2) + UP1 cohorts on a single GPU, speed preset
# (ucx + rmm pool). Harmony (--batch-key) corrects at the sample-kit level; cohort
# is nested within kit, so kit-level correction absorbs the technical cross-cohort
# shift without regressing out cohort-level biology. Results (obs incl. leiden +
# obsm/X_umap, no X) land at
#   <STORE>/subsets/cohort.cohortGuid_BR1_BR2_UP1
# Override via env: STORE=/path/to.zarr GPU=1 PRESET=capacity ./run_combined_bri_up1.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

STORE="${STORE:-/mnt/external_megazarr_v1.0.zarr}"
GPU="${GPU:-0}"
PRESET="${PRESET:-speed}"

pixi run python rapids_benchmark.py \
  --data-path "$STORE" \
  --gpus "$GPU" \
  --preset "$PRESET" \
  --subset-column cohort.cohortGuid \
  --subset-value BR1,BR2,UP1 \
  --batch-key sample.sampleKitGuid \
  --label "combined_BR1_BR2_UP1_g${GPU}_${PRESET}"
