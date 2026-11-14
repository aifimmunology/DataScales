#!/usr/bin/env bash
# One command on the GPU capsule: both notebooks headless (notebooks/test_notebooks.sh), then the optional pipeline
# (process/tests: config, store, and every case end to end through ./run on real RAPIDS and scVI).
#   ./test_gpu.sh          notebooks + pipeline (needs a GPU and internet for OmniPath)
#   ./test_gpu.sh big      also times PCA+harmony and scVI (5 epochs, batch 1024) on a synthetic 1M x 10k store
# WORK=<dir> keeps the stores, executed notebooks and outputs there.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK="${WORK:-$(mktemp -d)}"
echo "Working in $WORK"
status=0

"$HERE/notebooks/test_notebooks.sh" || status=1
python -m pytest "$HERE/process/tests" -v -p no:cacheprovider --basetemp="$WORK/pytest" -W ignore::FutureWarning || status=1

if [[ "${1:-}" == big ]]; then
  BIG="$WORK/big"
  mkdir -p "$BIG"
  [[ -d "$BIG/big.zarr" ]] || python - "$BIG/big.zarr" <<'PY'
import sys

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp

N, G, ROWS, TYPES, PER_CELL = 1_000_000, 10_000, 24_000, 12, 1_000
rng = np.random.default_rng(0)
profiles = rng.dirichlet(np.full(G, 0.05), TYPES)
cell_type = rng.integers(0, TYPES, N)


def block(block_info=None):
    start = block_info[None]["array-location"][0][0]
    types = cell_type[start:start + ROWS]
    r = np.random.default_rng(start)
    rows, cols = [], []
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        cols.append(r.choice(G, size=idx.size * PER_CELL, p=profiles[t]))
        rows.append(np.repeat(idx, PER_CELL))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(types.size, G))


X = da.map_blocks(block, chunks=((ROWS,) * (N // ROWS) + ((N % ROWS,) if N % ROWS else ()), (G,)),
                  dtype=np.float32, meta=sp.csr_matrix((0, 0), dtype=np.float32))
obs = pd.DataFrame({"sample_id": pd.Categorical([f"S{i % 8}" for i in range(N)]),
                    "chip_id": pd.Categorical([f"C{i % 3}" for i in range(N)])}, index=[f"c{i}" for i in range(N)])
ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(G)])).write_zarr(sys.argv[1])
print(f"wrote {sys.argv[1]}")
PY
  for method in PCA "scVI (train)"; do
    echo "=== 1M cells: $method"
    SCVI_MAX_EPOCHS=5 SCVI_BATCH_SIZE=1024 "$HERE/run" --input "$BIG/big.zarr" --method "$method" \
      --batch_key chip_id --leiden_resolutions 0.5,1.0 --results_dir "$BIG/results" || status=1
  done
  cat "$BIG/results/process_report.txt"
fi

[[ $status == 0 ]] && echo "ALL PASSED" || echo "SOME FAILED"
exit $status
