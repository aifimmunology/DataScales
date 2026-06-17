# zarr_query_benchmarking

CLI tool to benchmark **query time** against a Zarr store's `X` matrix (dense or
sparse CSR/CSC). It times *read selection + convert to the requested final format*,
so comparisons across layouts are fair (a `dense` query from a CSR store pays the
densify cost; a `csr` query from a dense store pays the sparsify cost).

Run via pixi (`python` is not on PATH directly):

```bash
pixi run zarr-bench --store <path.zarr> --axis row --count 1000 --format csr
pixi run zarr-bench-inspect --store <path.zarr>     # layout only, no timing
```

## Arguments

| Flag | Values | Default | Meaning |
|------|--------|---------|---------|
| `--store` | path | (required) | `.zarr` store to query (reads its `X`). |
| `--axis` | `row` \| `col` | (required) | Query rows (obs/cells) or columns (var/genes). |
| `--count` | int | (required) | Number of rows/columns to select. |
| `--mode` | `sequential` \| `random` | `sequential` | `sequential` = first N; `random` = N seeded random indices. |
| `--format` | `csr` \| `dense` | (required) | Final format the result is converted to (**included in the timing**). |
| `--concurrency` | int | zarr default (10) | `zarr` `async.concurrency` — parallel chunk fetches. |
| `--repeats` | int | 5 | Timed repeats (reports median/min/p95). |
| `--warmup` | int | 1 | Warmup runs discarded before timing. |
| `--seed` | int | 0 | RNG seed for `--mode random`. |
| `--json` | flag | off | Emit one JSON object (raw timings + provenance) instead of text. |
| `--inspect` | flag | off | Print `X` layout (format/shape/chunks/codec) and exit. |

## Metrics reported

- **Wall time** — median / min / p95 over `--repeats` (warm cache).
- **Chunks fetched** — real `store.get()` calls, counted in a *separate untimed pass*
  through a `WrapperStore` so counting never inflates the timing.
- **Bytes read** — compressed bytes pulled from the store (≈ GCS egress for remote stores).
- **Result shape / nnz**, source format, dtype, concurrency, library versions, git commit.

## How reads work

- **Dense** `X` → `zarr` array orthogonal indexing (`X[idx, :]` / `X[:, idx]`).
- **Sparse** `X` → `anndata.io.sparse_dataset(group)` slicing (the realistic downstream
  read path), returning scipy CSR/CSC.

Querying the *aligned* axis is cheap (CSR→rows, CSC→cols); the *cross* axis
(CSR→cols, CSC→rows) reads most of the store and is intentionally expensive — that
contrast is the point of the benchmark.

## Example: sweep row + col across many stores

`dev/run_query_sweep.sh` runs both axes against every `.zarr` in a directory and
appends one JSON line per run:

```bash
# dev/run_query_sweep.sh [STORE_DIR] [COUNT] [FORMAT] [MODE] [OUT]
dev/run_query_sweep.sh zarr_dbs 1000 csr sequential bench_results.jsonl
```

Summarize the JSONL into a table:

```bash
pixi run python - <<'PY'
import json
rows = [json.loads(l) for l in open("bench_results.jsonl")]
print(f"{'store':40} {'axis':4} {'chunks':>8} {'MB':>7} {'median_s':>9}")
for r in rows:
    print(f"{r['store'].split('/')[-1]:40} {r['axis']:4} "
          f"{r['chunks_fetched']:>8} {r['bytes_read']/1e6:>7.1f} {r['median_s']:>9.4f}")
PY
```

> Note: cross-axis sparse queries (e.g. `--axis col` on a CSR store) are slow by
> design — keep `--count` modest when sweeping sparse stores on both axes.
