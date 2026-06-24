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
| `--axis` | `row` \| `col` | (required¹) | Query rows (obs/cells) or columns (var/genes). |
| `--count` | int | (required¹) | Number of rows/columns to select. |
| `--mode` | `sequential` \| `random` \| `celltype` | `sequential` | `sequential` = first N; `random` = N seeded random indices; `celltype` = all obs rows whose `--obs-column` equals `--cell-type` (see below). |
| `--obs-column` | str | (required for `celltype`) | obs column to filter on (e.g. a cell-type annotation). |
| `--cell-type` | str | (required for `celltype`) | Value in `--obs-column` to select rows by. |
| `--format` | `csr` \| `dense` | (required) | Final format the result is converted to (**included in the timing**). |
| `--concurrency` | int | zarr default (10) | `zarr` `async.concurrency` — parallel chunk fetches. |
| `--repeats` | int | 5 | Timed repeats (reports median/min/p95). |
| `--warmup` | int | 1 | Warmup runs discarded before timing. |
| `--seed` | int | 0 | RNG seed for `--mode random`. |
| `--json` | flag | off | Emit one JSON object (raw timings + provenance) instead of text. |
| `--inspect` | flag | off | Print `X` layout (format/shape/chunks/codec) and exit. |

¹ `--axis` and `--count` are required for `sequential`/`random`. `--mode celltype`
forces `--axis row` (rejects `--axis col`), **ignores `--count`**, and requires
`--obs-column` + `--cell-type` instead.

## Selecting by cell type (`--mode celltype`)

```bash
pixi run zarr-bench --store <path.zarr> --mode celltype \
    --obs-column AIFI_L1 --cell-type Platelet --format csr
```

Reads `obs[<obs-column>]`, builds the `== <cell-type>` mask, and selects **all**
matching obs rows (the row count is whatever matches; `--count` is ignored). The
obs read + mask build happen **inside the timed region**, so the wall time models
a real "filter cells by type, then fetch their `X`" query rather than a bare slice.
A typo in the column or cell-type name exits non-zero and prints the available
columns / values. The JSON output adds `obs_column`, `cell_type`, and `selected`
(number of matched rows).

## Metrics reported

- **Wall time** — median / min / p95 over `--repeats` (warm cache).
- **Chunks fetched** — real `store.get()` calls, counted in a *separate untimed pass*
  through a `WrapperStore` so counting never inflates the timing.
- **Bytes read** — compressed bytes pulled from the store (≈ GCS egress for remote stores).
- **Decompressed result bytes** (`result_decompressed_bytes`) — working-set size of the
  materialized result (dense: `rows*cols`; CSR: `data+indices+indptr`). This — **not**
  compressed `bytes_read` — is what drives RAM and host→device (GPU) transfer. scRNA dense
  data Blosc-compresses extremely well, so `bytes_read` can look similar across formats while
  this reveals the real (often ~10×) gap.
- **Peak RSS** (`peak_rss_bytes`) — peak resident memory of a single read+convert, measured
  in an *isolated subprocess* (so warmup/repeats don't contaminate the high-water mark) and
  normalized to bytes across macOS/Linux. `null` if the probe fails.
- **Result shape / nnz**, source format, dtype, concurrency, library versions, git commit.

## How reads work

- **Dense** `X` → `zarr` array orthogonal indexing (`X[idx, :]` / `X[:, idx]`).
- **Sparse** `X` → `anndata.io.sparse_dataset(group)` slicing (the realistic downstream
  read path), returning scipy CSR/CSC.

Querying the *aligned* axis is cheap (CSR→rows, CSC→cols); the *cross* axis
(CSR→cols, CSC→rows) reads most of the store and is intentionally expensive — that
contrast is the point of the benchmark.

## Comparing dense vs. sparse fairly

`--format` is the **output** format and is *included in the timing*. A common `--format`
across two stores answers "cost to get format X out, regardless of how it's stored" — but
it charges the CSR store a `.toarray()` densify cost (and a dense store a sparsify cost)
that real **rapids-singlecell + dask** analysis *never pays*: that pipeline keeps each
store in its native on-device representation (cupyx CSR vs. dense CuPy) and reads
row-blocks (the aligned axis).

So for an **analysis-faithful read comparison**, query each store at its **native** format
on **`--axis row`** — CSR store → `--format csr`, dense store → `--format dense`. The
common-format mode answers a different question and will understate CSR's advantage.

Either way, read the comparison off **`result_decompressed_bytes`** and **`peak_rss_bytes`**,
not just `bytes_read`: the decompressed footprint is what real analysis pays in RAM and
PCIe transfer, and it's where the dense-vs-sparse difference actually shows up.

## Example: sweep row + col across many stores

`dev/run_query_sweep.sh` runs both axes against every `.zarr` in a directory and
appends one JSON line per run:

```bash
# dev/run_query_sweep.sh [STORE_DIR] [COUNT] [FORMAT] [THREAD_CONCURRENCY] [MODE] [OUT]
dev/run_query_sweep.sh zarr_dbs 1000 csr 32 sequential bench_results.jsonl
```

> Note: cross-axis sparse queries (e.g. `--axis col` on a CSR store) are slow by
> design — keep `--count` modest when sweeping sparse stores on both axes.

## Comparing runs (`compare`)

`compare` reads one or more `--json` result files and prints an aligned comparison
table, so differences between store layouts / axes / thread counts are easy to
eyeball. It accepts a single JSON object, a JSON **array** of them (e.g. a
hand-collected `output1.json`), or **JSON Lines** (the format
`dev/run_query_sweep.sh` appends) — and globs are expanded.

```bash
# one file, sorted by median wall time
pixi run python -m zarr_query_benchmarking.compare bench_results.jsonl --sort median_s

# several files at once (a "file" column is added automatically)
pixi run python -m zarr_query_benchmarking.compare 'runs/*.json' --sort store

# emit a GitHub-flavored markdown table for a PR / notes
pixi run python -m zarr_query_benchmarking.compare output1.json --md > table.md
```

| Flag | Meaning |
|------|---------|
| `files…` | One or more JSON / JSONL result files; shell globs are expanded. |
| `--sort FIELD` | Sort rows ascending by any run field (e.g. `median_s`, `store`, `chunks_fetched`). |
| `--md` | Emit a GitHub-flavored markdown table (numeric columns right-aligned). |

Columns: `store` · `src` (source format) · `shape` · `axis` · `mode` · `out`
(final format) · `conc` · `n` · `result` shape · `chunks` fetched · `read_MB` ·
`rss_GB` (peak) · `med_s` · `p95_s` · `commit`, plus a trailing **`xslow`** column
— each run's median relative to the fastest run in the table (`1.00x` = fastest).
That last column is usually the quickest way to see how much a layout choice costs:

```
store                  src    axis  ...  read_MB  rss_GB    med_s    xslow
5M_sparse_9.zarr       csr    row   ...     1391    10.5    2.875    1.00x
13M_..._soundlife.zarr csr    row   ...     1523    12.0    4.111    1.43x
5M_dense_5x11.zarr     dense  row   ...     1637    58.1  132.419   46.06x
```

(Read the dense-vs-sparse gap off `read_MB`/`rss_GB` alongside `xslow`, per the
fairness note above — `bytes_read` alone understates it.)
