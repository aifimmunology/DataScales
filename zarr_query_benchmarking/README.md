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
| `--store` | path \| URL | (required) | `.zarr` store to query (reads its `X`). Local path **or** an fsspec URL — `gs://bucket/store.zarr`, `s3://…` (see [Remote stores](#remote-stores)). |
| `--axis` | `row` \| `col` | (required¹) | Query rows (obs/cells) or columns (var/genes). |
| `--count` | int | (required¹) | Number of rows/columns to select. |
| `--mode` | `sequential` \| `random` \| `celltype` | `sequential` | `sequential` = first N; `random` = N seeded random indices; `celltype` = all obs rows whose `--obs-column` equals `--obs-value` (see below). |
| `--select-mode` | `auto` \| `slice` \| `fancy` | `auto` | **celltype only.** How matched rows are read (both first read obs + build the mask, so they differ *only* in the X read). `auto` picks `slice` when the cells form few long contiguous runs (sorted store — one cheap grab) and `fancy` when scattered into many short runs; `slice` reads `X[start:end]` per run (re-fetches shared chunks when scattered); `fancy` builds a `flatnonzero` index and gathers each chunk once. Ignored for sequential/random. |
| `--obs-column` | str | (required for `celltype`) | obs column to filter on (e.g. a cell-type annotation). |
| `--obs-value` | str | (required for `celltype`) | Value in `--obs-column` to select rows by. |
| `--format` | `csr` \| `dense` | (required¹) | Final format the result is converted to (**included in the timing**; broken out as `convert_s`). |
| `--native` | flag | off | Read at the store's **native** format — no conversion (`convert_s` ≈ 0), so you measure the layout, not the conversion tax. Overrides `--format`. Use for fair CSR-vs-dense / row-vs-col comparisons. |
| `--concurrency` | int | zarr default (10) | `zarr` `async.concurrency` — how many chunks are *dispatched* at once (a semaphore). Pair with `--max-workers`. |
| `--max-workers` | int | `min(32, cpu+4)` | `zarr` `threading.max_workers` — size of the pool that actually *decodes* chunks (Blosc/zstd). **The dominant stage of a dense read.** Set ≈ physical cores. |
| `--repeats` | int | 5 | Timed repeats (reports median/min/p95). |
| `--warmup` | int | 1 | Warmup runs discarded before timing. |
| `--seed` | int | 0 | RNG seed for `--mode random`. |
| `--json` | flag | off | Emit one JSON object (raw timings + provenance) instead of text. |
| `--inspect` | flag | off | Print `X` layout (format/shape/chunks/codec) and exit. |

¹ `--axis` and `--count` are required for `sequential`/`random`. `--mode celltype`
forces `--axis row` (rejects `--axis col`), **ignores `--count`**, and requires
`--obs-column` + `--obs-value` instead. Either `--format` **or** `--native` is required.

## Selecting by cell type (`--mode celltype`)

```bash
pixi run zarr-bench --store <path.zarr> --mode celltype \
    --obs-column AIFI_L1 --obs-value Platelet --format csr
```

Reads `obs[<obs-column>]`, builds the `== <obs-value>` mask, and selects **all**
matching obs rows (the row count is whatever matches; `--count` is ignored). The
obs read + mask build happen **inside the timed region**, so the wall time models
a real "filter cells by type, then fetch their `X`" query rather than a bare slice.
A typo in the column or obs-value name exits non-zero and prints the available
columns / values. The JSON output adds `obs_column`, `obs_value`, and `selected`
(number of matched rows).

## Two knobs, both required for parallelism

A dense read uses **two independent** zarr settings, and raising only one pins you near 1 core:

- `--concurrency` (`async.concurrency`) only *dispatches* chunks — it's an `asyncio.Semaphore`,
  it runs no CPU work.
- `--max-workers` (`threading.max_workers`) sizes the `ThreadPoolExecutor` that runs **Blosc/zstd
  decompression**, which is **~70% of a warm dense read** (`BloscCodec._decode_single` →
  `asyncio.to_thread`). Decode releases the GIL, so it scales across cores.

So `--concurrency 60 --max-workers 1` ≈ 1 core (decode serialized); `--concurrency 1` starves the
pool regardless of workers. **Set both** (e.g. `--concurrency 64 --max-workers <physical cores>`).
Measured on a 12-core box, `X[0:100000, :]` (500k×34k dense, 1k×1k chunks): `conc=60/mw=1` = 10.1 s
@ 1.1 cores → `conc=60/mw=12` = 2.63 s @ 5.2 cores. See `thread_scaling_probe.py` for the full
stage decomposition (IO / decompress / assemble) and the dask comparison.

## Metrics reported

- **Wall time** — median / min / p95 over `--repeats` (warm cache).
- **I/O vs CPU vs convert split** (`io_wall_median_s` / `cpu_wall_median_s` / `convert_median_s`) —
  every timed read runs through a `TimingStore` that records each `store.get` interval, and the
  final format conversion is timed separately. The three sum to the wall time:
  - `io_wall` = wall time with **≥1 fetch in flight** (the *union* of fetch intervals —
    concurrency-correct, not a naive sum of overlapping durations). This is the store-engine / network cost.
  - `cpu_wall` = **decompress + gather** (the read-side CPU: wall with no fetch in flight, not converting).
  - `convert_median_s` = the `.toarray()` / `.tocsr()` step (0 with `--native`). This is the
    format-conversion tax — often the *dominant, layout-independent* term in a CSR-vs-dense
    comparison at a common `--format`, which is why it's broken out.

  Together they tell you whether a query is **network/IO-bound** (→ layout / chunk locality is the
  lever), **read-CPU-bound** (→ read path / decompress), or **conversion-bound** (→ you're measuring
  the format tax, not the layout — switch to `--native`). For a perfectly clean I/O split run
  `--concurrency 1` (no overlap); under high concurrency I/O and CPU pipeline, so `io_wall` counts
  any wall time with a fetch outstanding.
- **Chunks fetched** — real `store.get()` calls, counted by the same `TimingStore` in the timed
  pass (deterministic, so it doesn't inflate the timing).
- **Runs** (`n_spans`, celltype only) — number of contiguous row-runs the matched cells form: a
  **locality metric**. A sorted store yields few long runs (slice reads are cheap); an unsorted
  store yields many length-1 runs (slice degenerates — use `--select-mode fancy`).
- **Peak RSS** (`peak_rss_bytes`) — peak resident memory of a single read+convert, measured
  in an *isolated subprocess* (so warmup/repeats don't contaminate the high-water mark) and
  normalized to bytes across macOS/Linux. `null` if the probe fails.
- **Result shape / nnz**, source format, dtype, select mode, concurrency, versions, git commit.

## How reads work

- **Dense** `X` → `zarr` array orthogonal indexing (`X[idx, :]` / `X[:, idx]`).
- **Sparse** `X` → `anndata.io.sparse_dataset(group)` slicing (the realistic downstream
  read path), returning scipy CSR/CSC.

For `--mode celltype`, `--select-mode` chooses how the matched rows are read (both first read
obs + build the `== value` mask, so they differ **only** in the X read):

- **`slice`** (default) finds the contiguous `[start, end)` run(s) of matched rows and reads
  `X[start:end]` per run. On a **sorted** store the cell type is one run → a single contiguous
  grab that takes anndata's fast path (`_get_contiguous_compressed_slice`: no per-row gather,
  no coordinate indexer). This is what `datascale.reader` does.
- **`fancy`** builds a `flatnonzero` integer index and gathers per row (anndata's coordinate
  path — the per-row `indptr` loop + zarr `CoordinateIndexer`).

The catch: `slice` reads each run independently, so when the cell type is **scattered**
(unsorted → hundreds/thousands of runs) it re-fetches shared chunks per run and is *far slower*
than one `fancy` gather (measured: 995 runs → 2008 chunk GETs and ~120× the wall time of the
26-GET fancy read on the same data). So the rule is **sorted store → `slice`, unsorted store →
`fancy`**; the `runs` metric tells you which regime you're in.

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

Better yet, pass **`--native`**: it reads each store at its own format (no conversion), so
`convert_s` ≈ 0 and you measure the layout instead of the conversion tax — the cleanest way to
compare CSR vs dense (and row vs col) without one side paying a `.toarray()`/`.tocsr()` it never
pays in real analysis. Then read the comparison off **`peak_rss_bytes`** and the
**`io_wall`/`cpu_wall`** split: peak RSS is what real analysis pays in RAM and host→device (GPU)
transfer, and the I/O-vs-CPU split shows whether the cost is the store engine (network / chunks
fetched) or the read path (decompress + reassembly).

## Remote stores

`--store` accepts an fsspec URL, not just a local path. A URL with a scheme is
opened through zarr's `FsspecStore` (the documented, stable remote backend —
`ObjectStore`/`obstore` is still flagged experimental):

```bash
pixi run zarr-bench --store gs://my-bucket/health_atlas_csr.zarr \
    --axis row --count 1000 --format csr --json
```

- **Google Cloud Storage** (`gs://`) requires `gcsfs` (a pixi dependency) and uses
  **gcloud Application Default Credentials** automatically — run
  `gcloud auth application-default login` once; no token is passed in code.
- **`chunks_fetched`** is the number of object GETs and **`io_wall`** is the wall time spent
  on them (fetch in flight) — the numbers that matter for remote cost.
- The "warm cache" in the timing reflects gcsfs/OS caching after the warmup read,
  **not** a cold first-touch network read. For a cold-vs-warm split, run with
  `--warmup 0 --repeats 1` in a fresh process for cold, and the normal settings
  for warm. (Object stores can't be page-cache-dropped from userspace.)

## Example: sweep row + col across many stores

`dev/run_query_sweep.sh` runs both axes against every `.zarr` in a directory and
appends one JSON line per run:

```bash
# dev/run_query_sweep.sh [STORE_DIR] [COUNT] [FORMAT] [THREAD_CONCURRENCY] [MODE] [OUT]
dev/run_query_sweep.sh zarr_dbs 1000 csr 32 sequential bench_results.jsonl

# STORE_DIR may be a gs:// prefix — every *.zarr at that prefix is listed via
# gcsfs (a bucket can't be shell-globbed) and benchmarked in turn:
dev/run_query_sweep.sh gs://my-bucket/stores 1000 csr 32 sequential bench_results.jsonl
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
(final format) · `smode` (select mode) · `conc` · `n` · `result` shape · `chunks` fetched ·
`runs` · `rss_GB` (peak) · `med_s` · `io_s` / `cpu_s` (I/O vs CPU split) · `p95_s` · `commit`,
plus a trailing **`xslow`** column — each run's median relative to the fastest run in the table
(`1.00x` = fastest). That last column is usually the quickest way to see how much a layout /
read-path choice costs:

```
store           src  axis  smode  ...  chunks  runs  rss_GB  med_s   io_s   cpu_s   xslow
sorted.zarr     csr  row   slice  ...      16     1     0.1  0.005  0.001   0.004   1.00x
sorted.zarr     csr  row   fancy  ...      16     1     0.1  0.009  0.001   0.008   1.78x
unsorted.zarr   csr  row   fancy  ...      26   995     0.1  0.010  0.003   0.007   1.94x
unsorted.zarr   csr  row   slice  ...    2008   995     0.1  1.204  0.152   1.052 240.8x
```

Read it off `runs` + `chunks` + the `io_s`/`cpu_s` split: `slice` on a low-`runs` (sorted)
store is the win; a high `runs` count means the cells are scattered — use `fancy` there
(`slice` re-fetches shared chunks per run and blows up, as the last row shows).
