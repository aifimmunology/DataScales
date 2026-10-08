# zarr-query-bench

CLI tool to benchmark **query time** against a Zarr store's `X` matrix (dense or sparse
CSR/CSC). It times *read selection + convert to the requested final format*, so comparisons
across layouts are fair (a `dense` query from a CSR store pays the densify cost, and vice versa).

> Findings produced with this tool (dense vs sparse, row vs column axis, sorted vs unsorted,
> read threading) live in
> [`benchmarking_results/zarr_layouts/`](../../benchmarking_results/zarr_layouts/README.md).

Run via pixi (`python` is not on PATH directly):

```bash
pixi run zarr-bench --store <path.zarr> --axis row --count 1000 --format csr
pixi run zarr-bench-inspect --store <path.zarr>     # layout only, no timing
```

Output is a text summary, or one JSON object with `--json` — the raw per-run timings (wall,
plus an I/O / decompress / convert split) and full provenance (versions, shape, dtype, chunks,
codec, git commit) for later comparison.

## Arguments

| Flag | Values | Default | Meaning |
|------|--------|---------|---------|
| `--store` | path \| URL | (required) | `.zarr` store to query. Local path or fsspec URL (`gs://…`, `s3://…`). |
| `--axis` | `row` \| `col` | (required¹) | Query rows (cells) or columns (genes). |
| `--count` | int | (required¹) | Number of rows/columns to select. |
| `--mode` | `sequential` \| `random` \| `celltype` | `sequential` | `sequential` = first N; `random` = N seeded indices; `celltype` = all rows matching an obs value (see below). |
| `--obs-column` / `--obs-value` | str | (required for `celltype`) | obs column + value to filter rows by. |
| `--format` | `csr` \| `dense` | (required¹) | Final format the result is converted to (**included in the timing**). |
| `--native` | flag | off | Read at the store's native format — no conversion. Measures the layout, not the format tax. Overrides `--format`. |
| `--concurrency` | int | 10 | zarr `async.concurrency` — chunks dispatched at once. Pair with `--max-workers`. |
| `--max-workers` | int | `min(32, cpu+4)` | zarr `threading.max_workers` — the Blosc/zstd **decode** pool. Set ≈ physical cores. |
| `--repeats` / `--warmup` | int | 5 / 1 | Timed repeats (median/min/p95) / warmup runs discarded. |
| `--seed` | int | 0 | RNG seed for `--mode random`. |
| `--json` / `--inspect` | flag | off | Emit JSON instead of text / print layout and exit. |

¹ `--axis` + `--count` are required for `sequential`/`random`. `celltype` forces `--axis row`,
ignores `--count`, and needs `--obs-column` + `--obs-value`. Either `--format` or `--native` is required.

## Usage

**Select by cell type** — reads `obs[<column>]`, builds the `== <value>` mask, and fetches all
matching rows' `X` by slicing each contiguous run of matched rows (`X[start:end]`, anndata's fast
path; the mask build is inside the timed region, modeling a real "filter then fetch"):

```bash
pixi run zarr-bench --store <path.zarr> --mode celltype \
    --obs-column AIFI_L1 --obs-value Platelet --format csr
```

The `runs` metric reports locality: a store **sorted** by that column yields a few long runs (one
cheap grab per run); an unsorted store scatters into many short runs, so the per-run slices
re-fetch shared chunks and the query degrades — a warning fires when that happens.

**Remote stores** — a URL with a scheme is opened through zarr's `FsspecStore`. `gs://` needs
`gcsfs` (a pixi dep) and uses gcloud Application Default Credentials — run
`gcloud auth application-default login` once; no token is passed in code.

```bash
pixi run zarr-bench --store gs://my-bucket/atlas_csr.zarr --axis row --count 1000 --format csr --json
```

**Sweep many stores** — `./run_query_sweep.sh` runs both axes over
every `.zarr` in a directory (or `gs://` prefix) and appends one JSON line per run:

```bash
# [STORE_DIR] [COUNT] [FORMAT] [THREAD_CONCURRENCY] [MODE] [OUT]
./run_query_sweep.sh zarr_dbs 1000 csr 32 sequential bench.jsonl
```

For an explicit few-stores × few-parameters matrix instead (edit the CONFIG block, run),
use `./sweep_stores.sh`.

**Compare runs** — `compare` reads `--json` files (a single object, an array, or JSON Lines) and
prints an aligned table, sortable, with a trailing `xslow` column (each run's median vs the fastest):

```bash
pixi run python -m zarr_query_bench.compare bench.jsonl --sort median_s
pixi run python -m zarr_query_bench.compare 'runs/*.json' --md > table.md   # markdown
```

## Notes

**Two knobs for parallelism.** A dense read needs *both* zarr settings; raising one alone pins you
near 1 core. `--concurrency` only *dispatches* chunks (a semaphore, no CPU); `--max-workers` sizes
the pool that runs Blosc/zstd **decode** — the dominant cost of a warm dense read, and it scales
across cores. Set both, e.g. `--concurrency 64 --max-workers <physical cores>`.

**Fair dense-vs-sparse comparison.** `--format` is *included in the timing*, so a common `--format`
charges the CSR store a `.toarray()` (and a dense store a `.tocsr()`) that real rapids-singlecell +
dask analysis never pays. For an analysis-faithful read, pass **`--native`** so each store is read
in its own layout (no conversion), then compare on peak RSS and the store-engine cost rather than
the format tax. Query on `--axis row` (the aligned axis for CSR row-block pipelines).

**Cold vs warm.** Timed repeats are warm (OS/gcsfs cache hot after warmup). For a cold first-touch
number, run `--warmup 0 --repeats 1` in a fresh process. Object stores can't be page-cache-dropped
from userspace, so on `gs://` the warm number reflects client caching, not a cold network read.

> Cross-axis sparse queries (e.g. `--axis col` on a CSR store) are slow by design — keep `--count`
> modest when sweeping sparse stores on both axes.
