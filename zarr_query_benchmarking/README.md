# Zarr Query Benchmarking

Compare read speed across zarr storage strategies for single-cell AnnData stores.

A **request** says: which store, which axis to pull along (`obs` = cells/rows or
`var` = genes/cols), how many items to pull, contiguous vs. random selection, and
the final in-memory format (`dense` or `csr`). The tool **validates** the request
against the store, then **times** the full pull — including converting the pulled
data to the final format. Any format conversion (e.g. sparse→dense, or
dense→csr) is part of the timed region.

This isolates the cost of the *storage layout* (dense vs. CSR vs. CSC, chunk
shape) from AnnData overhead: reads go through **direct zarr slicing**, doing the
CSR/CSC index math by hand rather than via `anndata.read_zarr`.

## Streaming model

The benchmark mimics a tool that processes a large number of chunks **without
holding the whole selection in dense form**. Reads are streamed in **bands**
along the query axis: every chunk in the selection is touched, but only one band
is dense in memory at a time.

- **dense source** → bands are read at the store's native chunk extent along the
  query axis; each band is materialised and then released.
- **sparse source** → the selection is first assembled as a *compact* CSR matrix
  (no full densify), then densified one row-band at a time.

The `final_format` decides what is kept:

| final format | what happens | what's returned |
|--------------|--------------|-----------------|
| `dense` | each band is touched (sum + nnz) and **discarded** | a summary: shape, nnz, checksum, band count — the full dense block is never all in RAM |
| `csr`   | dense bands are sparsified as they arrive and accumulated; a sparse source stays sparse end-to-end | a single compact `scipy.sparse.csr_matrix` |

A *minor-axis* sparse query (e.g. `var` on a CSR store) must read every stored
nonzero — the inherent cost of the layout — but those stay sparse, and the dense
materialisation is still streamed. Sparse sources have no native row/col chunk
grid (their `data` is chunked along the flat nonzero array), so band size there
is derived from a 64 MB dense-footprint cap.

## Supported `X` encodings (Zarr v3, AnnData conventions)

| encoding-type | format | efficient pull | expensive pull |
|---------------|--------|----------------|----------------|
| `array`       | dense  | both axes      | —              |
| `csr_matrix`  | CSR    | rows (`obs`)   | cols (`var`) — reads full matrix |
| `csc_matrix`  | CSC    | cols (`var`)   | rows (`obs`) — reads full matrix |

The result is always oriented with the non-selected axis in full: an `obs` query
returns `(count, n_vars)`, a `var` query returns `(n_obs, count)`.

> Other storage formats can be added by extending the readers in
> [query.py](query.py); the request/validation/timing layers are format-agnostic.

## Usage

```bash
# Compare a CSR, CSC, and dense store pulling 1000 cells (rows)
pixi run python -m zarr_query_benchmarking \
  --store zarr_dbs/health_atlas_csr_1000.zarr \
  --store zarr_dbs/health_atlas_csc_1000.zarr \
  --store zarr_dbs/health_atlas_dense_1k_1k.zarr \
  --axis obs --count 1000 --mode contiguous --repeats 5

# Random sample of 500 genes (cols), write results to JSON
pixi run python -m zarr_query_benchmarking \
  --store zarr_dbs/health_atlas_csc_1000.zarr \
  --axis var --count 500 --mode random --seed 0 --output results.json

# Pull 1000 cells into a single compact CSR matrix (dense stores convert on the fly)
pixi run python -m zarr_query_benchmarking \
  --store zarr_dbs/health_atlas_csr_1000.zarr \
  --store zarr_dbs/health_atlas_dense_1k_1k.zarr \
  --axis obs --count 1000 --final-format csr

# Just report each store's layout (no timing)
pixi run python -m zarr_query_benchmarking --inspect \
  --store zarr_dbs/health_atlas_csr_1000.zarr
```

### Options

| flag | default | meaning |
|------|---------|---------|
| `--store PATH` | (required) | Store to query. Repeat to compare setups. |
| `--axis {obs,var}` | `obs` | Pull cells (rows) or genes (cols). |
| `--count N` | `1000` | Number of rows/cols to pull. |
| `--mode {contiguous,random}` | `contiguous` | Block or random sample. |
| `--offset N` | `0` | Start index for contiguous mode. |
| `--final-format {dense,csr}` | `dense` | Format produced before timing ends. |
| `--array PATH` | `X` | Node to query, e.g. `layers/counts`. |
| `--repeats N` | `5` | Timed runs (summary = min / median). |
| `--warmup N` | `1` | Untimed runs before timing. |
| `--seed N` | `0` | RNG seed for random mode. |
| `--output PATH` | — | Write full results as JSON. |
| `--json` | — | Print results as JSON to stdout. |
| `--inspect` | — | Report layout only; skip benchmarking. |

## A note on caching

The OS page cache and zarr's own caches make a second read of the same chunks
warm, so default runs measure *warm* reads. For **cold-cache** numbers, drop the
page cache between runs and use `--repeats 1 --warmup 0`:

```bash
# Linux (requires root)
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
# macOS
sync && sudo purge
```

## As a library

```python
from zarr_query_benchmarking import QueryRequest, benchmark_request, run_query

# Time a query
req = QueryRequest(store="zarr_dbs/health_atlas_csr_1000.zarr",
                   axis="obs", count=1000, mode="contiguous")
result = benchmark_request(req, repeats=5, warmup=1)
print(result.min_s, result.result_shape, result.n_bands)

# Or run once and keep the data (csr output returns a compact matrix)
res = run_query(QueryRequest(store="zarr_dbs/health_atlas_dense_1k_1k.zarr",
                             axis="obs", count=1000, final_format="csr"))
print(res.matrix.shape, res.nnz)   # res.matrix is a scipy.sparse.csr_matrix
```
