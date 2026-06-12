# Zarr Query Benchmarking

Compare read speed across zarr storage strategies for single-cell AnnData stores.

A **request** says: which store, which axis to pull along (`obs` = cells/rows or
`var` = genes/cols), how many items to pull, contiguous vs. random selection, and
the final in-memory format. The tool **validates** the request against the store,
then **times** the full pull — including converting the pulled data to the final
format. If the store is sparse and the final format is dense, the
sparse→dense conversion is part of the timed region.

This isolates the cost of the *storage layout* (dense vs. CSR vs. CSC, chunk
shape) from AnnData overhead: reads go through **direct zarr slicing**, doing the
CSR/CSC index math by hand rather than via `anndata.read_zarr`.

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
| `--final-format {dense}` | `dense` | Format produced before timing ends. |
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
from zarr_query_benchmarking import QueryRequest, benchmark_request

req = QueryRequest(store="zarr_dbs/health_atlas_csr_1000.zarr",
                   axis="obs", count=1000, mode="contiguous")
result = benchmark_request(req, repeats=5, warmup=1)
print(result.min_s, result.result_shape)
```
