# convert-to-zarr

A configurable converter for h5 data(currently only single cell) to Zarr stores, non-spatial single-cell AnnData.

> **This README covers the convert-to-zarr tool.** The repo also ships a separate,
> CLI-only **[zarr query benchmark tool](../zarr-query-bench/README.md)** — used to time any
> query type (row/column, sequential/random/cell-type) against a Zarr store's `X` so you can
> compare setups (dense vs CSR/CSC, chunking, sharding). See its own README; everything below is
> about conversion.

> The CLI entrypoint is currently `datascale` (package rename to `convert-to-zarr` pending).

## Scope

- Converts `.h5ad` and 10x Genomics Cell Ranger `.h5` files to Zarr v3
- Rejects spatial AnnData inputs during validation
- Supports sparse (CSR/CSC) and dense output storage formats with configurable chunking and optional sharding for dense X (`--x-shard-factor`)
- Input expected to be CSR-formatted AnnData (`adata.X` in CSR) or it is converted to it (slower)
- Optional 'backed' HDF5 loading (`--backed`) streams X from disk without loading it into RAM — useful for large files or memory-constrained environments
- Optional Icechunk storage backend (`--icechunk`) — writes the store through a transactional, versioned repository instead of a plain zarr directory
- Optional sort + partition of rows by obs column(s) (`--sort-by`) — physically groups each key tuple into a contiguous block for fast subset reads; the output is a plain sorted AnnData (no datascale-specific metadata), so you derive the range from the sorted obs column and slice `X[start:end]` with stock anndata/zarr
- Config via TOML or YAML with CLI overrides; all options can also be passed as CLI flags

## Install

Requires Python 3.10+. Run commands from this tool's directory (`tools/convert-to-zarr`) after cloning.

**pixi (recommended)** — handles all dependencies automatically:
```bash
cd tools/convert-to-zarr
pixi install
```

## Commands

### Convert `.h5ad` to Zarr

```bash
pixi run datascale convert-h5ad \
  --input path/to/input.h5ad \
  --output path/to/output.zarr \
  --config example_config.toml
```

### Convert 10x Cell Ranger `.h5` to Zarr

```bash
pixi run datascale convert-10x-h5 \
  --input path/to/filtered_feature_bc_matrix.h5 \
  --output path/to/output.zarr \
  --config example_config.toml
```

### Concatenate multiple `.h5ad` files into one Zarr

Combines several `.h5ad` files (e.g. one per sample/batch) into a single Zarr
store along the `obs` axis (rows). Each file's `X` is streamed directly into
its slice of the output — the full concatenated matrix is never materialised in
memory.

```bash
pixi run datascale concat-h5ads \
  --inputs path/to/sample_A.h5ad path/to/sample_B.h5ad path/to/sample_C.h5ad \
  --output path/to/combined.zarr \
  --backed \
  --x-storage sparse-csr
```

Requirements (strict; conversion errors otherwise):

- All inputs must share the **same `var`** — identical gene names *and* order.
- **`obs` columns** — by default all inputs must share an identical `obs` schema
  (same column names). Pass `--obs-columns COL...` to instead keep only those
  columns: each input must *contain* them, `obs` is projected to exactly those (in
  the given order) and all other columns are dropped. Lets files with differing
  *extra* obs columns be joined; a coercion warning is emitted if a kept column has
  mixed dtypes across files (e.g. categoricals with differing categories → string).
- `adata.X` in each file must be CSR or CSC (CSC is auto-converted to CSR).
- All `X` matrices must share the same dtype.

# Argument passing

## Config (TOML or YAML)
See `example_config.toml` for a full reference. Key options:

```toml
[io]
overwrite = false
consolidate_metadata = false
# x_storage: Zarr output storage format: "sparse-csr" (default), "sparse-csc" (force CSC), "dense" (force dense). 
x_storage = "sparse-csr"
backed = false
# backend: "zarr" (default, plain on-disk) or "icechunk" (transactional/versioned repo,
# one commit per conversion). icechunk requires eager input (not backed) for now.
backend = "zarr"
icechunk_storage = "local"   # "gcs" is scaffolded but not wired up yet

[chunks]
#Chunk size for 2d dense arrays
x_row_chunk = 2048
x_col_chunk = 2048
#Tune 1d Shunk size for sparse array storage. #Reccomended to tune to median nnz per row.
sparse_flat_chunk = 1000000
#Pack dense X chunks into shards of (x_row_chunk, x_col_chunk) * factor. 1 = no sharding.
#Use >1 with small chunks to keep read granularity fine while cutting file/object count.
x_shard_factor = 1

[validation]
reject_spatial = true
require_non_empty = true
min_obs = 1
min_vars = 1

# Sort + partition X by obs columns (convert-h5ad, sparse-csr or dense, eager only). When
# enabled, rows are physically sorted by sort_by (primary key first) so each distinct key tuple
# is a contiguous block. The output is a plain sorted AnnData with no datascale-specific
# metadata: to read a subset, derive the contiguous range from the (now sorted) obs column and
# slice X[start:end] with stock anndata/zarr — no datascale dependency to query.
[grouping]
enabled = false
sort_by = ["AIFI_L1", "batch_id"]
```

## CLI options

All commands share the same optional flags:

| Flag | Required? | Description |
|---|---|---|
| `--input` | **Required** (`convert-h5ad`, `convert-10x-h5`) | Path to input file (`.h5ad` or `.h5`) |
| `--inputs` | **Required** (`concat-h5ads`) | Two or more input `.h5ad` paths, space-separated |
| `--output` | **Required** | Path to output `.zarr` directory |
| `--config` | Optional | Path to TOML/YAML config file |
| `--overwrite` | Optional | Overwrite output path if it already exists, else it throws error that folder already exists |
| `--x-storage` | Optional | `sparse-csr` (default) \| `sparse-csc` \| `dense` (note: `concat-h5ads` does not support `sparse-csc`) |
| `--backed` | Optional | Stream X from disk without loading into RAM. Available on `convert-h5ad` and `concat-h5ads`. Saves peak memory at a small speed cost |
| `--cpus` | Optional | Threads for parallel matrix chunk writes (dense + sparse). No effect with `--backed` |
| `--x-row-chunk` | Optional | Row chunk size for dense X (auto-capped at 64 MB per chunk) |
| `--x-col-chunk` | Optional | Column chunk size for dense X |
| `--sparse-flat-chunk` | Optional | Flat array chunk size for sparse arrays. Best tuned to median nnz per row |
| `--x-shard-factor` | Optional | Pack dense X chunks into shards of `(x_row_chunk, x_col_chunk)` × factor. `1` (default) = no sharding. Use >1 with small chunks to keep read granularity fine while cutting file/object count (dense X only) |
| `--consolidate-metadata` | Optional | False by default - Write consolidated zarr metadata. useful for remote stores |
| `--icechunk` | Optional | Write the output through an Icechunk repository (transactional, versioned) instead of a plain zarr directory. Commits to the `main` branch. Local storage; eager input only (not `--backed`). All subcommands |
| `--sort-by` | Optional (`convert-h5ad`) | Sort + partition rows by these obs column(s), primary key first (e.g. `--sort-by AIFI_L1 batch_id`). Physically sorts rows so each distinct key tuple is a contiguous block; output is a plain sorted AnnData (no datascale index) — derive ranges from the sorted obs column and slice `X[start:end]` with stock anndata/zarr. Requires eager load + `sparse-csr` or `dense` |
| `--obs-columns` | Optional (`concat-h5ads`) | obs columns to keep and join on (e.g. `--obs-columns cell_type donor`). Omitted = require an identical obs schema across all inputs. When given, each input must contain these columns; `obs` is projected to exactly these (in this order) and all other columns are dropped |

```bash
pixi run datascale convert-h5ad --help
pixi run datascale convert-10x-h5 --help
pixi run datascale concat-h5ads --help
```

## Development

```bash
pixi run -e dev pytest tests/ -v
```

## License

Released under the [MIT License](../../LICENSE).
