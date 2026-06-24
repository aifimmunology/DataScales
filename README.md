# DataScale

A configurable converter for single-cell gene expression data to Zarr stores, focused on non-spatial single-cell AnnData.

## Scope

- Converts `.h5ad` and 10x Genomics Cell Ranger `.h5` files to Zarr v3
- Rejects spatial AnnData inputs during validation
- Supports sparse (CSR/CSC) and dense output storage formats with configurable chunking
- Input expected to be CSR-formatted AnnData (`adata.X` in CSR) or it is converted to it
- Optional 'backed' HDF5 loading (`--backed`) streams X from disk without loading it into RAM — useful for large files or memory-constrained environments
- Optional [Icechunk](#icechunk-storage-backend) storage backend (`--icechunk`) — writes the store through a transactional, versioned repository instead of a plain zarr directory
- Optional [sort + partition](#sorted-stores---sort-by) of rows by obs column(s) (`--sort-by`) — physically groups each key tuple into a contiguous row range for fast subset reads via `datascale.open_sorted`
- Config via TOML or YAML with CLI overrides; all options can also be passed as CLI flags

## Install

Requires Python 3.10+. Run commands in repository directory after cloning.

**pixi (recommended)** — handles all dependencies automatically:
```bash
pixi install
```

**pip** — with a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

**From source** — for contributors:
```bash
git clone https://github.com/yourname/DataScale.git
cd DataScale
pip install -e .
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
- All inputs must share the **same `obs` schema** — identical column names.
- `adata.X` in each file must be CSR or CSC (CSC is auto-converted to CSR).
- All `X` matrices must share the same dtype.

What gets written:

- `X` — concatenated along rows, written in the configured `--x-storage`
  format (`sparse-csr` or `dense`; `sparse-csc` is not supported here).
- `obs` — concatenated via `pandas.concat` (duplicate `obs_names` are kept as-is).
- `var` — taken from the first file (already verified identical across all files).
- Empty `obsm` / `varm` / `uns` / `obsp` / `varp` groups for anndata-zarr
  compatibility. **`layers`, `raw`, `obsm`, etc. are not concatenated.** Use
  `convert-h5ad` per-file if you need those.

`--backed` is recommended for large inputs: each file's `X` is streamed from
HDF5 instead of being fully loaded.

## Sorted stores (`--sort-by`)

`convert-h5ad` can physically **sort and partition** the output by one or more
`obs` columns. Rows are reordered so that every distinct key tuple becomes a
single contiguous row range, and a range table is written under
`uns/datascale_sort_index`. All obs-aligned arrays (`obs`, `obsm`, …) are
reordered consistently, so the store stays a valid AnnData.

```bash
pixi run datascale convert-h5ad \
  --input path/to/input.h5ad \
  --output path/to/sorted.zarr \
  --sort-by AIFI_L1 batch_id             # primary sort key first
```

Constraints: requires an **eager** (non-`--backed`) load and **`sparse-csr`**
storage.

Once written, query subsets without materialising the full matrix using the
importable reader:

```python
from datascale import open_sorted

store = open_sorted("path/to/sorted.zarr")          # or open_sorted("repo", icechunk=True)
store.groups()                                       # range table: key tuple → start/end
adata = store.select(AIFI_L1="T cell")               # one contiguous block → in-memory AnnData
adata = store.select(AIFI_L1="T cell", batch_id="B001")   # narrower sub-range
adata = store.select(batch_id="B001")                # cross-cut: gathered from runs + concatenated
```

Only the matching `X[start:end]` rows (plus the corresponding `obs`/`var`/`obsm`)
are read, so subsets of a large store come back fast.

## Icechunk storage backend

Pass `--icechunk` to write the output through an
[Icechunk](https://icechunk.io) repository — a transactional, versioned Zarr
store — instead of a plain zarr directory. The whole conversion is staged and
made durable as a **single commit** on the `main` branch.

```bash
pixi run datascale convert-h5ad \
  --input path/to/input.h5ad \
  --output path/to/repo \
  --icechunk
```

Notes:

- Available on all subcommands via `--icechunk`. The conversion always commits to `main`.
- Local storage only for now (GCS is scaffolded but not wired up).
- Requires an **eager** input (not compatible with `--backed`).
- Read an icechunk-backed sorted store with `open_sorted(path, icechunk=True)`.

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

[validation]
reject_spatial = true
require_non_empty = true
min_obs = 1
min_vars = 1

# Sort + partition X by obs columns (convert-h5ad, sparse-csr, eager only). When enabled,
# rows are physically sorted by sort_by (primary key first) so each distinct key tuple is a
# contiguous row range, recorded under uns/datascale_sort_index and queryable via
# datascale.open_sorted(...).select(...).
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
| `--consolidate-metadata` | Optional | False by default - Write consolidated zarr metadata. useful for remote stores |
| `--icechunk` | Optional | Write the output through an Icechunk repository (transactional, versioned) instead of a plain zarr directory. Commits to the `main` branch. Local storage; eager input only (not `--backed`). All subcommands |
| `--sort-by` | Optional (`convert-h5ad`) | Sort + partition rows by these obs column(s), primary key first (e.g. `--sort-by AIFI_L1 batch_id`). Each distinct key tuple becomes a contiguous, queryable row range. Requires eager load + `sparse-csr` |

```bash
pixi run datascale convert-h5ad --help
pixi run datascale convert-10x-h5 --help
pixi run datascale concat-h5ads --help
```

## Development

```bash
pixi run -e dev pytest tests/ -v
```
