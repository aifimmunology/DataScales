# DataScale

A configurable converter for single-cell gene expression data to Zarr stores, focused on non-spatial single-cell AnnData.

## Scope

- Converts `.h5ad` and 10x Genomics Cell Ranger `.h5` files to Zarr v3
- Rejects spatial AnnData inputs during validation
- Supports sparse (CSR/CSC) and dense output storage formats with configurable chunking
- Input expected to be CSR-formatted AnnData (`adata.X` in CSR) or it is converted to it
- Optional 'backed' HDF5 loading (`--backed`) streams X from disk without loading it into RAM — useful for large files or memory-constrained environments
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

## Config (TOML or YAML)

See `example_config.toml` for a full reference. Key options:

```toml
[io]
overwrite = false
consolidate_metadata = false
# x_storage: Zarr output storage format: "sparse-csr" (default), "sparse-csc" (force CSC), "dense" (force dense). 
x_storage = "sparse-csr"

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

```bash
pixi run datascale convert-h5ad --help
pixi run datascale convert-10x-h5 --help
pixi run datascale concat-h5ads --help
```

## Development

```bash
pixi run -e dev pytest tests/ -v
```
