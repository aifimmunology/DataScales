# DataScale

A configurable converter for single-cell gene expression data to Zarr stores, focused on non-spatial single-cell AnnData.

## Scope

- Converts `.h5ad` and 10x Genomics Cell Ranger `.h5` files to Zarr v3
- Rejects spatial AnnData inputs during validation
- Supports sparse (CSR/CSC) and dense output storage formats with configurable chunking
- Input must be CSR-formatted AnnData (`adata.X` in CSR); backed HDF5 loading is used automatically
- Config via TOML or YAML with CLI overrides

## Install

Requires Python 3.10+.

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

## Config (TOML or YAML)

See `example_config.toml` for a full reference. Key options:

```toml
[io]
overwrite = false
consolidate_metadata = false
# x_storage: "sparse-csr" (default), "sparse-csc" (force CSC), "dense" (force dense)
x_storage = "sparse-csr"

[chunks]
#Chunk size for 2d dense arrays
x_row_chunk = 1000
x_col_chunk = 1000
#Tune 1d Shunk size for sparse array storage. #Reccomended to tune to median nnz per row.
sparse_flat_chunk = 4096

[validation]
reject_spatial = true
require_non_empty = true
min_obs = 1
min_vars = 1
```

## CLI options

All commands share the same optional flags:

| Flag | Description |
|---|---|
| `--config` | Path to TOML/YAML config file |
| `--overwrite` | Overwrite output path if it already exists |
| `--x-storage` | `sparse-csr` (default) \| `sparse-csc` \| `dense` |
| `--x-row-chunk` | Row chunk size for dense X |
| `--x-col-chunk` | Column chunk size for dense X |
| `--sparse-flat-chunk` | Flat array chunk size for sparse X |
| `--consolidate-metadata` | Force zarr metadata consolidation after write |

```bash
pixi run datascale convert-h5ad --help
pixi run datascale convert-10x-h5 --help
```

## Development

```bash
pixi run -e dev pytest tests/ -v
```
