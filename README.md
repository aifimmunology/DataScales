# DataScale

A configurable converter for AnnData (`.h5ad`) to Zarr focused on single-cell gene matrices.

## Scope (v1)

- Supports non-spatial single-cell AnnData inputs.
- Rejects likely spatial AnnData inputs during validation.
- Supports config via TOML or YAML plus CLI overrides.

## Install

```bash
pixi init .
pixi install
```

## Quickstart

```bash
pixi run datascale convert \
  --input path/to/input.h5ad \
  --output path/to/output.zarr \
  --config path/to/config.toml
```

## Example config (TOML or YAML)
# see example_config.toml

```toml
[io]
overwrite = false
consolidate_metadata = true
x_storage = "auto" # auto | sparse | dense

[chunks]
x_row_chunk = 2048
x_col_chunk = 2048

[validation]
reject_spatial = true
require_non_empty = true
min_obs = 1
min_vars = 1
```

## Command help

```bash
datascale --help
datascale convert --help
```
