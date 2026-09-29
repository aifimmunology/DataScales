# AnniZarr

Convert, edit, and version AnnData Zarr stores — streaming, memory-bounded, and
Icechunk-versioned.

## Install

Not yet on PyPI — install from this directory:

```bash
pip install .                # core: convert, add-expr, rechunk, sort, append
pip install ".[icechunk]"    # + Icechunk-backed stores (versioning, branches, S3/GCS)
pip install ".[s3]"          # + `ic copy` to/from s3://
pip install ".[all]"         # icechunk + s3
```

## Quickstart

```bash
# convert: format is sniffed from HDF5 contents (h5ad vs 10x); override with --from
annizarr convert sample.h5ad -o sample.zarr

# two or more inputs = concat
annizarr convert a.h5ad b.h5ad -o merged.zarr

# add a log-normalized layer (layers/gexp) derived from CSR X
annizarr add-expr merged.zarr --format csr

# rewrite X with new chunking; everything else streams through unchanged
annizarr rechunk merged.zarr -o rechunked.zarr --x-row-chunk 2000

# physically sort rows by obs column(s) into a new store
annizarr sort merged.zarr -o sorted.zarr --by cell_type

# append cells in place; errors if it would drop obsm/obsp/layers unless you consent
annizarr append sorted.zarr more_cells.zarr --drop-derived -y

# --- Icechunk: same ops, but each one lands as a commit ---
annizarr convert sample.h5ad -o repo.icechunk --ic -m "initial import"
annizarr add-expr repo.icechunk --format csr -m "add lognorm layer" --branch dev
annizarr ic log repo.icechunk --oneline
annizarr ic checkout repo.icechunk main -b
annizarr ic cherrypick repo.icechunk <snapshot-id>
annizarr ic copy repo.icechunk s3://bucket/prefix
```

## Python API

```python
import annizarr as az

result = az.convert("sample.h5ad", output="sample.zarr")        # -> OpResult
az.add_expr("sample.zarr", fmt="csr")
plan = az.plan_append("sample.zarr", cells="more_cells.zarr")   # pure: metadata only
az.append("sample.zarr", cells="more_cells.zarr", drop_derived=True)

repo = az.Repo("repo.icechunk")   # git-like: branch, checkout, log, tree, cherrypick, copy
repo.log(branch="main")

from annizarr import sources
sources.register_source("my_kind", my_loader, sniffer=my_sniffer)  # extension point
```

`OpResult(path, n_obs, n_vars, snapshot_id)` — `snapshot_id` is `None` for plain zarr,
the committed Icechunk snapshot id otherwise.

## How stores are written

Every store is **anndata-readable** zarr v3: root/array/sparse-group `encoding-type` /
`encoding-version` attrs are set by hand to match anndata 0.12.x's on-disk spec. Writes stream
in chunk-aligned bands; a matrix loads eagerly only when its on-disk size is under
`io.eager_max_bytes` (default 2 GiB) — above that, every op streams instead, bounded by memory
regardless of store size. `convert`/`rechunk`/`sort` write a new store to a sibling
`OUT.tmp-<uuid>`, verify it opens, then atomically rename onto the target — a killed run never
leaves a partial store at the destination. On Icechunk, each op is exactly one commit.
Idempotency: `add-expr` on a store that already has the layer errors unless `--overwrite`;
`append` errors if every appended obs name is already present ("already appended?") and only
warns on partial overlap (barcodes legitimately collide across samples). `--x-storage
csr|csc|dense` picks X's on-disk layout — CSR for row-wise/cuML access, CSC/dense for column
queries.

## Configuration

Defaults < config file < CLI flags. See [`example_config.toml`](example_config.toml) for every
`[io]`/`[chunks]`/`[validation]`/`[grouping]`/`[concat]` key; pass it with `--config FILE`.

## Compatibility

| Requirement | Constraint | Why |
|---|---|---|
| Python | `>=3.12` | every `zarr>=3.2` release and every icechunk wheel require it |
| anndata | `>=0.12.10,<0.13` | a phantom `None` layer key on `>=0.13` breaks the layer writer (upstream issue TBD) |
| zarr | `>=3.3,<4`, v3 only | uses v3-only APIs (`create_array`, `shards=`, `compressors=`) |
| icechunk | optional, `>=2.1.2,<3` | `pip install "annizarr[icechunk]"` |
| S3 | optional, `boto3>=1.28` | `pip install "annizarr[s3]"`; `ic copy` supports local and `s3://` only |
| GCS | `gs://` repos open/read/write via icechunk | no `ic copy` to/from `gs://` |

## Development

```bash
pixi install
pixi run -e default pytest              # excludes -m slow by default
pixi run -e default pytest -m slow      # large synthetic-store memory-ceiling test
pixi run -e default ruff check .
pixi run -e default mypy --strict src/annizarr
pre-commit run --all-files
```

Golden stores for regression tests live in `tests/golden/`.

## License

MIT — see [LICENSE](LICENSE).
