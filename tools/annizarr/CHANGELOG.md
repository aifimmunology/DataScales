# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Breaking:** `--backed` / `io.backed` renamed `--lazy` / `io.lazy`; lazy is the default and
  `--eager` loads the whole input first.
- `cpus` defaults to all cores (`--cpus N` overrides); h5py reads align to the source chunk grid.
- Matrix I/O goes through one reader protocol: every input layout (dense, CSR, CSC; eager or
  lazy) writes every `--x-storage`, so lazy dense input streams to CSR/CSC instead of erroring and
  a multi-input `convert` accepts mixed h5ad/10x inputs of any layout, including CSC output.
- Lazy h5py input into an Icechunk repo streams through a read-ahead pipeline (reader processes
  feed the writer threads) instead of erroring.
- `--auto-shard` leaves a config file's `auto_shard = true` in force when omitted.

### Added

- `Repo(..., anonymous=True)` / `Repo.exists(..., anonymous=True)` for public buckets; open
  failures distinguish a missing repo from missing credentials and name the fix.
- Git-style reprs for `Repo`, `branches()`, `log()`, `tree()` (icechunk's commit graph) and the
  root group from `open_zarr()`, which anndata's `read_elem`/`write_elem` still accept.

### Removed

- **Breaking:** the `ic` CLI (`init|log|tree|checkout|cherrypick|copy`) and HEAD persistence
  (`annizarr_head`, `$ANNIZARR_HOME`, `$ANNIZARR_ORIGIN`). The branch lives on the `Repo` object:
  `Repo(path)` opens `main`, `branch=` pins, `checkout` switches in-process. `convert --ic`,
  `--branch` and `-m` on the ops are unchanged.
- `io.eager_max_bytes` and the size-based eager/lazy auto-select.

### Fixed

- `az.convert(adata)` on a backed AnnData with sparse output loaded the whole matrix.
- Output verification before the atomic rename / Icechunk commit now checks X's shape.
- zarr thread sizing is applied on every op, not only the first in a process.

## [0.1.0] - 2026-09-29

Merges `convert-to-zarr`, `zarrsmith`, and `scizarr_IC` into one package, `annizarr`.

### Added

- `convert`: h5ad/10x input(s) to an AnnData zarr (or Icechunk) store; content-sniffed format
  detection (`--from` to override); two or more inputs concatenate.
- `add-expr`: a log-normalized expression layer derived from CSR X.
- `rechunk`: rewrite one matrix element with new chunking; stream-copy the rest as-is.
- `sort`: physically sort a store's rows by obs column(s) into a new store.
- `append`: append another store's cells in place, with a loss plan for derived elements.
- `annizarr.ic.Repo`: git-like Icechunk version control from Python.
- `--branch`/`-m` on every write op; one Icechunk commit per op.
- `io.eager_max_bytes` auto-select between eager and streamed (backed) input.

### Changed

- **Breaking:** package/CLI renamed to `annizarr` (alias `anz`); old tool names
  (`convert-to-zarr`, `zarrsmith`, `scizarr_ic`) are gone, no forwarders.
- **Breaking:** flat CLI — `convert`, `add-expr`, `rechunk`, `sort`, `append`, `ic <subcommand>`
  replace the three tools' separate entry points.
- **Breaking:** `x_storage`/`--x-storage`/`--format` values are `csr`/`csc`/`dense`; the old
  `sparse-csr`/`sparse-csc` spellings are no longer accepted.
- **Breaking:** the Python API is keyword-only after the leading path argument and every op
  returns a frozen `OpResult(path, n_obs, n_vars, snapshot_id)` instead of `None`/ad hoc values.
  Warnings go through `logging`, not return values.
- **Breaking:** env var renamed `ANNIZARR_ORIGIN` (was the old tool's origin override); HEAD
  persists in an `annizarr_head` file (was the old tool's head file name); per-user sidecar now
  under `$ANNIZARR_HOME`.
- **Breaking:** `layers/gexp` records its normalization target under the `annizarr_target_sum`
  attr; the old `zarrsmith_target_sum` key is still read for backward compatibility.
- **Breaking:** `requires-python >=3.12` (raised from the old tools' floor — every installable
  `zarr>=3.2`/icechunk wheel already required it).

### Removed

- `scanpy` and `dask` dependencies — 10x HDF5 reading is a plain h5py reader; writers use a
  `ThreadPoolExecutor` (in-memory) / process pool (backed h5ad) instead of dask.
- The three tools' forwarding/compat CLIs and the `scizarr_ic` import name.
- `ic copy` to/from `gs://` (still supported for local paths and `s3://`).

[Unreleased]: https://github.com/aifimmunology/DataScales
[0.1.0]: https://github.com/aifimmunology/DataScales
