---
name: dask
description: >-
  Use when building or tuning dask.array pipelines in this repo — `da.store`, `da.from_delayed`,
  `da.from_array`, `da.concatenate`, `rechunk`, region writes, choosing a scheduler
  (threads vs synchronous), setting `num_workers`, ProgressBar, or diagnosing why a parallel
  write isn't scaling / is using too much RAM. Invoke before changing any dask call in the
  streaming writers. Verifies against vendored dask 2026.3.0 source.
---

# Dask (dask.array, 2026.3.0) — streaming writes & parallelism

DataScale streams every large matrix to zarr through dask so the full dense/sparse matrix is
never materialized and writes run in parallel. The correctness of that streaming (chunk
alignment, scheduler choice, worker count) lives here.

## Where to look (ground truth)

- **Source:** `.claude/vendor/dask/src/dask/` (the package dir; dask has no upstream `src/`
  layout, so it was placed under `src/` here for consistency with the other vendored libs)
- **Docs:** `.claude/vendor/dask/docs/`
- **Installed fallback:** `pixi run python -c "import dask; print(dask.__version__)"`

| Question | File |
|----------|------|
| `store`, `from_array`, `from_delayed`, `concatenate`, `Array` | `dask/array/core.py` |
| `rechunk` semantics | `dask/array/rechunk.py` |
| Slicing / region math | `dask/array/slicing.py` |
| Graph optimization / fusion | `dask/array/optimization.py` |
| Threaded scheduler | `dask/threaded.py` |
| Synchronous + base local scheduler | `dask/local.py` |
| `compute` / scheduler selection | `dask/base.py` |
| Global config | `dask/config.py` |
| `ProgressBar` | `dask/diagnostics/` |

### Grep recipes

```bash
V=.claude/vendor/dask/src/dask
rg -n "def store" $V/array/core.py        # regions=, lock=, compute=, return_stored=
rg -n "def from_delayed|def from_array|def concatenate" $V/array/core.py
rg -n "scheduler|num_workers" $V/base.py  # how scheduler= / num_workers= are resolved
```

## Project-relevant essentials

- **`da.store(sources, targets, regions=, lock=, scheduler=, num_workers=)`** is the write
  primitive. `scheduler=`/`num_workers=` flow through to `compute`. Confirmed params in
  2026.3.0: `sources, targets, lock, regions, compute, return_stored, load_stored, **kwargs`.
- **Scheduler rules (mandatory, not tuning):**
  - In-memory scipy sparse / ndarray → `scheduler="threads"`, `num_workers=cfg.chunks.cpus`.
    Works because the heavy ops (`.toarray()`, blosc) release the GIL.
  - Backed HDF5 (`_CSRDataset`) is **never threaded** (**h5py is not thread-safe**), but it is
    **not** run on dask at all in the single-file converter: it's parallelised across
    **processes** (`_run_parallel` in converter.py — each worker opens its own read-only handle
    and writes a chunk-aligned region, so no lock is needed). The multi-h5ad concat path
    (`_append_*`) is still `scheduler="synchronous"`, 1 worker.
- **`da.from_delayed`** is used (not `from_array`) for in-memory sparse tiling
  (`_build_tiled_dense_dask`) because scipy sparse blocks have no ndim / array protocol until
  densified. Each delayed block must declare correct `shape` and `dtype`.
- **2D tiling rule:** blocks must match the zarr `(row_chunk, col_chunk)` grid. See
  `_build_tiled_dense_dask` in [converter.py](../../../datascale/converter.py). A dask array
  chunked `(row_chunk, n_cols)` (full-width) is the *bug* we fixed — it causes
  read-modify-write and unbounded per-block RAM. Verify `.chunks` is 2D-tiled before storing.
- **Region writes:** `regions=(slice(r0, r1), slice(None))` writes one matrix into a slice of
  a larger pre-created zarr array (multi-h5ad concat). One target → pass the tuple directly.
- **ProgressBar:** `from dask.diagnostics import ProgressBar; with ProgressBar(out=sys.stderr, ...)`.
  Always to **stderr** (stdout stays clean).

## Reading for analysis (zarr → dask blocks)

The streaming writers are one direction; query/benchmark code reads back through dask, and the
block structure decides how much is read per query.

- **Dense:** `da.from_zarr(url, component=..., chunks=None)` (`dask/array/core.py`, ~L3737)
  builds a dask array whose blocks **default to the zarr chunk grid**. A row slice or a
  (scattered) gene-column selection only materializes the blocks it overlaps — so the zarr
  chunk shape directly sets read granularity *and* parallelism. Avoid `rechunk` on the read
  path; it shuffles data across blocks.
- **Sparse:** the dask array is built by anndata's `read_sparse_as_dask` (see the `anndata`
  skill) — **one CSR/CSC block per major-axis stride, minor axis whole**. Which query is cheap
  is fixed by the stored *format*, not by a dask kwarg.
- **Slicing** lives in `dask/array/slicing.py`; fancy/scattered indexing pulls whole blocks, so
  contiguous targets touch fewer blocks (ties to the row-locality lever in the `zarr`/`anndata`
  skills). These CPU blocks become GPU blocks via rapids `map_blocks` (see `rapids` skill).

```bash
V=.claude/vendor/dask/src/dask
rg -n "def from_zarr|def to_zarr" $V/array/core.py
rg -n "def slice_array|def take|def slice_wrap" $V/array/slicing.py
```

## Why a parallel write might not scale (checklist)

1. **Thread oversubscription** — dask `num_workers` × BLAS/OMP threads × Blosc threads. Pin
   `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`. This is the #1 silent killer.
2. **GIL-bound tasks** — pure-Python work in a delayed fn won't parallelize under `threads`.
3. **Chunk/grid misalignment** → serialized read-modify-write at the zarr layer.
4. **One giant block** — no parallelism + RAM spike. Check `arr.chunks` / `arr.numblocks`.
5. **Graph fusion collapsing** independent writes — inspect with `dask.visualize` if unsure.

## ✍️ Maintainer notes — ADD YOURS

<!-- TODO(you): fill in / delete prompts -->

- **Default `cfg.chunks.cpus` per environment** (laptop vs HPC node) and the matching BLAS thread caps:
- **Whether to ever use the distributed scheduler** (currently threads/synchronous only):
- **Memory budget per worker** you target, and how it maps to chunk size:
- **Vendored doc pages worth reading first** (e.g. array best-practices, scheduler docs):
