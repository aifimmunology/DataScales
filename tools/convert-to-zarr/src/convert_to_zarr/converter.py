from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import anndata as ad
import scipy.sparse as sp
import zarr

from .config import AppConfig
from .storage import open_output_store
from .validation import validate_single_cell_anndata


class ConversionError(RuntimeError):
    """Raised when conversion cannot be completed."""


def _x_compressors():
    """Blosc(zstd) + byte-shuffle. zarr's default is bare zstd level-0 with no
    shuffle; the shuffle gives a large ratio/throughput win on numeric matrices."""
    from zarr.codecs import BloscCodec
    return (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)


def _dense_shards(row_chunk, col_chunk, n_rows, n_cols, factor):
    """Resolve the zarr v3 shard shape and the write-block shape for a dense X array.

    With sharding on (``factor`` > 1) the inner chunk stays (row_chunk, col_chunk) — that
    remains the read granularity — and many inner chunks are packed into one shard object,
    cutting file/object count. zarr requires the shard shape to be an integer multiple of the
    inner chunk shape, so the shard is ``chunk * factor`` per axis, capped at the number of
    chunks the array actually spans (no point in a shard reaching far past the data).

    Returns ``(shards, block_row, block_col)`` where ``shards`` is the shards= kwarg (None when
    no sharding) and (block_row, block_col) is the granularity callers must write at. Writing a
    *partial* shard makes zarr's sharding codec read-modify-write the whole shard (silent perf
    killer #2), so the block shape equals the shard shape when sharding is on, and the inner
    chunk shape otherwise. Peak dense RAM per write block therefore grows by ~factor**2 when
    sharding — the documented cost of fewer, larger objects.
    """
    if factor <= 1:
        return None, row_chunk, col_chunk
    import math
    rf = min(factor, math.ceil(n_rows / row_chunk))
    cf = min(factor, math.ceil(n_cols / col_chunk))
    shard_row = row_chunk * rf
    shard_col = col_chunk * cf
    return (shard_row, shard_col), shard_row, shard_col


# ── Backed-sparse parallel workers ───────────────────────────────────────────
# These run in separate processes (h5py is not thread-safe, but independent
# read-only file handles across processes are). They are module-level so the
# process pool can pickle them. Each worker writes a chunk-aligned region, so
# no two workers ever touch the same zarr chunk and no lock is needed.

def _copy_sparse_segment(out_root, data_path, indices_path, src_file, src_group,
                         s0, s1, indices_dtype):
    """Copy a chunk-aligned nnz segment [s0:s1) from a backed sparse h5ad to zarr.

    CSR→CSR / CSC→CSC keeps row/col order, so source and output flat positions
    map 1:1 — this is a straight flat copy, no scipy needed.
    """
    import h5py
    import numpy as np
    import zarr
    from zarr.storage import LocalStore

    with h5py.File(src_file, "r") as f:
        g = f[src_group]
        data = g["data"][s0:s1]
        indices = np.asarray(g["indices"][s0:s1], dtype=indices_dtype)
    root = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")
    root[data_path][s0:s1] = data
    root[indices_path][s0:s1] = indices


def _densify_band_segment(out_root, data_path, src_file, src_group, r0, r1,
                          col_chunk, n_cols):
    """Read CSR row band [r0:r1) from a backed sparse h5ad and write it dense,
    one column tile at a time so dense RAM is bounded by one chunk."""
    import zarr
    import h5py
    from anndata.io import sparse_dataset
    from zarr.storage import LocalStore

    with h5py.File(src_file, "r") as f:
        band = sparse_dataset(f[src_group])[r0:r1]
    arr = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")[data_path]
    for c0 in range(0, n_cols, col_chunk):
        c1 = min(c0 + col_chunk, n_cols)
        arr[r0:r1, c0:c1] = band[:, c0:c1].toarray()


def _run_parallel(worker, jobs, cpus):
    """Run worker(*job) for each job — in a process pool when cpus>1, else inline.
    With a single job the pool would be pure spawn overhead, so run inline."""
    if cpus <= 1 or len(jobs) <= 1:
        for job in jobs:
            worker(*job)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=cpus) as ex:
        for fut in [ex.submit(worker, *job) for job in jobs]:
            fut.result()


@contextmanager
def _stage(label: str):
    """Print a labelled progress line with elapsed time. Flushes immediately."""
    print(f"→ {label} ...", flush=True, file=sys.stderr)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True, file=sys.stderr)


def _resolve_backend_cfg(cfg: AppConfig) -> AppConfig:
    """Validate + adapt config for the chosen storage backend.

    The icechunk backend writes through an in-process session; the backed-input writers
    fan out to worker processes that reopen the store by filesystem path, which an
    IcechunkStore can't provide — so backed + icechunk is rejected for now. To keep the
    single-session write thread-safe we also force a single worker for icechunk.
    """
    if cfg.chunks.x_shard_factor > 1 and cfg.io.x_storage != "dense":
        print(
            f"→ x_shard_factor={cfg.chunks.x_shard_factor} only applies to dense X; "
            f"x_storage={cfg.io.x_storage!r} is sparse, so sharding is ignored.",
            flush=True, file=sys.stderr,
        )
    if cfg.io.backend != "icechunk":
        return cfg
    if cfg.io.backed:
        raise ConversionError(
            "backend='icechunk' does not support --backed input yet (backed writers use "
            "worker processes that reopen the store by path; the icechunk session is "
            "in-process only). Convert eagerly (omit --backed), or use backend='zarr'."
        )
    if cfg.chunks.cpus > 1:
        print(
            f"→ icechunk backend: forcing cpus=1 (was {cfg.chunks.cpus}; single-session "
            "writes are not parallelised yet).",
            flush=True, file=sys.stderr,
        )
        cfg = replace(cfg, chunks=replace(cfg.chunks, cpus=1))
    return cfg


def _load_h5ad_for_conversion(input_path: Path, cfg: AppConfig) -> tuple[ad.AnnData, list[str]]:
    mode = "backed (streaming)" if cfg.io.backed else "eager (full load)"
    with _stage(f"Reading {input_path} [{mode}]"):
        if cfg.io.backed:
            try:
                return ad.read_h5ad(input_path, backed="r"), []
            except Exception as exc:
                raise ConversionError(
                    f"Backed load failed for {input_path} ({type(exc).__name__}: {exc}). "
                    "Remove --backed / set backed=false to load eagerly."
                ) from exc
        return ad.read_h5ad(input_path), []



def _build_tiled_dense_dask(matrix: Any, row_chunk: int, col_chunk: int) -> Any:
    """Build a 2D-tiled dask array from a (backed or in-memory) CSR matrix.

    Each block is exactly (row_chunk × col_chunk), matching the zarr chunk grid so
    da.store() writes whole chunks (no read-modify-write) and never materialises more
    than one chunk-sized dense block per worker — peak RAM is bounded by the chunk
    size, independent of total column count.

    A single delayed row-slice (sparse CSR) is shared across that band's column tiles,
    so each row band is sliced from source storage only once. da.from_delayed is used
    (not da.from_array) because backed _CSRDataset has no ndim / array protocol.
    """
    import numpy as np
    import dask
    import dask.array as da

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    def _row_band(r0: int, r1: int):
        # Row-slicing returns scipy CSR for both backed and in-memory inputs.
        return matrix[r0:r1]

    row_blocks = []
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        band = dask.delayed(_row_band)(r0, r1)  # sparse; computed once, reused per tile
        col_blocks = []
        for c0 in range(0, n_cols, col_chunk):
            c1 = min(c0 + col_chunk, n_cols)
            col_blocks.append(da.from_delayed(
                dask.delayed(lambda b, a=c0, z=c1: np.asarray(b[:, a:z].toarray()))(band),
                shape=(r1 - r0, c1 - c0),
                dtype=dtype,
            ))
        row_blocks.append(da.concatenate(col_blocks, axis=1))
    return da.concatenate(row_blocks, axis=0)


def _write_sparse_as_dense_dask(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Write a CSR matrix (in-memory or backed _CSRDataset) as a dense zarr array.

    Blocks match the zarr chunk grid (row_chunk × col_chunk) so the full dense
    matrix is never materialised. In-memory CSR streams through a 2D-tiled dask
    array (cfg.chunks.cpus threads). Backed _CSRDataset is written by row band in
    parallel processes (each opens its own h5py handle; bands are chunk-aligned).
    """
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    shards, block_row, block_col = _dense_shards(
        row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor
    )

    zarr_arr = group.require_array(
        key,
        shape=(n_rows, n_cols),
        dtype=dtype,
        chunks=(row_chunk, col_chunk),
        shards=shards,
        compressors=_x_compressors(),
        overwrite=True,
    )
    zarr_arr.attrs["encoding-type"] = "array"
    zarr_arr.attrs["encoding-version"] = "0.2.0"

    backed = not hasattr(matrix, "indices")  # backed _CSRDataset lacks .indices
    if backed:
        # Bands are block_row tall and densified in block_col-wide tiles so each write
        # covers whole shards (or whole chunks when unsharded) — no read-modify-write,
        # and disjoint bands never share a shard so the parallel workers need no lock.
        src = matrix.group
        out_root = zarr_arr.store_path.store.root
        jobs = [
            (out_root, zarr_arr.store_path.path, src.file.filename, src.name,
             r0, min(r0 + block_row, n_rows), block_col, n_cols)
            for r0 in range(0, n_rows, block_row)
        ]
        _run_parallel(_densify_band_segment, jobs, cfg.chunks.cpus)
        return

    dask_dense = _build_tiled_dense_dask(matrix, block_row, block_col)
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        # lock=False: this single-file path tiles from row 0 with block == the shard shape
        # (_dense_shards), so every da.store task writes one whole, disjoint shard — no shared
        # chunk, so no write lock is needed. dask's default lock=True serializes the
        # compress+write and pins the threaded path to ~1 core (measured ~6.5x slower on the
        # unsharded dense path). NOTE: do NOT copy lock=False to the concat/_append_* paths —
        # those write at a misaligned row/nnz offset and read-modify-write the seam chunk, so
        # they must keep the lock or concurrent writes corrupt data.
        da.store(dask_dense, zarr_arr, scheduler="threads",
                 num_workers=cfg.chunks.cpus, lock=False)


def _write_dense_streaming(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Stream an already-dense matrix to a dense zarr array via da.store.

    anndata's write_elem assigns the whole array at once (full materialisation);
    da.from_array + da.store writes chunk-by-chunk to the zarr grid instead.
    """
    import numpy as np
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    shards, block_row, block_col = _dense_shards(
        row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor
    )

    zarr_arr = group.require_array(
        key, shape=(n_rows, n_cols), dtype=matrix.dtype,
        chunks=(row_chunk, col_chunk), shards=shards,
        compressors=_x_compressors(), overwrite=True,
    )
    zarr_arr.attrs["encoding-type"] = "array"
    zarr_arr.attrs["encoding-version"] = "0.2.0"

    # Block to the shard grid (or chunk grid when unsharded) so each da.store write covers
    # whole shards and never triggers a read-modify-write of a partial shard.
    backed = not isinstance(matrix, np.ndarray)  # h5py-backed dense isn't thread-safe
    arr = da.from_array(matrix, chunks=(block_row, block_col))
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            arr, zarr_arr,
            scheduler="synchronous" if backed else "threads",
            num_workers=1 if backed else cfg.chunks.cpus,
            lock=False,  # single-file, block==shard tiled from 0 -> disjoint whole-shard writes (see _write_sparse_as_dense_dask)
        )


def _write_matrix_direct(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Write a single matrix to zarr in the target format, directly.

    Input may be any format (CSR, CSC, dense ndarray, backed SparseDataset) -> because vvv
    NOTE adata.X is guaranteed CSR by the caller, but other layers and raw.X may not be.

    Dispatches based on cfg.io.x_storage:
      dense      → dask row-chunked write; CSC converted to CSR first for row slicing
      sparse-csr → ensure CSR, stream row-batches via dask (parallel for in-memory input)
      sparse-csc → ensure CSC, stream col-batches via dask (parallel for in-memory input)
    """
    mode = cfg.io.x_storage

    # Backed SparseDataset (anndata HDF5-backed): has .format but is not a scipy sparse matrix.
    is_backed_sparse = not sp.issparse(matrix) and hasattr(matrix, "format")

    if mode == "dense":
        if sp.issparse(matrix) or is_backed_sparse:
            if is_backed_sparse and getattr(matrix, "format", None) != "csr":
                matrix = matrix.tocsr()  # backed CSC: load into memory and convert
            elif sp.issparse(matrix) and not sp.isspmatrix_csr(matrix):
                matrix = matrix.tocsr() 
            _write_sparse_as_dense_dask(group, matrix, key, cfg)
            
        else:  # already dense
            _write_dense_streaming(group, matrix, key, cfg)
        return

    # Sparse output (sparse-csr, sparse-csc).
    # For backed input whose format doesn't match the target, we have no choice but to
    # load into memory and convert (no incremental transpose). For matching format
    # (CSR→CSR or CSC→CSC), streaming works directly from backed storage.
    if is_backed_sparse:
        dataset_format = getattr(matrix, "format", None)
        if mode == "sparse-csc" and dataset_format != "csc":
            # _CSRDataset has no tocsc(); load into memory first then convert.
            matrix = matrix[:].tocsc()
        elif mode == "sparse-csr" and dataset_format != "csr":
            matrix = matrix[:].tocsr()
    elif mode == "sparse-csr" and sp.issparse(matrix) and not sp.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    elif mode == "sparse-csc" and sp.issparse(matrix) and not sp.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()

    _write_sparse_streaming(group, matrix, key, cfg, csr=(mode == "sparse-csr"))


def _write_sparse_streaming(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig, csr: bool
) -> None:
    """Stream a sparse matrix to zarr as a CSR/CSC group with per-batch dask writes.

    Reads ``indptr`` upfront (small, ~8B per row/col) to know exact output offsets.
    In-memory scipy sparse is written in row/col batches via dask threads. Backed
    _CSRDataset/_CSCDataset is the same format as the output, so its data/indices
    map 1:1 to the output — workers flat-copy chunk-aligned nnz segments in parallel
    processes (h5py is not thread-safe, but independent process handles are).
    """
    import numpy as np
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    # CSR iterates over rows; CSC iterates over columns.
    n_major = n_rows if csr else n_cols

    # indptr is small (~8B per row); load fully to compute exact offsets.
    # Backed _CSRDataset/_CSCDataset doesn't expose .indptr directly — read from
    # the underlying h5py group instead.
    if sp.issparse(matrix):
        indptr_full = np.asarray(matrix.indptr)
    elif hasattr(matrix, "indptr"):
        indptr_full = np.asarray(matrix.indptr[:])
    elif hasattr(matrix, "group"):
        indptr_full = np.asarray(matrix.group["indptr"][:])
    else:
        raise ConversionError(
            f"Cannot locate indptr on sparse input of type {type(matrix).__name__}"
        )
    nnz_total = int(indptr_full[-1])

    backed = not sp.issparse(matrix)  # backed _CSRDataset / _CSCDataset
    indices_dtype = np.int32  # match scipy default; values fit unless > 2^31 cols/rows

    # ── Create output group + arrays ──────────────────────────────────────────
    sp_group = group.require_group(key)
    sp_group.attrs["encoding-type"] = "csr_matrix" if csr else "csc_matrix"
    sp_group.attrs["encoding-version"] = "0.1.0"
    sp_group.attrs["shape"] = [n_rows, n_cols]

    flat_chunk = min(cfg.chunks.sparse_flat_chunk, max(1, nnz_total))
    data_arr = sp_group.require_array(
        "data", shape=(nnz_total,), dtype=matrix.dtype,
        chunks=(flat_chunk,), compressors=_x_compressors(), overwrite=True,
    )
    indices_arr = sp_group.require_array(
        "indices", shape=(nnz_total,), dtype=indices_dtype,
        chunks=(flat_chunk,), compressors=_x_compressors(), overwrite=True,
    )
    indptr_arr = sp_group.require_array(
        "indptr", shape=(n_major + 1,), dtype=indptr_full.dtype,
        chunks=(n_major + 1,), overwrite=True,
    )
    for a in (data_arr, indices_arr, indptr_arr):
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"

    # indptr is small — write it directly.
    indptr_arr[:] = indptr_full

    _TARGET_BATCH_BYTES = 256 * 1024 * 1024

    if backed:
        # Flat-copy chunk-aligned nnz segments in parallel processes. Segments are
        # multiples of flat_chunk, so each zarr chunk is owned by exactly one worker.
        src = matrix.group
        out_root = data_arr.store_path.store.root
        bytes_per_nnz = np.dtype(matrix.dtype).itemsize + np.dtype(indices_dtype).itemsize
        seg = max(1, _TARGET_BATCH_BYTES // (flat_chunk * bytes_per_nnz)) * flat_chunk
        jobs = [
            (out_root, data_arr.store_path.path, indices_arr.store_path.path,
             src.file.filename, src.name, s, min(s + seg, nnz_total), indices_dtype)
            for s in range(0, nnz_total, seg)
        ]
        _run_parallel(_copy_sparse_segment, jobs, cfg.chunks.cpus)
        return

    # ── In-memory: build dask arrays for data + indices via delayed row/col batches.
    # Auto-tune batch size: target ~256 MB per batch in RAM. Clamped to [1k, 200k] majors.
    avg_nnz_per_major = max(1, nnz_total // max(1, n_major))
    bytes_per_major = avg_nnz_per_major * (
        np.dtype(matrix.dtype).itemsize + np.dtype(indices_dtype).itemsize
    )
    batch_size = max(1_000, min(200_000, _TARGET_BATCH_BYTES // max(1, bytes_per_major)))
    batch_starts = list(range(0, n_major, batch_size))

    def _load_batch(m, b0: int, b1: int):
        # Slicing returns scipy sparse for both backed and in-memory inputs.
        return m[b0:b1] if csr else m[:, b0:b1]

    data_parts = []
    indices_parts = []
    nonempty = 0
    for b0 in batch_starts:
        b1 = min(b0 + batch_size, n_major)
        batch_nnz = int(indptr_full[b1] - indptr_full[b0])
        if batch_nnz == 0:
            continue
        nonempty += 1
        # One delayed batch shared by both data and indices to avoid loading twice.
        batch = dask.delayed(_load_batch)(matrix, b0, b1)
        data_parts.append(da.from_delayed(
            dask.delayed(lambda b: np.asarray(b.data))(batch),
            shape=(batch_nnz,), dtype=matrix.dtype,
        ))
        indices_parts.append(da.from_delayed(
            dask.delayed(lambda b: np.asarray(b.indices, dtype=indices_dtype))(batch),
            shape=(batch_nnz,), dtype=indices_dtype,
        ))

    if nonempty == 0:
        # all-zero matrix: indptr already written, data/indices are empty
        return

    data_dask = da.concatenate(data_parts)
    indices_dask = da.concatenate(indices_parts)

    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            [data_dask, indices_dask],
            [data_arr, indices_arr],
            scheduler="threads",
            num_workers=cfg.chunks.cpus,
        )


def _write_csr_adata_direct(
    adata: ad.AnnData,
    store: zarr.Group,
    cfg: AppConfig,
    x_override: Any | None = None,
) -> None:
    """Write an AnnData with CSR X (in-memory or backed SparseDataset) into ``store``.

    Matrices are written with the target format and chunking in one step. The caller
    owns opening the store and finalising it (consolidate / icechunk commit).
    """
    from anndata._io.specs import write_elem  # not part of the public API

    # Root encoding attrs — marks this as a valid anndata zarr store.
    store.attrs["encoding-type"] = "anndata"
    store.attrs["encoding-version"] = "0.1.0"

    # Metadata components — all small, written as one stage.
    with _stage("Writing metadata (obs/var/uns/obsm/varm/obsp/varp)"):
        write_elem(store, "obs", adata.obs)
        write_elem(store, "var", adata.var)
        write_elem(store, "uns", dict(adata.uns))
        write_elem(store, "obsm", dict(adata.obsm))
        write_elem(store, "varm", dict(adata.varm))
        write_elem(store, "obsp", dict(adata.obsp))
        write_elem(store, "varp", dict(adata.varp))

    # X: the heavy step — sub-progress comes from dask's ProgressBar inside.
    x_matrix = x_override if x_override is not None else adata.X
    x_nnz = getattr(x_matrix, "nnz", None)
    x_info = f"shape={x_matrix.shape}, {cfg.io.x_storage}"
    if x_nnz is not None:
        x_info += f", nnz={x_nnz}"
    with _stage(f"Writing X ({x_info})"):
        _write_matrix_direct(store, x_matrix, "X", cfg)

    if adata.layers:
        write_elem(store, "layers", {})
        layers_group = store["layers"]
        for name, data in adata.layers.items():
            with _stage(f"Writing layers/{name} (shape={data.shape})"):
                _write_matrix_direct(layers_group, data, name, cfg)

    if adata.raw is not None:
        raw_group = store.require_group("raw")
        raw_group.attrs["encoding-type"] = "raw"
        raw_group.attrs["encoding-version"] = "0.1.0"
        write_elem(raw_group, "var", adata.raw.var)
        write_elem(raw_group, "varm", dict(adata.raw.varm))
        with _stage(f"Writing raw/X (shape={adata.raw.X.shape})"):
            _write_matrix_direct(raw_group, adata.raw.X, "X", cfg)


def _close_backed_if_needed(adata: ad.AnnData) -> None:
    if getattr(adata, "isbacked", False):
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()


# =============================================================================
# Sort + partition by obs columns (Feature B)
# =============================================================================

def _compute_sort(adata: ad.AnnData, sort_by: tuple[str, ...]):
    """Compute the row permutation + contiguous range table for sorting by ``sort_by``.

    Returns ``(perm, ranges_df)`` where ``perm`` is the int64 permutation (original row
    position for each sorted position) and ``ranges_df`` has one row per distinct key tuple
    with the sort-key columns plus ``start``/``end`` (half-open) row offsets into the
    sorted store. Sort is lexicographic with ``sort_by[0]`` as the primary key, stable.
    """
    import numpy as np
    import pandas as pd

    obs = adata.obs
    missing = [c for c in sort_by if c not in obs.columns]
    if missing:
        raise ConversionError(
            f"grouping sort_by columns not found in obs: {missing}. "
            f"Available: {list(obs.columns)}"
        )

    keys = []
    for col in sort_by:
        codes, _ = pd.factorize(obs[col], sort=True)  # codes follow sorted value order
        if (np.asarray(codes) < 0).any():
            raise ConversionError(
                f"obs column '{col}' has missing (NaN) values; cannot sort by it."
            )
        keys.append(np.asarray(codes))

    # np.lexsort treats the LAST key as primary, so reverse to make sort_by[0] primary.
    perm = np.lexsort(keys[::-1]).astype(np.int64)

    obs_sorted = obs.iloc[perm]
    sizes = obs_sorted.groupby(list(sort_by), sort=False, observed=True).size()
    ends = np.cumsum(sizes.to_numpy())
    starts = ends - sizes.to_numpy()
    ranges = sizes.index.to_frame(index=False)
    ranges["start"] = starts.astype(np.int64)
    ranges["end"] = ends.astype(np.int64)
    return perm, ranges


def _maybe_sort_adata(
    adata: ad.AnnData, cfg: AppConfig, warnings: list[str]
) -> ad.AnnData:
    """If grouping is enabled, reorder all obs-aligned arrays by the sort keys.

    Reordering uses anndata fancy indexing so obs/obsm/obsp/layers/raw all share one
    permutation and the store stays a valid AnnData. No convert-to-zarr-specific index is written:
    the result is a plain, physically sorted AnnData, so each distinct key tuple is a
    contiguous row block that a downstream reader derives from the sorted obs column(s).
    """
    if not cfg.grouping.enabled:
        return adata

    sort_by = cfg.grouping.sort_by
    if cfg.io.x_storage not in ("sparse-csr", "dense"):
        raise ConversionError(
            f"grouping (sort_by) requires x_storage='sparse-csr' or 'dense'; "
            f"got '{cfg.io.x_storage}'."
        )
    if cfg.io.backed:
        raise ConversionError(
            "grouping (sort_by) requires an eager (in-memory) load; not supported with "
            "--backed yet. Omit --backed to sort."
        )

    perm, ranges_df = _compute_sort(adata, sort_by)
    with _stage(f"Sorting {adata.n_obs} cells by {list(sort_by)} ({len(ranges_df)} groups)"):
        adata = adata[perm].copy()  # reorders X/obs/obsm/obsp/layers/raw consistently
    warnings.append(
        f"Rows sorted by {list(sort_by)} into {len(ranges_df)} contiguous groups; "
        "obs/obsm/obsp/layers/raw reordered to match. Store is a plain sorted AnnData "
        "(no convert-to-zarr index); derive ranges from the sorted obs column(s) if needed."
    )
    return adata


def _write_sorted_backed(
    adata: ad.AnnData,
    output_path: Path,
    cfg: AppConfig,
    warnings: list[str],
) -> list[str]:
    """Streamed, memory-bounded sort for --backed input (Option C: bucket + concat).

    The eager sort (:func:`_maybe_sort_adata`) does ``adata[perm].copy()`` — a full in-memory
    reorder that transiently holds ~2x X. For a backed load X stays on the h5py handle, so we
    keep it there: one *sequential* pass over X buckets each source row into a temporary
    per-group CSR zarr store (contiguous append — no random scatter, no read-modify-write of
    output chunks), then the groups are concatenated in sorted order into the final store via
    the existing concat writer. Peak RAM is one row-batch of X, not the whole matrix.

    Scope (raises otherwise): sparse-csr X only; the backed input's X must be CSR on disk; and
    layers / raw / obsp must be absent (those are obs-aligned and would need their own reorder).
    obs/obsm are reordered in memory (backed mode already loads them); var/varm/varp/uns are not
    obs-aligned and are written as-is. Dense or CSC sort still works eagerly (omit --backed).
    """
    import shutil
    import tempfile

    import numpy as np
    from anndata._io.specs import write_elem  # private API — see anndata skill
    from anndata.io import sparse_dataset

    sort_by = cfg.grouping.sort_by
    if cfg.io.x_storage != "sparse-csr":
        raise ConversionError(
            f"--backed --sort-by supports x_storage='sparse-csr' only (got '{cfg.io.x_storage}'). "
            "Omit --backed to sort dense/CSC eagerly."
        )
    if adata.layers or adata.raw is not None or len(adata.obsp) > 0:
        raise ConversionError(
            "--backed --sort-by does not reorder layers/raw/obsp yet (they are obs-aligned and "
            "would need their own streamed reorder). Omit --backed to sort eagerly, or drop them."
        )
    x = adata.X
    if sp.issparse(x) or getattr(x, "format", None) != "csr":
        got = "in-memory " + type(x).__name__ if sp.issparse(x) else (getattr(x, "format", None) or type(x).__name__)
        raise ConversionError(
            f"--backed --sort-by requires the backed input's X to be CSR on disk; got {got}. "
            "Omit --backed to sort eagerly."
        )

    n_obs, n_vars = adata.shape
    x_dtype = x.dtype
    indices_dtype = np.int32  # matches the rest of the converter (fits unless > 2^31 cols)

    # Permutation + contiguous group ranges — obs-only, so backed-safe (obs is in memory).
    perm, ranges = _compute_sort(adata, sort_by)
    n_groups = len(ranges)
    starts = ranges["start"].to_numpy()
    ends = ranges["end"].to_numpy()

    # For each SOURCE row, the group (in sorted-group order) it routes to. perm[start:end] lists
    # a group's source rows in output order, which for a stable lexsort is ascending source order.
    group_of_source = np.empty(n_obs, dtype=np.int64)
    group_rows = []  # source-row ids per group, ascending (== stable within-group order)
    for gi in range(n_groups):
        rows = perm[starts[gi]:ends[gi]]
        group_of_source[rows] = gi
        group_rows.append(rows)

    # Per-group nnz + full indptr, precomputed from the (small) source indptr — no data pass
    # needed for structure, only for the data/indices values.
    row_nnz = np.diff(_get_indptr(x)).astype(np.int64)
    n_rows_each = [int(r.size) for r in group_rows]
    indptr_each = [
        np.concatenate([[0], np.cumsum(row_nnz[r])]).astype(np.int64) for r in group_rows
    ]
    nnz_each = [int(ip[-1]) for ip in indptr_each]

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    ad.settings.zarr_write_format = 3
    print(
        f"Converting (backed, streamed sort) → {output_path} "
        f"(n_obs={n_obs}, n_vars={n_vars}, sparse-csr, {n_groups} groups, backend={cfg.io.backend})",
        flush=True, file=sys.stderr,
    )
    t0 = time.perf_counter()

    tmp_root = Path(tempfile.mkdtemp(prefix="convert-to-zarr_sort_", dir=str(output_path.parent)))
    try:
        # ── Create temp per-group CSR stores (indptr known upfront; data filled by the pass) ──
        temp_groups = []
        for gi in range(n_groups):
            tg = zarr.open_group(str(tmp_root / f"g{gi}"), mode="w")
            tg.attrs["encoding-type"] = "csr_matrix"
            tg.attrs["encoding-version"] = "0.1.0"
            tg.attrs["shape"] = [n_rows_each[gi], n_vars]
            flat = min(cfg.chunks.sparse_flat_chunk, max(1, nnz_each[gi]))
            tg.require_array("data", shape=(nnz_each[gi],), dtype=x_dtype, chunks=(flat,), overwrite=True)
            tg.require_array("indices", shape=(nnz_each[gi],), dtype=indices_dtype, chunks=(flat,), overwrite=True)
            ip = tg.require_array(
                "indptr", shape=(n_rows_each[gi] + 1,), dtype=np.int64,
                chunks=(n_rows_each[gi] + 1,), overwrite=True,
            )
            for name in ("data", "indices", "indptr"):
                tg[name].attrs["encoding-type"] = "array"
                tg[name].attrs["encoding-version"] = "0.2.0"
            ip[:] = indptr_each[gi]
            temp_groups.append(tg)

        # ── Single sequential pass over X: bucket each row-batch into its groups ──
        # Batch to ~256 MB of nnz like the other streaming writers.
        nnz_total = int(row_nnz.sum())
        bpm = max(1, nnz_total // max(1, n_obs)) * (np.dtype(x_dtype).itemsize + np.dtype(indices_dtype).itemsize)
        batch_size = max(1_000, min(200_000, (256 * 1024 * 1024) // max(1, bpm)))
        cursors = [0] * n_groups  # nnz write cursor per group
        with _stage(f"Bucketing {n_obs} rows into {n_groups} groups (backed, streamed)"):
            for b0 in range(0, n_obs, batch_size):
                b1 = min(b0 + batch_size, n_obs)
                batch = x[b0:b1]  # backed CSR slice -> in-memory scipy CSR (one batch bounds RAM)
                if not sp.isspmatrix_csr(batch):
                    batch = batch.tocsr()
                g_batch = group_of_source[b0:b1]
                for gi in np.unique(g_batch):
                    gi = int(gi)
                    sub = batch[g_batch == gi]  # this group's rows, in source (== output) order
                    m = sub.nnz
                    if m == 0:
                        continue
                    c = cursors[gi]
                    temp_groups[gi]["data"][c:c + m] = sub.data
                    temp_groups[gi]["indices"][c:c + m] = sub.indices.astype(indices_dtype, copy=False)
                    cursors[gi] = c + m

        # ── Concat the groups (in sorted order) into the final store ──
        store, finalize = open_output_store(
            output_path, cfg, commit_message=f"convert-to-zarr convert-h5ad (sorted) → {output_path.name}",
        )
        store.attrs["encoding-type"] = "anndata"
        store.attrs["encoding-version"] = "0.1.0"
        with _stage("Writing metadata (sorted obs/obsm; var/varm/varp/uns as-is)"):
            write_elem(store, "obs", adata.obs.iloc[perm])
            write_elem(store, "var", adata.var)
            write_elem(store, "uns", dict(adata.uns))
            write_elem(store, "obsm", {k: (v.iloc[perm] if hasattr(v, "iloc") else v[perm])
                                       for k, v in adata.obsm.items()})
            write_elem(store, "varm", dict(adata.varm))
            write_elem(store, "obsp", {})   # empty (non-empty obsp is rejected above)
            write_elem(store, "varp", dict(adata.varp))

        temp_mats = [sparse_dataset(tg) for tg in temp_groups]
        with _stage(f"Writing X (n_obs={n_obs}, n_vars={n_vars}, sparse-csr, concat {n_groups} groups)"):
            _write_concatenated_csr(store, "X", temp_mats, n_rows_each, n_vars, x_dtype, cfg)
        finalize()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"Done in {time.perf_counter() - t0:.1f}s", flush=True, file=sys.stderr)
    warnings.append(
        f"Rows sorted by {list(sort_by)} into {n_groups} contiguous groups via backed streamed "
        "bucketing (X never fully materialised); obs/obsm reordered to match. Store is a plain "
        "sorted AnnData (no convert-to-zarr index)."
    )
    return [*warnings, *validation_result.warnings]


def _write_adata_to_zarr(
    adata: ad.AnnData,
    output_path: Path,
    cfg: AppConfig,
    load_warnings: list[str],
    allow_grouping: bool = False,
) -> list[str]:
    """Write AnnData to zarr (or icechunk).

    adata.X is expected to be CSR, but CSC is accepted and converted to CSR in memory with
    a warning. When ``allow_grouping`` and grouping is enabled, rows are sorted first.
    """
    cfg = _resolve_backend_cfg(cfg)
    warnings = list(load_warnings)

    if allow_grouping:
        if cfg.grouping.enabled and cfg.io.backed:
            # Backed input: sort X without ever materialising it (streamed bucket + concat).
            return _write_sorted_backed(adata, output_path, cfg, warnings)
        adata = _maybe_sort_adata(adata, cfg, warnings)
    elif cfg.grouping.enabled:
        raise ConversionError(
            "grouping (sort_by) is only supported by convert-h5ad for now."
        )

    x_for_write: Any | None = None
    is_csr = sp.isspmatrix_csr(adata.X) or (
        not sp.issparse(adata.X) and getattr(adata.X, "format", None) == "csr"  # backed mode
    )
    is_csc = sp.isspmatrix_csc(adata.X) or (
        not sp.issparse(adata.X) and getattr(adata.X, "format", None) == "csc"  # backed mode
    )

    if not is_csr:
        if is_csc:
            if sp.issparse(adata.X):
                x_for_write = adata.X.tocsr()
            else:
                x_for_write = adata.X[:].tocsr()
            warnings.append(
                "adata.X was CSC and has been converted to CSR in memory before zarr conversion."
            )
        else:
            raise ConversionError(
                f"adata.X must be CSR or CSC format. Got: {type(adata.X).__name__}"
            )

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    ad.settings.zarr_write_format = 3

    print(
        f"Converting → {output_path} "
        f"(n_obs={adata.n_obs}, n_vars={adata.n_vars}, {cfg.io.x_storage}, backend={cfg.io.backend})",
        flush=True, file=sys.stderr,
    )
    t0 = time.perf_counter()
    store, finalize = open_output_store(
        output_path, cfg, commit_message=f"convert-to-zarr convert-h5ad → {output_path.name}",
    )
    _write_csr_adata_direct(adata, store, cfg, x_override=x_for_write)
    finalize()
    print(
        f"Done in {time.perf_counter() - t0:.1f}s",
        flush=True, file=sys.stderr,
    )
    return [*warnings, *validation_result.warnings]


def convert_h5ad_to_zarr(input_h5ad: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """"Converts a .h5ad file to zarr, using the configuration for storage format,
    chunk sizes, and validation. Returns a list of warnings encountered"""
    adata = None
    load_warnings: list[str] = []
    try:
        adata, load_warnings = _load_h5ad_for_conversion(Path(input_h5ad), cfg) #attempts to load in 'backed' mode, where its not fully memory loaded
        return _write_adata_to_zarr(adata, Path(output_zarr), cfg, load_warnings, allow_grouping=True)
    
    except Exception as e:
        load_warnings = f"Warnings: {load_warnings}" if load_warnings else ""
        raise ConversionError(f"Failed to convert .h5ad file: {e}.{load_warnings}") from e
    finally: #closing 'backed' mode
        if adata is not None:
            _close_backed_if_needed(adata)


def convert_10x_h5_to_zarr(input_h5: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Converts .h5 file from 10x preprocessing into zarr with configured storage options and chunking"""
    #TODO edit to not load into anndata object, but write directly to zarr chunks
    
    try:
        import scanpy as sc
        #NOTE Expecting CSR from this load
        adata = sc.read_10x_h5(str(input_h5))
        
    except Exception as e:
        raise ConversionError(f"Failed to read 10x H5 file: {e}") from e
    
    return _write_adata_to_zarr(adata, Path(output_zarr), cfg, [])


# =============================================================================
# Multi-h5ad concatenation along obs (rows)
# =============================================================================

def _get_indptr(matrix: Any):
    """Return indptr as a numpy array for in-memory or backed sparse input."""
    import numpy as np
    if sp.issparse(matrix):
        return np.asarray(matrix.indptr)
    if hasattr(matrix, "indptr"):
        return np.asarray(matrix.indptr[:])
    if hasattr(matrix, "group"):
        return np.asarray(matrix.group["indptr"][:])
    raise ConversionError(
        f"Cannot locate indptr on sparse input of type {type(matrix).__name__}"
    )


def _ensure_csr(matrix: Any, label: str) -> tuple[Any, str | None]:
    """Return (csr_matrix, optional warning). Accepts CSR or CSC (in-memory or backed)."""
    is_csr = sp.isspmatrix_csr(matrix) or (
        not sp.issparse(matrix) and getattr(matrix, "format", None) == "csr"
    )
    is_csc = sp.isspmatrix_csc(matrix) or (
        not sp.issparse(matrix) and getattr(matrix, "format", None) == "csc"
    )
    if is_csr:
        return matrix, None
    if is_csc:
        if sp.issparse(matrix):
            converted = matrix.tocsr()
        else:
            converted = matrix[:].tocsr()
        return converted, f"[{label}] adata.X was CSC and converted to CSR in memory."
    raise ConversionError(
        f"[{label}] adata.X must be CSR or CSC; got {type(matrix).__name__}"
    )


def _append_dense_region(
    zarr_arr: Any,
    matrix: Any,
    row_offset: int,
    cfg: AppConfig,
) -> None:
    """Write a single matrix into zarr_arr[row_offset:row_offset+n_rows, :]."""
    import numpy as np
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    # Block to the array's shard grid (or chunk grid when unsharded) so writes cover whole
    # shards. NB: each input file starts at an arbitrary row_offset, so the shard straddling
    # a file seam is read-modify-written — unavoidable with per-file concat region writes.
    block_row, block_col = zarr_arr.shards or zarr_arr.chunks
    block_row = min(block_row, n_rows)
    block_col = min(block_col, n_cols)
    region = (slice(row_offset, row_offset + n_rows), slice(None))

    is_backed_sparse = not sp.issparse(matrix) and hasattr(matrix, "format")
    if not sp.issparse(matrix) and not is_backed_sparse:
        # Already dense (ndarray-like); 2D-tile to the zarr chunk grid via region store.
        arr = da.from_array(np.asarray(matrix), chunks=(block_row, block_col))
        with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
            da.store(
                arr, zarr_arr,
                regions=region,
                scheduler="threads",
                num_workers=cfg.chunks.cpus,
            )
        return

    backed = is_backed_sparse
    dask_dense = _build_tiled_dense_dask(matrix, block_row, block_col)
    scheduler = "synchronous" if backed else "threads"
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            dask_dense, zarr_arr,
            regions=region,
            scheduler=scheduler,
            num_workers=1 if backed else cfg.chunks.cpus,
        )


def _append_sparse_csr_region(
    data_arr: Any,
    indices_arr: Any,
    matrix: Any,
    indptr_full: Any,
    nnz_offset: int,
    n_rows: int,
    nnz_total: int,
    data_dtype: Any,
    indices_dtype: Any,
    cfg: AppConfig,
) -> None:
    """Stream one CSR matrix's data + indices into existing zarr arrays at nnz_offset."""
    import numpy as np
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar

    backed = not sp.issparse(matrix)

    _TARGET_BATCH_BYTES = 256 * 1024 * 1024
    avg_nnz = max(1, nnz_total // max(1, n_rows))
    bpm = avg_nnz * (np.dtype(data_dtype).itemsize + np.dtype(indices_dtype).itemsize)
    batch_size = max(1_000, min(200_000, _TARGET_BATCH_BYTES // max(1, bpm)))

    data_parts, indices_parts = [], []
    for b0 in range(0, n_rows, batch_size):
        b1 = min(b0 + batch_size, n_rows)
        bnnz = int(indptr_full[b1] - indptr_full[b0])
        if bnnz == 0:
            continue
        batch = dask.delayed(lambda m, a, b: m[a:b])(matrix, b0, b1)
        data_parts.append(da.from_delayed(
            dask.delayed(lambda b: np.asarray(b.data, dtype=data_dtype))(batch),
            shape=(bnnz,), dtype=data_dtype,
        ))
        indices_parts.append(da.from_delayed(
            dask.delayed(lambda b: np.asarray(b.indices, dtype=indices_dtype))(batch),
            shape=(bnnz,), dtype=indices_dtype,
        ))

    if not data_parts:
        return

    data_dask = da.concatenate(data_parts)
    indices_dask = da.concatenate(indices_parts)
    region = (slice(nnz_offset, nnz_offset + nnz_total),)
    scheduler = "synchronous" if backed else "threads"
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            [data_dask, indices_dask],
            [data_arr, indices_arr],
            regions=[region, region],
            scheduler=scheduler,
            num_workers=1 if backed else cfg.chunks.cpus,
        )


def _write_concatenated_csr(
    group: zarr.Group,
    key: str,
    matrices: list[Any],
    n_obs_each: list[int],
    n_vars: int,
    data_dtype: Any,
    cfg: AppConfig,
) -> None:
    import numpy as np

    indptrs = [_get_indptr(m) for m in matrices]
    nnz_each = [int(ip[-1]) for ip in indptrs]
    nnz_total = sum(nnz_each)
    n_obs_total = sum(n_obs_each)

    indices_dtype = np.int32
    indptr_dtype = np.int64 if nnz_total > np.iinfo(np.int32).max else np.int32

    sp_group = group.require_group(key)
    sp_group.attrs["encoding-type"] = "csr_matrix"
    sp_group.attrs["encoding-version"] = "0.1.0"
    sp_group.attrs["shape"] = [n_obs_total, n_vars]

    flat_chunk = min(cfg.chunks.sparse_flat_chunk, max(1, nnz_total))
    data_arr = sp_group.require_array(
        "data", shape=(nnz_total,), dtype=data_dtype,
        chunks=(flat_chunk,), compressors=_x_compressors(), overwrite=True,
    )
    indices_arr = sp_group.require_array(
        "indices", shape=(nnz_total,), dtype=indices_dtype,
        chunks=(flat_chunk,), compressors=_x_compressors(), overwrite=True,
    )
    indptr_arr = sp_group.require_array(
        "indptr", shape=(n_obs_total + 1,), dtype=indptr_dtype,
        chunks=(n_obs_total + 1,), overwrite=True,
    )
    for a in (data_arr, indices_arr, indptr_arr):
        a.attrs["encoding-type"] = "array"
        a.attrs["encoding-version"] = "0.2.0"

    # Build full indptr in memory (small: ~8B per row) then write once.
    full_indptr = np.empty(n_obs_total + 1, dtype=indptr_dtype)
    full_indptr[0] = 0
    row_offset = 0
    nnz_offset = 0
    for ip, n_obs_i, nnz_i in zip(indptrs, n_obs_each, nnz_each):
        full_indptr[row_offset + 1 : row_offset + 1 + n_obs_i] = (
            ip[1:].astype(indptr_dtype, copy=False) + nnz_offset
        )
        row_offset += n_obs_i
        nnz_offset += nnz_i
    indptr_arr[:] = full_indptr

    # Stream each matrix's data + indices to its nnz region.
    nnz_offset = 0
    for matrix, ip, n_obs_i, nnz_i in zip(matrices, indptrs, n_obs_each, nnz_each):
        if nnz_i > 0:
            _append_sparse_csr_region(
                data_arr, indices_arr, matrix, ip,
                nnz_offset, n_obs_i, nnz_i,
                data_dtype, indices_dtype, cfg,
            )
        nnz_offset += nnz_i


def _write_concatenated_dense(
    group: zarr.Group,
    key: str,
    matrices: list[Any],
    n_obs_each: list[int],
    n_vars: int,
    data_dtype: Any,
    cfg: AppConfig,
) -> None:
    n_obs_total = sum(n_obs_each)

    # _append_dense_region 2D-tiles each file to the array's write-block grid (shard shape
    # when sharding, else chunk shape), so peak RAM is bounded by one block — no need to
    # shrink row_chunk for wide matrices.
    row_chunk = min(cfg.chunks.x_row_chunk, n_obs_total)
    col_chunk = min(cfg.chunks.x_col_chunk, n_vars)
    shards, _, _ = _dense_shards(
        row_chunk, col_chunk, n_obs_total, n_vars, cfg.chunks.x_shard_factor
    )

    zarr_arr = group.require_array(
        key, shape=(n_obs_total, n_vars), dtype=data_dtype,
        chunks=(row_chunk, col_chunk), shards=shards,
        compressors=_x_compressors(), overwrite=True,
    )
    zarr_arr.attrs["encoding-type"] = "array"
    zarr_arr.attrs["encoding-version"] = "0.2.0"

    row_offset = 0
    for matrix, n_rows in zip(matrices, n_obs_each):
        _append_dense_region(zarr_arr, matrix, row_offset, cfg)
        row_offset += n_rows


def convert_h5ads_to_zarr(
    input_h5ads: list[str], output_zarr: str, cfg: AppConfig
) -> list[str]:
    """Concatenate multiple .h5ad files along obs (rows) into a single zarr store.

    Requirements:
      - All inputs must share the same `var` (gene names + order, strict match).
      - obs columns: by default all inputs must share an identical obs schema
        (same names + order). If ``cfg.concat.obs_columns`` is set, each input must
        instead merely *contain* those columns; obs is projected to exactly those
        (in that order) and all other columns are dropped before concatenation. A
        selected column that is categorical must have the same categories in every
        input, else it would degrade to a string array on concat — a hard error.
      - Only X, obs, var are written. layers/raw/uns/obsm/etc. are ignored.

    Sparse output uses CSR; dense output is supported. CSC output is not supported
    for multi-file concat (would require costly transpose).
    """
    from anndata._io.specs import write_elem
    import pandas as pd

    if not input_h5ads:
        raise ConversionError("convert_h5ads_to_zarr requires at least one input file.")

    if cfg.io.x_storage == "sparse-csc":
        raise ConversionError(
            "x_storage='sparse-csc' is not supported for multi-h5ad concat. "
            "Use 'sparse-csr' or 'dense'."
        )

    cfg = _resolve_backend_cfg(cfg)
    if cfg.grouping.enabled:
        raise ConversionError("grouping (sort_by) is only supported by convert-h5ad for now.")

    inputs = [Path(p) for p in input_h5ads]
    output_path = Path(output_zarr)
    # Fail fast on a pre-existing output before the (expensive) multi-file load; the actual
    # prepare/overwrite happens in open_output_store below.
    if output_path.exists() and not cfg.io.overwrite:
        raise ConversionError(
            f"Output path already exists: {output_path}. "
            "Use overwrite=true in config or --overwrite flag."
        )
    ad.settings.zarr_write_format = 3

    adatas: list[ad.AnnData] = []
    all_warnings: list[str] = []
    try:
        # ── Pass 1: load + validate ───────────────────────────────────────────
        for p in inputs:
            adata, _ = _load_h5ad_for_conversion(p, cfg)
            adatas.append(adata)

        ref_var_names = adatas[0].var_names
        ref_var = adatas[0].var
        n_vars = adatas[0].n_vars
        for i, a in enumerate(adatas[1:], start=1):
            if a.n_vars != n_vars or not (a.var_names == ref_var_names).all():
                raise ConversionError(
                    f"var mismatch in {inputs[i]}: expected {n_vars} vars matching "
                    f"{inputs[0].name}, got {a.n_vars} (names+order must be identical)."
                )

        obs_columns = list(cfg.concat.obs_columns)
        if obs_columns:
            # Explicit selection: every input must contain the named columns; obs is then
            # projected down to exactly these (in this order) at concat time — all other
            # columns are dropped. Lets files with differing *extra* columns be joined.
            for i, a in enumerate(adatas):
                missing = [c for c in obs_columns if c not in a.obs.columns]
                if missing:
                    raise ConversionError(
                        f"obs columns not found in {inputs[i].name}: {missing}. "
                        f"Requested via obs_columns; available: {list(a.obs.columns)}."
                    )
                dropped = [c for c in a.obs.columns if c not in obs_columns]
                if dropped:
                    all_warnings.append(
                        f"[{inputs[i].name}] dropping {len(dropped)} obs column(s) not in "
                        f"obs_columns: {dropped}."
                    )
            # Categorical columns must line up across inputs. If they don't (mixed
            # categorical/non-categorical, or differing category *sets*), pandas coerces the
            # column to a string (object) array on concat — dropping the compact categorical
            # `codes` encoding and making per-cell-type access far slower. Fail loudly instead.
            for c in obs_columns:
                is_cat = [isinstance(a.obs[c].dtype, pd.CategoricalDtype) for a in adatas]
                if not any(is_cat):
                    continue
                if not all(is_cat):
                    have = [inputs[i].name for i, v in enumerate(is_cat) if v]
                    lack = [inputs[i].name for i, v in enumerate(is_cat) if not v]
                    raise ConversionError(
                        f"obs column '{c}' is categorical in {have} but not in {lack}; "
                        f"concatenating would coerce it to a string array (dropping the "
                        f"categorical encoding). Make '{c}' categorical in all inputs, or "
                        f"drop it from obs_columns."
                    )
                cat0 = set(adatas[0].obs[c].cat.categories)
                bad = [inputs[i].name for i, a in enumerate(adatas)
                       if set(a.obs[c].cat.categories) != cat0]
                if bad:
                    raise ConversionError(
                        f"obs column '{c}' has mismatched categorical categories across "
                        f"inputs ({bad} differ from {inputs[0].name}); concatenating would "
                        f"coerce it to a string array (dropping the categorical encoding). "
                        f"Reconcile the categories (union them) across inputs, or drop "
                        f"'{c}' from obs_columns."
                    )
        else:
            # Default: strict identical obs schema (names + order) against file 0.
            ref_obs_cols = list(adatas[0].obs.columns)
            for i, a in enumerate(adatas[1:], start=1):
                if list(a.obs.columns) != ref_obs_cols:
                    raise ConversionError(
                        f"obs schema mismatch in {inputs[i]}: expected columns "
                        f"{ref_obs_cols}, got {list(a.obs.columns)}."
                    )

        for i, a in enumerate(adatas):
            vr = validate_single_cell_anndata(a, cfg.validation)
            all_warnings.extend(f"[{inputs[i].name}] {w}" for w in vr.warnings)

        # ── Ensure CSR for X; verify common dtype ─────────────────────────────
        x_matrices: list[Any] = []
        x_dtype = None
        for i, a in enumerate(adatas):
            x, warn = _ensure_csr(a.X, inputs[i].name)
            if warn:
                all_warnings.append(warn)
            if x_dtype is None:
                x_dtype = x.dtype
            elif x.dtype != x_dtype:
                raise ConversionError(
                    f"X dtype mismatch: {inputs[i].name} has {x.dtype}, "
                    f"expected {x_dtype}."
                )
            x_matrices.append(x)

        n_obs_each = [a.n_obs for a in adatas]
        n_obs_total = sum(n_obs_each)

        # ── Concat obs (small; pandas) ────────────────────────────────────────
        if obs_columns:
            # Project each obs to the selected columns (fixes output order), then concat.
            # (Categorical mismatches already errored out above; any coercion left here is
            # numeric, e.g. int+float -> float — harmless, but worth a heads-up.)
            obs_concat = pd.concat([a.obs[obs_columns] for a in adatas], axis=0)
            for c in obs_columns:
                in_dtypes = {str(a.obs[c].dtype) for a in adatas}
                out_dtype = str(obs_concat[c].dtype)
                if in_dtypes != {out_dtype}:
                    all_warnings.append(
                        f"obs column '{c}' coerced on concat: {sorted(in_dtypes)} -> {out_dtype}."
                    )
        else:
            obs_concat = pd.concat([a.obs for a in adatas], axis=0)

        print(
            f"Concatenating {len(inputs)} h5ads → {output_path} "
            f"(n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})",
            flush=True, file=sys.stderr,
        )
        t0 = time.perf_counter()

        # ── Open store + write metadata ───────────────────────────────────────
        store, finalize = open_output_store(
            output_path, cfg, commit_message=f"convert-to-zarr concat-h5ads → {output_path.name}",
        )
        store.attrs["encoding-type"] = "anndata"
        store.attrs["encoding-version"] = "0.1.0"

        with _stage("Writing metadata (obs, var, empty obsm/varm/uns/obsp/varp)"):
            write_elem(store, "obs", obs_concat)
            write_elem(store, "var", ref_var)
            write_elem(store, "uns", {})
            write_elem(store, "obsm", {})
            write_elem(store, "varm", {})
            write_elem(store, "obsp", {})
            write_elem(store, "varp", {})

        with _stage(f"Writing X (n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})"):
            if cfg.io.x_storage == "dense":
                _write_concatenated_dense(
                    store, "X", x_matrices, n_obs_each, n_vars, x_dtype, cfg,
                )
            else:  # sparse-csr
                _write_concatenated_csr(
                    store, "X", x_matrices, n_obs_each, n_vars, x_dtype, cfg,
                )

        finalize()
        print(
            f"Done in {time.perf_counter() - t0:.1f}s",
            flush=True, file=sys.stderr,
        )
        return all_warnings

    except Exception as e:
        raise ConversionError(f"Failed to concatenate h5ads: {e}") from e
    finally:
        for a in adatas:
            _close_backed_if_needed(a)
