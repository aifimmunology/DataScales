from __future__ import annotations

import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anndata as ad
import scipy.sparse as sp
import zarr

from .config import AppConfig
from .validation import validate_single_cell_anndata


class ConversionError(RuntimeError):
    """Raised when conversion cannot be completed."""


@contextmanager
def _stage(label: str):
    """Print a labelled progress line with elapsed time. Flushes immediately."""
    print(f"→ {label} ...", flush=True, file=sys.stderr)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True, file=sys.stderr)


def _prepare_output_path(output_path: Path, overwrite: bool) -> None:
    """"removes existing output path if overwrite is enabled, otherwise raises error"""
    if output_path.exists():
        if not overwrite:
            raise ConversionError(
                f"Output path already exists: {output_path}. "
                "Use overwrite=true in config or --overwrite flag."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()


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

    The matrix is streamed through a 2D-tiled dask array whose blocks match the zarr
    chunk grid (row_chunk × col_chunk), so the full dense matrix is never materialised
    and peak RAM per worker is bounded by one chunk regardless of column count.

    For in-memory CSR, cfg.chunks.cpus threads process tiles in parallel. For backed
    _CSRDataset the synchronous scheduler is enforced because h5py is not thread-safe.
    """
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)

    dask_dense = _build_tiled_dense_dask(matrix, row_chunk, col_chunk)

    zarr_arr = group.require_array(
        key,
        shape=(n_rows, n_cols),
        dtype=dtype,
        chunks=(row_chunk, col_chunk),
        overwrite=True,
    )
    zarr_arr.attrs["encoding-type"] = "array"
    zarr_arr.attrs["encoding-version"] = "0.2.0"

    backed = not hasattr(matrix, "indices")  # backed _CSRDataset lacks .indices
    scheduler = "synchronous" if backed else "threads"
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            dask_dense, zarr_arr,
            scheduler=scheduler,
            num_workers=1 if backed else cfg.chunks.cpus,
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
    from anndata._io.specs import write_elem  # not part of the public API

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
            
        else:  #already dense #TODO make way for dense to be loaded chunk by chunk
            write_elem(group, key, matrix, dataset_kwargs={
                "chunks": (cfg.chunks.x_row_chunk, cfg.chunks.x_col_chunk)
            })
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

    Reads ``matrix.indptr`` upfront (small, ~8B per row/col) to know exact output
    offsets, then writes ``data`` and ``indices`` in row/col batches in parallel.
    Peak RAM per worker ≈ batch_rows × avg_nnz_per_row × (data_itemsize + 4).

    Works for in-memory scipy sparse and backed _CSRDataset/_CSCDataset (h5py).
    Backed input uses synchronous scheduler (h5py is not thread-safe);
    in-memory uses ``cfg.chunks.cpus`` threads.
    """
    import warnings
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
        chunks=(flat_chunk,), overwrite=True,
    )
    indices_arr = sp_group.require_array(
        "indices", shape=(nnz_total,), dtype=indices_dtype,
        chunks=(flat_chunk,), overwrite=True,
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

    # ── Build dask arrays for data + indices via delayed row/col batches ──────
    # Auto-tune batch size: target ~256 MB per batch in RAM so multiple workers
    # fit comfortably even on tight memory. RAM per batch ≈ batch × avg_nnz ×
    # (data_itemsize + indices_itemsize). Clamped to [1k, 200k] majors.
    _TARGET_BATCH_BYTES = 256 * 1024 * 1024
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

    # ── Store: parallel threads for in-memory, synchronous for backed (h5py) ──
    if backed and cfg.chunks.cpus > 1:
        warnings.warn(
            f"Backed sparse input forces synchronous writes "
            f"(cpus={cfg.chunks.cpus} ignored; h5py is not thread-safe).",
            UserWarning, stacklevel=2,
        )

    scheduler = "synchronous" if backed else "threads"
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            [data_dask, indices_dask],
            [data_arr, indices_arr],
            scheduler=scheduler,
            num_workers=1 if backed else cfg.chunks.cpus,
        )


def _write_csr_adata_direct(
    adata: ad.AnnData,
    output_path: Path,
    cfg: AppConfig,
    x_override: Any | None = None,
) -> None:
    """Write an AnnData with CSR X (in-memory or backed SparseDataset) directly to zarr.

    Matrices are written with the target format and chunking in one step.
    """
    from anndata._io.specs import write_elem  # not part of the public API

    store = zarr.open_group(str(output_path), mode="w")

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


def _write_adata_to_zarr(
    adata: ad.AnnData,
    output_path: Path,
    cfg: AppConfig,
    load_warnings: list[str],
) -> list[str]:
    """Write AnnData to zarr.

    adata.X is expected to be CSR, but CSC is accepted and converted to CSR
    in memory with a warning.
    """
    warnings = list(load_warnings)
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
    _prepare_output_path(output_path, cfg.io.overwrite)
    ad.settings.zarr_write_format = 3

    print(
        f"Converting → {output_path} "
        f"(n_obs={adata.n_obs}, n_vars={adata.n_vars}, {cfg.io.x_storage})",
        flush=True, file=sys.stderr,
    )
    t0 = time.perf_counter()
    _write_csr_adata_direct(adata, output_path, cfg, x_override=x_for_write)
    zarr.consolidate_metadata(str(output_path))
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
        return _write_adata_to_zarr(adata, Path(output_zarr), cfg, load_warnings)
    
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
    row_chunk: int,
    cfg: AppConfig,
) -> None:
    """Write a single matrix into zarr_arr[row_offset:row_offset+n_rows, :]."""
    import numpy as np
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    row_chunk = min(row_chunk, n_rows)
    region = (slice(row_offset, row_offset + n_rows), slice(None))

    is_backed_sparse = not sp.issparse(matrix) and hasattr(matrix, "format")
    if not sp.issparse(matrix) and not is_backed_sparse:
        # Already dense (ndarray-like); 2D-tile to the zarr chunk grid via region store.
        arr = da.from_array(np.asarray(matrix), chunks=(row_chunk, col_chunk))
        with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
            da.store(
                arr, zarr_arr,
                regions=region,
                scheduler="threads",
                num_workers=cfg.chunks.cpus,
            )
        return

    backed = is_backed_sparse
    dask_dense = _build_tiled_dense_dask(matrix, row_chunk, col_chunk)
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
        chunks=(flat_chunk,), overwrite=True,
    )
    indices_arr = sp_group.require_array(
        "indices", shape=(nnz_total,), dtype=indices_dtype,
        chunks=(flat_chunk,), overwrite=True,
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

    # Blocks are 2D-tiled to (row_chunk × col_chunk) by _append_dense_region, so peak
    # RAM is bounded by chunk size — no need to shrink row_chunk for wide matrices.
    row_chunk = min(cfg.chunks.x_row_chunk, n_obs_total)
    col_chunk = min(cfg.chunks.x_col_chunk, n_vars)

    zarr_arr = group.require_array(
        key, shape=(n_obs_total, n_vars), dtype=data_dtype,
        chunks=(row_chunk, col_chunk), overwrite=True,
    )
    zarr_arr.attrs["encoding-type"] = "array"
    zarr_arr.attrs["encoding-version"] = "0.2.0"

    row_offset = 0
    for matrix, n_rows in zip(matrices, n_obs_each):
        _append_dense_region(zarr_arr, matrix, row_offset, row_chunk, cfg)
        row_offset += n_rows


def convert_h5ads_to_zarr(
    input_h5ads: list[str], output_zarr: str, cfg: AppConfig
) -> list[str]:
    """Concatenate multiple .h5ad files along obs (rows) into a single zarr store.

    Requirements:
      - All inputs must share the same `var` (gene names + order, strict match).
      - All inputs must share the same `obs` columns (strict schema match).
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

    inputs = [Path(p) for p in input_h5ads]
    output_path = Path(output_zarr)
    _prepare_output_path(output_path, cfg.io.overwrite)
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
        obs_concat = pd.concat([a.obs for a in adatas], axis=0)

        print(
            f"Concatenating {len(inputs)} h5ads → {output_path} "
            f"(n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})",
            flush=True, file=sys.stderr,
        )
        t0 = time.perf_counter()

        # ── Open store + write metadata ───────────────────────────────────────
        store = zarr.open_group(str(output_path), mode="w")
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

        zarr.consolidate_metadata(str(output_path))
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
