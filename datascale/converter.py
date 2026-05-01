from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import scipy.sparse as sp
import zarr

from .config import AppConfig
from .validation import validate_single_cell_anndata


class ConversionError(RuntimeError):
    """Raised when conversion cannot be completed."""


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
    if cfg.io.backed:
        try:
            return ad.read_h5ad(input_path, backed="r"), []
        except Exception as exc:
            raise ConversionError(
                f"Backed load failed for {input_path} ({type(exc).__name__}: {exc}). "
                "Remove --backed / set backed=false to load eagerly."
            ) from exc
    return ad.read_h5ad(input_path), []



def _write_sparse_as_dense_dask(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Write a CSR matrix (in-memory or backed _CSRDataset) as a dense zarr array.

    Uses da.store() to write one row-chunk at a time — the full dense matrix is never
    materialised. For in-memory CSR, cfg.chunks.n_dense_workers threads process chunks
    in parallel (each thread: .toarray() one slice, write one zarr chunk). For backed
    _CSRDataset, synchronous scheduler is enforced because h5py is not thread-safe.

    row_chunk is capped adaptively so one dense chunk stays under 64 MB regardless
    of column count.
    """
    import numpy as np
    import dask
    import dask.array as da

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    # Cap row_chunk so one dense chunk stays under 64 MB.
    bytes_per_row = n_cols * np.dtype(dtype).itemsize
    adaptive_row_chunk = max(1, (64 * 1024 * 1024) // bytes_per_row)
    row_chunk = min(cfg.chunks.x_row_chunk, adaptive_row_chunk)
    col_chunk = cfg.chunks.x_col_chunk

    # Build a dask array from delayed row slices.
    # da.from_delayed is used (not da.from_array) because backed _CSRDataset has no
    # ndim / array protocol.
    slices = [
        da.from_delayed(
            dask.delayed(lambda s=start: matrix[s : s + row_chunk].toarray())(),
            shape=(min(row_chunk, n_rows - start), n_cols),
            dtype=dtype,
        )
        for start in range(0, n_rows, row_chunk)
    ]
    dask_dense = da.concatenate(slices, axis=0)

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
    if backed:
        # h5py is not thread-safe: one chunk at a time.
        da.store(dask_dense, zarr_arr, scheduler="synchronous")
    else:
        da.store(
            dask_dense, zarr_arr,
            scheduler="threads",
            num_workers=cfg.chunks.n_dense_workers,
        )


def _write_matrix_direct(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Write a single matrix to zarr in the target format, directly.

    Input may be any format (CSR, CSC, dense ndarray, backed SparseDataset) -> because vvv
    NOTE adata.X is guaranteed CSR by the caller, but other layers and raw.X may not be.

    Dispatches based on cfg.io.x_storage:
      dense      → dask row-chunked write; CSC converted to CSR first for row slicing
      sparse-csr → ensure CSR, write_elem with flat chunks
      sparse-csc → ensure CSC, write_elem with flat chunks
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
    if is_backed_sparse:
        dataset_format = getattr(matrix, "format", None)
        if mode == "sparse-csc" and dataset_format != "csc":
            # _CSRDataset has no tocsc(); load into memory first then convert.
            matrix = matrix[:].tocsc()
        elif mode == "sparse-csr" and dataset_format != "csr":
            matrix = matrix[:].tocsr()
        

        nnz = getattr(matrix, "nnz", None)
        capped = min(cfg.chunks.sparse_flat_chunk, max(1, nnz)) if nnz is not None else cfg.chunks.sparse_flat_chunk
        write_elem(group, key, matrix, dataset_kwargs={"chunks": (capped,)})
        return
    
    elif mode == "sparse-csr" and not sp.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    elif mode == "sparse-csc" and not sp.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()

    capped = min(cfg.chunks.sparse_flat_chunk, max(1, matrix.nnz))  # matrix is always in-memory scipy here
    write_elem(group, key, matrix, dataset_kwargs={"chunks": (capped,)})


def _write_csr_adata_direct(adata: ad.AnnData, output_path: Path, cfg: AppConfig) -> None:
    """Write an AnnData with CSR X (in-memory or backed SparseDataset) directly to zarr.

    Matrices are written with the target format and chunking in one step.
    """
    from anndata._io.specs import write_elem  # not part of the public API

    store = zarr.open_group(str(output_path), mode="w")

    # Root encoding attrs — marks this as a valid anndata zarr store.
    store.attrs["encoding-type"] = "anndata"
    store.attrs["encoding-version"] = "0.1.0"

    # Metadata components: write_elem sets the correct encoding-type on each.
    write_elem(store, "obs", adata.obs)
    write_elem(store, "var", adata.var)
    write_elem(store, "uns", dict(adata.uns))
    write_elem(store, "obsm", dict(adata.obsm))
    write_elem(store, "varm", dict(adata.varm))
    write_elem(store, "obsp", dict(adata.obsp))
    write_elem(store, "varp", dict(adata.varp))

    # X: written directly in target format (dask for dense, write_elem for sparse).
    _write_matrix_direct(store, adata.X, "X", cfg)

    #Other layers are written with same storage format, outside of the X matrix.
    if adata.layers:
        write_elem(store, "layers", {})
        layers_group = store["layers"]
        for name, data in adata.layers.items():
            _write_matrix_direct(layers_group, data, name, cfg)

    if adata.raw is not None:
        raw_group = store.require_group("raw")
        raw_group.attrs["encoding-type"] = "raw"
        raw_group.attrs["encoding-version"] = "0.1.0"
        write_elem(raw_group, "var", adata.raw.var)
        write_elem(raw_group, "varm", dict(adata.raw.varm))
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
    """Write AnnData to zarr. Requires adata.X to be CSR (in-memory or backed SparseDataset)."""
    is_csr = sp.isspmatrix_csr(adata.X) or (
        not sp.issparse(adata.X) and getattr(adata.X, "format", None) == "csr" #backed mode
    )
    if not is_csr:
        raise ConversionError(
            f"adata.X must be CSR format. Got: {type(adata.X).__name__}"
        )

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    _prepare_output_path(output_path, cfg.io.overwrite)
    ad.settings.zarr_write_format = 3

    _write_csr_adata_direct(adata, output_path, cfg)
    zarr.consolidate_metadata(str(output_path))
    return [*load_warnings, *validation_result.warnings]


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
