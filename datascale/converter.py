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


def _convert_matrix_storage(matrix: Any, mode: str) -> Any:
    if mode == "auto":
        return matrix

    if mode == "sparse-csr":
        return matrix if sp.issparse(matrix) else sp.csr_matrix(matrix)

    if mode == "sparse-csc":
        return matrix.tocsc() if sp.issparse(matrix) else sp.csc_matrix(matrix)

    if mode == "dense":
        return matrix.toarray() if sp.issparse(matrix) else matrix

    raise ConversionError(f"Unsupported x_storage mode: {mode}")


def _apply_x_storage_mode(adata: ad.AnnData, mode: str) -> None:
    if mode == "auto":
        return

    adata.X = _convert_matrix_storage(adata.X, mode)

    for layer_name in list(adata.layers.keys()):
        adata.layers[layer_name] = _convert_matrix_storage(adata.layers[layer_name], mode)

    if adata.raw is not None:
        raw_adata = adata.raw.to_adata()
        raw_adata.X = _convert_matrix_storage(raw_adata.X, mode)
        adata.raw = raw_adata


def _prepare_output_path(output_path: Path, overwrite: bool) -> None:
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
    warnings: list[str] = []

    #auto is used when its stored the same way as original h5ad, so we know we can 'backed' load the anndata
    if cfg.io.x_storage == "auto":
        try:
            return ad.read_h5ad(input_path, backed="r"), warnings
        except Exception as exc:
            warnings.append(
                "Backed read was unavailable; falling back to in-memory load "
                f"({type(exc).__name__}: {exc})."
            )

    return ad.read_h5ad(input_path), warnings


def _load_10x_h5(input_path: Path) -> tuple[ad.AnnData, list[str]]:
    import scanpy as sc

    adata = sc.read_10x_h5(str(input_path))
    return adata, []


def _apply_sparse_flat_chunks(output_path: Path, adata: ad.AnnData, sparse_flat_chunk: int) -> None:
    """Re-write sparse matrix groups in the output zarr with a controlled flat chunk size.

    This runs after write_zarr so that obs/var/etc. are already written, then only the
    sparse flat arrays (data/indices/indptr) are deleted and rewritten with the requested
    chunk size.  Dense matrices are not affected.
    """
    from anndata._io.specs import write_elem  # local import – not part of the public API

    store = zarr.open_group(str(output_path), mode="r+", use_consolidated=False)

    def _rechunk(group: zarr.Group, matrix: Any, key: str) -> None:
        # Cap chunk size at nnz so zarr v3 never creates a chunk larger than the array.
        capped = min(sparse_flat_chunk, max(1, matrix.nnz))
        del group[key]
        write_elem(group, key, matrix, dataset_kwargs={"chunks": (capped,)})

    if sp.issparse(adata.X):
        _rechunk(store, adata.X, "X")

    if adata.layers:
        layers_group = store["layers"]
        for layer_name, layer_data in adata.layers.items():
            if sp.issparse(layer_data):
                _rechunk(layers_group, layer_data, layer_name)

    if adata.raw is not None:
        raw_X = adata.raw.X
        if sp.issparse(raw_X):
            _rechunk(store["raw"], raw_X, "X")


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
    """Shared write core used by all three converter entry points."""
    validation_result = validate_single_cell_anndata(adata, cfg.validation)

    _prepare_output_path(output_path, cfg.io.overwrite)

    ad.settings.zarr_write_format = 3

    _apply_x_storage_mode(adata, cfg.io.x_storage)

    chunks = (cfg.chunks.x_row_chunk, cfg.chunks.x_col_chunk)
    adata.write_zarr(str(output_path), chunks=chunks)


    rechunked = False
    if not getattr(adata, "isbacked", False):
        _apply_sparse_flat_chunks(output_path, adata, cfg.chunks.sparse_flat_chunk)
        rechunked = True

    # zarr v3 auto-consolidates during write_zarr.  Re-consolidate when rechunking
    # happened (stale root zarr.json) or when the user explicitly requested it.
    if rechunked or cfg.io.consolidate_metadata:
        zarr.consolidate_metadata(str(output_path))

    return [*load_warnings, *validation_result.warnings]


def convert_h5ad_to_zarr(input_h5ad: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    input_path = Path(input_h5ad)
    output_path = Path(output_zarr)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".h5ad":
        raise ConversionError("Input must be an .h5ad file for this command.")

    adata, load_warnings = _load_h5ad_for_conversion(input_path, cfg)
    try:
        return _write_adata_to_zarr(adata, output_path, cfg, load_warnings)
    finally:
        _close_backed_if_needed(adata)


def convert_10x_h5_to_zarr(input_h5: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    input_path = Path(input_h5)
    output_path = Path(output_zarr)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ConversionError("Input for convert-10x-h5 must be a .h5 or .hdf5 file.")

    adata, load_warnings = _load_10x_h5(input_path)
    return _write_adata_to_zarr(adata, output_path, cfg, load_warnings)
