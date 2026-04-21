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

    if mode == "sparse":
        return matrix if sp.issparse(matrix) else sp.csr_matrix(matrix)

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


def convert_h5ad_to_zarr(input_h5ad: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    input_path = Path(input_h5ad)
    output_path = Path(output_zarr)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() != ".h5ad":
        raise ConversionError("Input must be an .h5ad file for this command.")

    adata = ad.read_h5ad(input_path)
    validation_result = validate_single_cell_anndata(adata, cfg.validation)

    _prepare_output_path(output_path, cfg.io.overwrite)

    ad.settings.zarr_write_format = 3

    _apply_x_storage_mode(adata, cfg.io.x_storage)

    chunks = (cfg.chunks.x_row_chunk, cfg.chunks.x_col_chunk)
    adata.write_zarr(str(output_path), chunks=chunks)

    if cfg.io.consolidate_metadata:
        zarr.consolidate_metadata(str(output_path))

    return validation_result.warnings
