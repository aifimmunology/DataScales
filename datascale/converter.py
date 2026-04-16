from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad
import zarr

from .config import AppConfig
from .validation import validate_single_cell_anndata


class ConversionError(RuntimeError):
    """Raised when conversion cannot be completed."""


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

    chunks = (cfg.chunks.x_row_chunk, cfg.chunks.x_col_chunk)
    adata.write_zarr(str(output_path), chunks=chunks)

    if cfg.io.consolidate_metadata:
        zarr.consolidate_metadata(str(output_path))

    return validation_result.warnings
