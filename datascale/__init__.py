"""DataScale: AnnData conversion tools."""

from .config import AppConfig, load_config
from .converter import (
    ConversionError,
    convert_10x_h5_to_zarr,
    convert_h5ad_to_zarr,
    convert_h5ads_to_zarr,
)
from .validation import ValidationError, validate_single_cell_anndata

# Exportable Library API
__all__ = [
    "AppConfig",
    "load_config",
    "ConversionError",
    "ValidationError",
    "convert_h5ad_to_zarr",
    "convert_h5ads_to_zarr",
    "convert_10x_h5_to_zarr",
    "validate_single_cell_anndata",
]
