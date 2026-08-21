"""zarrsmith: build and rework AnnData zarr stores."""

from .config import AppConfig, GroupingConfig, load_config
from .errors import ConversionError
from .ops import convert_10x_h5_to_zarr, convert_h5ad_to_zarr, convert_h5ads_to_zarr
from .validation import ValidationError, validate_single_cell_anndata

__all__ = [
    "AppConfig",
    "GroupingConfig",
    "load_config",
    "ConversionError",
    "ValidationError",
    "convert_h5ad_to_zarr",
    "convert_h5ads_to_zarr",
    "convert_10x_h5_to_zarr",
    "validate_single_cell_anndata",
]
