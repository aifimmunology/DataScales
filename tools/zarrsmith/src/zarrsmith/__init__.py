"""zarrsmith: build and rework AnnData zarr stores."""

from .config import AppConfig, GroupingConfig, load_config
from .errors import ConversionError
from .ops import (
    add_expr_layer,
    append_cells,
    convert_10x_h5_to_zarr,
    convert_h5ad_to_zarr,
    convert_h5ads_to_zarr,
    rechunk_store,
    sort_store,
)
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
    "add_expr_layer",
    "rechunk_store",
    "sort_store",
    "append_cells",
]
