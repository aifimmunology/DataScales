from __future__ import annotations

from pathlib import Path

from convert_to_zarr.config import AppConfig
from convert_to_zarr.errors import ConversionError
from convert_to_zarr.sorting import _stream_sorted_store


def sort_store(input_store: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Physically sort an existing zarr store by obs column(s) into a new store."""
    from anndata.io import read_elem, sparse_dataset

    from convert_to_zarr.config import _resolve_backend_cfg
    from convert_to_zarr.storage import open_input_group
    from .expr import _introspect_gexp, add_expr_layer

    cfg = _resolve_backend_cfg(cfg)
    if not cfg.grouping.sort_by:
        raise ConversionError("sort requires --by OBS_COLUMN [OBS_COLUMN ...].")
    if cfg.io.x_storage != "sparse-csr":
        raise ConversionError(
            f"sort supports x_storage='sparse-csr' only (got '{cfg.io.x_storage}')."
        )

    src = open_input_group(input_store)
    if "X" not in src:
        raise ConversionError(f"no X in {input_store} — not an AnnData zarr store?")
    if src["X"].attrs.get("encoding-type") != "csr_matrix":
        raise ConversionError(
            f"sort requires CSR X; got encoding {src['X'].attrs.get('encoding-type')!r}."
        )
    # a lone gexp layer is re-derived on the sorted output; anything else is refused
    gexp_params = None
    layer_keys = list(src["layers"]) if "layers" in src else []
    if layer_keys == ["gexp"]:
        gexp_params = _introspect_gexp(src["layers"]["gexp"])
    elif layer_keys:
        raise ConversionError(
            f"sort does not reorder layers {layer_keys}; only a gexp layer is re-derived."
        )
    for key in ("raw", "obsp"):
        if key in src and len(list(src[key])) > 0:
            raise ConversionError(
                f"sort does not reorder {key} yet; drop it or sort at convert time."
            )

    def _read(key):
        return read_elem(src[key]) if key in src else {}

    x = sparse_dataset(src["X"])
    warnings = _stream_sorted_store(
        x, read_elem(src["obs"]), read_elem(src["var"]), _read("uns"), _read("obsm"),
        _read("varm"), _read("varp"), Path(output_zarr), cfg, [], [],
    )
    if gexp_params is not None:
        fmt, chunk_elems, target_sum = gexp_params
        if target_sum is None:
            warnings.append("layers/gexp has no recorded target_sum; re-deriving at 1e4.")
            target_sum = 1e4
        add_expr_layer(output_zarr, cfg, fmt=fmt, chunk_elems=chunk_elems, target_sum=target_sum)
        warnings.append(f"layers/gexp re-derived ({fmt}) on the sorted store.")
    return warnings
