from __future__ import annotations

from pathlib import Path

from .._config import AppConfig
from .._sorting import stream_sorted_store
from ..errors import ConversionError


def sort(input_store: str, output_zarr: str, cfg: AppConfig) -> list[str]:
    """Physically sort an existing zarr store by obs column(s) into a new store."""
    from anndata.io import read_elem, sparse_dataset

    from .._config import resolve_backend_cfg
    from .._storage import is_s3_url, open_input_group
    from ._expr import add_expr, introspect_gexp

    cfg = resolve_backend_cfg(cfg)
    if is_s3_url(output_zarr):
        raise ConversionError(
            "sort streams through local temp stores; s3:// output is not supported yet."
        )
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
        gexp_params = introspect_gexp(src["layers"]["gexp"])
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
    warnings = stream_sorted_store(
        x, read_elem(src["obs"]), read_elem(src["var"]), _read("uns"), _read("obsm"),
        _read("varm"), _read("varp"), Path(output_zarr), cfg, [], [],
    )
    if gexp_params is not None:
        fmt, chunk_elems, target_sum = gexp_params
        if target_sum is None:
            warnings.append("layers/gexp has no recorded target_sum; re-deriving at 1e4.")
            target_sum = 1e4
        add_expr(output_zarr, cfg, fmt=fmt, chunk_elems=chunk_elems, target_sum=target_sum)
        warnings.append(f"layers/gexp re-derived ({fmt}) on the sorted store.")
    return warnings
