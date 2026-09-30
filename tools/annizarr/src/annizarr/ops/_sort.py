from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from annizarr._core._config import AppConfig, load_config, resolve_backend_cfg
from annizarr._core._sorting import stream_sorted_store
from annizarr._core._zarr import get_group
from annizarr._storage import check_output_target, is_remote, open_input_group, store_name
from annizarr.errors import ConversionError
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import zarr

    from annizarr.typing import PathLike, XStorage

logger = logging.getLogger(__name__)


def sort(
    store: PathLike,
    *,
    output: PathLike,
    by: Sequence[str],
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Physically sort an existing zarr store by obs column(s) into a new store.

    Parameters
    ----------
    store
        Existing AnnData zarr (or Icechunk) store, with CSR X, to read from.
    output
        Destination store path or URI.
    by
        Obs column name(s) to sort by, primary key first.
    cfg
        Resolved configuration; ``None`` loads :func:`~annizarr.config.load_config` defaults.
    branch
        Icechunk branch to write ``output`` to; created off the current tip if it
        doesn't exist yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the sort keys.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``by`` is empty, ``output`` is a remote (s3://, gs://) URI, ``store`` has no CSR X,
        or ``store`` carries layers other than a lone ``gexp`` (re-derived on the
        sorted output), non-empty ``raw``, or non-empty ``obsp`` (none of those are
        reordered).
    """
    from anndata.io import read_elem, sparse_dataset

    from annizarr.ops._expr import introspect_gexp, write_expr_layer

    if cfg is None:
        cfg = load_config()
    cfg = resolve_backend_cfg(cfg)
    by = tuple(by)
    if is_remote(output):
        raise ConversionError("sort streams through local temp stores; a remote output is not supported yet.")
    if not by:
        raise ConversionError("sort requires by=[OBS_COLUMN, ...].")
    if cfg.io.x_storage != "csr":
        raise ConversionError(f"sort supports x_storage='csr' only (got '{cfg.io.x_storage}').")
    check_output_target(output, cfg)

    src = open_input_group(store)
    if "X" not in src:
        raise ConversionError(f"no X in {store} — not an AnnData zarr store?")
    if src["X"].attrs.get("encoding-type") != "csr_matrix":
        raise ConversionError(f"sort requires CSR X; got encoding {src['X'].attrs.get('encoding-type')!r}.")
    # a lone gexp layer is re-derived on the sorted output; anything else is refused
    gexp_params = None
    layer_keys = list(get_group(src, "layers")) if "layers" in src else []
    if layer_keys == ["gexp"]:
        gexp_params = introspect_gexp(get_group(src, "layers")["gexp"])
    elif layer_keys:
        raise ConversionError(f"sort does not reorder layers {layer_keys}; only a gexp layer is re-derived.")
    for key in ("raw", "obsp"):
        if key in src and len(list(get_group(src, key))) > 0:
            raise ConversionError(f"sort does not reorder {key} yet; drop it or sort at convert time.")

    def _read(key: str) -> Any:
        return read_elem(src[key]) if key in src else {}

    x = sparse_dataset(src["X"])
    n_obs, n_vars = x.shape
    commit_message = message or f"annizarr sort by {','.join(by)} → {store_name(output)}"

    after_write = None
    if gexp_params is not None:
        fmt, chunk_elems, target_sum = gexp_params
        if target_sum is None:
            logger.warning("layers/gexp has no recorded target_sum; re-deriving at 1e4.")
            target_sum = 1e4

        def after_write(
            root: zarr.Group, fmt: XStorage = fmt, chunk_elems: int = chunk_elems, target_sum: float = target_sum
        ) -> None:
            # re-derives gexp before stream_sorted_store's finalize(), so it lands in the
            # same icechunk commit rather than a separate add_expr call afterwards.
            write_expr_layer(root, cfg, fmt=fmt, chunk_elems=chunk_elems, target_sum=target_sum)
            logger.warning(f"layers/gexp re-derived ({fmt}) on the sorted store.")

    snapshot_id = stream_sorted_store(
        x,
        read_elem(src["obs"]),
        read_elem(src["var"]),
        _read("uns"),
        _read("obsm"),
        _read("varm"),
        _read("varp"),
        Path(output),
        cfg,
        sort_by=by,
        commit_message=commit_message,
        branch=branch,
        after_write=after_write,
    )
    return OpResult(path=str(output), n_obs=n_obs, n_vars=n_vars, snapshot_id=snapshot_id)
