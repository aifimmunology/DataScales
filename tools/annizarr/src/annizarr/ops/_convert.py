from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad

from annizarr._core._config import AppConfig, load_config, resolve_backend_cfg
from annizarr._core._runtime import configure_runtime
from annizarr._core._sorting import _write_sorted_lazy, maybe_sort_adata
from annizarr._core._validation import validate_single_cell_anndata
from annizarr._sources import close_lazy_if_needed, detect_format, load_10x_h5, load_h5ad, open_source
from annizarr._storage import check_output_target, open_output_store, store_name
from annizarr._writers import write_adata
from annizarr.errors import AnzError, ConversionError
from annizarr.ops._concat import concat
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from annizarr.typing import PathLike

logger = logging.getLogger(__name__)


def write_adata_to_store(
    adata: ad.AnnData,
    output_path: PathLike,
    cfg: AppConfig,
    *,
    allow_grouping: bool = False,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    cfg = resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)

    if allow_grouping:
        if cfg.grouping.enabled and cfg.io.lazy:
            snapshot_id = _write_sorted_lazy(adata, Path(output_path), cfg, branch=branch, message=message)
            return OpResult(path=str(output_path), n_obs=adata.n_obs, n_vars=adata.n_vars, snapshot_id=snapshot_id)
        adata = maybe_sort_adata(adata, cfg)
    elif cfg.grouping.enabled:
        raise ConversionError("grouping (sort_by) is only supported by convert for now.")

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    for w in validation_result.warnings:
        logger.warning(w)
    ad.settings.zarr_write_format = 3

    logger.info(
        f"Converting → {output_path} (n_obs={adata.n_obs}, n_vars={adata.n_vars}, "
        f"{cfg.io.x_storage}, backend={cfg.io.backend})"
    )
    t0 = time.perf_counter()
    commit_message = message or f"annizarr convert → {store_name(output_path)}"
    out = open_output_store(
        output_path, cfg, commit_message=commit_message, branch=branch, expected_shape=(adata.n_obs, adata.n_vars)
    )
    try:
        write_adata(adata, out.root, cfg)
        snapshot_id = out.finalize()
    except BaseException:
        out.abort()
        raise
    logger.info(f"Done in {time.perf_counter() - t0:.1f}s")
    return OpResult(path=str(output_path), n_obs=adata.n_obs, n_vars=adata.n_vars, snapshot_id=snapshot_id)


def convert_adata(
    adata: ad.AnnData,
    *,
    output: PathLike,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Write an AnnData (in-memory or already backed) to a zarr (or icechunk) store."""
    if cfg is None:
        cfg = load_config()
    cfg = replace(cfg, io=replace(cfg.io, lazy=adata.isbacked))
    check_output_target(output, cfg)
    return write_adata_to_store(adata, output, cfg, allow_grouping=True, branch=branch, message=message)


def convert_h5ad(
    path: PathLike,
    *,
    output: PathLike,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Convert a .h5ad file to zarr."""
    if cfg is None:
        cfg = load_config()
    cfg = resolve_backend_cfg(cfg)  # before check_output_target's Repo.exists(), a heavier check
    check_output_target(output, cfg)
    adata = None
    try:
        adata, load_warnings = load_h5ad(Path(path), cfg)
        for w in load_warnings:
            logger.warning(w)
        return write_adata_to_store(adata, output, cfg, allow_grouping=True, branch=branch, message=message)
    except AnzError:
        raise
    except Exception as e:
        raise ConversionError(f"Failed to convert .h5ad file: {e}") from e
    finally:
        if adata is not None:
            close_lazy_if_needed(adata)


def convert_10x_h5(
    path: PathLike,
    *,
    output: PathLike,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Convert a 10x Cell Ranger .h5 to zarr; expects CSR from the 10x load."""
    if cfg is None:
        cfg = load_config()
    check_output_target(output, cfg)
    try:
        adata = load_10x_h5(path)
    except AnzError:
        raise
    except Exception as e:
        raise ConversionError(f"Failed to read 10x H5 file: {e}") from e

    return write_adata_to_store(adata, output, cfg, allow_grouping=False, branch=branch, message=message)


def convert(
    inputs: PathLike | ad.AnnData | Sequence[PathLike],
    *,
    output: PathLike,
    cfg: AppConfig | None = None,
    fmt: Literal["h5ad", "10x"] | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Convert one or more inputs into a single AnnData zarr (or icechunk) store.

    Dispatches on ``inputs``: an in-memory AnnData goes through :func:`convert_adata`; a
    single path (or a one-element sequence) is sniffed with :func:`~annizarr.sources.detect_format`
    (or forced via ``fmt``) and routed to :func:`convert_h5ad`, :func:`convert_10x_h5`, or —
    for any other registered kind (see :func:`~annizarr.sources.register_source`) — through
    :func:`~annizarr.sources.open_source`; a sequence of two or more paths concatenates them
    (any mix of registered input kinds).

    Parameters
    ----------
    inputs
        A single path/URI, an in-memory :class:`~anndata.AnnData`, or a sequence of paths.
    output
        Destination store path or URI.
    cfg
        Resolved configuration; ``None`` loads :func:`~annizarr.config.load_config` defaults.
    fmt
        ``"h5ad"`` or ``"10x"``, overriding content detection for a single input; ignored
        for an AnnData input or a multi-input concat.
    branch
        Icechunk branch to write to; created off the current tip if it doesn't exist
        yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the destination.

    Returns
    -------
    OpResult

    Raises
    ------
    ConversionError
        ``inputs`` is an empty sequence, or a single input already is a zarr/icechunk store
        (use rechunk or sort instead).
    """
    if cfg is None:
        cfg = load_config()

    if isinstance(inputs, ad.AnnData):
        return convert_adata(inputs, output=output, cfg=cfg, branch=branch, message=message)

    paths: list[PathLike] = [inputs] if isinstance(inputs, (str, os.PathLike)) else list(inputs)
    if not paths:
        raise ConversionError("convert requires at least one input.")

    if len(paths) == 1:
        return _convert_one(paths[0], output, cfg, fmt, branch=branch, message=message)

    return concat([str(p) for p in paths], output=output, cfg=cfg, branch=branch, message=message)


def _convert_one(
    path: PathLike, output: PathLike, cfg: AppConfig, fmt: str | None, *, branch: str | None, message: str | None
) -> OpResult:
    kind = fmt or detect_format(path)
    if kind == "h5ad":
        return convert_h5ad(path, output=output, cfg=cfg, branch=branch, message=message)
    if kind == "10x":
        return convert_10x_h5(path, output=output, cfg=cfg, branch=branch, message=message)
    if kind in ("zarr", "icechunk"):
        raise ConversionError(f"{path} is already a store; use rechunk or sort.")

    source = open_source(path, cfg, fmt=kind)
    try:
        for w in source.warnings:
            logger.warning(w)
        return write_adata_to_store(source.adata, output, cfg, allow_grouping=False, branch=branch, message=message)
    finally:
        source.close()
