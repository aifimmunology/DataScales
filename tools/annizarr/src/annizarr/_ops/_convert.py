from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp

from annizarr._config import AppConfig, load_config, resolve_backend_cfg
from annizarr._ops._concat import concat
from annizarr._ops._result import OpResult
from annizarr._runtime import configure_runtime
from annizarr._sorting import _write_sorted_backed, maybe_sort_adata
from annizarr._sources import close_backed_if_needed, detect_format, load_10x_h5, load_h5ad, open_source
from annizarr._sources._matrix import is_backed, matrix_format
from annizarr._storage import check_output_target, open_output_store, store_name
from annizarr._validation import validate_single_cell_anndata
from annizarr._writers import write_adata
from annizarr.errors import AnzError, ConversionError

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
    """Write AnnData to zarr (or icechunk).

    adata.X is expected to be CSR; CSC is converted to CSR in memory with a warning, and
    dense X is accepted (streamed for dense output, sparsified in memory for sparse output
    on eager loads). When ``allow_grouping`` and grouping is enabled, rows are sorted first.

    Parameters
    ----------
    branch
        Icechunk branch to write to; created off the current tip if it doesn't exist
        yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the destination.
    """
    cfg = resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)

    if allow_grouping:
        if cfg.grouping.enabled and cfg.io.backed:
            # Backed input: sort X without ever materialising it (streamed bucket + concat).
            snapshot_id = _write_sorted_backed(adata, Path(output_path), cfg, branch=branch, message=message)
            return OpResult(path=str(output_path), n_obs=adata.n_obs, n_vars=adata.n_vars, snapshot_id=snapshot_id)
        adata = maybe_sort_adata(adata, cfg)
    elif cfg.grouping.enabled:
        raise ConversionError("grouping (sort_by) is only supported by convert for now.")

    x_for_write: Any | None = None
    x = adata.X
    fmt = matrix_format(x)

    if fmt != "csr":
        # masked arrays would classify as "dense" below; silently dropping a mask would
        # corrupt values, so refuse outright instead of the dense/sparse-conversion paths.
        if np.ma.isMaskedArray(x):
            raise ConversionError(f"adata.X must be CSR, CSC, or dense. Got: {type(x).__name__}")
        if fmt == "csc":
            if not is_backed(x):
                x_for_write = x.tocsr()
            # backed: left as the raw CSC dataset; write_matrix streams it instead (no in-memory
            # .tocsr() — anndata's backed _CSCDataset doesn't have one anyway).
            logger.warning("adata.X was CSC and has been converted to CSR in memory before zarr conversion.")
        elif fmt == "dense":
            if cfg.io.x_storage == "dense":
                pass  # the writer streams dense input to dense output directly
            elif not cfg.io.backed:
                # eager dense (issue #4): X is already in memory — sparsify for sparse output,
                # directly in the target format (no CSR->CSC double conversion)
                to_sparse = sp.csr_matrix if cfg.io.x_storage == "csr" else sp.csc_matrix
                x_for_write = to_sparse(np.asarray(x))
                logger.warning("adata.X was dense and has been converted to sparse in memory for sparse output.")
            else:
                raise ConversionError(
                    "backed dense X with sparse output is not supported: use --x-storage dense "
                    "(streamed) or omit --backed to convert in memory."
                )

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
    out = open_output_store(output_path, cfg, commit_message=commit_message, branch=branch)
    try:
        write_adata(adata, out.root, cfg, x_override=x_for_write)
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
    """Write an in-memory AnnData to a zarr (or icechunk) store.

    Public library entry for data that doesn't start as .h5ad/10x: X may be CSR, CSC
    (converted), or dense; honors x_storage/backend/sort_by exactly like convert_h5ad.

    Parameters
    ----------
    branch
        Icechunk branch to write to; created off the current tip if it doesn't exist
        yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the destination.

    Raises
    ------
    ConversionError
        ``cfg.io.backed`` is set (backed loading does not apply to an in-memory input).
    """
    if cfg is None:
        cfg = load_config()
    # pure config check first: no icechunk import needed to reject this, unlike
    # check_output_target's icechunk-backend branch (Repo.exists()).
    if cfg.io.backed:
        raise ConversionError("convert_adata takes an in-memory AnnData; io.backed does not apply.")
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
    """Convert a .h5ad file to zarr.

    Parameters
    ----------
    branch
        Icechunk branch to write to; created off the current tip if it doesn't exist
        yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the destination.
    """
    if cfg is None:
        cfg = load_config()
    # resolve_backend_cfg's backed+icechunk validation first: it's a pure config check (no
    # icechunk import), so it must fail before check_output_target's Repo.exists() would
    # otherwise require icechunk to be installed just to reach that same error.
    cfg = resolve_backend_cfg(cfg)
    check_output_target(output, cfg)
    adata = None
    try:
        adata, load_warnings = load_h5ad(Path(path), cfg)
        for w in load_warnings:
            logger.warning(w)
        if cfg.io.backed is None:
            # load_h5ad auto-selected; mirror its decision back onto cfg so every
            # downstream `cfg.io.backed` check (grouping dispatch, dense/sparse
            # conversion) sees the real, resolved value instead of "unset".
            cfg = replace(cfg, io=replace(cfg.io, backed=adata.isbacked))
        return write_adata_to_store(adata, output, cfg, allow_grouping=True, branch=branch, message=message)
    except AnzError:
        raise
    except Exception as e:
        raise ConversionError(f"Failed to convert .h5ad file: {e}") from e
    finally:
        if adata is not None:
            close_backed_if_needed(adata)


def convert_10x_h5(
    path: PathLike,
    *,
    output: PathLike,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Convert a 10x Cell Ranger .h5 to zarr; expects CSR from the 10x load.

    Parameters
    ----------
    branch
        Icechunk branch to write to; created off the current tip if it doesn't exist
        yet. Ignored for plain zarr.
    message
        Icechunk commit message; ``None`` names the op and the destination.
    """
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
    (h5ad only).

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
        ``inputs`` is an empty sequence; a single input already is a zarr/icechunk store
        (use rechunk or sort instead); or a multi-input sequence mixes non-h5ad formats.
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

    for p in paths:
        if detect_format(p) != "h5ad":
            raise ConversionError("concat supports h5ad inputs only.")
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
