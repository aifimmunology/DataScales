from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr

from annizarr._runtime import configure_runtime, stage
from annizarr._sources._matrix import get_indptr
from annizarr._storage import open_output_store
from annizarr._validation import validate_single_cell_anndata
from annizarr._writers._concat import _write_concatenated_csr
from annizarr._writers._encoding import make_sparse_group, set_array_attrs, write_elem
from annizarr._zarr import get_array
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from annizarr._config import AppConfig

logger = logging.getLogger(__name__)


def compute_sort(obs: pd.DataFrame, sort_by: tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    # returns (perm, ranges): perm is the int64 permutation (original row position for each
    # sorted position); ranges has one row per distinct key tuple, with the sort-key columns
    # plus start/end (half-open) row offsets into the sorted store. Sort is lexicographic
    # with sort_by[0] as the primary key, stable.
    missing = [c for c in sort_by if c not in obs.columns]
    if missing:
        raise ConversionError(f"grouping sort_by columns not found in obs: {missing}. Available: {list(obs.columns)}")

    keys = []
    for col in sort_by:
        codes, _ = pd.factorize(obs[col], sort=True)  # codes follow sorted value order
        if (np.asarray(codes) < 0).any():
            raise ConversionError(f"obs column '{col}' has missing (NaN) values; cannot sort by it.")
        keys.append(np.asarray(codes))

    # np.lexsort treats the LAST key as primary, so reverse to make sort_by[0] primary.
    perm = np.lexsort(keys[::-1]).astype(np.int64)

    obs_sorted = obs.iloc[perm]
    sizes = obs_sorted.groupby(list(sort_by), sort=False, observed=True).size()
    ends = np.cumsum(sizes.to_numpy())
    starts = ends - sizes.to_numpy()
    ranges = sizes.index.to_frame(index=False)
    ranges["start"] = starts.astype(np.int64)
    ranges["end"] = ends.astype(np.int64)
    return perm, ranges


def maybe_sort_adata(adata: ad.AnnData, cfg: AppConfig) -> ad.AnnData:
    # if grouping is enabled, reorders all obs-aligned arrays by the sort keys. Uses anndata
    # fancy indexing so obs/obsm/obsp/layers/raw share one permutation and the store stays a
    # valid AnnData; no tool-specific index is written — the result is a plain, physically
    # sorted AnnData, so each distinct key tuple is a contiguous row block a downstream
    # reader derives from the sorted obs column(s). Raises if x_storage isn't csr/dense, or
    # if cfg.io.backed is set (backed grouping goes through stream_sorted_store instead).
    if not cfg.grouping.enabled:
        return adata

    sort_by = cfg.grouping.sort_by
    if cfg.io.x_storage not in ("csr", "dense"):
        raise ConversionError(f"grouping (sort_by) requires x_storage='csr' or 'dense'; got '{cfg.io.x_storage}'.")
    if cfg.io.backed:
        raise ConversionError(
            "grouping (sort_by) requires an eager (in-memory) load; not supported with "
            "--backed yet. Omit --backed to sort."
        )

    perm, ranges_df = compute_sort(adata.obs, sort_by)
    with stage(f"Sorting {adata.n_obs} cells by {list(sort_by)} ({len(ranges_df)} groups)"):
        adata = adata[perm].copy()  # reorders X/obs/obsm/obsp/layers/raw consistently
    logger.warning(
        f"Rows sorted by {list(sort_by)} into {len(ranges_df)} contiguous groups; "
        "obs/obsm/obsp/layers/raw reordered to match. Store is a plain sorted AnnData "
        "(no tool index); derive ranges from the sorted obs column(s) if needed."
    )
    return adata


def _write_sorted_backed(
    adata: ad.AnnData, output_path: Path, cfg: AppConfig, *, branch: str | None = None, message: str | None = None
) -> str | None:
    # streamed, memory-bounded sort for --backed input (bucket + concat): the eager sort
    # (maybe_sort_adata) does adata[perm].copy() — a full in-memory reorder transiently
    # holding ~2x X. For a backed load X stays on the h5py handle, so we keep it there: one
    # sequential pass over X buckets each source row into a temporary per-group CSR zarr
    # store (contiguous append — no random scatter, no read-modify-write of output chunks),
    # then the groups are concatenated in sorted order into the final store via the existing
    # concat writer. Peak RAM is one row-batch of X, not the whole matrix.
    # Scope (raises otherwise): csr X only, on disk; layers/raw/obsp must be absent (obs-
    # aligned, would need their own reorder). obs/obsm reordered in memory (backed mode
    # already loads them); var/varm/varp/uns are not obs-aligned and written as-is. Dense or
    # CSC sort still works eagerly (omit --backed). Returns the icechunk snapshot id, or
    # None for a plain zarr store.
    if cfg.io.x_storage != "csr":
        raise ConversionError(
            f"--backed --sort-by supports x_storage='csr' only (got '{cfg.io.x_storage}'). "
            "Omit --backed to sort dense/CSC eagerly."
        )
    if adata.layers or adata.raw is not None or len(adata.obsp) > 0:
        raise ConversionError(
            "--backed --sort-by does not reorder layers/raw/obsp yet (they are obs-aligned and "
            "would need their own streamed reorder). Omit --backed to sort eagerly, or drop them."
        )
    x = adata.X
    if sp.issparse(x) or getattr(x, "format", None) != "csr":
        got = "in-memory " + type(x).__name__ if sp.issparse(x) else (getattr(x, "format", None) or type(x).__name__)
        raise ConversionError(
            f"--backed --sort-by requires the backed input's X to be CSR on disk; got {got}. "
            "Omit --backed to sort eagerly."
        )

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    for w in validation_result.warnings:
        logger.warning(w)
    sort_by = cfg.grouping.sort_by
    snapshot_id = stream_sorted_store(
        x,
        adata.obs,
        adata.var,
        dict(adata.uns),
        dict(adata.obsm),
        dict(adata.varm),
        dict(adata.varp),
        output_path,
        cfg,
        sort_by=sort_by,
        commit_message=message or f"annizarr convert (sorted by {','.join(sort_by)}) → {output_path.name}",
        branch=branch,
    )
    return snapshot_id


def stream_sorted_store(
    x: Any,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict[str, Any],
    obsm: dict[str, Any],
    varm: dict[str, Any],
    varp: dict[str, Any],
    output_path: Path,
    cfg: AppConfig,
    *,
    sort_by: tuple[str, ...],
    commit_message: str | None = None,
    branch: str | None = None,
    after_write: Callable[[zarr.Group], None] | None = None,
) -> str | None:
    # buckets rows into temp per-group CSR stores, then concats them in sorted order;
    # returns the icechunk snapshot id, or None for a plain zarr store. after_write, if
    # given, runs on the still-open output root before the single finalize() below — e.g.
    # sort's own re-derivation of a lone gexp layer, so it lands in the same commit as the
    # sort itself instead of a second one.
    from anndata.io import sparse_dataset

    from annizarr import _layout

    # the target-exists check happens in each caller (sort op / convert's --backed
    # --sort-by dispatch) before their own expensive work, via check_output_target
    configure_runtime(cfg.chunks.cpus)

    n_obs, n_vars = x.shape
    x_dtype = x.dtype
    indices_dtype = np.int32  # matches the rest of the writers (fits unless > 2^31 cols)

    perm, ranges = compute_sort(obs, sort_by)
    n_groups = len(ranges)
    starts = ranges["start"].to_numpy()
    ends = ranges["end"].to_numpy()

    # For each SOURCE row, the group (in sorted-group order) it routes to. perm[start:end] lists
    # a group's source rows in output order, which for a stable lexsort is ascending source order.
    group_of_source = np.empty(n_obs, dtype=np.int64)
    group_rows = []  # source-row ids per group, ascending (== stable within-group order)
    for gi in range(n_groups):
        rows = perm[starts[gi] : ends[gi]]
        group_of_source[rows] = gi
        group_rows.append(rows)

    # Per-group nnz + full indptr, precomputed from the (small) source indptr — no data pass
    # needed for structure, only for the data/indices values.
    row_nnz = np.diff(get_indptr(x)).astype(np.int64)
    n_rows_each = [int(r.size) for r in group_rows]
    indptr_each = [np.concatenate([[0], np.cumsum(row_nnz[r])]).astype(np.int64) for r in group_rows]
    nnz_each = [int(ip[-1]) for ip in indptr_each]

    ad.settings.zarr_write_format = 3
    logger.info(
        f"Sorting (streamed) → {output_path} (n_obs={n_obs}, n_vars={n_vars}, csr, "
        f"{n_groups} groups, backend={cfg.io.backend})"
    )
    t0 = time.perf_counter()

    tmp_root = Path(tempfile.mkdtemp(prefix="annizarr_sort_", dir=str(output_path.parent)))
    try:
        # Create temp per-group CSR stores (indptr known upfront; data filled by the pass) as
        # subgroups of one shared temp store.
        tmp_store_root = zarr.open_group(str(tmp_root), mode="w")
        temp_groups = []
        for gi in range(n_groups):
            tg = make_sparse_group(tmp_store_root, f"g{gi}", csr=True, shape=(n_rows_each[gi], n_vars))
            tg.require_array("data", shape=(nnz_each[gi],), dtype=x_dtype, chunks="auto", overwrite=True)
            tg.require_array("indices", shape=(nnz_each[gi],), dtype=indices_dtype, chunks="auto", overwrite=True)
            ip = tg.require_array(
                "indptr",
                shape=(n_rows_each[gi] + 1,),
                dtype=np.int64,
                chunks=(n_rows_each[gi] + 1,),
                overwrite=True,
            )
            for name in ("data", "indices", "indptr"):
                set_array_attrs(get_array(tg, name))
            ip[:] = indptr_each[gi]
            temp_groups.append(tg)

        # Single sequential pass over X: bucket each row-batch into its groups.
        # Batch to ~256 MB of nnz like the other streaming writers.
        nnz_total = int(row_nnz.sum())
        bpm = max(1, nnz_total // max(1, n_obs)) * (np.dtype(x_dtype).itemsize + np.dtype(indices_dtype).itemsize)
        batch_size = max(1_000, min(200_000, _layout.BATCH_BYTES // bpm))
        cursors = [0] * n_groups  # nnz write cursor per group
        with stage(f"Bucketing {n_obs} rows into {n_groups} groups (backed, streamed)"):
            for b0 in range(0, n_obs, batch_size):
                b1 = min(b0 + batch_size, n_obs)
                batch = x[b0:b1]  # backed CSR slice -> in-memory scipy CSR (one batch bounds RAM)
                if not sp.isspmatrix_csr(batch):
                    batch = batch.tocsr()
                # one stable argsort per batch instead of a boolean mask per group —
                # O(rows·log) not O(rows·groups) at high-cardinality keys
                g_batch = group_of_source[b0:b1]
                order = np.argsort(g_batch, kind="stable")
                sorted_g = g_batch[order]
                starts_b = np.flatnonzero(np.concatenate(([True], sorted_g[1:] != sorted_g[:-1])))
                ends_b = np.concatenate((starts_b[1:], [sorted_g.size]))
                for lo, hi in zip(starts_b, ends_b, strict=True):
                    gi = int(sorted_g[lo])
                    sub = batch[order[lo:hi]]  # stable ties keep source (== output) order
                    m = sub.nnz
                    if m == 0:
                        continue
                    c = cursors[gi]
                    get_array(temp_groups[gi], "data")[c : c + m] = sub.data
                    get_array(temp_groups[gi], "indices")[c : c + m] = sub.indices.astype(indices_dtype, copy=False)
                    cursors[gi] = c + m

        # Concat the groups (in sorted order) into the final store.
        out = open_output_store(
            output_path,
            cfg,
            commit_message=commit_message or f"annizarr sort → {output_path.name}",
            branch=branch,
        )
        try:
            store = out.root
            store.attrs["encoding-type"] = "anndata"
            store.attrs["encoding-version"] = "0.1.0"
            with stage("Writing metadata (sorted obs/obsm; var/varm/varp/uns as-is)"):
                write_elem(store, "obs", obs.iloc[perm])
                write_elem(store, "var", var)
                write_elem(store, "uns", dict(uns))
                write_elem(
                    store, "obsm", {k: (v.iloc[perm] if hasattr(v, "iloc") else v[perm]) for k, v in obsm.items()}
                )
                write_elem(store, "varm", dict(varm))
                write_elem(store, "obsp", {})  # empty (non-empty obsp is rejected by callers)
                write_elem(store, "varp", dict(varp))

            temp_mats = [sparse_dataset(tg) for tg in temp_groups]
            with stage(f"Writing X (n_obs={n_obs}, n_vars={n_vars}, csr, concat {n_groups} groups)"):
                _write_concatenated_csr(store, "X", temp_mats, n_rows_each, n_vars, x_dtype, cfg)
            if after_write is not None:
                after_write(store)
            snapshot_id = out.finalize()
        except BaseException:
            out.abort()
            raise
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    logger.info(f"Done in {time.perf_counter() - t0:.1f}s")
    logger.info(
        f"Rows sorted by {list(sort_by)} into {n_groups} contiguous groups via backed streamed "
        "bucketing (X never fully materialised); obs/obsm reordered to match. Store is a plain "
        "sorted AnnData (no tool index)."
    )
    return snapshot_id
