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

from annizarr._core._runtime import configure_runtime, stage
from annizarr._core._validation import validate_single_cell_anndata
from annizarr._core._zarr import get_array
from annizarr._sources._readers import ConcatReader, CSRZarrReader, as_reader
from annizarr._storage import open_output_store
from annizarr._writers._encoding import autoshard_setting, make_sparse_group, set_array_attrs, write_elem
from annizarr._writers._matrix import write_matrix
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from annizarr._core._config import AppConfig

logger = logging.getLogger(__name__)


def compute_sort(obs: pd.DataFrame, sort_by: tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
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
    if not cfg.grouping.enabled:
        return adata

    sort_by = cfg.grouping.sort_by
    if cfg.io.x_storage not in ("csr", "dense"):
        raise ConversionError(f"grouping (sort_by) requires x_storage='csr' or 'dense'; got '{cfg.io.x_storage}'.")
    if cfg.io.lazy:
        raise ConversionError(
            "grouping (sort_by) requires an eager (in-memory) load; not supported with --lazy yet. Omit --lazy to sort."
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


def _write_sorted_lazy(
    adata: ad.AnnData, output_path: Path, cfg: AppConfig, *, branch: str | None = None, message: str | None = None
) -> str | None:
    # streams X into temp per-group CSR stores (peak RAM one row-batch), unlike
    # maybe_sort_adata's adata[perm].copy() (~2x X in RAM).
    if cfg.io.x_storage != "csr":
        raise ConversionError(
            f"--lazy --sort-by supports x_storage='csr' only (got '{cfg.io.x_storage}'). "
            "Omit --lazy to sort dense/CSC eagerly."
        )
    if adata.layers or adata.raw is not None or len(adata.obsp) > 0:
        raise ConversionError(
            "--lazy --sort-by does not reorder layers/raw/obsp yet (they are obs-aligned and "
            "would need their own streamed reorder). Omit --lazy to sort eagerly, or drop them."
        )
    x = adata.X
    if sp.issparse(x) or getattr(x, "format", None) != "csr":
        got = "in-memory " + type(x).__name__ if sp.issparse(x) else (getattr(x, "format", None) or type(x).__name__)
        raise ConversionError(
            f"--lazy --sort-by requires the lazy input's X to be CSR on disk; got {got}. Omit --lazy to sort eagerly."
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
    from annizarr._core import _layout

    configure_runtime(cfg.chunks.cpus)

    src_reader = as_reader(x, cfg=cfg)
    n_obs, n_vars = x.shape
    x_dtype = x.dtype
    indices_dtype = np.int32  # matches the rest of the writers (fits unless > 2^31 cols)

    perm, ranges = compute_sort(obs, sort_by)
    n_groups = len(ranges)
    starts = ranges["start"].to_numpy()
    ends = ranges["end"].to_numpy()

    group_of_source = np.empty(n_obs, dtype=np.int64)
    group_rows = []
    for gi in range(n_groups):
        rows = perm[starts[gi] : ends[gi]]
        group_of_source[rows] = gi
        group_rows.append(rows)

    row_nnz = np.diff(src_reader.indptr).astype(np.int64)
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

        # per-group cursors aren't write-grid aligned, but this pass is serial: a perf cost only.
        nnz_total = int(row_nnz.sum())
        bpm = max(1, nnz_total // max(1, n_obs)) * (np.dtype(x_dtype).itemsize + np.dtype(indices_dtype).itemsize)
        batch_size = max(1_000, min(200_000, _layout.BATCH_BYTES // bpm))
        cursors = [0] * n_groups  # nnz write cursor per group
        with stage(f"Bucketing {n_obs} rows into {n_groups} groups (lazy, streamed)"):
            for b0 in range(0, n_obs, batch_size):
                b1 = min(b0 + batch_size, n_obs)
                batch = src_reader.csr_rows(b0, b1)  # -> in-memory scipy CSR (one batch bounds RAM)
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

        out = open_output_store(
            output_path,
            cfg,
            commit_message=commit_message or f"annizarr sort → {output_path.name}",
            branch=branch,
            expected_shape=(n_obs, n_vars),
        )
        try:
            store = out.root
            store.attrs["encoding-type"] = "anndata"
            store.attrs["encoding-version"] = "0.1.0"
            with autoshard_setting(cfg.chunks.auto_shard):
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

                concat_reader = ConcatReader([CSRZarrReader(tg) for tg in temp_groups])
                with stage(f"Writing X (n_obs={n_obs}, n_vars={n_vars}, csr, concat {n_groups} groups)"):
                    write_matrix(store, "X", concat_reader, cfg)
                if after_write is not None:
                    after_write(store)
            snapshot_id = out.finalize()
        except BaseException:
            out.abort()
            raise
    finally:
        src_reader.close()
        shutil.rmtree(tmp_root, ignore_errors=True)

    logger.info(f"Done in {time.perf_counter() - t0:.1f}s")
    logger.info(
        f"Rows sorted by {list(sort_by)} into {n_groups} contiguous groups via lazy streamed "
        "bucketing (X never fully materialised); obs/obsm reordered to match. Store is a plain "
        "sorted AnnData (no tool index)."
    )
    return snapshot_id
