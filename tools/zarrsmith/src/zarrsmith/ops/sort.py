from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import scipy.sparse as sp
import zarr

from ..config import AppConfig
from ..engine import _stage
from ..errors import ConversionError
from ..sources import _get_indptr
from ..storage import open_output_store
from ..validation import validate_single_cell_anndata
from ..writers import _write_concatenated_csr


def _compute_sort(adata: ad.AnnData, sort_by: tuple[str, ...]):
    """Compute the row permutation + contiguous range table for sorting by ``sort_by``.

    Returns ``(perm, ranges_df)`` where ``perm`` is the int64 permutation (original row
    position for each sorted position) and ``ranges_df`` has one row per distinct key tuple
    with the sort-key columns plus ``start``/``end`` (half-open) row offsets into the
    sorted store. Sort is lexicographic with ``sort_by[0]`` as the primary key, stable.
    """
    import numpy as np
    import pandas as pd

    obs = adata.obs
    missing = [c for c in sort_by if c not in obs.columns]
    if missing:
        raise ConversionError(
            f"grouping sort_by columns not found in obs: {missing}. "
            f"Available: {list(obs.columns)}"
        )

    keys = []
    for col in sort_by:
        codes, _ = pd.factorize(obs[col], sort=True)  # codes follow sorted value order
        if (np.asarray(codes) < 0).any():
            raise ConversionError(
                f"obs column '{col}' has missing (NaN) values; cannot sort by it."
            )
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


def _maybe_sort_adata(
    adata: ad.AnnData, cfg: AppConfig, warnings: list[str]
) -> ad.AnnData:
    """If grouping is enabled, reorder all obs-aligned arrays by the sort keys.

    Reordering uses anndata fancy indexing so obs/obsm/obsp/layers/raw all share one
    permutation and the store stays a valid AnnData. No tool-specific index is written:
    the result is a plain, physically sorted AnnData, so each distinct key tuple is a
    contiguous row block that a downstream reader derives from the sorted obs column(s).
    """
    if not cfg.grouping.enabled:
        return adata

    sort_by = cfg.grouping.sort_by
    if cfg.io.x_storage not in ("sparse-csr", "dense"):
        raise ConversionError(
            f"grouping (sort_by) requires x_storage='sparse-csr' or 'dense'; "
            f"got '{cfg.io.x_storage}'."
        )
    if cfg.io.backed:
        raise ConversionError(
            "grouping (sort_by) requires an eager (in-memory) load; not supported with "
            "--backed yet. Omit --backed to sort."
        )

    perm, ranges_df = _compute_sort(adata, sort_by)
    with _stage(f"Sorting {adata.n_obs} cells by {list(sort_by)} ({len(ranges_df)} groups)"):
        adata = adata[perm].copy()  # reorders X/obs/obsm/obsp/layers/raw consistently
    warnings.append(
        f"Rows sorted by {list(sort_by)} into {len(ranges_df)} contiguous groups; "
        "obs/obsm/obsp/layers/raw reordered to match. Store is a plain sorted AnnData "
        "(no tool index); derive ranges from the sorted obs column(s) if needed."
    )
    return adata


def _write_sorted_backed(
    adata: ad.AnnData,
    output_path: Path,
    cfg: AppConfig,
    warnings: list[str],
) -> list[str]:
    """Streamed, memory-bounded sort for --backed input (Option C: bucket + concat).

    The eager sort (:func:`_maybe_sort_adata`) does ``adata[perm].copy()`` — a full in-memory
    reorder that transiently holds ~2x X. For a backed load X stays on the h5py handle, so we
    keep it there: one *sequential* pass over X buckets each source row into a temporary
    per-group CSR zarr store (contiguous append — no random scatter, no read-modify-write of
    output chunks), then the groups are concatenated in sorted order into the final store via
    the existing concat writer. Peak RAM is one row-batch of X, not the whole matrix.

    Scope (raises otherwise): sparse-csr X only; the backed input's X must be CSR on disk; and
    layers / raw / obsp must be absent (those are obs-aligned and would need their own reorder).
    obs/obsm are reordered in memory (backed mode already loads them); var/varm/varp/uns are not
    obs-aligned and are written as-is. Dense or CSC sort still works eagerly (omit --backed).
    """
    import shutil
    import tempfile

    import numpy as np
    from anndata._io.specs import write_elem  # private API — see anndata skill
    from anndata.io import sparse_dataset

    sort_by = cfg.grouping.sort_by
    if cfg.io.x_storage != "sparse-csr":
        raise ConversionError(
            f"--backed --sort-by supports x_storage='sparse-csr' only (got '{cfg.io.x_storage}'). "
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

    n_obs, n_vars = adata.shape
    x_dtype = x.dtype
    indices_dtype = np.int32  # matches the rest of the writers (fits unless > 2^31 cols)

    # Permutation + contiguous group ranges — obs-only, so backed-safe (obs is in memory).
    perm, ranges = _compute_sort(adata, sort_by)
    n_groups = len(ranges)
    starts = ranges["start"].to_numpy()
    ends = ranges["end"].to_numpy()

    # For each SOURCE row, the group (in sorted-group order) it routes to. perm[start:end] lists
    # a group's source rows in output order, which for a stable lexsort is ascending source order.
    group_of_source = np.empty(n_obs, dtype=np.int64)
    group_rows = []  # source-row ids per group, ascending (== stable within-group order)
    for gi in range(n_groups):
        rows = perm[starts[gi]:ends[gi]]
        group_of_source[rows] = gi
        group_rows.append(rows)

    # Per-group nnz + full indptr, precomputed from the (small) source indptr — no data pass
    # needed for structure, only for the data/indices values.
    row_nnz = np.diff(_get_indptr(x)).astype(np.int64)
    n_rows_each = [int(r.size) for r in group_rows]
    indptr_each = [
        np.concatenate([[0], np.cumsum(row_nnz[r])]).astype(np.int64) for r in group_rows
    ]
    nnz_each = [int(ip[-1]) for ip in indptr_each]

    validation_result = validate_single_cell_anndata(adata, cfg.validation)
    ad.settings.zarr_write_format = 3
    print(
        f"Converting (backed, streamed sort) → {output_path} "
        f"(n_obs={n_obs}, n_vars={n_vars}, sparse-csr, {n_groups} groups, backend={cfg.io.backend})",
        flush=True, file=sys.stderr,
    )
    t0 = time.perf_counter()

    tmp_root = Path(tempfile.mkdtemp(prefix="zarrsmith_sort_", dir=str(output_path.parent)))
    try:
        # Create temp per-group CSR stores (indptr known upfront; data filled by the pass).
        temp_groups = []
        for gi in range(n_groups):
            tg = zarr.open_group(str(tmp_root / f"g{gi}"), mode="w")
            tg.attrs["encoding-type"] = "csr_matrix"
            tg.attrs["encoding-version"] = "0.1.0"
            tg.attrs["shape"] = [n_rows_each[gi], n_vars]
            tg.require_array("data", shape=(nnz_each[gi],), dtype=x_dtype, chunks="auto", overwrite=True)
            tg.require_array("indices", shape=(nnz_each[gi],), dtype=indices_dtype, chunks="auto", overwrite=True)
            ip = tg.require_array(
                "indptr", shape=(n_rows_each[gi] + 1,), dtype=np.int64,
                chunks=(n_rows_each[gi] + 1,), overwrite=True,
            )
            for name in ("data", "indices", "indptr"):
                tg[name].attrs["encoding-type"] = "array"
                tg[name].attrs["encoding-version"] = "0.2.0"
            ip[:] = indptr_each[gi]
            temp_groups.append(tg)

        # Single sequential pass over X: bucket each row-batch into its groups.
        # Batch to ~256 MB of nnz like the other streaming writers.
        nnz_total = int(row_nnz.sum())
        bpm = max(1, nnz_total // max(1, n_obs)) * (np.dtype(x_dtype).itemsize + np.dtype(indices_dtype).itemsize)
        batch_size = max(1_000, min(200_000, (256 * 1024 * 1024) // max(1, bpm)))
        cursors = [0] * n_groups  # nnz write cursor per group
        with _stage(f"Bucketing {n_obs} rows into {n_groups} groups (backed, streamed)"):
            for b0 in range(0, n_obs, batch_size):
                b1 = min(b0 + batch_size, n_obs)
                batch = x[b0:b1]  # backed CSR slice -> in-memory scipy CSR (one batch bounds RAM)
                if not sp.isspmatrix_csr(batch):
                    batch = batch.tocsr()
                g_batch = group_of_source[b0:b1]
                for gi in np.unique(g_batch):
                    gi = int(gi)
                    sub = batch[g_batch == gi]  # this group's rows, in source (== output) order
                    m = sub.nnz
                    if m == 0:
                        continue
                    c = cursors[gi]
                    temp_groups[gi]["data"][c:c + m] = sub.data
                    temp_groups[gi]["indices"][c:c + m] = sub.indices.astype(indices_dtype, copy=False)
                    cursors[gi] = c + m

        # Concat the groups (in sorted order) into the final store.
        store, finalize = open_output_store(
            output_path, cfg, commit_message=f"zarrsmith convert (sorted) → {output_path.name}",
        )
        store.attrs["encoding-type"] = "anndata"
        store.attrs["encoding-version"] = "0.1.0"
        with _stage("Writing metadata (sorted obs/obsm; var/varm/varp/uns as-is)"):
            write_elem(store, "obs", adata.obs.iloc[perm])
            write_elem(store, "var", adata.var)
            write_elem(store, "uns", dict(adata.uns))
            write_elem(store, "obsm", {k: (v.iloc[perm] if hasattr(v, "iloc") else v[perm])
                                       for k, v in adata.obsm.items()})
            write_elem(store, "varm", dict(adata.varm))
            write_elem(store, "obsp", {})   # empty (non-empty obsp is rejected above)
            write_elem(store, "varp", dict(adata.varp))

        temp_mats = [sparse_dataset(tg) for tg in temp_groups]
        with _stage(f"Writing X (n_obs={n_obs}, n_vars={n_vars}, sparse-csr, concat {n_groups} groups)"):
            _write_concatenated_csr(store, "X", temp_mats, n_rows_each, n_vars, x_dtype, cfg)
        finalize()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"Done in {time.perf_counter() - t0:.1f}s", flush=True, file=sys.stderr)
    warnings.append(
        f"Rows sorted by {list(sort_by)} into {n_groups} contiguous groups via backed streamed "
        "bucketing (X never fully materialised); obs/obsm reordered to match. Store is a plain "
        "sorted AnnData (no tool index)."
    )
    return [*warnings, *validation_result.warnings]
