"""Out-of-core seurat_v3 HVG (batch-aware), matching scanpy's `_highly_variable_genes_seurat_v3`.

Two streamed CPU passes (numba, genes partitioned over threads so every (batch, gene) is
updated in row order; rows with sorted indices jump to their gene range by binary search): Welford mean/var as scanpy's `aggregate`, loess per batch, then clipped
sum/sumsq as scanpy's `_sum_and_sum_squares_clipped`. Bitwise-equal inputs to loess matter:
many genes share identical log-means, and loess neighbour ties flip on 1e-15 differences.
"""
import numba
import numpy as np
import pandas as pd

from . import csr_stream


@numba.njit(cache=True, inline="always")
def _span(indptr, indices, r, g0, g1, sorted_rows):
    lo, hi = indptr[r], indptr[r + 1]
    if sorted_rows:
        a = lo + np.searchsorted(indices[lo:hi], g0)
        b = a + np.searchsorted(indices[a:hi], g1)
        return a, b
    return lo, hi


@numba.njit(parallel=True, cache=True)
def _welford_block(indptr, indices, data, rowcode, bounds, sorted_rows, cnt, mean, m2):
    for p in numba.prange(len(bounds) - 1):
        g0, g1 = bounds[p], bounds[p + 1]
        for r in range(len(rowcode)):
            b = rowcode[r]
            if b < 0:
                continue
            a, e = _span(indptr, indices, r, g0, g1, sorted_rows)
            for j in range(a, e):
                c = indices[j]
                if c < g0 or c >= g1:
                    continue
                value = np.float64(data[j])
                n = cnt[b, c] + 1
                cnt[b, c] = n
                m = mean[b, c]
                delta = value - m
                m += delta / n
                mean[b, c] = m
                m2[b, c] += delta * (value - m)


@numba.njit(parallel=True, cache=True)
def _clipped_block(indptr, indices, data, rowcode, bounds, sorted_rows, clip, s, sq):
    for p in numba.prange(len(bounds) - 1):
        g0, g1 = bounds[p], bounds[p + 1]
        for r in range(len(rowcode)):
            b = rowcode[r]
            if b < 0:
                continue
            a, e = _span(indptr, indices, r, g0, g1, sorted_rows)
            for j in range(a, e):
                c = indices[j]
                if c < g0 or c >= g1:
                    continue
                element = min(np.float64(data[j]), clip[b, c])
                sq[b, c] += element ** 2
                s[b, c] += element


@numba.njit(parallel=True, cache=True)
def _rows_sorted(indptr, indices):
    bad = 0
    for r in numba.prange(len(indptr) - 1):
        for j in range(indptr[r] + 1, indptr[r + 1]):
            if indices[j] <= indices[j - 1]:
                bad += 1
    return bad == 0


def _rowcodes(blk, codes):
    rc = np.full(blk.r1 - blk.r0, -1, np.int32)
    rc[blk.sel] = codes[blk.out0:blk.out0 + len(blk.sel)]
    return rc


def _bounds(n_genes, parts):
    return np.linspace(0, n_genes, parts + 1).astype(np.int64)


def _stream(X, rows, codes, kernel, args, block_nnz, parts, stats):
    bounds = _bounds(args[-1].shape[1], parts)
    for blk in csr_stream.blocks(X, rows, block_nnz, stats=stats):
        srt = _rows_sorted(blk.indptr, blk.indices)
        kernel(blk.indptr, blk.indices, blk.data, _rowcodes(blk, codes), bounds, srt, *args)


def _mean_var(X, rows, codes, n_b, n_genes, block_nnz, parts, stats):
    """Per-(batch, gene) mean and ddof=1 var, bit-identical to scanpy's Welford `mean_var_csr`."""
    shape = (len(n_b), n_genes)
    cnt, mean, m2 = np.zeros(shape, np.int64), np.zeros(shape), np.zeros(shape)
    _stream(X, rows, codes, _welford_block, (cnt, mean, m2), block_nnz, parts, stats)
    n_obs = n_b[:, None].astype(np.int64)
    k = n_obs - cnt
    merge = (k > 0) & (cnt > 0)
    mean_a = mean.copy()
    mean = np.where(merge, mean_a * cnt / n_obs, mean)
    m2 = np.where(merge, m2 + mean_a * mean_a * cnt * k / n_obs, m2)
    var = m2 / n_obs
    denom = np.where(n_obs > 1, n_obs - 1, np.nan)
    return mean, var * (n_obs / denom), cnt.sum(0)


def _clipped_sums(X, rows, codes, clip, block_nnz, parts, stats):
    s, sq = np.zeros(clip.shape), np.zeros(clip.shape)
    _stream(X, rows, codes, _clipped_block, (clip, s, sq), block_nnz, parts, stats)
    return s, sq


SPAN_RETRY = (0.5, 0.7, 0.9)


def _loess_reg_std(mean, var, span, bumped=None, batch=None):
    """scanpy's per-batch loess; where scanpy would raise (near-singular fit, e.g. a batch with very few
    cells in the subset) retry with wider spans and record (batch, span) in `bumped`."""
    from skmisc.loess import loess
    est = np.zeros(mean.shape[0], dtype=np.float64)
    not_const = var > 0
    if not_const.any():
        x, y = np.log10(mean[not_const]), np.log10(var[not_const])
        for sp in (span, *SPAN_RETRY):
            try:
                model = loess(x, y, span=sp, degree=2)
                model.fit()
                break
            except ValueError:
                if sp == SPAN_RETRY[-1]:
                    raise
        if sp != span and bumped is not None:
            bumped.append((str(batch), sp))
        est[not_const] = model.outputs.fitted_values
    return np.sqrt(10 ** est)


def _rank(norm_gene_vars, n_top_genes):
    ranked = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1).astype(np.float32)
    nbatches = np.sum((ranked < n_top_genes).astype(int), axis=0)
    ranked[ranked >= n_top_genes] = np.nan
    median = np.ma.median(np.ma.masked_invalid(ranked), axis=0).filled(np.nan)
    return nbatches, median


def seurat_v3(X, var_names, batch, rows=None, n_top_genes=4000, span=0.3, block_nnz=csr_stream.BLOCK_NNZ,
              parts=None, stats=None):
    """DataFrame like scanpy's seurat_v3 var output (+ per-gene `nnz`) for X[rows]; X is a zarr CSR
    group (or csr_stream.from_scipy(m)), batch is per row of X, rows sorted (None = all).
    stats: see csr_stream.blocks."""
    parts = parts or numba.get_num_threads()
    var_names = pd.Index(var_names)
    n_genes = len(var_names)
    cat = pd.Categorical(np.asarray(batch) if rows is None else np.asarray(batch)[rows])
    present = np.flatnonzero(np.bincount(cat.codes, minlength=len(cat.categories)) > 0)
    remap = np.full(len(cat.categories), -1, np.int32)
    remap[present] = np.arange(len(present), dtype=np.int32)
    codes = remap[cat.codes]
    n_b = np.bincount(codes, minlength=len(present))
    mean, var, nnz = _mean_var(X, rows, codes, n_b, n_genes, block_nnz, parts, stats)
    bumped = []
    reg_std = np.stack([_loess_reg_std(mean[b], var[b], span, bumped, cat.categories[present[b]]) for b in range(len(present))])
    if stats is not None and bumped:
        stats["loess_span_bumped"] = bumped
    norm = np.empty_like(mean)
    clip = np.stack([reg_std[b] * np.sqrt(n_b[b]) + mean[b] for b in range(len(present))])
    cs, csq = _clipped_sums(X, rows, codes, clip, block_nnz, parts, stats)
    for b in range(len(present)):
        n_obs = int(n_b[b])
        norm[b] = (1 / ((n_obs - 1) * np.square(reg_std[b]))) * (
            (n_obs * np.square(mean[b])) + csq[b] - 2 * cs[b] * mean[b])

    w = n_b[:, None] / n_b.sum()
    g_mean = (mean * w).sum(0)
    g_var = ((var * (n_b[:, None] - 1) + n_b[:, None] * (mean - g_mean) ** 2).sum(0)) / (n_b.sum() - 1)
    nbatches, median = _rank(norm, n_top_genes)
    df = pd.DataFrame({"means": g_mean, "variances": g_var, "gene_name": var_names,
                       "highly_variable_nbatches": nbatches, "highly_variable_rank": median,
                       "variances_norm": norm.mean(0), "nnz": nnz}, index=var_names)
    cols = ["highly_variable_rank", "highly_variable_nbatches"]
    order = df[cols].sort_values(cols, ascending=[True, False], na_position="last").index
    df["highly_variable"] = False
    df.loc[order[:int(n_top_genes)], "highly_variable"] = True
    return df


def hvg_columns(df):
    """Sorted integer column positions of the selected genes."""
    return np.flatnonzero(df["highly_variable"].to_numpy())
