"""The downsampled embedding: a chunk-aware cell sample, then normalize → HVG → PCA → UMAP → leiden on CPU."""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import zarr

N_HVG = 2000
N_PCS = 50
NEIGHBORS = 15
NEIGHBOR_PCS = 30
RESOLUTION = 1.0
MIN_CELLS = 30


@dataclass
class Sampled:
    rows: np.ndarray
    counts: sp.csr_matrix


def _chunk_span(indptr: np.ndarray, rows: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    first = indptr[rows] // chunk
    last = np.maximum(indptr[rows + 1] - 1, indptr[rows]) // chunk
    return first, last


def _from_chunks(rows: np.ndarray, indptr: np.ndarray, chunk: int, quota: int, rng: np.random.Generator) -> np.ndarray:
    """Rows lying wholly inside a run of consecutive storage chunks, starting at a random one, just long enough for `quota` cells."""
    first, last = _chunk_span(indptr, rows, chunk)
    lo, hi = int(first.min()), int(last.max())
    a = b = int(rng.integers(lo, hi + 1))
    inside = (first >= a) & (last <= b)
    while inside.sum() < quota:
        if b < hi:
            b += 1
        else:
            a -= 1
        inside = (first >= a) & (last <= b)
    return rng.choice(rows[inside], size=quota, replace=False)


def choose_rows(sample: pd.Series, indptr: np.ndarray, chunk: int, per_sample: int, max_cells: int,
                seed: int) -> np.ndarray:
    """Up to `per_sample` cells per sample (fewer when that would pass `max_cells`), sorted by position."""
    rng = np.random.default_rng(seed)
    codes = sample.cat.codes.to_numpy()
    order = np.argsort(codes, kind="stable")
    present, starts = np.unique(codes[order], return_index=True)
    groups = dict(zip(present, np.split(order, starts[1:])))
    groups.pop(-1, None)
    quota = max(1, min(per_sample, max_cells // max(len(groups), 1)))
    chosen = []
    for rows in groups.values():
        take = min(quota, len(rows))
        if take == len(rows):
            chosen.append(rows)
        elif rows[-1] - rows[0] + 1 == len(rows):
            chosen.append(_from_chunks(rows, indptr, chunk, take, rng))
        else:
            chosen.append(rng.choice(rows, size=take, replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=np.int64)


def read_rows(X: zarr.Group, indptr: np.ndarray, rows: np.ndarray) -> sp.csr_matrix:
    """Reads sorted `rows`, one contiguous slice of `data` and `indices` per run of nearby rows."""
    n_vars = int(X.attrs["shape"][1])
    gaps = indptr[rows[1:]] - indptr[rows[:-1] + 1] > X["data"].chunks[0]
    blocks = []
    for run in np.split(rows, np.flatnonzero(gaps) + 1) if len(rows) else []:
        lo, hi = int(indptr[run[0]]), int(indptr[run[-1] + 1])
        data, indices = X["data"][lo:hi], X["indices"][lo:hi]
        starts, stops = indptr[run] - lo, indptr[run + 1] - lo
        take = np.concatenate([np.arange(s, e) for s, e in zip(starts, stops)])
        run_indptr = np.concatenate([[0], np.cumsum(stops - starts)])
        blocks.append(sp.csr_matrix((data[take], indices[take], run_indptr), shape=(len(run), n_vars)))
    return sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, n_vars))


def sample_cells(root: zarr.Group, sample: pd.Series, per_sample: int, max_cells: int, seed: int) -> Sampled:
    X = root["X"]
    indptr = X["indptr"][:].astype(np.int64)
    rows = choose_rows(sample, indptr, X["data"].chunks[0], per_sample, max_cells, seed)
    return Sampled(rows, read_rows(X, indptr, rows).astype(np.float32))


def embed(counts: sp.csr_matrix, var_names: np.ndarray, seed: int) -> pd.DataFrame | None:
    """UMAP coordinates and leiden clusters, one row per sampled cell; None when there are too few cells."""
    if counts.shape[0] < MIN_CELLS:
        return None
    adata = ad.AnnData(counts.copy(), var=pd.DataFrame(index=pd.Index(var_names, dtype=str)))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.n_vars > N_HVG:
        sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    adata.X = adata.X.toarray()
    sc.pp.scale(adata, max_value=10)
    n_comps = min(N_PCS, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=min(NEIGHBORS, adata.n_obs - 1), n_pcs=min(NEIGHBOR_PCS, n_comps),
                    random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    sc.tl.leiden(adata, resolution=RESOLUTION, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    umap = adata.obsm["X_umap"]
    return pd.DataFrame({"umap_1": umap[:, 0], "umap_2": umap[:, 1],
                         "leiden": pd.Categorical(adata.obs["leiden"].astype(str))})
