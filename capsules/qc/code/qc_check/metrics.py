"""Per-cell QC metrics: the store's own columns, or, when allowed, computed from X in row bands."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import zarr
from annizarr import AppConfig

from profiles import COUNTS, GENES, MITO, TRANSCRIPTS

MITO_PREFIX = "MT-"
FROM_X = {GENES: GENES, COUNTS: COUNTS, MITO: MITO, TRANSCRIPTS: COUNTS}


class MetricsError(RuntimeError):
    """The store's metrics are missing and can't (or may not) be computed."""


def missing(obs: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    """The required metrics that are absent, or present with NA for some cells."""
    return [c for c in required if c not in obs or obs[c].isna().any()]


def mito_mask(var_names: np.ndarray) -> np.ndarray:
    return np.char.startswith(np.asarray(var_names, dtype=str), MITO_PREFIX)


def bands(n_obs: int, band_cells: int) -> list[tuple[int, int]]:
    return [(start, min(start + band_cells, n_obs)) for start in range(0, n_obs, band_cells)]


def _band_metrics(X: zarr.Group, start: int, stop: int, mito: np.ndarray) -> np.ndarray:
    indptr = X["indptr"][start:stop + 1].astype(np.int64)
    lo, hi = int(indptr[0]), int(indptr[-1])
    data = X["data"][lo:hi]
    if not np.issubdtype(data.dtype, np.integer) and not np.array_equal(data, np.round(data)):
        raise MetricsError("X doesn't hold integer counts; QC metrics need raw UMI counts.")
    data = data.astype(np.float64)
    indices = X["indices"][lo:hi]
    offsets = indptr - lo
    lengths = np.diff(offsets)
    empty = lengths == 0

    def row_sum(values: np.ndarray) -> np.ndarray:
        sums = np.add.reduceat(np.append(values, 0.0), offsets[:-1]) if len(lengths) else np.zeros(0)
        sums[empty] = 0
        return sums

    genes = row_sum((data > 0).astype(np.float64))
    total = row_sum(data)
    mt = row_sum(np.where(mito[indices], data, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = mt / total * 100
    return np.column_stack([genes, total, pct])


def compute(root: zarr.Group, var_names: np.ndarray, band_cells: int) -> pd.DataFrame:
    """`n_genes_by_counts`, `total_counts` and `pct_counts_mt` as `scanpy.pp.calculate_qc_metrics` defines them."""
    X = root["X"]
    n_obs = int(X.attrs["shape"][0])
    mito = mito_mask(var_names)
    with ThreadPoolExecutor(max_workers=AppConfig().chunks.cpus) as pool:
        parts = list(pool.map(lambda b: _band_metrics(X, *b, mito), bands(n_obs, band_cells)))
    values = np.vstack(parts) if parts else np.empty((0, 3))
    return pd.DataFrame({GENES: values[:, 0].astype(np.int64), COUNTS: values[:, 1], MITO: values[:, 2]})
