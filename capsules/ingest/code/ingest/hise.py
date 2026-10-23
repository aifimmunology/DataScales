"""HISE H5 files (10x matrix plus a per-cell `observations` group) as an annizarr input source.

Both observation layouts are read onto the older HISE column names: tenx-scrna-pipeline's
(`Batch`, `Pool`, `Chip`, `uuid`, ...) is renamed through `_ALIASES`."""
from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from annizarr import AppConfig
from annizarr.config import IOConfig
from annizarr.sources import Source, open_source, register_source

KIND = "hise"
_GEX = "Gene Expression"
_SAMPLE_ID = re.compile(r"_([A-Z0-9]+)-\d+$")
_ALIASES = {
    "sample_name": "pbmc_sample_id", "Batch": "batch_id", "Pool": "pool_id", "Chip": "chip_id",
    "hashtag": "hto_barcode", "uuid": "cell_uuid", "barcodes": "original_barcodes",
    "n_mitochondrial_counts": "n_mito_umis",
}


def _text(values: np.ndarray) -> np.ndarray:
    """h5py returns fixed-length strings as bytes ("S") and variable-length ones as an object array of bytes."""
    if values.dtype.kind == "S":
        return values.astype(str)
    if values.dtype.kind == "O":
        return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)
    return values


def sniff(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as f:
            matrix = f.get("matrix")
            return KIND if isinstance(matrix, h5py.Group) and "observations" in matrix else None
    except OSError:
        return None


def load(path: Path | str, cfg: AppConfig) -> Source:
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        counts = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
        names = _text(m["features/name"][:])
        if "feature_type" in m["features"]:
            keep = np.asarray(_text(m["features/feature_type"][:]) == _GEX)
            counts, names = counts[keep], names[keep]
        var = pd.DataFrame(index=ad.utils.make_index_unique(pd.Index(names)))
        obs = pd.DataFrame(
            {col: _text(ds[:]) for col, ds in m["observations"].items()},
            index=pd.Index(_text(m["barcodes"][:]), name="barcodes"),
        )
        obs = obs.rename(columns={k: v for k, v in _ALIASES.items() if k in obs and v not in obs})
    adata = ad.AnnData(counts.T.tocsr(), obs=obs, var=var)
    return Source(adata=adata, kind=KIND, lazy=False, warnings=())


register_source(KIND, load, sniffer=sniff)


def read_sample(path: Path | str) -> ad.AnnData:
    """Any input annizarr can open (HISE, 10x, h5ad) as an in-memory AnnData of raw counts.

    Loaded eagerly: annizarr's lazy default leaves a 10x source's X on a separate reader and an
    h5ad's X backed by a file that close() shuts, and every step after this needs X in memory."""
    source = open_source(path, AppConfig(io=IOConfig(lazy=False)))
    try:
        return source.adata
    finally:
        source.close()


def sample_id(path: Path | str) -> str:
    stem = Path(path).stem
    match = _SAMPLE_ID.search(stem)
    return match.group(1) if match else stem
