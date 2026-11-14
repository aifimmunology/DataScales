"""Sample sources: HISE H5 files (10x matrix plus a per-cell `observations` group) as an annizarr
input source, AnnData `.h5ad`/`.zarr`, and the table of a SpatialData `.zarr`, read directly with zarr.

Both HISE observation layouts are read onto the older HISE column names: tenx-scrna-pipeline's
(`Batch`, `Pool`, `Chip`, `uuid`, ...) is renamed through `_ALIASES`. Only Gene Expression
features are kept, so Xenium control probes and codewords never reach the store."""
from __future__ import annotations

import contextlib
import functools
import re
from pathlib import Path

import anndata as ad
import h5py
import zarr
from anndata.io import read_elem
import numpy as np
import pandas as pd
import scipy.sparse as sp
from annizarr import AppConfig
from annizarr.config import IOConfig
from annizarr.sources import Source, open_source, register_source

KIND = "hise"
_GEX = "Gene Expression"
_SAMPLE_ID = re.compile(r"_([A-Z0-9]+)-\d+$")
SPATIAL_TABLE = "tables/table"
# 10x/Xenium output names that say nothing about the sample; the sample ID comes from the folder instead.
_GENERIC_STEMS = {"cell_feature_matrix", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "sample_filtered_feature_bc_matrix"}
_GENERIC_DIRS = {"outs", "extract", "outputs", "output", "data", "count"}
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


def zarr_kind(path: Path | str) -> str | None:
    """What a zarr folder holds: "spatialdata", "anndata", "mudata", "zarr" (anything else), or None."""
    return _zarr_kind(str(path))


@functools.lru_cache(maxsize=None)
def _zarr_kind(path: str) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        root = zarr.open_group(path, mode="r")
    except Exception:
        return None
    if "spatialdata_attrs" in root.attrs and SPATIAL_TABLE in root:
        return "spatialdata"
    if root.attrs.get("encoding-type") == "MuData" or "mod" in root:
        return "mudata"
    if "X" in root and "obs" in root:
        return "anndata"
    return "zarr"


def read_object(path: Path | str) -> ad.AnnData:
    """An existing AnnData (`.h5ad`, AnnData `.zarr`, or a SpatialData table) as it is, in memory."""
    path = Path(path)
    kind = zarr_kind(path)
    if kind == "spatialdata":
        return ad.io.read_elem(zarr.open_group(path, mode="r")[SPATIAL_TABLE])
    return ad.read_zarr(path) if kind == "anndata" else ad.read_h5ad(path)


@contextlib.contextmanager
def object_obs(path: Path | str):
    """The obs group of an existing AnnData (`.h5ad`, AnnData `.zarr`, or a SpatialData table)."""
    path = Path(path)
    kind = zarr_kind(path)
    group = zarr.open_group(path, mode="r") if kind else h5py.File(path, "r")
    try:
        yield group[f"{SPATIAL_TABLE}/obs" if kind == "spatialdata" else "obs"]
    finally:
        if not kind:
            group.close()


def object_n_obs(path: Path | str) -> int:
    with object_obs(path) as obs:
        return len(read_elem(obs[obs.attrs["_index"]]))


def object_sample_ids(path: Path | str, column: str) -> list[str | None] | None:
    """The distinct values of obs `column` in an existing AnnData (None for a missing value), read from
    obs alone, or None without that column."""
    with object_obs(path) as obs:
        if column not in obs:
            return None
        return sorted({None if pd.isna(v) else str(v) for v in pd.unique(read_elem(obs[column]))},
                      key=lambda v: (v is None, v))


def gene_mask(var: pd.DataFrame) -> np.ndarray:
    """True for the Gene Expression features; everything is a gene when var has no feature type."""
    for col in ("feature_types", "feature_type"):
        if col in var:
            return np.asarray(var[col].astype(str) == _GEX)
    return np.ones(len(var), dtype=bool)


def genes_only(adata: ad.AnnData) -> ad.AnnData:
    keep = gene_mask(adata.var)
    return adata if keep.all() else adata[:, keep].copy()


def read_sample(path: Path | str) -> ad.AnnData:
    """Any input annizarr can open (HISE, 10x, h5ad), or a SpatialData `.zarr`'s table, as an
    in-memory AnnData of raw counts, Gene Expression features only.

    Loaded eagerly: annizarr's lazy default leaves a 10x source's X on a separate reader and an
    h5ad's X backed by a file that close() shuts, and every step after this needs X in memory."""
    if zarr_kind(path) in ("spatialdata", "anndata"):
        return genes_only(read_object(path))
    source = open_source(path, AppConfig(io=IOConfig(lazy=False)))
    try:
        return genes_only(source.adata)
    finally:
        source.close()


def gene_names(path: Path | str) -> np.ndarray:
    """The genes `read_sample` will return, read from var alone when the layout allows, for checking
    panels before any work."""
    path = Path(path)
    try:
        return _gene_names(path)
    except KeyError:
        return np.asarray(read_sample(path).var_names, dtype=object)


def _gene_names(path: Path) -> np.ndarray:
    kind = zarr_kind(path)
    if kind in ("spatialdata", "anndata"):
        var = read_elem(zarr.open_group(path, mode="r")[f"{SPATIAL_TABLE}/var" if kind == "spatialdata" else "var"])
    elif path.suffix.lower() == ".h5ad":
        with h5py.File(path, "r") as f:
            var = read_elem(f["var"])
    else:
        with h5py.File(path, "r") as f:
            feat = f["matrix/features"]
            var = pd.DataFrame({"feature_type": _text(feat["feature_type"][:])} if "feature_type" in feat else {},
                               index=pd.Index(_text(feat["name"][:])))
            if "observations" in f["matrix"]:
                var = var[gene_mask(var)]
                var.index = ad.utils.make_index_unique(var.index)
    return np.asarray(var.index[gene_mask(var)], dtype=object)


def sample_id(path: Path | str) -> str:
    """`B092-P1_SP02425-01.h5 -> SP02425`; a generic 10x/Xenium file name (`cell_feature_matrix.h5`)
    takes its folder's name instead, skipping folders such as `outs` or `extract`."""
    path = Path(path)
    stem = path.stem
    if stem in _GENERIC_STEMS:
        parent = path.parent
        while parent.name.lower() in _GENERIC_DIRS and parent != parent.parent:
            parent = parent.parent
        stem = parent.name or stem
    match = _SAMPLE_ID.search(stem)
    return match.group(1) if match else stem
