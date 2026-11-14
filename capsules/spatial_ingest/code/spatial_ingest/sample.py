"""One sample made ready for the store (spec §3S.4, §3S.8): its elements renamed into the sample's own
coordinate system, and its per-sample table written as a complete AnnData zarr group on scratch, a row
band at a time — X (CSR raw counts, symbol-indexed Gene Expression var), obs (source columns + `sample_id`
+ QC metrics + optional CellTypist labels), `uns/spatial_ingest` (model, source, fingerprint, genes in
panel) and the spatialdata table attributes annotating the renamed regions. The count matrix is never
held whole: QC, labels and the fingerprint are computed band by band."""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem

import annotate
import existing
import layout
import matrix
from contracts import Loaded, Prepared, Sample

logger = logging.getLogger(__name__)

BUNDLE_REGION = "cell_boundaries"
BUNDLE_REGION_KEY = "region"
BUNDLE_INSTANCE_KEY = "cell_id"
RECORD = "spatial_ingest"
TABLE_ATTRS = {"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2"}
LABEL_ROWS = 10_000            # cells per CellTypist call (it densifies the model's genes)
FINGERPRINT_CHUNK = 1 << 24


class SampleError(ValueError):
    """A sample that can't be added as it is."""


def renamed(elements: dict[str, dict[str, object]], sid: str, tags: tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Elements named `<sid>_<name>`, with every coordinate system renamed for the sample."""
    from spatialdata.transformations import get_transformation, set_transformation

    out: dict[str, dict[str, object]] = {}
    sources: dict[str, str] = {}
    for kind, items in elements.items():
        for name, element in items.items():
            new = layout.element_name(name, sid, tags)
            if new in sources:
                raise SampleError(f"{sid}: elements {sources[new]} and {name} would both be named {new}.")
            sources[new] = name
            systems = get_transformation(element, get_all=True)
            set_transformation(element, {layout.system_name(cs, sid, tags): t for cs, t in systems.items()}, set_all=True)
            out.setdefault(kind, {})[new] = element
    return out


class Fingerprint:
    """sha256 over "n_obs:n_vars", indptr (int64), indices (int64), data (float64) and the "\\n"-joined cell
    IDs — the same value as the in-memory fingerprint of earlier versions, fed band by band (indices while
    writing, data from the written array afterwards), so it doesn't depend on how the rows are chunked."""

    def __init__(self, n_obs: int, n_vars: int, indptr: np.ndarray):
        self.digest = hashlib.sha256(f"{n_obs}:{n_vars}".encode())
        self.digest.update(np.ascontiguousarray(indptr, dtype=np.int64).tobytes())

    def indices(self, band: sp.csr_matrix) -> None:
        self.digest.update(np.ascontiguousarray(band.indices, dtype=np.int64).tobytes())

    def finish(self, data, ids: pd.Series) -> str:
        for a in range(0, data.shape[0], FINGERPRINT_CHUNK):
            self.digest.update(np.ascontiguousarray(data[a:a + FINGERPRINT_CHUNK], dtype=np.float64).tobytes())
        self.digest.update("\n".join(ids.astype(str)).encode())
        return self.digest.hexdigest()


def cell_ids(obs: pd.DataFrame) -> pd.Series:
    return obs["cell_id"].astype(str) if "cell_id" in obs else pd.Series(obs.index.astype(str), index=obs.index)


def fingerprint_of(group_path: Path, obs: pd.DataFrame) -> str:
    """The fingerprint of a CSR AnnData zarr (or its X) holding these cells, read band by band."""
    n_obs, n_vars = matrix.shape(group_path)
    fp = Fingerprint(n_obs, n_vars, matrix.read_indptr(group_path))
    for _, band in matrix.iter_rows(group_path):
        fp.indices(band)
    return fp.finish(matrix.x_node(group_path)["data"], cell_ids(obs))


def _source_obs(obs: pd.DataFrame, sid: str) -> pd.DataFrame:
    obs = obs.copy()
    if obs.index.name in obs.columns:          # a column named like the index: anndata refuses to write it
        obs.index.name = None
    obs.index = pd.Index(obs.index.astype(str))
    obs["sample_id"] = pd.Categorical([sid] * len(obs))
    return obs


def _annotations(band: sp.csr_matrix, obs: pd.DataFrame, var: pd.DataFrame, model: Path | None) -> pd.DataFrame:
    """QC metrics (and CellTypist labels) of one row band; every cell is independent."""
    import anndata as ad

    chunk = ad.AnnData(band, obs=pd.DataFrame(index=obs.index), var=pd.DataFrame(index=var.index))
    parts = [annotate.qc_metrics(chunk)]
    if model is not None:
        labels = [annotate.celltypist_labels(chunk[a:a + LABEL_ROWS].copy(), model)
                  for a in range(0, chunk.n_obs, LABEL_ROWS)]
        parts.append(pd.concat(labels))
    return pd.concat(parts, axis=1)


def _categorical(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.columns:
        if frame[col].dtype == object or isinstance(frame[col].dtype, pd.CategoricalDtype) \
                or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = pd.Categorical(frame[col].astype(str))
    return frame


def _regions(loaded: Loaded, sid: str, obs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    regions = {r: layout.element_name(r, sid, loaded.tags) for r in loaded.regions}
    if loaded.region_key and regions:
        if loaded.region_key in obs:
            obs[loaded.region_key] = pd.Categorical(obs[loaded.region_key].astype(str).map(regions))
        else:
            obs[loaded.region_key] = pd.Categorical([next(iter(regions.values()))] * len(obs))
        if obs[loaded.region_key].isna().any():
            raise SampleError(f"{sid}: some cells annotate a region that isn't one of {list(regions)}.")
    return obs, list(regions.values())


def _new_table_group(path: Path) -> zarr.Group:
    shutil.rmtree(path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    return group


def _check_counts(band: sp.csr_matrix, sid: str) -> None:
    if band.nnz and (band.data.min() < 0 or np.any(np.mod(band.data, 1))):
        raise SampleError(f"{sid}: the table holds non-count values (negative or fractional); the store takes "
                          "raw counts only.")


def prepare(loaded: Loaded, sample: Sample, model: Path | None, scratch: Path) -> Prepared:
    """The sample ready to install (see the module docstring); the table goes to
    `scratch/prepared/<sid>/<sid>_table`."""
    sid = sample.sid
    source = loaded.table
    src = zarr.open_group(str(source.path), mode="r")
    var = read_elem(src["var"])
    obsm = dict(read_elem(src["obsm"])) if "obsm" in src else {}
    obs = _source_obs(source.obs, sid)
    n_obs, n_vars = matrix.shape(source.path)
    if len(obs) != n_obs:
        raise SampleError(f"{sid}: obs has {len(obs)} rows but the count table {n_obs}.")
    renames = existing.safe_names(list(obs.columns))       # cell_ID beside cell_id, invalid characters
    if renames:
        logger.info("%s: obs columns renamed so spatialdata can read the table: %s", sid,
                    ", ".join(f"{a} -> {b}" for a, b in renames.items()))
        obs = obs.rename(columns=renames)
        loaded.region_key = renames.get(loaded.region_key, loaded.region_key)
        loaded.instance_key = renames.get(loaded.instance_key, loaded.instance_key)
    obs, regions = _regions(loaded, sid, obs)

    out = scratch / "prepared" / sid / layout.table_name(sid)
    group = _new_table_group(out)
    writer = matrix.CSRWriter(group, n_vars, matrix.x_node(source.path)["data"].dtype)
    fp = Fingerprint(n_obs, n_vars, matrix.read_indptr(source.path))
    annotations = []
    for start, band in matrix.iter_rows(source.path):
        _check_counts(band, sid)
        writer.append(band)
        fp.indices(band)
        annotations.append(_annotations(band, obs.iloc[start:start + band.shape[0]], var, model))
    writer.close()
    fingerprint = fp.finish(group["X"]["data"], cell_ids(obs))

    extra = _categorical(pd.concat(annotations)) if annotations else pd.DataFrame(index=obs.index)
    extra.index = obs.index
    obs = obs.drop(columns=[c for c in extra.columns if c in obs]).join(extra)
    write_elem(group, "obs", obs)
    write_elem(group, "var", var)
    write_elem(group, "obsm", {k: np.asarray(v) for k, v in obsm.items()})
    attrs = {"region": (regions[0] if len(regions) == 1 else regions) if regions else None,
             "region_key": loaded.region_key if regions else None,
             "instance_key": loaded.instance_key if regions else None}
    uns = {RECORD: {"celltypist_model": model.stem if model else "", "source": str(sample.candidate.origin),
                    "fingerprint": fingerprint, "genes_in_panel": int(n_vars)}}
    if regions:
        uns["spatialdata_attrs"] = dict(attrs)
    write_elem(group, "uns", uns)
    group.attrs.update({**TABLE_ATTRS, **attrs})

    label = f"{model.stem}_label" if model else None
    stats = {"n_cells": int(n_obs), "genes_in_panel": int(n_vars),
             "mean_genes": float(obs["n_genes_by_counts"].mean()) if n_obs else 0.0,
             "mean_total_counts": float(obs["total_counts"].mean()) if n_obs else 0.0,
             "n_celltypist_labels": int(obs[label].nunique()) if label and label in obs else 0}
    elements = renamed(loaded.elements, sid, loaded.tags)
    return Prepared(sid, elements, out, obs, var, fingerprint, loaded.region_key, stats)
