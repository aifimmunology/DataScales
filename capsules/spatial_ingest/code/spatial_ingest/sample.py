"""One sample, read and made ready for the store: elements renamed into the sample's coordinate system,
and its own table, symbol-indexed, with QC metrics and optional CellTypist labels (spec §3S.4, §3S.8)."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse as sp

import annotate
import existing
import genes
import layout

logger = logging.getLogger(__name__)

BUNDLE_REGION = "cell_boundaries"
BUNDLE_REGION_KEY = "region"
BUNDLE_INSTANCE_KEY = "cell_id"
RECORD = "spatial_ingest"


class SampleError(ValueError):
    """A sample that can't be added as it is."""


@dataclass
class Loaded:
    """A sample as read: elements by kind and name, its table, and how the table annotates them."""
    elements: dict[str, dict[str, object]]
    table: ad.AnnData
    regions: list[str]
    region_key: str | None
    instance_key: str | None
    tags: tuple[str, ...] = ()


def read_bundle(kind: str, folder: Path, n_jobs: int) -> Loaded:
    """A Xenium or Atera bundle; its table is re-pointed at `cell_boundaries` by `cell_id`, which both
    readers use as the shapes' index."""
    if kind == "xenium":
        from spatialdata_io import xenium

        sdata = xenium(folder, n_jobs=n_jobs)
    else:
        from vendor.atera import atera

        sdata = atera(folder, morphology_3d_images=(folder / "morphology_3d").is_dir(), transcripts=True)
    if BUNDLE_REGION not in sdata.shapes or "table" not in sdata.tables:
        raise SampleError(f"{folder}: the reader found no {BUNDLE_REGION} or no cell table.")
    elements = {k: dict(getattr(sdata, k)) for k in existing.ELEMENT_KINDS}
    table = sdata.tables["table"]
    table.obs[BUNDLE_REGION_KEY] = BUNDLE_REGION
    return Loaded(elements, table, [BUNDLE_REGION], BUNDLE_REGION_KEY, BUNDLE_INSTANCE_KEY)


def read_part(path: Path, part: existing.Part) -> Loaded:
    """One sample of an existing SpatialData, read lazily: only its elements and its table rows."""
    import spatialdata as sd

    sdata = sd.read_zarr(path, selection=existing.ELEMENT_KINDS)
    elements: dict[str, dict[str, object]] = {k: {} for k in existing.ELEMENT_KINDS}
    for kind, name in part.elements:
        elements[kind][name] = getattr(sdata, kind)[name]
    table = existing.read_table(path, part)
    attrs = table.uns.pop("spatialdata_attrs", {}) if part.regions else {}
    return Loaded(elements, table, list(part.regions), attrs.get("region_key"), attrs.get("instance_key"), part.tags)


def counts(table: ad.AnnData, sid: str) -> sp.csr_matrix:
    """The table's raw counts as CSR: X, or `layers["counts"]` when X isn't counts."""
    for source in ("X", "counts"):
        x = table.X if source == "X" else table.layers.get("counts")
        if x is None:
            continue
        if isinstance(x, da.Array):
            x = x.compute()
        x = x.tocsr() if sp.issparse(x) else sp.csr_matrix(np.asarray(x))
        if genes.holds_counts(x):
            if source == "counts":
                logger.info("%s: X isn't raw counts; using layers['counts']", sid)
            return x
    raise SampleError(f"{sid}: X holds non-count values and there is no layers['counts']; the joint table takes "
                      "raw counts only.")


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


def fingerprint(table: ad.AnnData) -> str:
    """Identifies a sample's cells whatever its name: its exact counts (CSR) and cell IDs."""
    ids = table.obs["cell_id"].astype(str) if "cell_id" in table.obs else pd.Series(table.obs_names)
    x = sp.csr_matrix(table.X)
    digest = hashlib.sha256(f"{table.n_obs}:{table.n_vars}".encode())
    for part in (x.indptr.astype(np.int64), x.indices.astype(np.int64), x.data.astype(np.float64)):
        digest.update(np.ascontiguousarray(part).tobytes())
    digest.update("\n".join(ids).encode())
    return digest.hexdigest()


def build_table(loaded: Loaded, sid: str, model: Path | None, source: str) -> ad.AnnData:
    """The per-sample table: Gene Expression counts on symbols, the source's obs, obsm and obsp, plus
    `sample_id`, QC metrics and CellTypist labels, annotating the renamed regions."""
    from spatialdata.models import TableModel

    table = loaded.table
    x = counts(table, sid)
    keep, var = genes.symbol_var(table.var, sid)
    dropped = [k for k in ("layers", "varm") if len(getattr(table, k))]
    if dropped:
        logger.info("%s: %s left out of the per-sample table (gene-indexed for the source's features)",
                    sid, " and ".join(dropped))
    uns = {k: v for k, v in table.uns.items() if k not in ("spatialdata_attrs", RECORD)}
    out = ad.AnnData(x[:, keep], obs=table.obs.copy(), var=var, obsm=dict(table.obsm), obsp=dict(table.obsp), uns=uns)
    out.uns[RECORD] = {"celltypist_model": model.stem if model else "", "source": source, "fingerprint": fingerprint(out)}
    if out.obs.index.name in out.obs.columns:
        out.obs.index.name = None
    out.obs["sample_id"] = pd.Categorical([sid] * out.n_obs)
    qc = annotate.qc_metrics(out)
    out.obs = out.obs.drop(columns=[c for c in qc.columns if c in out.obs]).join(qc)
    if model is not None:
        labels = annotate.celltypist_labels(out, model)
        out.obs = out.obs.drop(columns=[c for c in labels.columns if c in out.obs]).join(labels)
    if not loaded.regions:
        return TableModel.parse(out)
    regions = {r: layout.element_name(r, sid, loaded.tags) for r in loaded.regions}
    out.obs[loaded.region_key] = pd.Categorical(out.obs[loaded.region_key].astype(str).map(regions))
    region = list(regions.values())
    return TableModel.parse(out, region=region[0] if len(region) == 1 else region,
                            region_key=loaded.region_key, instance_key=loaded.instance_key)


def build(loaded: Loaded, sid: str, model: Path | None, source: str):
    """The sample as a SpatialData holding only its renamed elements and its table `<sid>_table`."""
    from spatialdata import SpatialData

    table = build_table(loaded, sid, model, source)
    elements = renamed(loaded.elements, sid, loaded.tags)
    return SpatialData(**elements, tables={layout.table_name(sid): table})
