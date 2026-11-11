"""The joint single-cell table `tables/table`: X, obs and var of every sample, no spatial elements
(spec §3S.8). Rows are staged as an AnnData zarr and added with annizarr, in place, so the table is
never read whole; the first sample seeds it."""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anndata as ad
import annizarr
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, write_elem
from annizarr import AppConfig
from annizarr.config import ChunkConfig, IOConfig

import genes
import store as stores

logger = logging.getLogger(__name__)

NAME = "table"
UNS_KEY = "spatial_ingest"
LEADING = ("sample_id", "cell_id", "n_genes_by_counts", "total_counts", "pct_counts_mt")
EXCLUDED = {"region", "cell_labels", "label_id", "barcode", "total_counts_mt"}
TABLE_ATTRS = {"spatialdata-encoding-type": "ngff:regions_table", "version": "0.2"}


class JointError(ValueError):
    """Rows the joint table can't take."""


@dataclass
class Schema:
    """What every sample's rows must match: the first sample's genes, X dtype and obs columns, the
    CellTypist model and species, plus the samples in table order."""
    genes: list[str]
    gene_ids: list[str]
    x_dtype: str
    obs: dict[str, str]
    panel_source: str
    celltypist_model: str = ""
    species: str = ""
    samples: list[str] = field(default_factory=list)

    @property
    def var(self) -> pd.DataFrame:
        return pd.DataFrame({"gene_ids": self.gene_ids, "feature_types": genes.GEX},
                            index=pd.Index(np.asarray(self.genes, dtype=object)))

    def record(self) -> dict:
        """The schema as stored in `uns/spatial_ingest` (genes come from var)."""
        out = {k: v for k, v in asdict(self).items() if k not in ("genes", "gene_ids")}
        out["obs"] = json.dumps(self.obs)
        return out


def table_path(store: Path) -> Path:
    return store / "tables" / NAME


def kinds(obs: pd.DataFrame) -> dict[str, str]:
    out = {}
    for c, dtype in obs.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            out[c] = "category"
        elif str(dtype) in ("Int64", "boolean", "bool"):
            out[c] = str(dtype)
        elif dtype == object or pd.api.types.is_string_dtype(dtype):
            out[c] = "str"
        else:
            out[c] = str(dtype)
    return out


def columns(table: ad.AnnData, model: str) -> list[str]:
    """The joint obs columns a first sample sets: the fixed ones, CellTypist's, then its remaining
    reader columns except region links and per-sample clusters."""
    labels = [f"{model}_label", f"{model}_prob"] if model else []
    skip = set(LEADING) | set(labels) | EXCLUDED | {table.uns.get("spatialdata_attrs", {}).get("region_key")}
    rest = [c for c in table.obs.columns if c not in skip and not c.startswith("leiden")]
    return [*LEADING, *labels, *rest]


def match_obs(obs: pd.DataFrame, target: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """obs with the schema's columns, in the schema's kinds; a column that can't convert is left out
    (returned by name), since an append needs matching types."""
    obs, failed = obs[[c for c in target if c in obs]].copy(), []
    for c, kind in target.items():
        if c not in obs:
            continue
        try:
            if kind == "category":
                obs[c] = obs[c].astype(str).where(obs[c].notna()).astype("category")
            elif kind == "str":
                obs[c] = obs[c].astype(str).where(obs[c].notna(), "")
            else:
                obs[c] = obs[c].astype(kind)
        except (TypeError, ValueError):
            obs = obs.drop(columns=c)
            failed.append(c)
    return obs, failed


def first_schema(table: ad.AnnData, sid: str, model: str, species: str | None) -> Schema:
    obs = rows_obs(table, sid, columns(table, model))
    return Schema(list(table.var_names), list(table.var["gene_ids"].astype(str)), str(table.X.dtype), kinds(obs),
                  sid, model, species or "")


def rows_obs(table: ad.AnnData, sid: str, cols: list[str]) -> pd.DataFrame:
    obs = table.obs.copy()
    obs["sample_id"] = pd.Categorical([sid] * len(obs))
    obs["cell_id"] = obs["cell_id"].astype(str) if "cell_id" in obs else np.asarray(obs.index, dtype=str)
    obs = obs[[c for c in dict.fromkeys(cols) if c in obs]]
    names = pd.Index([f"{sid}:{c}" for c in obs["cell_id"]], dtype=object)
    region_key = table.uns.get("spatialdata_attrs", {}).get("region_key")
    if not names.is_unique and region_key in table.obs:
        names = pd.Index([f"{sid}:{r}:{c}" for r, c in zip(table.obs[region_key].astype(str), obs["cell_id"])],
                         dtype=object)
    if not names.is_unique:
        raise JointError(f"{sid}: cell IDs repeat within the sample, so its joint cell names wouldn't be unique.")
    obs.index = names
    return obs


def rows(table: ad.AnnData, sid: str, schema: Schema) -> tuple[ad.AnnData, str]:
    """The sample's joint rows, fitted to the schema, and a note on how its genes were fitted."""
    diff = genes.panel_diff(schema.genes, table.var_names)
    try:
        x = genes.fit_counts(sp.csr_matrix(table.X), table.var_names, schema.genes, np.dtype(schema.x_dtype))
    except genes.PanelError as exc:
        raise JointError(f"{sid}: {exc}") from None
    obs = rows_obs(table, sid, list(table.obs.columns) + ["sample_id", "cell_id"])
    region_key = table.uns.get("spatialdata_attrs", {}).get("region_key")
    extra = [c for c in obs.columns if c not in schema.obs and c not in EXCLUDED and c != region_key
             and not c.startswith("leiden")]
    obs, failed = match_obs(obs, schema.obs)
    if extra:
        logger.warning("%s: obs columns the joint table doesn't have are left out of it: %s", sid, ", ".join(extra))
    if failed:
        logger.warning("%s: obs columns left empty in the joint table, since their values don't fit its types: %s",
                       sid, ", ".join(failed))
    return ad.AnnData(x, obs=obs, var=schema.var), genes.fit_note(diff)


def config(cpus: int) -> AppConfig:
    return AppConfig(io=IOConfig(overwrite=True), chunks=ChunkConfig(cpus=cpus))


def read_schema(store: Path) -> Schema | None:
    path = table_path(store)
    if not (path / "zarr.json").exists():
        return None
    group = zarr.open_group(path, mode="r")
    var = read_elem(group["var"])
    record = read_elem(group["uns"][UNS_KEY]) if "uns" in group and UNS_KEY in group["uns"] else None
    if record is None:
        raise JointError(f"{path} wasn't written by this capsule (no uns/{UNS_KEY}); start a new store.")
    record = dict(record)
    return Schema(list(var.index), list(var["gene_ids"].astype(str)), str(record["x_dtype"]),
                  json.loads(record["obs"]), str(record["panel_source"]), str(record.get("celltypist_model", "")),
                  str(record.get("species", "")), [str(s) for s in np.atleast_1d(record.get("samples", []))])


def write_schema(path: Path, schema: Schema) -> None:
    group = zarr.open_group(path, mode="r+", use_consolidated=False)
    write_elem(group["uns"] if "uns" in group else group.require_group("uns"), UNS_KEY, schema.record())


def add(store: Path, adata: ad.AnnData, sid: str, schema: Schema, staging: Path, cpus: int) -> None:
    """Adds one sample's rows: seeds the table (staged, then renamed into place) or appends in place."""
    cfg = config(cpus)
    staged = staging / f"{sid}.joint.zarr"
    annizarr.convert(adata, output=str(staged), cfg=cfg)
    target = table_path(store)
    schema.samples.append(sid)
    if (target / "zarr.json").exists():
        annizarr.append(str(target), cells=str(staged), cfg=cfg)
        write_schema(target, schema)
    else:
        group = zarr.open_group(staged, mode="r+", use_consolidated=False)
        group.attrs.update(TABLE_ATTRS)
        write_schema(staged, schema)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staged, target)
    shutil.rmtree(staged, ignore_errors=True)


def sample_counts(store: Path) -> pd.Series | None:
    """Cells per sample in the joint table, or None when its obs, X and indptr disagree (a killed append)."""
    path = table_path(store)
    try:
        group = zarr.open_group(path, mode="r", use_consolidated=False)
        sample = read_elem(group["obs"]["sample_id"])
        n_index = stores.n_obs(group)
        shape = group["X"].attrs["shape"]
        indptr = group["X"]["indptr"].shape[0]
    except Exception:
        return None
    if not (len(sample) == n_index == shape[0] == indptr - 1):
        return None
    return pd.Series(sample).value_counts()


def drop(store: Path) -> None:
    shutil.rmtree(table_path(store), ignore_errors=True)
