"""Preparing each new input for the store, before the store is touched: the store contract (a fixed obs
schema, CSR uint32 counts), per-sample annotation (QC metrics, CellTypist, Scrublet), the gene policy,
and staging each input as a zarr in the run's staging folder.

The gene policy: a store's genes are fixed when it is created. A sample with other genes is fitted:
store genes it lacks are zero-filled and its other genes dropped, with a warning and a note in the
commit. A sample sharing no gene is refused.

Existing AnnData objects are staged out of core: annizarr streams each one into a staging zarr, then
only obs and var (small) are edited in memory, and X is rewritten block by block only when its genes or
dtype must change."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import annizarr
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import zarr
from anndata.io import read_elem, sparse_dataset, write_elem
from annizarr import AppConfig, convert, rechunk

from config import Settings
from inputs import (SPATIAL_TABLE, IngestError, ObjectInput, RawInput, gene_mask, gene_names, read_sample,
                    zarr_kind)

logger = logging.getLogger("ingest")
# Keeps celltypist's progress lines quiet even where setup_logging isn't called (tests, notebooks).
logging.getLogger("celltypist").setLevel(logging.WARNING)

X_DTYPE = np.uint32
OBS_INDEX = "barcodes"
MITO_PREFIX = "MT-"
EXPECTED_DOUBLET_RATE = 0.06
PER_CELL = ("obsm", "obsp", "layers", "varm", "uns")
ROWS_PER_BLOCK = 20_000

STR, INT, FLOAT = "str", "Int64", "float64"

HISE_STR_COLS = (
    "adt_qc_flag", "batch_id", "cell_name", "cell_uuid", "chip_id", "hto_barcode",
    "hto_category", "original_barcodes", "pbmc_sample_id", "pool_id", "well_id",
)
HISE_INT_COLS = ("adt_umis", "n_genes", "n_mito_umis", "n_reads", "n_umis")
QC_COLS = {"n_genes_by_counts": INT, "total_counts": FLOAT, "total_counts_mt": FLOAT, "pct_counts_mt": FLOAT}
SCRUBLET_COLS = {"scrublet_score": FLOAT, "predicted_doublets": INT}

Schema = dict[str, str]


class ContractError(ValueError):
    """A sample the store cannot take as it is."""


@dataclass(frozen=True)
class Staged:
    """One input staged as a zarr, ready to seed or append: what it adds and how it was fitted."""
    sample_id: str
    path: Path
    summary: dict
    source: str = ""
    fitted: str = ""
    samples: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        return f"- {self.sample_id}: {self.source}" + (f" ({self.fitted})" if self.fitted else "")


def label_columns(model: str) -> Schema:
    return {f"{model}_label": STR, f"{model}_prob": FLOAT}


def requested_schema(run_scrublet: bool, models: list[str]) -> Schema:
    """The obs columns, in order; each CellTypist model adds `<model>_label` and `<model>_prob`."""
    schema: Schema = {"sample_id": STR}
    schema.update(dict.fromkeys(HISE_STR_COLS, STR))
    schema.update(dict.fromkeys(HISE_INT_COLS, INT))
    schema.update(QC_COLS)
    if run_scrublet:
        schema.update(SCRUBLET_COLS)
    for model in models:
        schema.update(label_columns(model))
    return schema


def store_models(store_columns: list[str]) -> list[str]:
    """The CellTypist models a store was built with, from its `<model>_label` columns."""
    return [c.removesuffix("_label") for c in store_columns if c.endswith("_label")]


def check_schema(store_columns: list[str], schema: Schema, run_scrublet: bool = True) -> None:
    """Appends need identical obs columns; a store's schema is fixed when it is created."""
    if store_columns == list(schema):
        return
    missing = sorted(set(store_columns) - set(schema))
    extra = sorted(set(schema) - set(store_columns))
    scrublet = "with" if "scrublet_score" in store_columns else "without"
    raise ContractError(
        "This run's obs schema differs from the store's, so its cells cannot be appended. "
        f"Store-only columns: {missing}; run-only columns: {extra}. The store was built {scrublet} Scrublet "
        f"and with CellTypist models {store_models(store_columns) or 'none'}; choose the same Celltypist "
        "option, or start a new store."
    )


def check_raw_schema(store_columns: list[str], schema: Schema, run_scrublet: bool) -> None:
    """A store built from raw samples keeps its exact schema; one seeded from an existing AnnData keeps
    its own columns, and raw samples fill only those."""
    if set(requested_schema(False, [])) <= set(store_columns):
        check_schema(store_columns, schema, run_scrublet)
        return
    missing = [c for c in schema if c not in store_columns]
    if missing:
        logger.warning("The store was built from existing AnnData, so these raw-sample columns aren't kept: %s",
                       ", ".join(missing))


def is_mito(var_names) -> np.ndarray:
    return np.char.startswith(np.asarray(var_names).astype(str), MITO_PREFIX)


def _empty_column(kind: str, n: int):
    if kind == STR:
        return np.full(n, "", dtype=object)
    if kind == FLOAT:
        return np.full(n, np.nan, dtype=np.float64)
    return pd.array([pd.NA] * n, dtype="Int64")


def _project_column(values: pd.Series | None, kind: str, n: int):
    if values is None:
        return _empty_column(kind, n)
    if kind == STR:
        return np.array(["" if pd.isna(v) else str(v) for v in values.to_numpy(dtype=object)], dtype=object)
    numeric = pd.to_numeric(values, errors="raise")
    if kind == FLOAT:
        return numeric.astype(np.float64).to_numpy()
    return pd.array(numeric, dtype="Int64")


def _as_counts(matrix) -> sp.csr_matrix:
    x = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    if x.data.size:
        if np.issubdtype(x.data.dtype, np.floating) and ((x.data < 0).any() or (x.data != np.round(x.data)).any()):
            raise ContractError("X holds negative or fractional values; the store takes raw counts only.")
        if int(x.data.max()) > np.iinfo(X_DTYPE).max:
            raise ContractError(f"X holds a count larger than {X_DTYPE.__name__} can store.")
    return x.astype(X_DTYPE)


def holds_counts(matrix) -> bool:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    return bool(np.all(values >= 0) and np.all(np.mod(values, 1) == 0))


def fit_x(matrix, dtype: np.dtype | None, counts: bool, name: str):
    """X in the store's dtype. A store of counts takes only counts; values must fit the dtype."""
    if counts and not holds_counts(matrix):
        raise ContractError(f"{name}: X holds non-count values, but the store's X holds counts; ingest raw "
                            "counts, or start a new store for normalized data.")
    if dtype is None or matrix.dtype == dtype:
        return matrix
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if np.issubdtype(dtype, np.integer) and values.size and values.max() > np.iinfo(dtype).max:
        raise ContractError(f"{name}: X values up to {values.max():g} don't fit the store's {dtype}.")
    return matrix.astype(dtype)


def frame_kinds(obs: pd.DataFrame) -> dict[str, str]:
    kinds = {}
    for c, dtype in obs.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            kinds[c] = "category"
        elif str(dtype) in ("Int64", "boolean"):
            kinds[c] = str(dtype)
        elif dtype == object or pd.api.types.is_string_dtype(dtype):
            kinds[c] = "str"
        else:
            kinds[c] = str(dtype)
    return kinds


def match_obs(obs: pd.DataFrame, kinds: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """obs with the store's columns converted to the store's kinds; a column that can't convert is left
    empty (returned by name), since an append needs matching types."""
    obs, failed = obs.copy(), []
    for c in [c for c in obs.columns if c in kinds]:
        kind = kinds[c]
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


def project(adata: ad.AnnData, schema: Schema, expected_var_names: np.ndarray | None = None) -> ad.AnnData:
    """A new AnnData holding exactly what the store accepts: CSR uint32 X, the schema's obs, and var."""
    var_names = np.asarray(adata.var_names, dtype=object)
    check_same_genes(var_names, expected_var_names)
    obs = pd.DataFrame(index=pd.Index(np.asarray(adata.obs_names, dtype=object), name=OBS_INDEX))
    for col, kind in schema.items():
        obs[col] = _project_column(adata.obs[col] if col in adata.obs else None, kind, adata.n_obs)
    var = pd.DataFrame(index=pd.Index(var_names))
    var["mt"] = is_mito(var_names)
    return ad.AnnData(X=_as_counts(adata.X), obs=obs, var=var)


def qc_metrics(adata: ad.AnnData) -> None:
    adata.var["mt"] = is_mito(adata.var_names)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)


def run_scrublet(adata: ad.AnnData, random_state: int = 0) -> None:
    """Scores doublets into obs. If Scrublet fails the columns stay absent and the contract pads them
    with NA; if it finds no threshold the scores are kept and the calls are NA."""
    import scrublet as scr

    counts = adata.X if sp.isspmatrix_csc(adata.X) else sp.csc_matrix(adata.X)
    try:
        scrub = scr.Scrublet(counts, expected_doublet_rate=EXPECTED_DOUBLET_RATE, random_state=random_state)
        scores, doublets = scrub.scrub_doublets(verbose=False)
    except Exception as exc:  # scrublet fails on tiny or empty samples
        logger.warning("Scrublet failed for this sample (%s: %s); its cells get NA doublet scores",
                       type(exc).__name__, exc)
        return
    adata.obs["scrublet_score"] = scores
    if doublets is None:
        logger.warning("Scrublet found no threshold; predicted_doublets left NA, scores kept")
        adata.obs["predicted_doublets"] = pd.array([pd.NA] * adata.n_obs, dtype="Int64")
        return
    adata.obs["predicted_doublets"] = doublets.astype(int)
    logger.info("Scrublet: %d / %d doublets", int(doublets.sum()), adata.n_obs)


def run_celltypist(adata: ad.AnnData, models: list[tuple[str, object]]) -> None:
    """Labels every cell with each CellTypist model, writing `<model>_label` and `<model>_prob` into obs.

    The models share one normalized copy (counts per 1e4, log1p), as the tenx-scrna-pipeline does."""
    import celltypist

    normalized = adata.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    for name, model in models:
        predictions = celltypist.annotate(normalized, model=model)
        labels = predictions.predicted_labels["predicted_labels"].astype(str).to_numpy()
        probs = predictions.probability_matrix
        picked = probs.columns.get_indexer(labels)
        label, prob = label_columns(name)
        adata.obs[label] = labels
        adata.obs[prob] = np.where(picked >= 0, probs.to_numpy()[np.arange(len(labels)), picked], np.nan)
        logger.info("CellTypist %s: %d labels", name, len(np.unique(labels)))


def check_same_genes(var_names: np.ndarray, expected: np.ndarray | None) -> None:
    """A last guard before a sample is staged: its genes are exactly the store's, in order."""
    if expected is None or np.array_equal(var_names, np.asarray(expected, dtype=object)):
        return
    if len(var_names) == len(expected) and set(var_names) == set(expected):
        found = "the same genes in a different order"
    else:
        found = f"{len(var_names)} genes, the store {len(expected)}, {len(set(var_names) ^ set(expected))} not shared"
    raise ContractError(f"var mismatch: the sample has {found}; gene names and order must match the store's.")


def panel_diff(store_genes, sample_genes, n_examples: int = 5) -> dict | None:
    """How a sample's genes differ from the store's, or None when names and order match."""
    store_genes, sample_genes = list(store_genes), list(sample_genes)
    if store_genes == sample_genes:
        return None
    in_sample, in_store = set(sample_genes), set(store_genes)
    only_store = [g for g in store_genes if g not in in_sample]
    only_sample = [g for g in sample_genes if g not in in_store]
    return {"n_shared": len(in_store & in_sample), "n_only_in_store": len(only_store),
            "n_only_in_sample": len(only_sample), "only_in_store": only_store[:n_examples],
            "only_in_sample": only_sample[:n_examples]}


def require_shared(diff: dict | None, n_genes: int, name: str) -> None:
    """Refuses a sample that shares no gene with the store (or, for a new store, no gene-expression
    feature at all): fitting it would store all-zero cells."""
    if n_genes == 0:
        raise ContractError(f"{name} has no Gene Expression features to store.")
    if diff and diff["n_shared"] == 0:
        raise ContractError(f"{name} shares no gene names with the store ({n_genes} genes, e.g. "
                            f"{', '.join(diff['only_in_store'][:3])}; the sample has e.g. "
                            f"{', '.join(diff['only_in_sample'][:3])}); its cells would be stored as all zeros. "
                            "Check that both use the same gene identifiers (symbols vs Ensembl IDs).")


def describe_diff(sid: str, diff: dict) -> str:
    if not diff["n_only_in_store"] and not diff["n_only_in_sample"]:
        return f"{sid}: the same {diff['n_shared']} genes in a different order"
    return (f"{sid}: {diff['n_shared']} shared, {diff['n_only_in_store']} only in the store "
            f"(e.g. {', '.join(diff['only_in_store']) or '-'}), {diff['n_only_in_sample']} only in the sample "
            f"(e.g. {', '.join(diff['only_in_sample']) or '-'})")


def fit_note(diff: dict) -> str:
    if not diff["n_only_in_store"] and not diff["n_only_in_sample"]:
        return "genes reordered to the store's"
    return (f"fitted to the store's genes: {diff['n_only_in_store']} zero-filled, "
            f"{diff['n_only_in_sample']} dropped")


def gene_picker(sample_genes, store_genes) -> sp.csr_matrix:
    """A 0/1 matrix that maps a sample's genes onto the store's: X @ picker gives the store's genes,
    zero for those the sample lacks. Duplicate sample genes map by their first occurrence."""
    src = pd.Index(ad.utils.make_index_unique(pd.Index(np.asarray(sample_genes, dtype=object)))).get_indexer(
        pd.Index(np.asarray(store_genes, dtype=object)))
    present = src >= 0
    return sp.csr_matrix((np.ones(present.sum(), dtype=np.float32), (src[present], np.flatnonzero(present))),
                         shape=(len(sample_genes), len(store_genes)))


def conform_var(adata: ad.AnnData, store_genes) -> ad.AnnData:
    """The sample's counts on the store's genes: store genes the sample lacks are zero-filled and its
    genes the store lacks are dropped. Annotation must run before this, on the sample's own genes."""
    x = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    fitted = sp.csr_matrix(x @ gene_picker(adata.var_names, store_genes)).astype(x.dtype)
    return ad.AnnData(fitted, obs=adata.obs.copy(), var=pd.DataFrame(index=pd.Index(np.asarray(store_genes, dtype=object))))


def fit_genes(adata: ad.AnnData, store_genes: np.ndarray | None) -> tuple[ad.AnnData, str]:
    """The sample on the store's genes (as it is for a new store), and a note on how they were fitted."""
    diff = None if store_genes is None else panel_diff(store_genes, adata.var_names)
    return (conform_var(adata, store_genes), fit_note(diff)) if diff else (adata, "")


def check_panels(new: list[RawInput | ObjectInput], store_genes: np.ndarray | None) -> None:
    """Warns, from var alone and before any sample is read, about each new sample whose genes differ
    from the store's (or, for a new store, the first sample's); refuses one sharing none."""
    reference = store_genes
    diffs = {}
    for item in new:
        names = gene_names(item.path)
        if reference is None:
            require_shared(None, len(names), item.name)
            reference = names
            continue
        diff = panel_diff(reference, names)
        require_shared(diff, len(reference), item.name)
        if diff:
            diffs[item.name] = diff
    if diffs:
        against = ("the store's" if store_genes is not None
                   else f"the first sample's ({new[0].name}, {len(reference)} genes)")
        logger.warning("Gene panel differs from %s for %d sample(s); they are fitted to its genes (missing genes "
                       "zero-filled, extra genes dropped): %s", against, len(diffs),
                       "; ".join(describe_diff(sid, d) for sid, d in diffs.items()))


@dataclass
class Target:
    """What staged cells must match: the store's genes, X dtype, whether X holds counts, and obs kinds.
    A new store takes them from its first staged sample."""
    var: np.ndarray | None = None
    x_dtype: np.dtype | None = None
    counts: bool = False
    obs: dict[str, str] | None = None

    def fit_cells(self, adata: ad.AnnData, name: str) -> ad.AnnData:
        """The sample's X and obs in the store's dtype and kinds; the first sample of a new store sets them."""
        adata.X = fit_x(adata.X, self.x_dtype, self.counts, name)
        if self.obs is not None:
            adata.obs, failed = match_obs(adata.obs, self.obs)
            if failed:
                logger.warning("%s: obs columns left empty, since their values don't fit the store's types: %s",
                               name, ", ".join(failed))
        if self.var is None:
            self.learn(np.asarray(adata.var_names, dtype=object), adata.X.dtype, holds_counts(adata.X),
                       frame_kinds(adata.obs))
        return adata

    def learn(self, var: np.ndarray, x_dtype: np.dtype, counts: bool, obs: dict[str, str]) -> None:
        self.var, self.x_dtype, self.counts, self.obs = var, x_dtype, counts, obs


def unique_names(names: list[str], name: str) -> pd.Index:
    """Cell names, with repeated barcodes (e.g. lanes concatenated without a suffix) made unique."""
    index = pd.Index(names)
    if index.is_unique:
        return index
    logger.warning("%s: %d repeated barcodes; suffixed -1, -2, ... to keep cell names unique", name,
                   int(index.duplicated().sum()))
    return pd.Index(ad.utils.make_index_unique(index))


def annotate_sample(adata: ad.AnnData, sid: str, settings: Settings, models: list) -> ad.AnnData:
    adata.obs_names = unique_names([f"{sid}:{bc}" for bc in adata.obs_names], sid)
    adata.obs["sample_id"] = sid
    qc_metrics(adata)
    if models:
        run_celltypist(adata, models)
    if settings.run_scrublet:
        run_scrublet(adata, random_state=settings.random_seed)
    return adata


def summarize(obs: pd.DataFrame) -> dict:
    row = {"n_cells": len(obs), "n_samples": obs["sample_id"].nunique()}
    if "n_genes_by_counts" in obs:
        row["mean_genes"] = obs["n_genes_by_counts"].mean()
        row["mean_pct_mt"] = obs["pct_counts_mt"].mean()
    if "predicted_doublets" in obs:
        row["n_doublets"] = obs["predicted_doublets"].sum(min_count=1)
        row["scrublet_scored"] = bool(obs["scrublet_score"].notna().any())
    for column in obs.columns[obs.columns.str.endswith("_label")]:
        row[f"n_{column.removesuffix('_label')}_types"] = obs[column].nunique()
    return row


def stage_samples(new: list[RawInput], settings: Settings, models: list, schema: Schema, target: Target,
                  cfg: AppConfig, staging: Path) -> list[Staged]:
    """Raw samples, one in memory at a time: annotated on their own genes, then fitted to the store's."""
    staged = []
    for item in new:
        sid = item.sample_id
        try:
            adata = read_sample(item.path)
        except (OSError, KeyError) as exc:
            raise IngestError(f"{item.path} can't be read ({type(exc).__name__}: {exc}); it may be truncated. "
                              "Nothing from this run was added.") from exc
        logger.info("%s: %d cells x %d genes", sid, adata.n_obs, adata.n_vars)
        if adata.n_obs == 0:
            logger.warning("%s has no cells; skipped", sid)
            continue
        adata, note = fit_genes(annotate_sample(adata, sid, settings, models), target.var)
        adata = target.fit_cells(project(adata, schema, target.var), sid)
        out = staging / f"{len(staged):04d}.zarr"
        annizarr.convert(adata, output=str(out), cfg=cfg)
        staged.append(Staged(sid, out, summarize(adata.obs), str(item.path), note, (sid,)))
        del adata
    return staged


def stage_objects(new: list[ObjectInput], target: Target, cfg: AppConfig, staging: Path) -> list[Staged]:
    """Existing AnnData objects, streamed by annizarr. The first object of a new store keeps everything;
    later ones keep X, the store's obs columns and var, since an append can't extend embeddings or layers."""
    staged = []
    for item in new:
        out = staging / f"{len(staged):04d}.zarr"
        obs, note, left_out = stage_object(item.path, out, item.column, item.sample_ids[0], target,
                                           keep_all=target.var is None, cfg=cfg, name=item.label)
        logger.info("%s: %d cells, samples %s", item.label, len(obs), list(item.sample_ids))
        if left_out:
            logger.warning("%s: not added, since the store has no place for them: %s", item.label, ", ".join(left_out))
        staged.append(Staged(item.label, out, summarize(obs), str(item.path), note, item.sample_ids))
    return staged


def stage_object(path: Path, out: Path, column: str | None, sample: str, target: Target, keep_all: bool,
                 cfg: AppConfig, name: str) -> tuple[pd.DataFrame, str, list[str]]:
    """Streams one existing AnnData into `out` and fits it to `target` (the store's genes, X dtype,
    counts and obs kinds). Returns its obs, a note on gene fitting, and what was left out."""
    try:
        left_out = copy_in(path, out, cfg, keep_all)
    except TypeError as exc:
        raise ContractError(f"{name}: annizarr can't copy {path.name} ({exc}).") from exc
    root = zarr.open_group(out, mode="r+", use_consolidated=False)
    var, obs = read_elem(root["var"]), read_elem(root["obs"])

    columns = np.asarray(var.index, dtype=object)
    wanted = columns[gene_mask(var)] if target.var is None else target.var
    diff = panel_diff(wanted, columns)
    require_shared(diff, len(wanted), name)
    x = matrix(root["X"])
    counts, top = x_stats(x)
    if target.counts and not counts:
        raise ContractError(f"{name}: X holds non-count values, but the store's X holds counts; ingest "
                            "raw counts, or start a new store for normalized data.")
    dtype = np.dtype(target.x_dtype or x.dtype)
    if np.issubdtype(dtype, np.integer) and top > np.iinfo(dtype).max:
        raise ContractError(f"{name}: X values up to {top:g} don't fit the store's {dtype}.")
    if diff or dtype != x.dtype:
        rewrite_x(root, out, "X", x, columns, wanted, dtype)
    if diff and keep_all:
        left_out += fit_per_gene(root, out, columns, wanted)
    if diff:
        write_elem(root, "var", var.reindex(wanted) if target.var is None else pd.DataFrame(index=pd.Index(wanted)))

    obs["sample_id"] = obs[column].astype(str) if column else sample
    obs.index = unique_names([n if n.startswith(f"{sid}:") else f"{sid}:{n}"
                              for sid, n in zip(obs["sample_id"], obs.index)], name)
    if target.obs is not None:
        left_out += [f"obs/{c}" for c in obs.columns if c not in target.obs]
        obs, failed = match_obs(obs[[c for c in obs.columns if c in target.obs]], target.obs)
        left_out += [f"obs/{c} (values don't fit the store's type)" for c in failed]
    write_elem(root, "obs", obs)
    zarr.consolidate_metadata(str(out))
    note = fit_note(diff) if diff and target.var is not None else ""
    if target.var is None:
        target.learn(wanted, dtype, counts, frame_kinds(obs))
    return obs, note, left_out


def left_behind(group: zarr.Group, keep_all: bool) -> list[str]:
    """What the store has no place for: `raw` always, per-cell elements unless the object is kept whole."""
    found = ["raw"] if "raw" in group else []
    return found + ([] if keep_all else [f"{key}/{k}" for key in PER_CELL if key in group for k in group[key]])


def matrix(node):
    return sparse_dataset(node) if isinstance(node, zarr.Group) else node


def copy_in(path: Path, out: Path, cfg: AppConfig, keep_all: bool) -> list[str]:
    """Streams one existing AnnData into the staging zarr `out`; returns what was left out."""
    kind = zarr_kind(path)
    source = path / SPATIAL_TABLE if kind == "spatialdata" else path
    src = zarr.open_group(source, mode="r") if kind else None
    if src is not None and src.metadata.zarr_format == 2:
        copy_v2(src, out, keep_all)
        return left_behind(src, keep_all)
    if kind:
        rechunk(str(source), output=str(out), cfg=cfg)
    else:
        convert(str(path), output=str(out), cfg=cfg, fmt="h5ad")
    root = zarr.open_group(out, mode="r+", use_consolidated=False)
    left_out = left_behind(root, keep_all)
    for key in left_out:
        del root[key]
    return left_out


def copy_v2(src: zarr.Group, out: Path, keep_all: bool) -> None:
    """A zarr v2 AnnData (which annizarr can't rechunk) as v3: matrices in row blocks, the rest whole."""
    root = zarr.open_group(out, mode="w", zarr_format=3)
    root.attrs.update(src.attrs.asdict())
    for key in src:
        if key == "raw" or (key in PER_CELL and not keep_all):
            continue
        if key == "X":
            x = matrix(src["X"])
            write_csr(root, "X", blocks(x), x.shape[1], x.dtype)
        elif key == "layers":
            layers = root.create_group("layers", attributes=src["layers"].attrs.asdict())
            for k in src["layers"]:
                x = matrix(src["layers"][k])
                write_csr(layers, k, blocks(x), x.shape[1], x.dtype)
        else:
            write_elem(root, key, read_elem(src[key]))


def blocks(x):
    for start in range(0, x.shape[0], ROWS_PER_BLOCK):
        block = x[start:start + ROWS_PER_BLOCK]
        yield block if sp.issparse(block) else sp.csr_matrix(block)


def x_stats(x) -> tuple[bool, float]:
    """Whether every stored value is a non-negative whole number, and the largest value."""
    counts, top = True, 0.0
    for block in blocks(x):
        v = block.data
        counts = counts and bool(np.all(v >= 0) and np.all(np.mod(v, 1) == 0))
        top = max(top, float(v.max()) if v.size else 0.0)
    return counts, top


def write_csr(parent: zarr.Group, name: str, rows, n_cols: int, dtype: np.dtype) -> None:
    """A CSR matrix written block by block from `rows`, an iterable of CSR row blocks."""
    group = parent.create_group(name)
    data = group.create_array("data", shape=(0,), dtype=dtype, chunks=(1_000_000,))
    indices = group.create_array("indices", shape=(0,), dtype=np.int64, chunks=(1_000_000,))
    indptr = [0]
    for block in rows:
        data.append(block.data.astype(dtype, copy=False))
        indices.append(block.indices.astype(np.int64))
        indptr.extend(indptr[-1] + block.indptr[1:])
    group.create_array("indptr", data=np.asarray(indptr, dtype=np.int64))
    group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": [len(indptr) - 1, n_cols]})


def rewrite_x(parent: zarr.Group, out: Path, name: str, x, columns: np.ndarray, wanted: np.ndarray,
              dtype: np.dtype) -> None:
    """`parent[name]` (X or a layer) restricted and reordered to `wanted` (missing ones zero-filled), in
    `dtype`, block by block; `out` is the staging zarr's folder."""
    pick = gene_picker(columns, wanted)
    write_csr(parent, f"{name}_fitted", (sp.csr_matrix(block @ pick, dtype=dtype) for block in blocks(x)),
              len(wanted), dtype)
    del parent[name]
    folder = out / parent.path if parent.path else out
    os.replace(folder / f"{name}_fitted", folder / name)


def fit_per_gene(root: zarr.Group, out: Path, columns: np.ndarray, wanted: np.ndarray) -> list[str]:
    """A whole object's layers on `wanted`, like X; its varm and varp (per feature, of the old features)
    are dropped and returned."""
    for k in list(root["layers"]) if "layers" in root else []:
        layer = matrix(root["layers"][k])
        rewrite_x(root["layers"], out, k, layer, columns, wanted, np.dtype(layer.dtype))
    dropped = [f"{key}/{k}" for key in ("varm", "varp") if key in root for k in root[key]]
    for key in ("varm", "varp"):
        if key in root:
            del root[key]
    return dropped
