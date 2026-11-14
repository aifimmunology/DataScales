"""Per-cell annotation on raw counts: QC metrics and optional CellTypist labels, plus the checks that a
CellTypist model fits a sample's gene panel before any sample is read.

Every function works on any row chunk of a sample: QC metrics and CellTypist predictions are per cell,
so a sample is annotated chunk by chunk with the same result as in one piece.

Importing celltypist turns off UserWarning, FutureWarning and RuntimeWarning globally
(celltypist/classifier.py); it is imported here only, inside `warnings.catch_warnings()`, so the run's
warning filters are the same with or without a model."""
from __future__ import annotations

import logging
import math
import warnings
from functools import lru_cache
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import genes

logger = logging.getLogger(__name__)

QC_COLUMNS = ("n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt")
MIN_SHARED = 10            # a model must share at least this many genes with a sample's panel ...
MIN_FRACTION = 0.05        # ... and at least this fraction of the model's genes
LOW_FRACTION = 0.5         # below this fraction of the model's genes, labels are less reliable: warn
HOW_TO_MAKE = ("A CellTypist model is the .pkl written by celltypist.train(...).write('model.pkl'), trained on "
               "log1p-normalized counts with the same gene symbols as your panel.")


class AnnotateError(ValueError):
    """A CellTypist model that can't be used for these samples."""


def _celltypist():
    """The celltypist modules, imported without leaving its global warning filters behind."""
    with warnings.catch_warnings():
        import celltypist
        from celltypist.models import Model
    return celltypist, Model


def qc_metrics(adata: ad.AnnData) -> pd.DataFrame:
    """scanpy's per-cell QC metrics (mito = `MT-` genes), without touching `adata`'s var."""
    probe = ad.AnnData(adata.X, obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names))
    probe.var["mt"] = genes.is_mito(adata.var_names)
    sc.pp.calculate_qc_metrics(probe, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return probe.obs[list(QC_COLUMNS)]


def label_columns(model: Path) -> tuple[str, str]:
    return f"{model.stem}_label", f"{model.stem}_prob"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float):
    _, Model = _celltypist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model.load(model=path)


def _detail(exc: Exception) -> str:
    text = str(exc).replace("🛑", "").strip()
    return text.splitlines()[0][:160] if text else type(exc).__name__


def load_model(model: Path):
    """The CellTypist model at `model` (loaded once per file); AnnotateError when it isn't one."""
    model = Path(model)
    if not model.is_file():
        raise AnnotateError(f"CellTypist model not found: {model}. Give the path of a .pkl file, e.g. "
                            "/data/models/my_panel.pkl, or leave the field empty for no labels.")
    try:
        loaded = _load(str(model), model.stat().st_mtime)
        features = loaded.features
    except Exception as exc:
        logger.debug("Loading %s as a CellTypist model: %s", model, _detail(exc))
        raise AnnotateError(f"{model} isn't a CellTypist model (it could not be loaded as one). {HOW_TO_MAKE}") from None
    if len(features) == 0:
        raise AnnotateError(f"{model} is a CellTypist model with no genes. {HOW_TO_MAKE}")
    return loaded


def model_features(model: Path) -> list[str]:
    return [str(g) for g in load_model(model).features]


def model_overlap(model: Path, var_names) -> tuple[int, int]:
    """(genes of the model found in `var_names`, genes in the model)."""
    features = model_features(model)
    return len(set(features) & {str(g) for g in var_names}), len(features)


def overlap_floor(n_model: int) -> int:
    """Fewest shared genes a model may have with a panel: max(10, 5% of the model's genes)."""
    return max(MIN_SHARED, math.ceil(MIN_FRACTION * n_model))


def check_overlap(model: Path, sid: str, var_names, store_model: bool = False) -> tuple[int, int, str | None]:
    """(shared, model genes, note or None) for one sample: AnnotateError when too few genes are shared,
    a note when fewer than half of the model's genes are in the panel. `store_model`: the store labels
    every sample with this model, so leaving the model out isn't an option for this store."""
    shared, n_model = model_overlap(model, var_names)
    floor = overlap_floor(n_model)
    if shared < floor:
        hint = ("another species, or Ensembl IDs instead of gene symbols?" if shared == 0
                else "a model trained on another panel or species?")
        what = (f"This store labels every sample with it, so add {sid} to a separate store (without a model, or with "
                "a model trained on its panel)." if store_model else
                "Give a model trained on this panel, or leave the model field empty.")
        raise AnnotateError(f"CellTypist model {Path(model).name} shares {shared} of its {n_model} genes with {sid}'s "
                            f"panel ({len(var_names)} genes); at least {floor} are needed ({hint}). {what}")
    if shared < LOW_FRACTION * n_model:
        return shared, n_model, (f"{sid}: {shared} of the model's {n_model} genes "
                                 f"({100 * shared / n_model:.0f}%) are in its panel")
    return shared, n_model, None


def celltypist_labels(adata: ad.AnnData, model: Path) -> pd.DataFrame:
    """Labels and their probability for every cell, from counts normalized to 1e4 and log1p."""
    celltypist, _ = _celltypist()
    loaded = load_model(model)
    normalized = ad.AnnData(adata.X.astype(np.float32), obs=pd.DataFrame(index=adata.obs_names),
                            var=pd.DataFrame(index=adata.var_names))
    with warnings.catch_warnings():   # cells with 0 counts are expected (they stay 0 and get a low-probability label)
        warnings.filterwarnings("ignore", message="Some cells have zero counts", category=UserWarning)
        sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    try:
        predictions = celltypist.annotate(normalized, model=loaded)
    except Exception as exc:
        raise AnnotateError(f"CellTypist could not label the cells with {Path(model).name}: {_detail(exc)}. "
                            f"{HOW_TO_MAKE}") from None
    labels = predictions.predicted_labels["predicted_labels"].astype(str).to_numpy()
    probs = predictions.probability_matrix
    picked = probs.columns.get_indexer(labels)
    label, prob = label_columns(Path(model))
    logger.debug("CellTypist %s: %d labels in %d cells", Path(model).stem, len(np.unique(labels)), len(labels))
    return pd.DataFrame({label: pd.Categorical(labels),
                         prob: np.where(picked >= 0, probs.to_numpy()[np.arange(len(labels)), picked], np.nan)},
                        index=adata.obs_names)
