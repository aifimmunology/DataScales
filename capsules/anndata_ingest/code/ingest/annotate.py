"""Per-sample annotation on raw counts: QC metrics, Scrublet doublet scores, CellTypist labels."""
from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import contract

logger = logging.getLogger(__name__)
logging.getLogger("celltypist").setLevel(logging.WARNING)

EXPECTED_DOUBLET_RATE = 0.06


class ModelError(ValueError):
    """A CellTypist model file that can't be loaded."""


def load_models(paths: list[Path]) -> list[tuple[str, object]]:
    """Each CellTypist model, loaded once before any sample is read: `(name, model)` pairs."""
    from celltypist.models import Model

    loaded, bad = [], []
    for path in paths:
        try:
            loaded.append((path.stem, Model.load(model=str(path))))
        except Exception as exc:  # celltypist raises plain Exception for an invalid file
            bad.append(f"{path} ({str(exc).strip()})")
    if bad:
        raise ModelError(f"CellTypist model(s) can't be loaded: {'; '.join(bad)}. Give a .pkl written by "
                         "celltypist (e.g. celltypist.train(...).write(path)).")
    return loaded


def qc_metrics(adata: ad.AnnData) -> None:
    adata.var["mt"] = contract.is_mito(adata.var_names)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)


def run_scrublet(adata: ad.AnnData, random_state: int = 0) -> None:
    """Scores doublets into obs. If Scrublet fails the columns stay absent and the contract pads them
    with NA; if it finds no threshold the scores are kept and the calls are NA."""
    import scrublet as scr

    counts = adata.X if sp.isspmatrix_csc(adata.X) else sp.csc_matrix(adata.X)
    try:
        scrub = scr.Scrublet(counts, expected_doublet_rate=EXPECTED_DOUBLET_RATE, random_state=random_state)
        scores, doublets = scrub.scrub_doublets(verbose=False)
    except Exception:
        logger.exception("Scrublet failed for this sample; its cells get NA doublet scores")
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
        label, prob = contract.label_columns(name)
        adata.obs[label] = labels
        adata.obs[prob] = np.where(picked >= 0, probs.to_numpy()[np.arange(len(labels)), picked], np.nan)
        logger.info("CellTypist %s: %d labels", name, len(np.unique(labels)))
