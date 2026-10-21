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


def run_celltypist(adata: ad.AnnData, models: list[Path]) -> None:
    """Labels every cell with each CellTypist model, writing `<model>_label` and `<model>_prob` into obs.

    The models share one normalized copy (counts per 1e4, log1p), as the tenx-scrna-pipeline does."""
    import celltypist
    from celltypist.models import Model

    normalized = adata.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    for path in models:
        predictions = celltypist.annotate(normalized, model=Model.load(model=str(path)))
        labels = predictions.predicted_labels["predicted_labels"].astype(str).to_numpy()
        probs = predictions.probability_matrix
        picked = probs.columns.get_indexer(labels)
        label, prob = contract.label_columns(path.stem)
        adata.obs[label] = labels
        adata.obs[prob] = np.where(picked >= 0, probs.to_numpy()[np.arange(len(labels)), picked], np.nan)
        logger.info("CellTypist %s: %d labels", path.stem, len(np.unique(labels)))
