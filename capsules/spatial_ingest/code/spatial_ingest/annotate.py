"""Per-sample annotation on raw counts: QC metrics and optional CellTypist labels."""
from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import genes

logger = logging.getLogger(__name__)
logging.getLogger("celltypist").setLevel(logging.WARNING)

QC_COLUMNS = ("n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt")


def qc_metrics(adata: ad.AnnData) -> pd.DataFrame:
    """scanpy's per-cell QC metrics (mito = `MT-` genes), without touching `adata`'s var."""
    probe = ad.AnnData(adata.X, obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names))
    probe.var["mt"] = genes.is_mito(adata.var_names)
    sc.pp.calculate_qc_metrics(probe, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return probe.obs[list(QC_COLUMNS)]


def label_columns(model: Path) -> tuple[str, str]:
    return f"{model.stem}_label", f"{model.stem}_prob"


def celltypist_labels(adata: ad.AnnData, model: Path) -> pd.DataFrame:
    """Labels and their probability for every cell, from counts normalized to 1e4 and log1p."""
    import celltypist
    from celltypist.models import Model

    normalized = ad.AnnData(adata.X.astype(np.float32), obs=pd.DataFrame(index=adata.obs_names),
                            var=pd.DataFrame(index=adata.var_names))
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    predictions = celltypist.annotate(normalized, model=Model.load(model=str(model)))
    labels = predictions.predicted_labels["predicted_labels"].astype(str).to_numpy()
    probs = predictions.probability_matrix
    picked = probs.columns.get_indexer(labels)
    label, prob = label_columns(model)
    logger.info("CellTypist %s: %d labels", model.stem, len(np.unique(labels)))
    return pd.DataFrame({label: pd.Categorical(labels),
                         prob: np.where(picked >= 0, probs.to_numpy()[np.arange(len(labels)), picked], np.nan)},
                        index=adata.obs_names)
