"""Staging: each new input is read, annotated (raw samples), fitted to the store, and written as a zarr
in the run's staging folder before the store is touched. `Target` is what staged cells must match."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import annizarr
import numpy as np
import pandas as pd
from annizarr import AppConfig

import annotate
import contract
import genes
import hise
import objects
import store as stores
from config import Settings
from contracts import IngestError, ObjectInput, RawInput, Staged

logger = logging.getLogger("ingest")


@dataclass
class Target:
    """What staged cells must match: the store's genes, X dtype, whether X holds counts, and obs kinds.
    A new store takes them from its first staged sample."""
    var: np.ndarray | None = None
    x_dtype: np.dtype | None = None
    counts: bool = False
    obs: dict[str, str] | None = None

    @classmethod
    def of(cls, root) -> "Target":
        if not stores.has_data(root):
            return cls()
        return cls(stores.var_names(root), stores.x_dtype(root), stores.x_holds_counts(root), stores.obs_kinds(root))

    def fit_cells(self, adata: ad.AnnData, name: str) -> ad.AnnData:
        """The sample's X and obs in the store's dtype and kinds; the first sample of a new store sets them."""
        adata.X = contract.fit_x(adata.X, self.x_dtype, self.counts, name)
        if self.obs is not None:
            adata.obs, failed = contract.match_obs(adata.obs, self.obs)
            if failed:
                logger.warning("%s: obs columns left empty, since their values don't fit the store's types: %s",
                               name, ", ".join(failed))
        if self.var is None:
            self.learn(np.asarray(adata.var_names, dtype=object), adata.X.dtype, contract.holds_counts(adata.X),
                       contract.frame_kinds(adata.obs))
        return adata

    def learn(self, var: np.ndarray, x_dtype: np.dtype, counts: bool, obs: dict[str, str]) -> None:
        self.var, self.x_dtype, self.counts, self.obs = var, x_dtype, counts, obs


def annotate_sample(adata: ad.AnnData, sid: str, settings: Settings, models: list) -> ad.AnnData:
    adata.obs_names = [f"{sid}:{bc}" for bc in adata.obs_names]
    adata.obs["sample_id"] = sid
    annotate.qc_metrics(adata)
    if models:
        annotate.run_celltypist(adata, models)
    if settings.run_scrublet:
        annotate.run_scrublet(adata, random_state=settings.random_seed)
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


def samples(new: list[RawInput], settings: Settings, models: list, schema: contract.Schema, target: Target,
            cfg: AppConfig, staging: Path) -> list[Staged]:
    """Raw samples, one in memory at a time: annotated on their own genes, then fitted to the store's."""
    staged = []
    for item in new:
        sid = item.sample_id
        try:
            adata = hise.read_sample(item.path)
        except (OSError, KeyError) as exc:
            raise IngestError(f"{item.path} can't be read ({type(exc).__name__}: {exc}); it may be truncated. "
                              "Nothing from this run was added.") from exc
        logger.info("%s: %d cells x %d genes", sid, adata.n_obs, adata.n_vars)
        if adata.n_obs == 0:
            logger.warning("%s has no cells; skipped", sid)
            continue
        adata, note = genes.fit(annotate_sample(adata, sid, settings, models), target.var)
        adata = target.fit_cells(contract.project(adata, schema, target.var), sid)
        out = staging / f"{len(staged):04d}.zarr"
        annizarr.convert(adata, output=str(out), cfg=cfg)
        staged.append(Staged(sid, out, summarize(adata.obs), str(item.path), note, (sid,)))
        del adata
    return staged


def existing_objects(new: list[ObjectInput], target: Target, cfg: AppConfig, staging: Path) -> list[Staged]:
    """Existing AnnData objects, streamed by annizarr. The first object of a new store keeps everything;
    later ones keep X, the store's obs columns and var, since an append can't extend embeddings or layers."""
    staged = []
    for item in new:
        out = staging / f"{len(staged):04d}.zarr"
        obs, note, left_out = objects.stage_object(item.path, out, item.column, item.sample_ids[0], target,
                                                   keep_all=target.var is None, cfg=cfg, name=item.label)
        logger.info("%s: %d cells, samples %s", item.label, len(obs), list(item.sample_ids))
        if left_out:
            logger.warning("%s: not added, since the store has no place for them: %s", item.label, ", ".join(left_out))
        staged.append(Staged(item.label, out, summarize(obs), str(item.path), note, item.sample_ids))
    return staged
