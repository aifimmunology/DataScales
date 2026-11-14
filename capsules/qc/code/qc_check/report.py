"""Everything a run writes to `/results` besides `run.json`: tables, the decisions file, the summary and the plots."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import celltypes
import plots
from ambient import Ambient
from profiles import Profile
from source import Source

ROW_GROUP = 1_000_000


@dataclass
class Review:
    """What a run computed: the cells' obs and verdicts, the per-sample table, the embedding and the decisions."""
    source: Source
    profile: Profile
    obs: pd.DataFrame
    decisions: dict
    fails: pd.DataFrame
    reason: pd.Categorical
    table: pd.DataFrame
    groups: list[str]
    label: str | None
    celltype_qc: bool
    computed_metrics: list[str]
    ambient: Ambient
    embedding: pd.DataFrame | None
    curation_line: str
    coords: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def defaults(self) -> dict:
        return {**self.profile.defaults, **self.decisions["defaults"]}

    @property
    def rules(self) -> list[str]:
        return list(self.fails.columns)

    @property
    def passed(self) -> np.ndarray:
        return ~self.fails.to_numpy().any(axis=1)

    def order(self) -> list[str]:
        """Samples with flags first, then the rest, each in store order."""
        flagged = self.table["flags"].map(bool)
        return [str(s) for s in self.table.index[flagged]] + [str(s) for s in self.table.index[~flagged]]

    def blockers(self) -> list[str]:
        out = []
        if self.decisions["pending"]:
            out.append(f"pending samples need a decision: {', '.join(self.decisions['pending'])}")
        if self.source.kind == "icechunk" and self.source.repo.origin is None:
            out.append("the input is read-only; apply needs its writable origin")
        return out


def cell_qc_digest(review: Review) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(review.passed).tobytes())
    digest.update(np.ascontiguousarray(review.reason.codes).tobytes())
    return digest.hexdigest()


def write_decisions(review: Review, results: Path) -> None:
    (results / "qc_decisions.json").write_text(json.dumps(review.decisions, indent=2, sort_keys=True) + "\n")


def write_cell_qc(review: Review, results: Path) -> None:
    columns = {
        "sample": pa.array(review.obs["sample_id"]),
        "qc_pass": pa.array(review.passed),
        "qc_fail_reason": pa.DictionaryArray.from_arrays(
            pa.array(np.asarray(review.reason.codes, dtype=np.int8), mask=np.asarray(review.reason.codes) < 0),
            pa.array(review.rules)),
    }
    for column in review.computed_metrics:
        columns[column] = pa.array(review.obs[column].to_numpy(dtype=np.float32))
    table = pa.table(columns).replace_schema_metadata({
        "fingerprint": review.source.fingerprint,
        "previewed_sha256": review.decisions["previewed_sha256"],
        "input": review.source.location,
    })
    pq.write_table(table, results / "cell_qc.parquet", compression="zstd", row_group_size=ROW_GROUP)


def write_tables(review: Review, results: Path) -> None:
    write_decisions(review, results)
    write_cell_qc(review, results)
    table = review.table.copy()
    table["flags"] = table["flags"].map(";".join)
    table.to_csv(results / "sample_qc.csv")
    if review.embedding is not None:
        review.embedding[["row", "sample_id", "umap_1", "umap_2", "leiden"]].to_parquet(
            results / "embedding.parquet", index=False)
    if review.celltype_qc and review.label:
        celltypes.label_table(review.obs, review.fails, review.profile.metrics, review.label).to_csv(
            results / "celltype_qc.csv")


def _limits(entries: dict, defaults: dict) -> dict[str, dict[str, float]]:
    return {str(name): {k: float(v) for k, v in {**defaults, **entry}.items()
                        if k in defaults and not isinstance(defaults[k], bool)}
            for name, entry in entries.items()}


def _lines(review: Review, column: str, own: dict | None = None) -> dict[str, float]:
    limits = {**review.defaults, **(own or {})}
    return {k: float(limits[k]) for k in review.profile.limits_for(column)}


def _metric_boxes(review: Review, by: pd.Series, order: list[str], own: dict, folder: Path, prefix: str) -> None:
    for name, (column, ylabel) in review.profile.metrics.items():
        stats = plots.quantile_boxes(review.obs[column].astype(np.float64), by)
        keys = review.profile.limits_for(column)
        mine = {g: {k: v for k, v in own.get(g, {}).items() if k in keys} for g in order}
        for page, chunk in enumerate(plots.pages(order), start=1):
            plots.boxes(stats, chunk, ylabel, f"{ylabel} by {by.name}", folder / f"{prefix}{name}_p{page}.png",
                        _lines(review, column), mine)


def _reason_pct(review: Review) -> pd.DataFrame:
    reason = pd.Series(review.reason, index=review.obs.index)
    table = pd.crosstab(review.obs["sample_id"], reason).reindex(index=review.table.index, columns=review.rules, fill_value=0)
    return table.div(review.table["n_cells"], axis=0) * 100


def sample_plots(review: Review, folder: Path) -> None:
    order = review.order()
    sample = review.obs["sample_id"]
    own = _limits(review.decisions["samples"], review.defaults)
    _metric_boxes(review, sample, order, own, folder, "")
    before_after = pd.DataFrame({"before": review.table["n_cells"], "after": review.table["n_pass"]})
    before_after.index = before_after.index.astype(str)
    reasons = _reason_pct(review)
    reasons.index = reasons.index.astype(str)
    colors = plots.reason_colors(review.rules)
    for page, chunk in enumerate(plots.pages(order), start=1):
        plots.bars(before_after.loc[chunk], "cells", "cells per sample, before and after QC",
                   folder / f"cells_before_after_p{page}.png")
        plots.bars(reasons.loc[chunk], "% of cells", "fail % by first failing rule",
                   folder / f"fail_pct_by_reason_p{page}.png", stacked=True, colors=colors)
    (x_column, x_label), (y_column, y_label) = (review.profile.metrics[m] for m in review.profile.scatter)
    for sid in [s for s in order if review.table.loc[s, "flags"]]:
        cells = (sample == sid).to_numpy()
        mine = review.decisions["samples"].get(sid, {})
        plots.scatter(review.obs[x_column].to_numpy(dtype=float)[cells], review.obs[y_column].to_numpy(dtype=float)[cells],
                      (x_label, _lines(review, x_column, mine)), (y_label, _lines(review, y_column, mine)),
                      f"{sid}: {', '.join(review.table.loc[sid, 'flags'])}",
                      folder / f"{review.profile.scatter[1]}_vs_{review.profile.scatter[0]}_{sid.replace('/', '_')}.png",
                      seed=0)


def group_plots(review: Review, folder: Path) -> None:
    for column in review.groups:
        by = review.obs[column].astype("category").cat.remove_unused_categories()
        order = [str(c) for c in by.cat.categories]
        _metric_boxes(review, by.rename(column), order, {}, folder, f"{column}_")
        per_rule = review.fails.groupby(by, observed=True).mean() * 100
        per_rule.index = per_rule.index.astype(str)
        plots.bars(per_rule, "% of cells failing", f"fail % per rule by {column}", folder / f"{column}_fail_pct.png",
                   colors=plots.reason_colors(review.rules))
        cells = pd.crosstab(review.obs["sample_id"], by).reindex(review.table.index, fill_value=0)
        cells.index = cells.index.astype(str)
        plots.bars(cells, "cells", f"cells per sample by {column}", folder / f"{column}_cells_per_sample.png",
                   stacked=True)


def embedding_plots(review: Review, folder: Path) -> None:
    emb = review.embedding
    plots.umap(emb, "sample_id", folder / "umap_sample.png")
    plots.umap(emb, "qc_fail_reason", folder / "umap_fail_reason.png",
               colors=dict(zip(["pass", *review.rules], plots.reason_colors(["pass", *review.rules]))))
    if review.label:
        plots.umap(emb, review.label, folder / "umap_label.png")
    plots.composition(emb, "sample_id", folder / "leiden_by_sample.png")
    for column in review.groups:
        plots.umap(emb, column, folder / f"umap_{column}.png")
        plots.umap_split(emb, column, folder / f"umap_split_{column}.png")
        plots.density(emb, column, folder / f"density_{column}.png", subtract_mean=False)
        plots.density(emb, column, folder / f"density_{column}_minus_mean.png", subtract_mean=True)
        plots.composition(emb, column, folder / f"leiden_by_{column}.png")


def ambient_plots(review: Review, folder: Path) -> None:
    table = review.ambient.table
    order = [s for s in review.order() if s in table.index]
    sets = [c.removeprefix("ambient_").removesuffix("_pct") for c in table.columns if c.endswith("_pct")]
    for name in sets:
        median = table[f"ambient_{name}_pct"].reindex(order)
        plots.bars(median.to_frame("median %"), "median % of UMIs", f"ambient {name} genes (outside their lineage)",
                   folder / f"ambient_{name}.png", line=1.5 * median.median())
    ratios = table[[f"ambient_{n}_ratio" for n in sets]].reindex(order).T
    ratios.index = sets
    plots.heatmap(ratios, "ambient ratio to cohort median", folder / "ambient_ratio_heatmap.png", cmap="RdBu_r",
                  center=1.0)


def celltype_plots(review: Review, folder: Path) -> None:
    label = review.label
    plots.heatmap(celltypes.fail_matrix(review.obs, review.fails, label), f"fail % by {label} and sample",
                  folder / "fail_pct_label_by_sample.png", cmap="magma")
    by = review.obs[label].astype("category").cat.remove_unused_categories().rename(label)
    order = [str(c) for c in by.cat.categories]
    own = _limits(review.decisions["cell_types"], review.defaults)
    for name in review.profile.scatter:
        column, ylabel = review.profile.metrics[name]
        keys = review.profile.limits_for(column)
        plots.boxes(plots.quantile_boxes(review.obs[column].astype(np.float64), by), order, ylabel,
                    f"{ylabel} by {label}", folder / f"{name}_by_label.png", _lines(review, column),
                    {g: {k: v for k, v in own.get(g, {}).items() if k in keys} for g in order})
    prob = celltypes.prob_column(label)
    if prob in review.obs:
        plots.boxes(plots.quantile_boxes(review.obs[prob].astype(np.float64), by), order, "label probability",
                    f"{prob} by {label}", folder / "prob_by_label.png")


def tissue_plots(review: Review, folder: Path) -> None:
    for page, chunk in enumerate(plots.pages(review.order(), plots.TISSUE_PAGE), start=1):
        plots.tissue(review.coords, review.passed, review.obs["sample_id"], chunk, folder / f"tissue_qc_p{page}.png")


def write_plots(review: Review, results: Path) -> None:
    root = results / "plots"
    sample_plots(review, root / "samples")
    if review.coords is not None:
        tissue_plots(review, root / "spatial")
    group_plots(review, root / "groups")
    if review.embedding is not None:
        embedding_plots(review, root / "embedding")
    if not review.ambient.table.empty:
        ambient_plots(review, root / "ambient")
    if review.celltype_qc and review.label:
        celltype_plots(review, root / "celltypes")


def summary(review: Review, apply: bool, applied: dict | None = None) -> str:
    n_cells, n_pass = len(review.obs), int(review.passed.sum())
    lines = [f"Mode: {'apply' if apply else 'report'}"]
    if apply:
        lines.append(f"Applied: {applied.get('commit') or applied.get('output')}" if applied else "Applied: nothing")
    else:
        blockers = review.blockers()
        lines.append("apply would be refused: " + "; ".join(blockers) if blockers
                     else "apply would be accepted with this run's qc_decisions.json")
    lines.append(review.curation_line)
    if review.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in review.warnings]
    lines += ["", f"Input: {review.source.location} ({review.source.kind}, fingerprint {review.source.fingerprint})",
              f"Cells: {n_pass:,} of {n_cells:,} pass ({100 * n_pass / max(n_cells, 1):.1f}%) "
              f"across {len(review.table)} samples"]
    fails = review.fails.sum()
    lines.append("Failing per rule (a cell can fail several): " + ", ".join(f"{r} {int(fails[r]):,}" for r in review.rules))
    flagged = review.table[review.table["flags"].map(bool)]
    lines += ["", f"Flagged samples ({len(flagged)}):"]
    lines += [f"  {sid} [{row['status']}]: {', '.join(row['flags'])}" for sid, row in flagged.iterrows()] or ["  none"]
    pending = review.decisions["pending"]
    lines += ["", f"Pending samples ({len(pending)}): {', '.join(pending) or 'none'}"]
    return "\n".join(lines) + "\n"


def write_summary(review: Review, results: Path, apply: bool, applied: dict | None = None) -> str:
    text = summary(review, apply, applied)
    (results / "report_summary.txt").write_text(text)
    return text
