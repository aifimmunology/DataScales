"""The optional per-cell-type section (`celltype_qc.csv`): counts, medians, MADs and fail % per label."""
from __future__ import annotations

import numpy as np
import pandas as pd

from samples import median_mad


def prob_column(label_column: str) -> str:
    return label_column.removesuffix("_label") + "_prob"


def label_table(obs: pd.DataFrame, fails: pd.DataFrame, metrics: dict[str, tuple[str, str]],
                label_column: str) -> pd.DataFrame:
    labels = obs[label_column]
    sizes = labels.value_counts(sort=False)
    table = pd.DataFrame({"n_cells": sizes[sizes > 0]})
    for name, (column, _) in metrics.items():
        median, mad = median_mad(obs[column].astype(np.float64), labels)
        table[f"median_{name}"] = median
        table[f"mad_{name}"] = mad
    prob = prob_column(label_column)
    if prob in obs:
        table["median_prob"] = obs[prob].astype(np.float64).groupby(labels, observed=True).median()
    per_rule = fails.groupby(labels, observed=True).mean() * 100
    per_rule.columns = [f"fail_pct_{rule}" for rule in fails.columns]
    per_rule["fail_pct"] = fails.any(axis=1).groupby(labels, observed=True).mean() * 100
    return table.join(per_rule).rename_axis("label")


def fail_matrix(obs: pd.DataFrame, fails: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Overall fail % per label (rows) and sample (columns)."""
    failed = fails.any(axis=1)
    return failed.groupby([obs[label_column], obs["sample_id"]], observed=True).mean().unstack() * 100
