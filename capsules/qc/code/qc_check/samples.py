"""The per-sample table (`sample_qc.csv`) and its advisory flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from profiles import DOUBLETS

REPORTED = ("hto_category", "adt_qc_flag")
MIN_CELLS = 500
DOUBLET_RATE_MAX = 0.15
OUTLIER_MADS = 3.0
MAD_FLOOR = 0.05
MIN_COHORT = 5


def median_mad(values: pd.Series, groups: pd.Series) -> tuple[pd.Series, pd.Series]:
    grouped = values.groupby(groups, observed=True)
    median = grouped.median()
    deviation = (values - grouped.transform("median")).abs()
    return median, deviation.groupby(groups, observed=True).median()


def _counts(column: pd.Series, groups: pd.Series, prefix: str) -> pd.DataFrame:
    table = pd.crosstab(groups, column.astype("category"))
    table.columns = [f"{prefix}{c}" for c in table.columns]
    return table


def sample_metrics(obs: pd.DataFrame, metrics: dict[str, tuple[str, str]], label_column: str | None) -> pd.DataFrame:
    """Cell count, medians and MADs, doublet rate and NA count, and composition, one row per sample."""
    groups = obs["sample_id"]
    sizes = groups.value_counts(sort=False)
    table = pd.DataFrame({"n_cells": sizes[sizes > 0]})
    for name, (column, _) in metrics.items():
        median, mad = median_mad(obs[column].astype(np.float64), groups)
        table[f"median_{name}"] = median
        table[f"mad_{name}"] = mad
    if DOUBLETS in obs:
        calls = obs[DOUBLETS].astype("Float64")
        called = calls.notna().groupby(groups, observed=True).sum()
        doublets = calls.fillna(0).groupby(groups, observed=True).sum().astype(np.float64)
        table["doublet_rate"] = doublets / called.replace(0, np.nan).astype(np.float64)
        table["doublet_na"] = table["n_cells"] - called
    parts = [table]
    if label_column and label_column in obs:
        parts.append(pd.crosstab(groups, obs[label_column], normalize="index").add_prefix("frac_"))
    for column in REPORTED:
        if column in obs and obs[column].notna().any():
            parts.append(_counts(obs[column], groups, f"{column}:"))
    return pd.concat(parts, axis=1).rename_axis("sample_id")


def fail_table(obs: pd.DataFrame, fails: pd.DataFrame) -> pd.DataFrame:
    """Per-rule and overall fail %, counting every failing rule, plus the passing cell count."""
    groups = obs["sample_id"]
    per_rule = fails.groupby(groups, observed=True).mean() * 100
    per_rule.columns = [f"fail_pct_{rule}" for rule in fails.columns]
    failed = fails.any(axis=1)
    per_rule["fail_pct"] = failed.groupby(groups, observed=True).mean() * 100
    per_rule["n_pass"] = (~failed).groupby(groups, observed=True).sum()
    return per_rule


def _outliers(table: pd.DataFrame) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {sid: [] for sid in table.index}
    if len(table) < MIN_COHORT:
        return flags
    for name in [c.removeprefix("median_") for c in table.columns if c.startswith("median_")]:
        medians = table[f"median_{name}"].astype(np.float64)
        centre = medians.median()
        spread = max((medians - centre).abs().median(), MAD_FLOOR * abs(centre))
        for sid in medians.index[(medians - centre).abs() > OUTLIER_MADS * spread]:
            flags[sid].append(f"outlier:{name}")
    return flags


def flag(table: pd.DataFrame, ambient_flags: dict[str, list[str]]) -> pd.Series:
    """Advisory flags per sample; they never change a decision, and fail % is never an input."""
    flags = _outliers(table)
    for sid, row in table.iterrows():
        own = []
        if row["n_cells"] < MIN_CELLS:
            own.append("few_cells")
        rate = row.get("doublet_rate", np.nan)
        if pd.notna(rate) and (rate == 0 or rate > DOUBLET_RATE_MAX):
            own.append("doublet_rate")
        if row.get("doublet_na", 0) > 0:
            own.append("doublet_unknown")
        flags[sid] = own + flags[sid] + list(ambient_flags.get(sid, []))
    return pd.Series(flags, name="flags").reindex(table.index)


def status(sample_ids: pd.Index, decisions: dict) -> pd.Series:
    def one(sid: str) -> str:
        if sid in decisions["pending"]:
            return "pending"
        if decisions["samples"].get(sid, {}).get("exclude"):
            return "excluded"
        return "reviewed" if sid in decisions["samples"] else "pending"
    return pd.Series([one(str(s)) for s in sample_ids], index=sample_ids, name="status")
