"""Ambient RNA indicator: another lineage's marker genes in the downsampled cells, per sample (detection only)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

RATIO_FLAG = 1.5
MIN_COHORT = 5


@dataclass(frozen=True)
class MarkerSet:
    name: str
    genes: tuple[str, ...]
    measured_in: tuple[str, ...] = ()
    not_in: tuple[str, ...] = ()

    def measured(self, labels: pd.Series) -> np.ndarray:
        values = labels.astype(str)
        if self.not_in:
            return (~values.isin(self.not_in) & labels.notna()).to_numpy()
        return values.isin(self.measured_in).to_numpy()


SETS = (
    MarkerSet("monocyte", ("LYZ", "S100A8", "S100A9"), measured_in=("T cell", "B cell", "NK cell")),
    MarkerSet("hemoglobin", ("HBB", "HBA1", "HBA2"), not_in=("Erythrocyte",)),
    MarkerSet("immunoglobulin", ("IGKC", "IGHM", "IGLC2"), measured_in=("T cell", "NK cell", "Monocyte")),
)


@dataclass
class Ambient:
    table: pd.DataFrame
    flags: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)


def shares(counts: sp.csr_matrix, var_names: np.ndarray, genes: tuple[str, ...]) -> np.ndarray:
    """Each cell's share of its UMIs in `genes`, in %."""
    columns = np.flatnonzero(np.isin(np.asarray(var_names, dtype=str), genes))
    total = np.asarray(counts.sum(axis=1)).ravel()
    in_set = np.asarray(counts[:, columns].sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, in_set / total * 100, np.nan)


def indicator(counts: sp.csr_matrix, var_names: np.ndarray, sample: pd.Series, labels: pd.Series | None,
              doublet: np.ndarray, label_column: str) -> Ambient:
    """`ambient_<set>_pct`, `_p90` and `_ratio` per sample, and the `ambient:<set>` flags."""
    if labels is None:
        return Ambient(pd.DataFrame(), {}, [f"No {label_column} column; ambient RNA indicator skipped."])
    names = set(np.asarray(var_names, dtype=str))
    columns, flags, warnings = {}, {}, []
    sample = pd.Series(np.asarray(sample, dtype=object))
    labels = pd.Series(np.asarray(labels, dtype=object))
    for marker in SETS:
        present = tuple(g for g in marker.genes if g in names)
        if not present:
            warnings.append(f"Ambient set '{marker.name}' skipped: none of {', '.join(marker.genes)} in var.")
            continue
        if len(present) < len(marker.genes):
            absent = [g for g in marker.genes if g not in names]
            warnings.append(f"Ambient set '{marker.name}' uses {', '.join(present)}; missing {', '.join(absent)}.")
        keep = marker.measured(labels) & ~doublet
        share = pd.Series(shares(counts, var_names, present))[keep]
        grouped = share.groupby(sample[keep].to_numpy())
        median, p90 = grouped.median(), grouped.quantile(0.9)
        cohort = median.median()
        ratio = median / cohort if cohort > 0 else median * np.nan
        columns[f"ambient_{marker.name}_pct"] = median
        columns[f"ambient_{marker.name}_p90"] = p90
        columns[f"ambient_{marker.name}_ratio"] = ratio
        if len(median) >= MIN_COHORT:
            for sid in ratio.index[ratio >= RATIO_FLAG]:
                flags.setdefault(sid, []).append(f"ambient:{marker.name}")
    table = pd.DataFrame(columns).rename_axis("sample_id")
    return Ambient(table, flags, warnings)
