"""QC profiles: the metrics a store is reviewed on, the cell rules, and their defaults, per assay."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GENES = "n_genes_by_counts"
COUNTS = "total_counts"
TRANSCRIPTS = "transcript_counts"
MITO = "pct_counts_mt"
DOUBLETS = "predicted_doublets"
NUCLEI = "nucleus_count"
EXCLUDED = "sample_excluded"


@dataclass(frozen=True)
class Rule:
    """A cell fails when `column` compares to the `key` threshold as `fails_when` says."""
    name: str
    column: str
    fails_when: str
    key: str

    def evaluate(self, values: pd.Series, limit: np.ndarray) -> np.ndarray:
        if self.fails_when == "called":
            return limit.astype(bool) & (values.astype("Float64").fillna(0).to_numpy() == 1)
        x = values.astype(np.float64).to_numpy()
        with np.errstate(invalid="ignore"):
            return {"<": x < limit, "<=": x <= limit, ">=": x >= limit}[self.fails_when]


@dataclass(frozen=True)
class Profile:
    name: str
    metrics: dict[str, tuple[str, str]]
    defaults: dict
    rules: tuple[Rule, ...]
    scatter: tuple[str, str]
    embed_per_sample: int
    compute_metrics: bool
    ambient: bool
    mito: bool

    @property
    def rule_names(self) -> list[str]:
        return [EXCLUDED, *(r.name for r in self.rules)]

    @property
    def thresholds(self) -> tuple[str, ...]:
        return tuple(self.defaults)

    @property
    def required(self) -> tuple[str, ...]:
        """The metric columns every cell needs; they can come from X when absent."""
        return tuple(column for column, _ in self.metrics.values())

    def limits_for(self, column: str) -> tuple[str, ...]:
        return tuple(r.key for r in self.rules if r.column == column and r.fails_when != "called")


SCRNA = Profile(
    name="scrna",
    metrics={"genes": (GENES, "genes detected"), "umis": (COUNTS, "total UMIs"), "mito": (MITO, "mito %")},
    defaults={"min_genes": 200, "max_genes": 5000, "max_mito": 10, "filter_doublets": True},
    rules=(Rule("doublet", DOUBLETS, "called", "filter_doublets"), Rule("high_mito", MITO, ">=", "max_mito"),
           Rule("low_genes", GENES, "<=", "min_genes"), Rule("high_genes", GENES, ">=", "max_genes")),
    scatter=("genes", "mito"),
    embed_per_sample=1000,
    compute_metrics=False,
    ambient=True,
    mito=True,
)

SPATIAL = Profile(
    name="spatial",
    metrics={"genes": (GENES, "genes detected"), "counts": (TRANSCRIPTS, "transcripts")},
    defaults={"min_counts": 20, "min_genes": 5, "min_nuclei": 1},
    rules=(Rule("low_counts", TRANSCRIPTS, "<", "min_counts"), Rule("low_genes", GENES, "<", "min_genes"),
           Rule("no_nucleus", NUCLEI, "<", "min_nuclei")),
    scatter=("counts", "genes"),
    embed_per_sample=20_000,
    compute_metrics=True,
    ambient=False,
    mito=False,
)

PROFILES = {p.name: p for p in (SCRNA, SPATIAL)}
