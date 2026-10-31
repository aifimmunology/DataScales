import numpy as np
import pandas as pd
import pytest

import samples
from profiles import SCRNA


def table(n: int = 6, **overrides) -> pd.DataFrame:
    base = pd.DataFrame({
        "n_cells": [1000] * n, "median_genes": [1500.0] * n, "median_umis": [4000.0] * n,
        "median_mito": [3.0] * n, "doublet_rate": [0.05] * n, "doublet_na": [0] * n,
    }, index=[f"S{i}" for i in range(n)])
    for column, values in overrides.items():
        for sid, value in values.items():
            base.loc[sid, column] = value
    return base


def test_no_flags_on_a_uniform_cohort():
    assert samples.flag(table(), {}).map(len).sum() == 0


def test_cell_count_and_doublet_flags():
    flags = samples.flag(table(n_cells={"S0": 499}, doublet_rate={"S1": 0.0, "S2": 0.151, "S3": 0.15},
                               doublet_na={"S4": 3}), {})
    assert flags["S0"] == ["few_cells"]
    assert flags["S1"] == ["doublet_rate"]
    assert flags["S2"] == ["doublet_rate"]
    assert flags["S3"] == []
    assert flags["S4"] == ["doublet_unknown"]


def test_no_doublet_rate_flag_without_any_calls():
    assert samples.flag(table(doublet_rate={"S0": np.nan}, doublet_na={"S0": 1000}), {})["S0"] == ["doublet_unknown"]


def test_outliers_need_five_samples():
    assert samples.flag(table(5, median_genes={"S0": 500.0}), {})["S0"] == ["outlier:genes"]
    assert samples.flag(table(4, median_genes={"S0": 500.0}), {})["S0"] == []


def test_mad_floor_of_five_percent():
    within = table(median_genes={"S0": 1500 * (1 - 0.149)})
    beyond = table(median_genes={"S0": 1500 * (1 - 0.151)})
    assert samples.flag(within, {})["S0"] == []
    assert samples.flag(beyond, {})["S0"] == ["outlier:genes"]


def test_ambient_flags_are_appended():
    assert samples.flag(table(), {"S1": ["ambient:monocyte"]})["S1"] == ["ambient:monocyte"]


def test_sample_metrics_and_fail_table():
    obs = pd.DataFrame({
        "sample_id": pd.Categorical(["A", "A", "A", "B"], categories=["A", "B", "unused"]),
        "n_genes_by_counts": [100, 200, 400, 300], "total_counts": [1.0, 2.0, 3.0, 4.0],
        "pct_counts_mt": [1.0, 2.0, 3.0, 4.0], "predicted_doublets": pd.array([1, 0, None, None], dtype="Int64"),
        "hto_category": pd.Categorical(["singlet", "doublet", "singlet", "singlet"]),
    })
    t = samples.sample_metrics(obs, SCRNA.metrics, None)
    assert t.index.tolist() == ["A", "B"]
    assert t.loc["A", "median_genes"] == 200 and t.loc["A", "mad_genes"] == 100
    assert t.loc["A", "doublet_rate"] == 0.5 and t.loc["A", "doublet_na"] == 1
    assert np.isnan(t.loc["B", "doublet_rate"]) and t.loc["B", "doublet_na"] == 1
    assert t.loc["A", "hto_category:singlet"] == 2
    fails = pd.DataFrame({r: [False] * 4 for r in SCRNA.rule_names})
    fails.loc[0, ["doublet", "low_genes"]] = True
    f = samples.fail_table(obs, fails)
    assert f.loc["A", "fail_pct"] == pytest.approx(100 / 3) and f.loc["A", "fail_pct_doublet"] == pytest.approx(100 / 3)
    assert f.loc["A", "n_pass"] == 2


def test_status():
    d = {"samples": {"A": {}, "B": {"exclude": True}}, "pending": ["C"]}
    assert samples.status(pd.Index(["A", "B", "C"]), d).tolist() == ["reviewed", "excluded", "pending"]
