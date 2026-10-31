import numpy as np
import pandas as pd
import scipy.sparse as sp

import ambient

VAR = np.array(["LYZ", "S100A8", "S100A9", "HBB", "HBA1", "HBA2", "IGKC", "IGHM", "IGLC2", "OTHER"], dtype=object)


def cell(lyz: int = 0, hb: int = 0, ig: int = 0, other: int = 100) -> list[int]:
    return [lyz, 0, 0, hb, 0, 0, ig, 0, 0, other]


def cohort(n_samples: int, shares: dict[str, int], label: str = "T cell", doublet_rows: dict | None = None):
    rows, sample, labels = [], [], []
    for i in range(n_samples):
        sid = f"S{i}"
        for _ in range(10):
            rows.append(cell(lyz=shares.get(sid, 1)))
            sample.append(sid)
            labels.append(label)
    counts = sp.csr_matrix(np.array(rows, dtype=np.float32))
    doublet = np.zeros(len(rows), bool)
    for index in (doublet_rows or {}):
        doublet[index] = True
    return counts, pd.Series(sample), pd.Series(labels), doublet


def test_known_shares_ratio_and_flag():
    counts, sample, labels, doublet = cohort(5, {"S4": 2})
    result = ambient.indicator(counts, VAR, sample, labels, doublet, "label")
    t = result.table
    np.testing.assert_allclose(t.loc["S0", "ambient_monocyte_pct"], 100 / 101)
    np.testing.assert_allclose(t.loc["S4", "ambient_monocyte_ratio"], (200 / 102) / (100 / 101))
    assert result.flags == {"S4": ["ambient:monocyte"]}
    assert t["ambient_immunoglobulin_pct"].eq(0).all() and t["ambient_immunoglobulin_ratio"].isna().all()


def test_needs_five_samples():
    counts, sample, labels, doublet = cohort(4, {"S3": 5})
    assert ambient.indicator(counts, VAR, sample, labels, doublet, "label").flags == {}


def test_doublets_are_left_out():
    counts, sample, labels, doublet = cohort(5, {}, doublet_rows=range(40, 50))
    counts = counts.tolil()
    counts[40:50, 0] = 500
    result = ambient.indicator(sp.csr_matrix(counts), VAR, sample, labels, doublet, "label")
    assert "S4" not in result.flags


def test_only_cells_of_other_lineages_are_measured():
    counts, sample, _, doublet = cohort(5, {"S4": 50})
    labels = pd.Series(["Monocyte"] * len(sample))
    t = ambient.indicator(counts, VAR, sample, labels, doublet, "label").table
    assert "ambient_monocyte_pct" not in t or t["ambient_monocyte_pct"].isna().all()


def test_skips_with_warnings():
    counts, sample, labels, doublet = cohort(5, {})
    assert ambient.indicator(counts, VAR, sample, None, doublet, "lbl").warnings == [
        "No lbl column; ambient RNA indicator skipped."]
    partial = VAR.copy()
    partial[1] = "X1"
    partial[3:6] = ["X2", "X3", "X4"]
    result = ambient.indicator(counts, partial, sample, labels, doublet, "label")
    assert any("'monocyte' uses LYZ, S100A9; missing S100A8" in w for w in result.warnings)
    assert any("'hemoglobin' skipped" in w for w in result.warnings)
    assert "ambient_hemoglobin_pct" not in result.table
