import json

import numpy as np
import pandas as pd
import pytest

import decisions as dec
from profiles import SCRNA, SPATIAL


def cells(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    base = {"sample_id": pd.Categorical(["S1"] * n), "n_genes_by_counts": [1000] * n, "pct_counts_mt": [1.0] * n,
            "predicted_doublets": pd.array([0] * n, dtype="Int64")}
    base.update(columns)
    return pd.DataFrame(base)


def decide(**sections) -> dict:
    return dec.merge(dec.empty(SCRNA), sections)


def test_strict_boundaries():
    obs = cells(n_genes_by_counts=[200, 201, 4999, 5000], pct_counts_mt=[9.99, 10.0, 0, 0])
    fails = dec.failures(obs, dec.empty(SCRNA), SCRNA)
    assert fails["low_genes"].tolist() == [True, False, False, False]
    assert fails["high_genes"].tolist() == [False, False, False, True]
    assert fails["high_mito"].tolist() == [False, True, False, False]


def test_doublet_na_passes_and_filter_can_be_turned_off():
    obs = cells(predicted_doublets=pd.array([1, 0, None], dtype="Int64"))
    assert dec.failures(obs, dec.empty(SCRNA), SCRNA)["doublet"].tolist() == [True, False, False]
    off = decide(defaults={"filter_doublets": False})
    assert not dec.failures(obs, off, SCRNA)["doublet"].any()


def test_reason_is_first_rule_but_counts_include_every_rule():
    obs = cells(n_genes_by_counts=[100], pct_counts_mt=[50.0], predicted_doublets=pd.array([1], dtype="Int64"))
    fails = dec.failures(obs, dec.empty(SCRNA), SCRNA)
    assert fails.loc[0, ["doublet", "high_mito", "low_genes"]].all()
    assert dec.first_reason(fails)[0] == "doublet"
    assert pd.isna(dec.first_reason(dec.failures(cells(n_genes_by_counts=[1000]), dec.empty(SCRNA), SCRNA))[0])


def test_precedence_sample_over_cell_type_over_defaults():
    obs = cells(sample_id=pd.Categorical(["S1", "S1", "S2", "S2"]), n_genes_by_counts=[350] * 4,
                label=pd.Categorical(["T", "B", "T", "B"]))
    d = decide(defaults={"min_genes": 300}, cell_types={"T": {"min_genes": 400}},
               samples={"S2": {"min_genes": 100}, "S1": {}}, label_column="label")
    assert dec.failures(obs, d, SCRNA)["low_genes"].tolist() == [True, False, False, False]
    d["label_column"] = None
    assert dec.failures(obs, d, SCRNA)["low_genes"].tolist() == [False, False, False, False]


def test_exclude_removes_every_cell_of_the_sample():
    obs = cells(sample_id=pd.Categorical(["S1", "S2", "S2"]))
    fails = dec.failures(obs, decide(samples={"S2": {"exclude": True}}), SCRNA)
    assert fails["sample_excluded"].tolist() == [False, True, True]


def test_merge_is_key_by_key():
    base = decide(samples={"S1": {"max_genes": 7000, "note": "deep"}}, defaults={"max_mito": 15})
    merged = dec.merge(base, {"samples": {"S1": {"note": "rechecked"}, "S2": {}}, "defaults": {"min_genes": 250}})
    assert merged["samples"] == {"S1": {"max_genes": 7000, "note": "rechecked"}, "S2": {}}
    assert merged["defaults"] == {**SCRNA.defaults, "max_mito": 15, "min_genes": 250}


def test_settle_new_samples():
    d = decide(samples={"S1": {"exclude": True}})
    settled = dec.settle_new_samples(d, ["S1", "S2", "S3"], flagged={"S1", "S3"})
    assert settled["samples"] == {"S1": {"exclude": True}, "S2": {}}
    assert settled["pending"] == ["S3"]


def test_digest_ignores_key_order_and_whitespace_but_not_content():
    sealed = dec.seal(decide(samples={"S1": {"max_genes": 7000}}))
    reordered = json.loads(json.dumps(sealed, indent=4, sort_keys=False))
    reordered = dict(reversed(list(reordered.items())))
    assert dec.is_sealed(reordered)
    edited = json.loads(json.dumps(sealed))
    edited["samples"]["S1"]["max_genes"] = 7001
    assert not dec.is_sealed(edited)
    assert not dec.is_sealed({k: v for k, v in sealed.items() if k != dec.HASH_KEY})


@pytest.mark.parametrize("bad, message", [
    ({"version": 2}, "version"),
    ({"version": 1, "samples": {"S1": {"max_gene": 1}}}, "unknown keys"),
    ({"version": 1, "samples": {"S1": {"exclude": "yes"}}}, "true or false"),
    ({"version": 1, "defaults": {"min_genes": "200"}}, "number"),
    ({"version": 1, "extra": 1}, "Unknown keys"),
])
def test_validate_rejects(bad, message):
    with pytest.raises(dec.DecisionsError, match=message):
        dec.validate(bad, SCRNA)


def test_load_errors(tmp_path):
    with pytest.raises(dec.DecisionsError, match="No decisions file"):
        dec.load(str(tmp_path / "missing.json"), SCRNA)
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(dec.DecisionsError, match="not valid JSON"):
        dec.load(str(tmp_path / "bad.json"), SCRNA)


def test_thresholds_are_per_cell():
    obs = cells(sample_id=pd.Categorical(["S1", "S2"]))
    limits = dec.thresholds(obs, decide(samples={"S2": {"max_mito": 20, "filter_doublets": False}}), SCRNA)
    assert limits["max_mito"].tolist() == [10, 20]
    assert limits["filter_doublets"].tolist() == [True, False]
    assert limits["filter_doublets"].dtype == np.bool_


def test_spatial_rules_and_profile_guard():
    obs = pd.DataFrame({"sample_id": pd.Categorical(["S1"] * 4), "transcript_counts": [19, 20, 50, 50],
                        "n_genes_by_counts": [5, 5, 5, 5], "nucleus_count": [1, 1, 0, None]})
    fails = dec.failures(obs, dec.empty(SPATIAL), SPATIAL)
    assert list(fails.columns) == ["sample_excluded", "low_counts", "low_genes", "no_nucleus"]
    assert fails["low_counts"].tolist() == [True, False, False, False]
    assert not fails["low_genes"].any()
    assert fails["no_nucleus"].tolist() == [False, False, True, False]
    assert not fails["low_genes"].any()
    reasons = dec.first_reason(fails)
    assert reasons[0] == "low_counts" and pd.isna(reasons[1]) and reasons[2] == "no_nucleus" and pd.isna(reasons[3])
    with pytest.raises(dec.DecisionsError, match="for a scrna store; this input is spatial"):
        dec.validate(dec.empty(SCRNA), SPATIAL)
    with pytest.raises(dec.DecisionsError, match="may only set"):
        dec.validate({**dec.empty(SPATIAL), "defaults": {"max_mito": 5}}, SPATIAL)


def test_a_rule_without_its_column_never_fails():
    obs = pd.DataFrame({"sample_id": pd.Categorical(["S1"]), "transcript_counts": [1], "n_genes_by_counts": [1]})
    assert not dec.failures(obs, dec.empty(SPATIAL), SPATIAL)["no_nucleus"].any()
