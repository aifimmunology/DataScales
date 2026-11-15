"""The store contract, annotation, the gene policy, and the bundled CellTypist models."""
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scrublet

import prepare
from config import BUNDLED_MODELS


def test_scrublet_without_threshold_keeps_scores_and_leaves_calls_na(monkeypatch):
    n = 50
    monkeypatch.setattr(scrublet.Scrublet, "scrub_doublets", lambda self, **kw: (np.linspace(0, 1, n), None))
    counts = sp.random(n, 20, density=0.3, format="csr", random_state=0)
    counts.data = np.ceil(counts.data * 10)
    adata = ad.AnnData(counts)
    prepare.run_scrublet(adata)
    assert adata.obs["scrublet_score"].notna().all()
    assert adata.obs["predicted_doublets"].isna().all()
    schema = {"scrublet_score": prepare.FLOAT, "predicted_doublets": prepare.INT}
    projected = prepare.project(adata, schema, None)
    assert projected.obs["predicted_doublets"].isna().all()
    assert str(projected.obs["predicted_doublets"].dtype) == "Int64"


def _adata(n=10, g=6, dtype=np.uint16):
    X = sp.random(n, g, density=0.5, format="csr", random_state=0)
    X.data = np.ones_like(X.data) * 3
    a = ad.AnnData(X.astype(dtype), var=pd.DataFrame(index=[f"MT-{i}" if i < 2 else f"G{i}" for i in range(g)]))
    a.obs["sample_id"] = "S"
    a.obs["n_umis"] = np.arange(n, dtype=np.uint32)
    a.obs["scrublet_score"] = np.linspace(0, 1, n)
    return a


def test_requested_schema_toggles():
    base = prepare.requested_schema(False, [])
    with_all = prepare.requested_schema(True, ["m1", "m2"])
    assert "scrublet_score" not in base and not any(c.endswith("_label") for c in base)
    assert with_all["predicted_doublets"] == prepare.INT and with_all["m2_prob"] == prepare.FLOAT
    assert list(with_all)[-4:] == ["m1_label", "m1_prob", "m2_label", "m2_prob"] and "leiden" not in with_all


def test_project_pads_and_casts():
    schema = prepare.requested_schema(True, ["m1"])
    out = prepare.project(_adata(), schema)
    assert out.X.dtype == np.uint32 and out.X.format == "csr"
    assert list(out.obs.columns) == list(schema)
    assert out.obs["predicted_doublets"].isna().all() and str(out.obs["predicted_doublets"].dtype) == "Int64"
    assert out.obs["m1_prob"].isna().all() and (out.obs["m1_label"] == "").all() and (out.obs["pool_id"] == "").all()
    assert out.obs["n_umis"].tolist() == list(range(10))
    assert out.var["mt"].sum() == 2 and out.obs.index.name == "barcodes"


def test_project_rejects_bad_counts_and_var():
    a = _adata(dtype=np.float32)
    a.X.data[0] = 1.5
    with pytest.raises(prepare.ContractError, match="raw counts"):
        prepare.project(a, prepare.requested_schema(False, []))
    with pytest.raises(prepare.ContractError, match="var mismatch"):
        prepare.project(_adata(), prepare.requested_schema(False, []), np.array(["other"] * 6, dtype=object))


def test_float_counts_that_are_whole_numbers_are_accepted():
    out = prepare.project(_adata(dtype=np.float64), prepare.requested_schema(False, []))
    assert out.X.dtype == np.uint32 and out.X.sum() == _adata().X.sum()


def test_check_schema_names_the_differing_columns():
    schema = prepare.requested_schema(False, [])
    prepare.check_schema(list(schema), schema)
    with pytest.raises(prepare.ContractError, match=r"Store-only columns: \[\'scrublet_score\'\]"):
        prepare.check_schema([*schema, "scrublet_score"], schema)


def test_var_mismatch_says_whether_only_the_order_differs():
    a = _adata()
    with pytest.raises(prepare.ContractError, match="same genes in a different order"):
        prepare.project(a, prepare.requested_schema(False, []), np.asarray(a.var_names[::-1], dtype=object))
    with pytest.raises(prepare.ContractError, match="1 not shared"):
        prepare.project(a[:, :5], prepare.requested_schema(False, []), np.asarray(a.var_names, dtype=object))


def test_counts_too_large_for_the_store_dtype_are_refused():
    with pytest.raises(prepare.ContractError, match="don't fit"):
        prepare.fit_x(sp.csr_matrix(np.array([[5e9]])), np.dtype(np.uint32), True, "big")


def test_fitting_handles_duplicate_gene_names():
    sample = ad.AnnData(sp.csr_matrix(np.array([[1, 2, 3]], dtype=np.uint32)),
                        var=pd.DataFrame(index=["A", "B", "A"]))
    fitted = prepare.conform_var(sample, ["B", "A", "C"])
    assert fitted.X.toarray().tolist() == [[2, 1, 0]] and fitted.X.dtype == np.uint32


def test_bundled_models_match_their_recorded_source():
    source = json.loads((BUNDLED_MODELS / "SOURCE.json").read_text())
    found = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BUNDLED_MODELS.glob("*.pkl")}
    assert found == source["sha256"]
