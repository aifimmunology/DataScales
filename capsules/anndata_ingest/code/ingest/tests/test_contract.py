import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import contract


def _adata(n=10, g=6, dtype=np.uint16):
    X = sp.random(n, g, density=0.5, format="csr", random_state=0)
    X.data = np.ones_like(X.data) * 3
    a = ad.AnnData(X.astype(dtype), var=pd.DataFrame(index=[f"MT-{i}" if i < 2 else f"G{i}" for i in range(g)]))
    a.obs["sample_id"] = "S"
    a.obs["n_umis"] = np.arange(n, dtype=np.uint32)
    a.obs["scrublet_score"] = np.linspace(0, 1, n)
    return a


def test_requested_schema_toggles():
    base = contract.requested_schema(False, [])
    with_all = contract.requested_schema(True, ["m1", "m2"])
    assert "scrublet_score" not in base and not any(c.endswith("_label") for c in base)
    assert with_all["predicted_doublets"] == contract.INT and with_all["m2_prob"] == contract.FLOAT
    assert list(with_all)[-4:] == ["m1_label", "m1_prob", "m2_label", "m2_prob"] and "leiden" not in with_all


def test_project_pads_and_casts():
    schema = contract.requested_schema(True, ["m1"])
    out = contract.project(_adata(), schema)
    assert out.X.dtype == np.uint32 and out.X.format == "csr"
    assert list(out.obs.columns) == list(schema)
    assert out.obs["predicted_doublets"].isna().all() and str(out.obs["predicted_doublets"].dtype) == "Int64"
    assert out.obs["m1_prob"].isna().all() and (out.obs["m1_label"] == "").all() and (out.obs["pool_id"] == "").all()
    assert out.obs["n_umis"].tolist() == list(range(10))
    assert out.var["mt"].sum() == 2 and out.obs.index.name == "barcodes"


def test_project_rejects_bad_counts_and_var():
    a = _adata(dtype=np.float32)
    a.X.data[0] = 1.5
    with pytest.raises(contract.ContractError, match="raw counts"):
        contract.project(a, contract.requested_schema(False, []))
    with pytest.raises(contract.ContractError, match="var mismatch"):
        contract.project(_adata(), contract.requested_schema(False, []), np.array(["other"] * 6, dtype=object))


def test_float_counts_that_are_whole_numbers_are_accepted():
    out = contract.project(_adata(dtype=np.float64), contract.requested_schema(False, []))
    assert out.X.dtype == np.uint32 and out.X.sum() == _adata().X.sum()


def test_check_schema_names_the_differing_columns():
    schema = contract.requested_schema(False, [])
    contract.check_schema(list(schema), schema)
    with pytest.raises(contract.ContractError, match=r"Store-only columns: \[\'scrublet_score\'\]"):
        contract.check_schema([*schema, "scrublet_score"], schema)


def test_var_mismatch_says_whether_only_the_order_differs():
    a = _adata()
    with pytest.raises(contract.ContractError, match="same genes in a different order"):
        contract.project(a, contract.requested_schema(False, []), np.asarray(a.var_names[::-1], dtype=object))
    with pytest.raises(contract.ContractError, match="1 not shared"):
        contract.project(a[:, :5], contract.requested_schema(False, []), np.asarray(a.var_names, dtype=object))
