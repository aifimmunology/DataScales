import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
import zarr
from anndata.io import write_elem

import metrics
from conftest import GENES, sample_adata


def zarr_x(tmp_path, X) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "x.zarr"), mode="w")
    write_elem(root, "X", X, dataset_kwargs={"chunks": (97,)})
    return root


def test_matches_scanpy(tmp_path):
    adata = sample_adata("S1", n_cells=230, with_metrics=False)
    X = adata.X.tolil()
    X[7, :] = 0
    X[229, :] = 0
    adata.X = sp.csr_matrix(X)
    root = zarr_x(tmp_path, adata.X)
    got = metrics.compute(root, np.array(GENES), band_cells=40)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    assert got[metrics.GENES].tolist() == adata.obs["n_genes_by_counts"].tolist()
    np.testing.assert_allclose(got[metrics.COUNTS], adata.obs["total_counts"])
    expected = adata.obs["pct_counts_mt"].to_numpy()
    np.testing.assert_allclose(got[metrics.MITO][expected == expected], expected[expected == expected])
    assert got[metrics.COUNTS][7] == 0 and got[metrics.GENES][229] == 0


def test_refuses_non_integer_counts(tmp_path):
    root = zarr_x(tmp_path, sp.csr_matrix(np.array([[0.5, 1.0], [2.0, 0.0]], dtype=np.float32)))
    with pytest.raises(metrics.MetricsError, match="integer counts"):
        metrics.compute(root, np.array(["A", "B"]), band_cells=10)


def test_integer_valued_floats_are_accepted(tmp_path):
    root = zarr_x(tmp_path, sp.csr_matrix(np.array([[0, 3.0], [2.0, 0.0]], dtype=np.float32)))
    got = metrics.compute(root, np.array(["MT-1", "B"]), band_cells=1)
    assert got[metrics.COUNTS].tolist() == [3.0, 2.0]
    assert got[metrics.MITO].tolist() == [0.0, 100.0]


def test_missing_and_mito_mask():
    obs = pd.DataFrame({"total_counts": [1.0], "pct_counts_mt": pd.array([None], dtype="Float64")})
    assert metrics.missing(obs, ("n_genes_by_counts", "total_counts", "pct_counts_mt")) == ["n_genes_by_counts", "pct_counts_mt"]
    assert metrics.mito_mask(np.array(["MT-CO1", "mt-co1", "GENE"], dtype=object)).tolist() == [True, False, False]
