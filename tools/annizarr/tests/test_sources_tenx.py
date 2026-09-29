from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from _builders import make_10x_v2_h5, make_10x_v3_h5
from annizarr._sources._tenx import load_10x_h5
from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.errors import ConversionError
from annizarr.sources import open_source

_EXPECTED_DENSE = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=np.float32)


def _cfg() -> AppConfig:
    return AppConfig(io=IOConfig(), chunks=ChunkConfig(x_row_chunk=4, x_col_chunk=4, sparse_flat_chunk=64))


def test_v3_reads_expected_x_obs_var_gex_filter_and_dtype_widen(tmp_path: Path) -> None:
    p = make_10x_v3_h5(tmp_path / "v3.h5")
    adata = load_10x_h5(p)

    assert sp.isspmatrix_csr(adata.X)
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X.toarray(), _EXPECTED_DENSE)
    assert list(adata.obs_names) == ["CELL1-1", "CELL2-1"]
    assert list(adata.var_names) == ["GENEA", "GENEB", "GENEC"]
    assert list(adata.var.columns) == ["gene_ids", "feature_types", "genome"]
    assert list(adata.var["gene_ids"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(adata.var["feature_types"]) == ["Gene Expression"] * 3
    assert list(adata.var["genome"]) == ["GRCh38"] * 3
    assert all(adata.var[c].dtype == object for c in adata.var.columns)

    # a non-Gene-Expression feature row is dropped by the GEX-only filter
    p_mixed = make_10x_v3_h5(tmp_path / "v3_mixed.h5", include_non_gex=True)
    mixed = load_10x_h5(p_mixed)
    assert mixed.shape == (2, 3)
    assert "ABC1" not in mixed.var_names
    assert set(mixed.var["feature_types"]) == {"Gene Expression"}
    np.testing.assert_array_equal(mixed.X.toarray(), _EXPECTED_DENSE)

    # int32 on-disk data is widened to float32
    p_int = make_10x_v3_h5(tmp_path / "v3_int.h5", int32_data=True)
    widened = load_10x_h5(p_int)
    assert widened.X.dtype == np.float32
    np.testing.assert_array_equal(widened.X.toarray(), _EXPECTED_DENSE)


def test_v2_single_genome_reads_expected(tmp_path: Path) -> None:
    p = make_10x_v2_h5(tmp_path / "v2.h5")
    adata = load_10x_h5(p)

    assert sp.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), _EXPECTED_DENSE)
    assert list(adata.obs_names) == ["CELL1-1", "CELL2-1"]
    assert list(adata.var_names) == ["GENEA", "GENEB", "GENEC"]
    assert list(adata.var.columns) == ["gene_ids"]
    assert list(adata.var["gene_ids"]) == ["ENSG001", "ENSG002", "ENSG003"]


def test_load_10x_h5_guards(tmp_path: Path) -> None:
    import h5py

    p_no_features = tmp_path / "no_features.h5"
    with h5py.File(p_no_features, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([b"CELL1-1"]))
        m.create_dataset("data", data=np.array([1.0], dtype=np.float32))
        m.create_dataset("indices", data=np.array([0], dtype=np.int32))
        m.create_dataset("indptr", data=np.array([0, 1], dtype=np.int32))
        m.create_dataset("shape", data=np.array([1, 1], dtype=np.int32))
    with pytest.raises(ValueError, match="no features group"):
        load_10x_h5(p_no_features)

    p_multi_genome = make_10x_v2_h5(tmp_path / "v2.h5", genomes=("GRCh38", "mm10"))
    with pytest.raises(ValueError, match="more than one genome"):
        load_10x_h5(p_multi_genome)

    p_empty = tmp_path / "empty.h5"
    with h5py.File(p_empty, "w"):
        pass
    with pytest.raises(ValueError, match="not a recognised 10x v2 layout"):
        load_10x_h5(p_empty)

    # load_10x_h5 raises the plain ValueError above; the source registry (the public,
    # end-user-facing entry point) wraps it into a ConversionError with a clear message.
    with pytest.raises(ConversionError, match="Failed to read 10x H5 file"):
        open_source(str(p_empty), _cfg(), fmt="10x")


# ---------------------------------------------------------------------------
# Reference test: pin our h5py reader against scanpy's own read_10x_h5
# ---------------------------------------------------------------------------


def test_matches_scanpy_v3_and_v2(tmp_path: Path) -> None:
    sc = pytest.importorskip("scanpy")
    import pandas.testing as pdt

    for p in (
        make_10x_v3_h5(tmp_path / "v3.h5", include_non_gex=True),
        make_10x_v2_h5(tmp_path / "v2.h5"),
    ):
        ours = load_10x_h5(p)
        theirs = sc.read_10x_h5(str(p))

        np.testing.assert_array_equal(ours.X.toarray(), theirs.X.toarray())
        assert ours.X.dtype == theirs.X.dtype
        assert list(ours.obs_names) == list(theirs.obs_names)
        assert list(ours.var_names) == list(theirs.var_names)
        pdt.assert_frame_equal(ours.var, theirs.var, check_dtype=True)
