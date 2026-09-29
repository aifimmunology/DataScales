from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from annizarr._ops import convert_adata
from annizarr._sources._matrix import matrix_format
from annizarr.config import AppConfig
from annizarr.errors import ConversionError


def test_matrix_format_csr() -> None:
    assert matrix_format(sp.csr_matrix(np.eye(3))) == "csr"


def test_matrix_format_csc() -> None:
    assert matrix_format(sp.csc_matrix(np.eye(3))) == "csc"


def test_matrix_format_dense_ndarray() -> None:
    assert matrix_format(np.zeros((3, 4))) == "dense"


def test_matrix_format_rejects_coo() -> None:
    with pytest.raises(ConversionError, match="CSR, CSC, or a 2-D dense array"):
        matrix_format(sp.coo_matrix(np.eye(3)))


def test_matrix_format_rejects_1d_array() -> None:
    with pytest.raises(ConversionError, match="CSR, CSC, or a 2-D dense array"):
        matrix_format(np.zeros(5))


def test_matrix_format_rejects_unknown_object() -> None:
    with pytest.raises(ConversionError, match="CSR, CSC, or a 2-D dense array"):
        matrix_format(object())


def test_matrix_format_backed_csr(tmp_path: Path) -> None:
    x = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    ad.AnnData(X=x).write_h5ad(tmp_path / "in.h5ad")
    backed = ad.read_h5ad(tmp_path / "in.h5ad", backed="r")
    try:
        assert matrix_format(backed.X) == "csr"
    finally:
        backed.file.close()


def test_convert_adata_coo_x_raises(tmp_path: Path) -> None:
    # anndata itself refuses non-CSR/CSC sparse X at construction/assignment time, so a
    # malformed COO X (e.g. loaded from a non-anndata path) is simulated via the private
    # attribute to exercise our own defensive check in matrix_format.
    adata = ad.AnnData(X=np.eye(3))
    adata._X = sp.coo_matrix(np.eye(3))
    with pytest.raises(ConversionError, match="CSR, CSC, or a 2-D dense array"):
        convert_adata(adata, output=tmp_path / "out.zarr", cfg=AppConfig())
