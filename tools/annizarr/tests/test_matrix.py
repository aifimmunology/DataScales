from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from annizarr._sources._matrix import matrix_format
from annizarr.errors import ConversionError


def test_matrix_format_csr_csc_and_dense() -> None:
    assert matrix_format(sp.csr_matrix(np.eye(3))) == "csr"
    assert matrix_format(sp.csc_matrix(np.eye(3))) == "csc"
    assert matrix_format(np.zeros((3, 4))) == "dense"


def test_matrix_format_rejects_coo_1d_and_unknown_objects() -> None:
    for bad in (sp.coo_matrix(np.eye(3)), np.zeros(5), object()):
        with pytest.raises(ConversionError, match="CSR, CSC, or a 2-D dense array"):
            matrix_format(bad)


def test_matrix_format_backed_csr(tmp_path: Path) -> None:
    x = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    ad.AnnData(X=x).write_h5ad(tmp_path / "in.h5ad")
    backed = ad.read_h5ad(tmp_path / "in.h5ad", backed="r")
    try:
        assert matrix_format(backed.X) == "csr"
    finally:
        backed.file.close()
