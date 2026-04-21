from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from datascale.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from datascale.converter import convert_h5ad_to_zarr


def _minimal_cfg(x_storage: str) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=False, consolidate_metadata=False, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=2, x_col_chunk=2),
        validation=ValidationConfig(),
    )


def _make_h5ad(path: Path, X: np.ndarray | sp.spmatrix) -> None:
    adata = ad.AnnData(X=X)
    adata.layers["counts"] = X.copy() if hasattr(X, "copy") else X
    adata.raw = adata.copy()
    adata.write_h5ad(path)


def test_convert_forces_sparse_output_from_dense_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input_dense.h5ad"
    output_path = tmp_path / "output_sparse.zarr"
    _make_h5ad(input_path, np.array([[1.0, 0.0], [0.0, 2.0]]))

    warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("sparse"))
    assert warnings == []

    out = ad.read_zarr(str(output_path))
    assert sp.issparse(out.X)
    assert sp.issparse(out.layers["counts"])


def test_convert_forces_dense_output_from_sparse_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input_sparse.h5ad"
    output_path = tmp_path / "output_dense.zarr"
    _make_h5ad(input_path, sp.csr_matrix(np.array([[3.0, 0.0], [0.0, 4.0]])))

    warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("dense"))
    assert warnings == []

    out = ad.read_zarr(str(output_path))
    assert not sp.issparse(out.X)
    assert not sp.issparse(out.layers["counts"])
