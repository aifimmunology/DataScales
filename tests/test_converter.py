from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import scipy.sparse as sp
import zarr

from datascale.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from datascale.converter import (
    ConversionError,
    convert_10x_h5_to_zarr,
    convert_h5ad_to_zarr,
)


def _minimal_cfg(x_storage: str, sparse_flat_chunk: int = 2048) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=False, consolidate_metadata=False, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=2, x_col_chunk=2, sparse_flat_chunk=sparse_flat_chunk),
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

    warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("sparse-csr"))
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


def test_convert_auto_mode_prefers_backed_read(tmp_path: Path) -> None:
    input_path = tmp_path / "input_auto.h5ad"
    output_path = tmp_path / "output_auto.zarr"
    _make_h5ad(input_path, np.array([[1.0, 0.0], [0.0, 2.0]]))

    with patch("datascale.converter.ad.read_h5ad", wraps=ad.read_h5ad) as mocked:
        warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("auto"))

    assert warnings == []
    assert mocked.call_args_list[0].kwargs.get("backed") == "r"


def test_convert_non_auto_mode_uses_eager_read(tmp_path: Path) -> None:
    input_path = tmp_path / "input_eager.h5ad"
    output_path = tmp_path / "output_eager.zarr"
    _make_h5ad(input_path, np.array([[1.0, 0.0], [0.0, 2.0]]))

    with patch("datascale.converter.ad.read_h5ad", wraps=ad.read_h5ad) as mocked:
        warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("sparse-csr"))

    assert warnings == []
    assert "backed" not in mocked.call_args_list[0].kwargs


# ---------------------------------------------------------------------------
# CSC format tests
# ---------------------------------------------------------------------------

def test_convert_forces_csc_output_from_dense_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input_dense.h5ad"
    output_path = tmp_path / "output_csc.zarr"
    _make_h5ad(input_path, np.array([[1.0, 0.0], [0.0, 2.0]]))

    warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("sparse-csc"))
    assert warnings == []

    out = ad.read_zarr(str(output_path))
    assert sp.isspmatrix_csc(out.X)
    assert sp.isspmatrix_csc(out.layers["counts"])


def test_convert_forces_csc_output_from_csr_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input_csr.h5ad"
    output_path = tmp_path / "output_csc.zarr"
    _make_h5ad(input_path, sp.csr_matrix(np.array([[3.0, 0.0], [0.0, 4.0]])))

    warnings = convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("sparse-csc"))
    assert warnings == []

    out = ad.read_zarr(str(output_path))
    assert sp.isspmatrix_csc(out.X)
    assert sp.isspmatrix_csc(out.layers["counts"])


def test_convert_sparse_mode_produces_csr(tmp_path: Path) -> None:
    """Existing 'sparse' mode converts dense → CSR (not CSC)."""
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.zarr"
    _make_h5ad(input_path, np.array([[1.0, 0.0], [0.0, 2.0]]))

    convert_h5ad_to_zarr(str(input_path), str(output_path), _minimal_cfg("sparse-csr"))

    out = ad.read_zarr(str(output_path))
    assert sp.isspmatrix_csr(out.X)


# ---------------------------------------------------------------------------
# sparse_flat_chunk tests
# ---------------------------------------------------------------------------

def _make_large_sparse_h5ad(path: Path) -> None:
    """Write a moderately large sparse matrix so nnz > any small chunk size."""
    rng = np.random.default_rng(0)
    dense = rng.random((50, 40))
    dense[dense < 0.7] = 0.0  # ~30% density → ~600 nnz
    X = sp.csr_matrix(dense)
    adata = ad.AnnData(X=X)
    adata.layers["counts"] = X.copy()
    adata.write_h5ad(path)


def _sparse_flat_chunks(zarr_path: Path, group_path: str) -> tuple[int, ...]:
    """Return the chunks tuple of the 'data' flat array for a sparse zarr group."""
    store = zarr.open_group(str(zarr_path), mode="r")
    node = store
    for part in group_path.split("/"):
        node = node[part]
    return node["data"].chunks


def test_sparse_flat_chunk_applied_for_sparse_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.zarr"
    _make_large_sparse_h5ad(input_path)

    flat_chunk = 100
    convert_h5ad_to_zarr(
        str(input_path), str(output_path), _minimal_cfg("sparse-csr", sparse_flat_chunk=flat_chunk)
    )

    assert _sparse_flat_chunks(output_path, "X") == (flat_chunk,)
    assert _sparse_flat_chunks(output_path, "layers/counts") == (flat_chunk,)


def test_sparse_flat_chunk_applied_for_csc_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.zarr"
    _make_large_sparse_h5ad(input_path)

    flat_chunk = 75
    convert_h5ad_to_zarr(
        str(input_path), str(output_path), _minimal_cfg("sparse-csc", sparse_flat_chunk=flat_chunk)
    )

    assert _sparse_flat_chunks(output_path, "X") == (flat_chunk,)
    assert _sparse_flat_chunks(output_path, "layers/counts") == (flat_chunk,)


def test_sparse_flat_chunk_not_applied_for_dense_output(tmp_path: Path) -> None:
    """Dense output should not be affected by sparse_flat_chunk."""
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.zarr"
    _make_h5ad(input_path, np.array([[1.0, 0.0], [0.0, 2.0]]))

    convert_h5ad_to_zarr(
        str(input_path), str(output_path), _minimal_cfg("dense", sparse_flat_chunk=10)
    )

    out = ad.read_zarr(str(output_path))
    assert not sp.issparse(out.X)


# ---------------------------------------------------------------------------
# 10x HDF5 tests
# ---------------------------------------------------------------------------

def _make_10x_h5(base: Path) -> Path:
    """Create a minimal Cell Ranger v3 HDF5 file (3 genes x 2 barcodes, 3 nnz)."""
    import h5py

    h5_path = base / "matrix.h5"
    # CSC stored as (n_genes x n_barcodes): columns = barcodes, rows = genes
    # barcode 0: genes 0, 2 → values 1.0, 3.0
    # barcode 1: gene  1 → value  2.0
    with h5py.File(h5_path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
        m.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
        m.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
        m.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
        m.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))  # [n_genes, n_barcodes]
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array([b"ENSG001", b"ENSG002", b"ENSG003"]))
        feat.create_dataset("name", data=np.array([b"GENEA", b"GENEB", b"GENEC"]))
        feat.create_dataset("feature_type", data=np.array([b"Gene Expression"] * 3))
    return h5_path


# ---------------------------------------------------------------------------
# 10x HDF5 tests
# ---------------------------------------------------------------------------

def test_convert_10x_h5_sparse(tmp_path: Path) -> None:
    h5_path = _make_10x_h5(tmp_path)
    output_path = tmp_path / "output.zarr"

    warnings = convert_10x_h5_to_zarr(str(h5_path), str(output_path), _minimal_cfg("sparse-csr"))
    assert warnings == []

    out = ad.read_zarr(str(output_path))
    assert out.n_obs == 2
    assert out.n_vars == 3
    assert sp.issparse(out.X)


def test_convert_10x_h5_csc(tmp_path: Path) -> None:
    h5_path = _make_10x_h5(tmp_path)
    output_path = tmp_path / "output.zarr"

    convert_10x_h5_to_zarr(str(h5_path), str(output_path), _minimal_cfg("sparse-csc"))

    out = ad.read_zarr(str(output_path))
    assert sp.isspmatrix_csc(out.X)


def test_convert_10x_h5_dense(tmp_path: Path) -> None:
    h5_path = _make_10x_h5(tmp_path)
    output_path = tmp_path / "output.zarr"

    convert_10x_h5_to_zarr(str(h5_path), str(output_path), _minimal_cfg("dense"))

    out = ad.read_zarr(str(output_path))
    assert not sp.issparse(out.X)


def test_convert_10x_h5_rejects_wrong_extension(tmp_path: Path) -> None:
    fake = tmp_path / "matrix.h5ad"
    fake.write_bytes(b"not real")

    try:
        convert_10x_h5_to_zarr(str(fake), str(tmp_path / "out.zarr"), _minimal_cfg("sparse-csr"))
        assert False, "Expected ConversionError"
    except ConversionError:
        pass
