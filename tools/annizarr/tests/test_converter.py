from pathlib import Path
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from annizarr._ops import convert_10x_h5, convert_adata, convert_h5ad
from annizarr.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from annizarr.errors import ConversionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(x_storage: str, sparse_flat_chunk: int = 2048) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=False, consolidate_metadata=False, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=2, x_col_chunk=2, sparse_flat_chunk=sparse_flat_chunk),
        validation=ValidationConfig(),
    )


def _cfg_backed(x_storage: str) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=False, consolidate_metadata=False, x_storage=x_storage, backed=True),
        chunks=ChunkConfig(x_row_chunk=2, x_col_chunk=2, sparse_flat_chunk=2048),
        validation=ValidationConfig(),
    )


def _make_h5ad(path: Path, dtype=np.float64) -> None:
    """Write a small CSR h5ad with a layer and raw."""
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=dtype))
    adata = ad.AnnData(X=X)
    adata.layers["counts"] = X.copy()
    adata.raw = adata.copy()
    adata.write_h5ad(path)


def _make_csc_h5ad(path: Path) -> None:
    """Write a small CSC h5ad where X requires CSR conversion."""
    X = sp.csc_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]))
    adata = ad.AnnData(X=X)
    adata.layers["counts"] = sp.csr_matrix(X)
    adata.raw = adata.copy()
    adata.write_h5ad(path)


def _make_large_h5ad(path: Path) -> None:
    """Write a larger CSR h5ad so nnz > any small flat_chunk."""
    rng = np.random.default_rng(0)
    dense = rng.random((50, 40))
    dense[dense < 0.7] = 0.0  # ~30% density → ~600 nnz
    X = sp.csr_matrix(dense)
    adata = ad.AnnData(X=X)
    adata.layers["counts"] = X.copy()
    adata.write_h5ad(path)


def _make_10x_h5(base: Path) -> Path:
    """Create a minimal Cell Ranger v3 HDF5 file (3 genes x 2 barcodes)."""
    h5_path = base / "matrix.h5"
    with h5py.File(h5_path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
        m.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
        m.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
        m.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
        m.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array([b"ENSG001", b"ENSG002", b"ENSG003"]))
        feat.create_dataset("name", data=np.array([b"GENEA", b"GENEB", b"GENEC"]))
        feat.create_dataset("feature_type", data=np.array([b"Gene Expression"] * 3))
    return h5_path


def _flat_chunks(zarr_path: Path, group_path: str) -> tuple[int, ...]:
    """Return the chunks of the flat 'data' array for a sparse zarr group."""
    store = zarr.open_group(str(zarr_path), mode="r")
    node = store
    for part in group_path.split("/"):
        node = node[part]
    return node["data"].chunks


def _messages(caplog):
    return [r.message for r in caplog.records]


# ---------------------------------------------------------------------------
# convert x_storage x dtype matrix (a representative sample of the full grid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_storage", ["csr", "csc", "dense"])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_convert_h5ad_x_storage_by_dtype(tmp_path: Path, x_storage: str, dtype) -> None:
    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=dtype))
    ad.AnnData(X=X).write_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "out.zarr"
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=_cfg(x_storage))
    got = ad.read_zarr(str(out))
    got_x = np.asarray(got.X.todense() if sp.issparse(got.X) else got.X)
    np.testing.assert_array_equal(got_x, X.toarray())
    assert got_x.dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# Eager loading — default (h5ad → zarr)
# In-memory CSR matrix: exercises da.from_array + map_blocks dask path for
# dense, and direct write_elem for sparse.
# ---------------------------------------------------------------------------


def test_h5ad_eager_csr_output(tmp_path: Path) -> None:
    _make_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csr(out.X)
    assert sp.isspmatrix_csr(out.layers["counts"])


def test_h5ad_eager_csc_output(tmp_path: Path) -> None:
    _make_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("csc"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csc(out.X)
    assert sp.isspmatrix_csc(out.layers["counts"])


def test_h5ad_eager_dense_output(tmp_path: Path) -> None:
    """In-memory CSR written as dense via dask map_blocks path."""
    _make_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("dense"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert not sp.issparse(out.X)
    assert not sp.issparse(out.layers["counts"])


def test_h5ad_eager_csc_x_converts_to_csr_with_warning(tmp_path: Path, caplog) -> None:
    _make_csc_h5ad(tmp_path / "input_csc.h5ad")
    convert_h5ad(str(tmp_path / "input_csc.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csr(out.X)
    assert any("adata.X was CSC and has been converted to CSR in memory" in m for m in _messages(caplog))


# ---------------------------------------------------------------------------
# Backed loading (h5ad → zarr, --backed flag)
# _CSRDataset code paths in write_matrix and _write_sparse_as_dense_dask.
# ---------------------------------------------------------------------------


def test_h5ad_backed_flag_uses_backed_read(tmp_path: Path) -> None:
    """--backed causes backed="r" load; omitting it causes eager load."""
    _make_h5ad(tmp_path / "input.h5ad")
    with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
        convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg_backed("csr"))
    assert mocked.call_args_list[0].kwargs.get("backed") == "r"


def test_h5ad_eager_does_not_use_backed_read(tmp_path: Path) -> None:
    """Default (backed=False) uses eager load."""
    _make_h5ad(tmp_path / "input.h5ad")
    with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
        convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr"))
    assert "backed" not in mocked.call_args_list[0].kwargs


def test_h5ad_backed_csr_output(tmp_path: Path) -> None:
    """Backed _CSRDataset written as CSR — no format conversion needed."""
    _make_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg_backed("csr"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csr(out.X)
    assert sp.isspmatrix_csr(out.layers["counts"])


def test_h5ad_backed_csc_output(tmp_path: Path) -> None:
    """Backed _CSRDataset loaded into memory and converted to CSC on write."""
    _make_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg_backed("csc"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csc(out.X)
    assert sp.isspmatrix_csc(out.layers["counts"])


def test_h5ad_backed_dense_output(tmp_path: Path) -> None:
    """Backed _CSRDataset written as dense via dask delayed row-slice path."""
    _make_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg_backed("dense"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert not sp.issparse(out.X)
    assert not sp.issparse(out.layers["counts"])


def test_h5ad_backed_csc_x_converts_to_csr_with_warning(tmp_path: Path, caplog) -> None:
    _make_csc_h5ad(tmp_path / "input_csc.h5ad")
    convert_h5ad(str(tmp_path / "input_csc.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg_backed("csr"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csr(out.X)
    assert any("adata.X was CSC and has been converted to CSR in memory" in m for m in _messages(caplog))


# ---------------------------------------------------------------------------
# In-memory loading (10x h5 → zarr)
# scanpy.read_10x_h5 always returns an in-memory CSR matrix, exercising the
# da.from_array + map_blocks dask path for dense and direct write_elem for sparse.
# ---------------------------------------------------------------------------


def test_10x_csr_output(tmp_path: Path) -> None:
    convert_10x_h5(str(_make_10x_h5(tmp_path)), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert out.n_obs == 2 and out.n_vars == 3
    assert sp.isspmatrix_csr(out.X)


def test_10x_csc_output(tmp_path: Path) -> None:
    convert_10x_h5(str(_make_10x_h5(tmp_path)), output=str(tmp_path / "out.zarr"), cfg=_cfg("csc"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert sp.isspmatrix_csc(out.X)


def test_10x_dense_output(tmp_path: Path) -> None:
    """In-memory CSR written as dense via dask map_blocks path."""
    convert_10x_h5(str(_make_10x_h5(tmp_path)), output=str(tmp_path / "out.zarr"), cfg=_cfg("dense"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert not sp.issparse(out.X)


def test_10x_rejects_wrong_extension(tmp_path: Path) -> None:
    fake = tmp_path / "matrix.h5ad"
    fake.write_bytes(b"not real")
    with pytest.raises(ConversionError):
        convert_10x_h5(str(fake), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr"))


# ---------------------------------------------------------------------------
# Flat chunk sizing for sparse output
# ---------------------------------------------------------------------------


def test_flat_chunk_applied_csr(tmp_path: Path) -> None:
    _make_large_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(
        str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr", sparse_flat_chunk=100)
    )
    assert _flat_chunks(tmp_path / "out.zarr", "X") == (100,)
    assert _flat_chunks(tmp_path / "out.zarr", "layers/counts") == (100,)


def test_flat_chunk_applied_csc(tmp_path: Path) -> None:
    _make_large_h5ad(tmp_path / "input.h5ad")
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=_cfg("csc", sparse_flat_chunk=75))
    assert _flat_chunks(tmp_path / "out.zarr", "X") == (75,)
    assert _flat_chunks(tmp_path / "out.zarr", "layers/counts") == (75,)


# ---------------------------------------------------------------------------
# Dense adata.X input (issue #4)
# ---------------------------------------------------------------------------


def _make_dense_h5ad(path: Path) -> np.ndarray:
    """Write an h5ad whose X is a plain dense ndarray (issue #4's input shape)."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    ad.AnnData(X=dense.copy()).write_h5ad(path)
    return dense


def test_h5ad_eager_dense_input_csr_output(tmp_path: Path, caplog) -> None:
    """Issue #4: a dense-X h5ad converts to sparse output instead of erroring."""
    h5 = tmp_path / "in.h5ad"
    dense = _make_dense_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg("csr"))
    z = zarr.open(str(out), mode="r")
    assert z["X"].attrs["encoding-type"] == "csr_matrix"
    got = sp.csr_matrix(
        (z["X/data"][:], z["X/indices"][:], z["X/indptr"][:]), shape=tuple(z["X"].attrs["shape"])
    ).toarray()
    np.testing.assert_array_equal(got, dense)
    assert any("dense" in m for m in _messages(caplog))


def test_h5ad_eager_dense_input_dense_output(tmp_path: Path) -> None:
    h5 = tmp_path / "in.h5ad"
    dense = _make_dense_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg("dense"))
    z = zarr.open(str(out), mode="r")
    np.testing.assert_array_equal(z["X"][:], dense)


def test_h5ad_backed_dense_input_dense_output(tmp_path: Path) -> None:
    """Backed dense X streams to dense output without materialising."""
    h5 = tmp_path / "in.h5ad"
    dense = _make_dense_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg_backed("dense"))
    z = zarr.open(str(out), mode="r")
    np.testing.assert_array_equal(z["X"][:], dense)


def test_h5ad_backed_dense_input_sparse_rejected(tmp_path: Path) -> None:
    """Backed dense -> sparse would silently materialise; refuse with guidance."""
    h5 = tmp_path / "in.h5ad"
    _make_dense_h5ad(h5)
    with pytest.raises(ConversionError, match="dense"):
        convert_h5ad(str(h5), output=str(tmp_path / "out.zarr"), cfg=_cfg_backed("csr"))


def test_h5ad_eager_dense_input_with_dense_layer_csr_output(tmp_path: Path) -> None:
    """Dense X plus a dense layer both sparsify under sparse output (no mid-write crash)."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    adata = ad.AnnData(X=dense.copy())
    adata.layers["scaled"] = dense * 2
    h5 = tmp_path / "in.h5ad"
    adata.write_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg("csr"))
    z = zarr.open(str(out), mode="r")
    assert z["layers/scaled"].attrs["encoding-type"] == "csr_matrix"
    got = sp.csr_matrix(
        (z["layers/scaled/data"][:], z["layers/scaled/indices"][:], z["layers/scaled/indptr"][:]),
        shape=tuple(z["layers/scaled"].attrs["shape"]),
    ).toarray()
    np.testing.assert_array_equal(got, dense * 2)


def test_convert_adata_in_memory(tmp_path: Path) -> None:
    """Public library entry: an in-memory AnnData converts like convert_h5ad."""
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]))
    adata = ad.AnnData(X=X)
    out = tmp_path / "out.zarr"
    convert_adata(adata, output=str(out), cfg=_cfg("csr"))
    got = ad.read_zarr(str(out))
    np.testing.assert_array_equal(got.X.toarray(), X.toarray())

    with pytest.raises(ConversionError, match="in-memory"):
        convert_adata(adata, output=str(tmp_path / "b.zarr"), cfg=_cfg_backed("csr"))
