import filecmp
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

import annizarr._layout as _layout
from annizarr._ops import convert_10x_h5, convert_adata, convert_h5ad
from annizarr.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from annizarr.errors import ConversionError, StorageError

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


def _make_300x200_h5ad(path: Path, *, x_csc: bool) -> None:
    """A 300x200 h5ad; X is CSC when ``x_csc``, else CSR with a CSC 'cnt' layer either way —
    both exercise the backed-CSC-to-dense path (anndata's backed _CSCDataset has no .tocsr())."""
    rng = np.random.default_rng(7)
    dense = rng.random((300, 200)).astype(np.float32)
    dense[dense < 0.7] = 0.0  # ~30% density
    X = sp.csc_matrix(dense) if x_csc else sp.csr_matrix(dense)
    adata = ad.AnnData(X=X)
    adata.layers["cnt"] = sp.csc_matrix(dense * 2.0)
    adata.write_h5ad(path)


@pytest.mark.parametrize("x_csc", [False, True])
def test_h5ad_backed_csc_to_dense_matches_eager_no_tmp_dir_left(tmp_path: Path, x_csc: bool) -> None:
    """A backed CSC source (X, or the 'cnt' layer) converted with x_storage='dense' streams
    through a temporary CSC->CSR transpose (never materialising the whole matrix) and matches
    the eager (in-memory) conversion byte-for-byte; the temp dir is cleaned up either way."""
    # chunks sized for the 300x200 fixture (not the module _cfg's 2x2, which would tile it
    # into thousands of tiny writes)
    chunks = ChunkConfig(x_row_chunk=64, x_col_chunk=64)
    cfg_eager = AppConfig(io=IOConfig(x_storage="dense"), chunks=chunks, validation=ValidationConfig())
    cfg_backed = replace(cfg_eager, io=replace(cfg_eager.io, backed=True))

    _make_300x200_h5ad(tmp_path / "input.h5ad", x_csc=x_csc)
    out_eager = tmp_path / "eager.zarr"
    out_backed = tmp_path / "backed.zarr"
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(out_eager), cfg=cfg_eager)
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(out_backed), cfg=cfg_backed)

    got_eager = ad.read_zarr(str(out_eager))
    got_backed = ad.read_zarr(str(out_backed))
    np.testing.assert_array_equal(np.asarray(got_backed.X), np.asarray(got_eager.X))
    np.testing.assert_array_equal(np.asarray(got_backed.layers["cnt"]), np.asarray(got_eager.layers["cnt"]))
    assert not list(tmp_path.rglob("annizarr_csc2csr_*"))


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


def test_convert_h5ad_existing_target_fails_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """check_output_target fails fast, before load_h5ad ever runs (no wasted parse)."""
    _make_h5ad(tmp_path / "input.h5ad")
    out = tmp_path / "out.zarr"
    out.mkdir()  # a pre-existing target, overwrite not set

    import annizarr._ops._convert as _convert_mod

    def _fail_loudly(*_a, **_kw):
        raise AssertionError("load_h5ad should not run when the target already exists and overwrite is unset")

    monkeypatch.setattr(_convert_mod, "load_h5ad", _fail_loudly)
    with pytest.raises(StorageError, match="already exists"):
        convert_h5ad(str(tmp_path / "input.h5ad"), output=str(out), cfg=_cfg("csr"))


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


# ---------------------------------------------------------------------------
# Streamed sparse transposition (backed CSR<->CSC via write_transposed_sparse)
# ---------------------------------------------------------------------------


def _tree_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}


def _assert_byte_identical(a: Path, b: Path) -> None:
    fa, fb = _tree_files(a), _tree_files(b)
    assert set(fa) == set(fb), f"file set differs: {a.name} has {set(fa) - set(fb)}, {b.name} has {set(fb) - set(fa)}"
    for rel in sorted(fa):
        assert filecmp.cmp(fa[rel], fb[rel], shallow=False), f"differs: {rel}"


def test_h5ad_backed_csr_to_csc_streamed_matches_eager_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backed CSR X targeted at csc output streams through write_transposed_sparse (never
    materialises X); the resulting store must be byte-identical to converting the same
    file eagerly. A tiny BATCH_BYTES forces many target (column) bands as well as several
    output sparse_flat_chunk-sized chunks with a ragged tail; X's indices are scipy's
    default int32."""
    monkeypatch.setattr(_layout, "BATCH_BYTES", 2048)

    rng = np.random.default_rng(7)
    n_obs, n_vars = 300, 200
    dense = rng.random((n_obs, n_vars))
    dense[dense < 0.85] = 0.0  # ~15% density, ~9000 nnz
    X = sp.csr_matrix(dense.astype(np.float32))
    assert X.indices.dtype == np.int32
    h5 = tmp_path / "in.h5ad"
    ad.AnnData(X=X).write_h5ad(h5)

    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csc"),
        chunks=ChunkConfig(x_row_chunk=64, x_col_chunk=48, sparse_flat_chunk=37),
        validation=ValidationConfig(),
    )
    cfg_backed = replace(cfg, io=replace(cfg.io, backed=True))

    out_eager = tmp_path / "eager.zarr"
    out_backed = tmp_path / "backed.zarr"
    convert_h5ad(str(h5), output=str(out_eager), cfg=cfg)
    convert_h5ad(str(h5), output=str(out_backed), cfg=cfg_backed)

    _assert_byte_identical(out_eager, out_backed)
    got = ad.read_zarr(str(out_backed))
    assert sp.isspmatrix_csc(got.X)
    np.testing.assert_allclose(got.X.toarray(), dense.astype(np.float32))


def test_h5ad_backed_csc_x_to_csr_streamed_matches_eager_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backed CSC X targeted at csr output (item 5): convert no longer eagerly materialises
    it via .tocsr() first (anndata's backed _CSCDataset has no .tocsr() anyway) — write_matrix
    streams it through write_transposed_sparse instead, matching the eager conversion
    byte-for-byte."""
    monkeypatch.setattr(_layout, "BATCH_BYTES", 2048)

    rng = np.random.default_rng(9)
    n_obs, n_vars = 300, 200
    dense = rng.random((n_obs, n_vars))
    dense[dense < 0.85] = 0.0  # ~15% density, ~9000 nnz
    X = sp.csc_matrix(dense.astype(np.float32))
    h5 = tmp_path / "in.h5ad"
    ad.AnnData(X=X).write_h5ad(h5)

    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=ChunkConfig(x_row_chunk=64, x_col_chunk=48, sparse_flat_chunk=37),
        validation=ValidationConfig(),
    )
    cfg_backed = replace(cfg, io=replace(cfg.io, backed=True))

    out_eager = tmp_path / "eager.zarr"
    out_backed = tmp_path / "backed.zarr"
    convert_h5ad(str(h5), output=str(out_eager), cfg=cfg)
    convert_h5ad(str(h5), output=str(out_backed), cfg=cfg_backed)

    _assert_byte_identical(out_eager, out_backed)
    got = ad.read_zarr(str(out_backed))
    assert sp.isspmatrix_csr(got.X)
    np.testing.assert_allclose(got.X.toarray(), dense.astype(np.float32))


def _make_mixed_format_h5ad(path: Path) -> np.ndarray:
    """X is CSR on disk; the 'counts' layer is CSC on disk — both are backed once opened
    with backed="r", so this mixes the two directions in one file."""
    dense = np.array(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0], [6.0, 7.0, 0.0]],
        dtype=np.float32,
    )
    adata = ad.AnnData(X=sp.csr_matrix(dense))
    adata.layers["counts"] = sp.csc_matrix(dense * 2)
    adata.write_h5ad(path)
    return dense


def test_h5ad_backed_csc_layer_streamed_to_csr(tmp_path: Path) -> None:
    """A backed CSC layer targeted at csr output streams through write_transposed_sparse
    too (the CSC->CSR direction). X is CSR here so convert's own X-specific CSC-to-CSR
    normalisation (out of this unit's scope; always forces CSR eagerly regardless of
    x_storage) never fires — write_matrix's generic backed dispatch handles the layer."""
    h5 = tmp_path / "mixed.h5ad"
    dense = _make_mixed_format_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg_backed("csr"))
    got = ad.read_zarr(str(out))
    assert sp.isspmatrix_csr(got.layers["counts"])
    np.testing.assert_allclose(got.layers["counts"].toarray(), dense * 2)
    np.testing.assert_allclose(got.X.toarray(), dense)
