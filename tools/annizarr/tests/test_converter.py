import filecmp
import importlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

import annizarr._core._layout as _layout
from annizarr.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from annizarr.errors import ConversionError, StorageError
from annizarr.ops import convert_10x_h5, convert_adata, convert_h5ad

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
# convert x_storage x backed x dtype grid (the core dispatch matrix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_storage,backed,dtype",
    [
        ("csr", False, np.float32),
        ("csr", True, np.float32),
        ("csc", False, np.float32),
        ("csc", True, np.float32),
        ("dense", False, np.float32),
        ("dense", True, np.float32),
        ("csr", False, np.int32),  # the sole dtype-preservation case elsewhere in this suite
    ],
)
def test_convert_h5ad_x_storage_backed_dtype(tmp_path: Path, x_storage: str, backed: bool, dtype) -> None:
    """The core convert dispatch grid: eager/backed input x csr/csc/dense output. X (an
    in-memory or backed CSR source) and its 'counts' layer must both land in the target
    format with the source dtype preserved."""
    _make_h5ad(tmp_path / "input.h5ad", dtype=dtype)
    cfg = _cfg_backed(x_storage) if backed else _cfg(x_storage)
    convert_h5ad(str(tmp_path / "input.h5ad"), output=str(tmp_path / "out.zarr"), cfg=cfg)
    out = ad.read_zarr(str(tmp_path / "out.zarr"))

    if x_storage == "dense":
        assert not sp.issparse(out.X)
        assert not sp.issparse(out.layers["counts"])
    else:
        checker = sp.isspmatrix_csr if x_storage == "csr" else sp.isspmatrix_csc
        assert checker(out.X)
        assert checker(out.layers["counts"])
    got_x = np.asarray(out.X.todense() if sp.issparse(out.X) else out.X)
    assert got_x.dtype == np.dtype(dtype)


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


def test_h5ad_backed_csc_to_dense_matches_eager_no_tmp_dir_left(tmp_path: Path) -> None:
    """A backed CSC source (X, or the 'cnt' layer -- both directions checked) converted with
    x_storage='dense' streams through a temporary CSC->CSR transpose (never materialising the
    whole matrix) and matches the eager (in-memory) conversion byte-for-byte; the temp dir is
    cleaned up either way. Also the only regression guard that --backed actually causes a
    backed="r" load (an output-format-only assertion wouldn't catch a silently-ignored flag,
    since a small fixture converts identically either way)."""
    # chunks sized for the 300x200 fixture (not the module _cfg's 2x2, which would tile it
    # into thousands of tiny writes)
    chunks = ChunkConfig(x_row_chunk=64, x_col_chunk=64)
    for x_csc in (False, True):
        cfg_eager = AppConfig(io=IOConfig(x_storage="dense"), chunks=chunks, validation=ValidationConfig())
        cfg_backed = replace(cfg_eager, io=replace(cfg_eager.io, backed=True))

        input_h5 = tmp_path / f"input_{x_csc}.h5ad"
        _make_300x200_h5ad(input_h5, x_csc=x_csc)
        out_eager = tmp_path / f"eager_{x_csc}.zarr"
        out_backed = tmp_path / f"backed_{x_csc}.zarr"
        convert_h5ad(str(input_h5), output=str(out_eager), cfg=cfg_eager)
        with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
            convert_h5ad(str(input_h5), output=str(out_backed), cfg=cfg_backed)
        assert mocked.call_args_list[0].kwargs.get("backed") == "r"

        got_eager = ad.read_zarr(str(out_eager))
        got_backed = ad.read_zarr(str(out_backed))
        np.testing.assert_array_equal(np.asarray(got_backed.X), np.asarray(got_eager.X))
        np.testing.assert_array_equal(np.asarray(got_backed.layers["cnt"]), np.asarray(got_eager.layers["cnt"]))
    assert not list(tmp_path.rglob("annizarr_csc2csr_*"))


# ---------------------------------------------------------------------------
# In-memory loading (10x h5 → zarr)
# ---------------------------------------------------------------------------


def test_10x_csr_output(tmp_path: Path) -> None:
    """scanpy.read_10x_h5 always returns an in-memory CSR matrix — this just confirms
    convert_10x_h5 dispatches into the same write_matrix grid already covered above (the
    csc/dense output cases add nothing beyond that dispatch, so aren't repeated here), plus
    the wrong-extension guard on the 10x-specific loader."""
    convert_10x_h5(str(_make_10x_h5(tmp_path)), output=str(tmp_path / "out.zarr"), cfg=_cfg("csr"))
    out = ad.read_zarr(str(tmp_path / "out.zarr"))
    assert out.n_obs == 2 and out.n_vars == 3
    assert sp.isspmatrix_csr(out.X)

    fake = tmp_path / "matrix.h5ad"
    fake.write_bytes(b"not real")
    with pytest.raises(ConversionError):
        convert_10x_h5(str(fake), output=str(tmp_path / "out2.zarr"), cfg=_cfg("csr"))


# ---------------------------------------------------------------------------
# Flat chunk sizing for sparse output
# ---------------------------------------------------------------------------


def test_convert_h5ad_existing_target_fails_before_loading_and_flat_chunk_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_output_target fails fast, before load_h5ad ever runs (no wasted parse); plus
    flat chunk sizing is applied to both X and layers for csr and csc output."""
    _make_h5ad(tmp_path / "input.h5ad")
    out = tmp_path / "out.zarr"
    out.mkdir()  # a pre-existing target, overwrite not set

    # annizarr.ops._convert (the name) is shadowed by the convert function; fetch the module via sys.modules.
    _convert_mod = importlib.import_module("annizarr.ops._convert")

    def _fail_loudly(*_a, **_kw):
        raise AssertionError("load_h5ad should not run when the target already exists and overwrite is unset")

    monkeypatch.setattr(_convert_mod, "load_h5ad", _fail_loudly)
    with pytest.raises(StorageError, match="already exists"):
        convert_h5ad(str(tmp_path / "input.h5ad"), output=str(out), cfg=_cfg("csr"))
    monkeypatch.undo()

    _make_large_h5ad(tmp_path / "csr_input.h5ad")
    convert_h5ad(
        str(tmp_path / "csr_input.h5ad"), output=str(tmp_path / "csr_out.zarr"), cfg=_cfg("csr", sparse_flat_chunk=100)
    )
    assert _flat_chunks(tmp_path / "csr_out.zarr", "X") == (100,)
    assert _flat_chunks(tmp_path / "csr_out.zarr", "layers/counts") == (100,)

    _make_large_h5ad(tmp_path / "csc_input.h5ad")
    convert_h5ad(
        str(tmp_path / "csc_input.h5ad"), output=str(tmp_path / "csc_out.zarr"), cfg=_cfg("csc", sparse_flat_chunk=75)
    )
    assert _flat_chunks(tmp_path / "csc_out.zarr", "X") == (75,)
    assert _flat_chunks(tmp_path / "csc_out.zarr", "layers/counts") == (75,)


# ---------------------------------------------------------------------------
# Dense adata.X input (issue #4)
# ---------------------------------------------------------------------------


def _make_dense_h5ad(path: Path) -> np.ndarray:
    """Write an h5ad whose X is a plain dense ndarray (issue #4's input shape)."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    ad.AnnData(X=dense.copy()).write_h5ad(path)
    return dense


def test_h5ad_eager_dense_input_csr_output(tmp_path: Path, caplog) -> None:
    """Issue #4: a dense-X h5ad (X, and any dense layer) converts to sparse output instead
    of erroring — both the pre-normalized X (write_adata_to_store) and a layer hitting
    write_matrix's own internal dense-fmt branch must sparsify correctly."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    adata = ad.AnnData(X=dense.copy())
    adata.layers["scaled"] = dense * 2
    h5 = tmp_path / "in.h5ad"
    adata.write_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg("csr"))
    z = zarr.open(str(out), mode="r")

    assert z["X"].attrs["encoding-type"] == "csr_matrix"
    got_x = sp.csr_matrix(
        (z["X/data"][:], z["X/indices"][:], z["X/indptr"][:]), shape=tuple(z["X"].attrs["shape"])
    ).toarray()
    np.testing.assert_array_equal(got_x, dense)
    assert any("dense" in m for m in _messages(caplog))

    assert z["layers/scaled"].attrs["encoding-type"] == "csr_matrix"
    got_layer = sp.csr_matrix(
        (z["layers/scaled/data"][:], z["layers/scaled/indices"][:], z["layers/scaled/indptr"][:]),
        shape=tuple(z["layers/scaled"].attrs["shape"]),
    ).toarray()
    np.testing.assert_array_equal(got_layer, dense * 2)


def test_h5ad_eager_and_backed_dense_input_dense_output(tmp_path: Path) -> None:
    h5 = tmp_path / "in.h5ad"
    dense = _make_dense_h5ad(h5)

    out_eager = tmp_path / "out_eager.zarr"
    convert_h5ad(str(h5), output=str(out_eager), cfg=_cfg("dense"))
    np.testing.assert_array_equal(zarr.open(str(out_eager), mode="r")["X"][:], dense)

    # backed dense X streams to dense output without materialising; backed dense -> sparse
    # output would silently materialise, so it's refused with guidance instead
    out_backed = tmp_path / "out_backed.zarr"
    convert_h5ad(str(h5), output=str(out_backed), cfg=_cfg_backed("dense"))
    np.testing.assert_array_equal(zarr.open(str(out_backed), mode="r")["X"][:], dense)

    with pytest.raises(ConversionError, match="dense"):
        convert_h5ad(str(h5), output=str(tmp_path / "out2.zarr"), cfg=_cfg_backed("csr"))


def test_convert_adata_in_memory(tmp_path: Path) -> None:
    """Public library entry: an in-memory AnnData converts like convert_h5ad."""
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]))
    adata = ad.AnnData(X=X)
    out = tmp_path / "out.zarr"
    convert_adata(adata, output=str(out), cfg=_cfg("csr"))
    got = ad.read_zarr(str(out))
    np.testing.assert_array_equal(got.X.toarray(), X.toarray())

    # cfg.io.backed is synced from adata.isbacked, not taken at face value: a genuinely
    # in-memory AnnData converts fine even if the caller's cfg says backed=True.
    out_b = tmp_path / "b.zarr"
    convert_adata(adata, output=str(out_b), cfg=_cfg_backed("csr"))
    assert ad.read_zarr(str(out_b)).n_obs == adata.n_obs


def test_convert_adata_accepts_a_backed_anndata(tmp_path: Path) -> None:
    """convert_adata also takes an already-backed AnnData (e.g. ad.read_h5ad(path, backed='r')),
    syncing cfg.io.backed from it: backed dense X with sparse output hits the same interim
    guard as convert_h5ad; dense output streams without materialising."""
    h5 = tmp_path / "in.h5ad"
    dense = _make_dense_h5ad(h5)
    adata = ad.read_h5ad(h5, backed="r")
    try:
        out = tmp_path / "out.zarr"
        with pytest.raises(ConversionError, match="backed dense X with sparse output"):
            convert_adata(adata, output=str(out), cfg=AppConfig(io=IOConfig(x_storage="csr")))
        assert not out.exists()

        out_dense = tmp_path / "out_dense.zarr"
        convert_adata(adata, output=str(out_dense), cfg=AppConfig(io=IOConfig(x_storage="dense")))
        np.testing.assert_array_equal(zarr.open(str(out_dense), mode="r")["X"][:], dense)
    finally:
        adata.file.close()


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


def test_h5ad_backed_csc_x_and_csc_layer_streamed_to_csr_matches_eager_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Backed CSC X targeted at csr output (item 5): convert no longer eagerly materialises
    it via .tocsr() first (anndata's backed _CSCDataset has no .tocsr() anyway) — write_matrix
    streams it through write_transposed_sparse instead, matching the eager conversion
    byte-for-byte. Also covers the CSC-source X-normalization warning: write_adata_to_store
    logs it unconditionally whenever adata.X is CSC, whether or not it's ultimately streamed
    through write_transposed_sparse (backed) rather than converted in memory (eager). A backed
    CSC *layer* (X stays CSR here) streams through the same write_transposed_sparse CSC->CSR
    direction without ever triggering that X-specific warning -- write_matrix's generic backed
    dispatch handles the layer, distinct from convert's own X-specific CSC-to-CSR normalisation."""
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
    assert any("adata.X was CSC and has been converted to CSR in memory" in m for m in _messages(caplog))

    _assert_byte_identical(out_eager, out_backed)
    got = ad.read_zarr(str(out_backed))
    assert sp.isspmatrix_csr(got.X)
    np.testing.assert_allclose(got.X.toarray(), dense.astype(np.float32))

    caplog.clear()
    h5_mixed = tmp_path / "mixed.h5ad"
    mixed_dense = _make_mixed_format_h5ad(h5_mixed)
    out_mixed = tmp_path / "mixed_out.zarr"
    convert_h5ad(str(h5_mixed), output=str(out_mixed), cfg=_cfg_backed("csr"))
    got_mixed = ad.read_zarr(str(out_mixed))
    assert sp.isspmatrix_csr(got_mixed.layers["counts"])
    np.testing.assert_allclose(got_mixed.layers["counts"].toarray(), mixed_dense * 2)
    np.testing.assert_allclose(got_mixed.X.toarray(), mixed_dense)
    assert not any("was CSC and has been converted to CSR" in m for m in _messages(caplog))
