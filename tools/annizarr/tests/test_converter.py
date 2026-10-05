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

from annizarr.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from annizarr.errors import ConversionError, StorageError
from annizarr.ops import convert_10x_h5, convert_adata, convert_h5ad

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# The core (input format x lazy x x_storage) dispatch grid lives in test_convert_matrix.py,
# via the shared Reader abstraction every input now goes through. What's left here is
# behaviour outside that grid: the 10x loader, the existing-target fast-fail + flat-chunk
# sizing, convert_adata's own lazy-sync, and the lazy-CSC temp-dir cleanup + the --lazy ->
# backed="r" regression guard.


def _cfg(x_storage: str, sparse_flat_chunk: int = 2048) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=False, consolidate_metadata=False, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=2, x_col_chunk=2, sparse_flat_chunk=sparse_flat_chunk),
        validation=ValidationConfig(),
    )


def _cfg_lazy(x_storage: str) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=False, consolidate_metadata=False, x_storage=x_storage, lazy=True),
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


# ---------------------------------------------------------------------------
# In-memory loading (10x h5 → zarr)
# ---------------------------------------------------------------------------


def test_10x_csr_output(tmp_path: Path) -> None:
    """scanpy.read_10x_h5 always returns an in-memory CSR matrix — this just confirms
    convert_10x_h5 dispatches into the normal writer grid, plus the wrong-extension guard
    on the 10x-specific loader."""
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
# convert_adata: in-memory entry point + its own lazy-sync
# ---------------------------------------------------------------------------


def test_convert_adata_in_memory(tmp_path: Path) -> None:
    """Public library entry: an in-memory AnnData converts like convert_h5ad."""
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]))
    adata = ad.AnnData(X=X)
    out = tmp_path / "out.zarr"
    convert_adata(adata, output=str(out), cfg=_cfg("csr"))
    got = ad.read_zarr(str(out))
    np.testing.assert_array_equal(got.X.toarray(), X.toarray())

    # cfg.io.lazy is synced from adata.isbacked, not taken at face value: a genuinely
    # in-memory AnnData converts fine even if the caller's cfg says lazy=True.
    out_b = tmp_path / "b.zarr"
    convert_adata(adata, output=str(out_b), cfg=_cfg_lazy("csr"))
    assert ad.read_zarr(str(out_b)).n_obs == adata.n_obs


def test_convert_adata_accepts_a_backed_anndata(tmp_path: Path) -> None:
    """convert_adata also takes an already-backed AnnData (e.g. ad.read_h5ad(path,
    backed='r')), syncing cfg.io.lazy from it; dense output streams without materialising X."""
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    h5 = tmp_path / "in.h5ad"
    ad.AnnData(X=dense.copy()).write_h5ad(h5)
    adata = ad.read_h5ad(h5, backed="r")
    try:
        out_dense = tmp_path / "out_dense.zarr"
        convert_adata(adata, output=str(out_dense), cfg=AppConfig(io=IOConfig(x_storage="dense")))
        np.testing.assert_array_equal(zarr.open(str(out_dense), mode="r")["X"][:], dense)
    finally:
        adata.file.close()


# ---------------------------------------------------------------------------
# Lazy CSC: temp-dir cleanup + the --lazy -> backed="r" regression guard
# ---------------------------------------------------------------------------


def _make_300x200_h5ad(path: Path, *, x_csc: bool) -> None:
    """A 300x200 h5ad; X is CSC when ``x_csc``, else CSR with a CSC 'cnt' layer either way —
    both exercise the lazy-CSC-to-CSR transpose (anndata's backed _CSCDataset has no .tocsr())."""
    rng = np.random.default_rng(7)
    dense = rng.random((300, 200)).astype(np.float32)
    dense[dense < 0.7] = 0.0  # ~30% density
    X = sp.csc_matrix(dense) if x_csc else sp.csr_matrix(dense)
    adata = ad.AnnData(X=X)
    adata.layers["cnt"] = sp.csc_matrix(dense * 2.0)
    adata.write_h5ad(path)


def test_h5ad_lazy_csc_to_dense_matches_eager_no_tmp_dir_left(tmp_path: Path) -> None:
    """A lazy CSC source (X, or the 'cnt' layer -- both directions checked) converted with
    x_storage='dense' streams through a temporary CSC->CSR transpose (never materialising the
    whole matrix) and matches the eager (in-memory) conversion byte-for-byte; the temp dir is
    cleaned up either way. Also the only regression guard that --lazy actually causes a
    backed="r" load (an output-format-only assertion wouldn't catch a silently-ignored flag,
    since a small fixture converts identically either way)."""
    # chunks sized for the 300x200 fixture (not a 2x2 default, which would tile it into
    # thousands of tiny writes)
    chunks = ChunkConfig(x_row_chunk=64, x_col_chunk=64)
    for x_csc in (False, True):
        cfg_eager = AppConfig(io=IOConfig(x_storage="dense", lazy=False), chunks=chunks, validation=ValidationConfig())
        cfg_lazy = replace(cfg_eager, io=replace(cfg_eager.io, lazy=True))

        input_h5 = tmp_path / f"input_{x_csc}.h5ad"
        _make_300x200_h5ad(input_h5, x_csc=x_csc)
        out_eager = tmp_path / f"eager_{x_csc}.zarr"
        out_lazy = tmp_path / f"lazy_{x_csc}.zarr"
        convert_h5ad(str(input_h5), output=str(out_eager), cfg=cfg_eager)
        with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
            convert_h5ad(str(input_h5), output=str(out_lazy), cfg=cfg_lazy)
        assert mocked.call_args_list[0].kwargs.get("backed") == "r"

        got_eager = ad.read_zarr(str(out_eager))
        got_lazy = ad.read_zarr(str(out_lazy))
        np.testing.assert_array_equal(np.asarray(got_lazy.X), np.asarray(got_eager.X))
        np.testing.assert_array_equal(np.asarray(got_lazy.layers["cnt"]), np.asarray(got_eager.layers["cnt"]))
    assert not list(tmp_path.rglob("annizarr_csc2csr_*"))
