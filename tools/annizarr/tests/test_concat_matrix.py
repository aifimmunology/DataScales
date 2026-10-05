from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.ops import convert

N_OBS_A, N_OBS_B, N_VARS = 12, 9, 6
VAR_NAMES = [f"GENE{i}" for i in range(N_VARS)]

_ENCODING = {"dense": "array", "csr": "csr_matrix", "csc": "csc_matrix"}


def _block(seed: int, n_obs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dense = (rng.random((n_obs, N_VARS), dtype=np.float32) < 0.4) * rng.random((n_obs, N_VARS), dtype=np.float32)
    return dense.astype(np.float32)


def _write_h5ad(path: Path, dense: np.ndarray, fmt: str, *, obs_prefix: str) -> None:
    x: Any = {"dense": dense.copy(), "csr": sp.csr_matrix(dense), "csc": sp.csc_matrix(dense)}[fmt]
    obs = {"obs_names": [f"{obs_prefix}{i}" for i in range(dense.shape[0])]}
    var = {"var_names": VAR_NAMES}
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(path)


def _write_10x_v3(path: Path, dense: np.ndarray) -> None:
    """A minimal Cell Ranger v3 HDF5 (genes x barcodes on disk, Gene Expression only)."""
    csc_t = sp.csc_matrix(dense.T)  # on-disk shape is (n_features, n_barcodes)
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([f"TENX{i}-1".encode() for i in range(dense.shape[0])]))
        m.create_dataset("data", data=csc_t.data.astype(np.float32))
        m.create_dataset("indices", data=csc_t.indices.astype(np.int32))
        m.create_dataset("indptr", data=csc_t.indptr.astype(np.int32))
        m.create_dataset("shape", data=np.array([N_VARS, dense.shape[0]], dtype=np.int32))
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array([f"ENSG{i:03d}".encode() for i in range(N_VARS)]))
        feat.create_dataset("name", data=np.array([v.encode() for v in VAR_NAMES]))
        feat.create_dataset("feature_type", data=np.array([b"Gene Expression"] * N_VARS))


@pytest.mark.parametrize("x_storage", ["dense", "csr", "csc"])
@pytest.mark.parametrize(
    "mix",
    ["dense_csr", "csr_csc", "dense_csr_lazy", "dense_csr_eager", "tenx_h5ad"],
)
def test_concat_matrix_grid(tmp_path: Path, mix: str, x_storage: str) -> None:
    a_dense = _block(1, N_OBS_A)
    b_dense = _block(2, N_OBS_B)
    a_path, b_path = tmp_path / "a.in", tmp_path / "b.in"
    cfg = AppConfig(
        io=IOConfig(x_storage=x_storage), chunks=ChunkConfig(x_row_chunk=4, x_col_chunk=3, sparse_flat_chunk=5)
    )

    if mix == "dense_csr":
        _write_h5ad(a_path, a_dense, "dense", obs_prefix="a")
        _write_h5ad(b_path, b_dense, "csr", obs_prefix="b")
    elif mix == "csr_csc":
        _write_h5ad(a_path, a_dense, "csr", obs_prefix="a")
        _write_h5ad(b_path, b_dense, "csc", obs_prefix="b")
    elif mix == "dense_csr_lazy":
        # explicit lazy=True (no more auto-select): both inputs stream band-by-band.
        _write_h5ad(a_path, a_dense, "dense", obs_prefix="a")
        _write_h5ad(b_path, b_dense, "csr", obs_prefix="b")
        cfg = AppConfig(io=IOConfig(x_storage=x_storage, lazy=True), chunks=cfg.chunks)
    elif mix == "dense_csr_eager":
        # explicit lazy=False: both inputs load whole into memory first.
        _write_h5ad(a_path, a_dense, "dense", obs_prefix="a")
        _write_h5ad(b_path, b_dense, "csr", obs_prefix="b")
        cfg = AppConfig(io=IOConfig(x_storage=x_storage, lazy=False), chunks=cfg.chunks)
    else:  # tenx_h5ad
        _write_10x_v3(a_path, a_dense)
        _write_h5ad(b_path, b_dense, "csr", obs_prefix="b")

    out = tmp_path / "out.zarr"
    result = convert([str(a_path), str(b_path)], output=str(out), cfg=cfg)
    assert result.n_obs == N_OBS_A + N_OBS_B
    assert result.n_vars == N_VARS

    got = ad.read_zarr(str(out))
    got_dense = np.asarray(got.X.todense() if sp.issparse(got.X) else got.X)
    np.testing.assert_array_equal(got_dense, np.vstack([a_dense, b_dense]))
    assert list(got.var_names) == VAR_NAMES
    if mix != "tenx_h5ad":
        assert list(got.obs_names) == [f"a{i}" for i in range(N_OBS_A)] + [f"b{i}" for i in range(N_OBS_B)]

    root = zarr.open_group(str(out), mode="r")
    assert root["X"].attrs["encoding-type"] == _ENCODING[x_storage]
