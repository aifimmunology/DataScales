from __future__ import annotations

from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

from zarrsmith import (
    ConversionError,
    add_expr_layer,
    append_cells,
    convert_h5ad_to_zarr,
    rechunk_store,
    sort_store,
)
from zarrsmith.config import AppConfig, ChunkConfig, GroupingConfig, IOConfig


def _cfg(**io):
    return AppConfig(
        io=IOConfig(**io),
        chunks=ChunkConfig(x_row_chunk=16, x_col_chunk=4, sparse_flat_chunk=64),
    )


def _adata(n=40, v=6, seed=0):
    rng = np.random.default_rng(seed)
    x = sp.random(n, v, density=0.5, format="csr", dtype=np.float32, random_state=seed)
    x.data = np.abs(x.data) + 1.0
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["b", "a", "c"], n), categories=["a", "b", "c"])},
        index=[f"c{seed}_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(v)])
    return ad.AnnData(X=x, obs=obs, var=var, obsm={"X_umap": rng.random((n, 2)).astype(np.float32)})


def _store(tmp_path, adata, name="store.zarr", cfg=None):
    h5 = tmp_path / f"{name}.h5ad"
    adata.write_h5ad(h5)
    out = tmp_path / name
    convert_h5ad_to_zarr(str(h5), str(out), cfg or _cfg())
    return out


def _expected_gexp(x, target_sum=1e4):
    x = sp.csr_matrix(x, dtype=np.float64)
    sums = np.asarray(x.sum(axis=1)).ravel()
    sf = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
    out = x.multiply(sf[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out.astype(np.float32).toarray()


# ── add-expr ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt,enc", [("csc", "csc_matrix"), ("csr", "csr_matrix")])
def test_add_expr_sparse(tmp_path, fmt, enc):
    adata = _adata()
    out = _store(tmp_path, adata)
    add_expr_layer(str(out), _cfg(), fmt=fmt, chunk_elems=32)
    g = zarr.open_group(str(out), mode="r")
    assert g["layers/gexp"].attrs["encoding-type"] == enc
    got = ad.read_zarr(str(out))
    np.testing.assert_allclose(
        got.layers["gexp"].toarray(), _expected_gexp(adata.X), rtol=1e-5
    )
    np.testing.assert_allclose(got.X.toarray(), adata.X.toarray())


def test_add_expr_dense(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata)
    add_expr_layer(str(out), _cfg(), fmt="dense", chunk_elems=80)
    arr = zarr.open_group(str(out), mode="r")["layers/gexp"]
    assert arr.chunks == (adata.n_obs, 2)
    got = ad.read_zarr(str(out))
    np.testing.assert_allclose(np.asarray(got.layers["gexp"]), _expected_gexp(adata.X), rtol=1e-5)


def test_add_expr_existing_layer(tmp_path):
    out = _store(tmp_path, _adata())
    add_expr_layer(str(out), _cfg(), fmt="csc")
    with pytest.raises(ConversionError, match="already exists"):
        add_expr_layer(str(out), _cfg(), fmt="csc")
    add_expr_layer(str(out), _cfg(overwrite=True), fmt="dense")
    assert isinstance(zarr.open_group(str(out), mode="r")["layers/gexp"], zarr.Array)


# ── rechunk ──────────────────────────────────────────────────────────────────

def test_rechunk_sparse(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata)
    out2 = tmp_path / "rechunked.zarr"
    rechunk_store(str(out), str(out2), AppConfig(chunks=ChunkConfig(sparse_flat_chunk=16)))
    g = zarr.open_group(str(out2), mode="r")
    assert g["X/data"].chunks == (16,)
    got = ad.read_zarr(str(out2))
    np.testing.assert_allclose(got.X.toarray(), adata.X.toarray())
    assert list(got.obs["cell_type"]) == list(adata.obs["cell_type"])
    np.testing.assert_allclose(got.obsm["X_umap"], adata.obsm["X_umap"])


def test_rechunk_dense(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata, cfg=_cfg(x_storage="dense"))
    out2 = tmp_path / "rechunked.zarr"
    rechunk_store(str(out), str(out2), AppConfig(chunks=ChunkConfig(x_row_chunk=8, x_col_chunk=3, cpus=2)))
    g = zarr.open_group(str(out2), mode="r")
    assert g["X"].chunks == (8, 3)
    got = ad.read_zarr(str(out2))
    np.testing.assert_allclose(np.asarray(got.X), adata.X.toarray())


def test_rechunk_copies_layers(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata)
    add_expr_layer(str(out), _cfg(), fmt="csc", chunk_elems=32)
    out2 = tmp_path / "rechunked.zarr"
    rechunk_store(str(out), str(out2), AppConfig(chunks=ChunkConfig(sparse_flat_chunk=16)))
    g = zarr.open_group(str(out2), mode="r")
    assert g["layers/gexp/data"].chunks == (32,)  # non-target layer keeps its chunks
    got = ad.read_zarr(str(out2))
    np.testing.assert_allclose(got.layers["gexp"].toarray(), _expected_gexp(adata.X), rtol=1e-5)


# ── sort ─────────────────────────────────────────────────────────────────────

def test_sort_store(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata)
    out2 = tmp_path / "sorted.zarr"
    cfg = replace(_cfg(), grouping=GroupingConfig(enabled=True, sort_by=("cell_type",)))
    sort_store(str(out), str(out2), cfg)
    got = ad.read_zarr(str(out2))
    codes = got.obs["cell_type"].cat.codes.to_numpy()
    assert (np.diff(codes) >= 0).all()
    orig_x = {n: adata.X[i].toarray().ravel() for i, n in enumerate(adata.obs_names)}
    orig_um = {n: adata.obsm["X_umap"][i] for i, n in enumerate(adata.obs_names)}
    for i, n in enumerate(got.obs_names):
        np.testing.assert_allclose(got.X[i].toarray().ravel(), orig_x[n])
        np.testing.assert_allclose(got.obsm["X_umap"][i], orig_um[n])


def test_sort_store_requires_by(tmp_path):
    out = _store(tmp_path, _adata())
    with pytest.raises(ConversionError, match="--by"):
        sort_store(str(out), str(tmp_path / "s.zarr"), _cfg())


# ── append ───────────────────────────────────────────────────────────────────

def test_append(tmp_path):
    a, b = _adata(n=40, seed=0), _adata(n=15, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    append_cells(str(sa), str(sb), _cfg())
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 55
    np.testing.assert_allclose(got.X.toarray(), sp.vstack([a.X, b.X]).toarray())
    assert list(got.obs_names) == list(a.obs_names) + list(b.obs_names)
    assert list(got.obs["cell_type"]) == list(a.obs["cell_type"]) + list(b.obs["cell_type"])
    np.testing.assert_allclose(
        got.obsm["X_umap"], np.vstack([a.obsm["X_umap"], b.obsm["X_umap"]])
    )


def test_append_guards(tmp_path):
    sa = _store(tmp_path, _adata(n=20, seed=0), "a.zarr")
    sb_bad = _store(tmp_path, _adata(n=10, v=5, seed=1), "bad.zarr")
    with pytest.raises(ConversionError, match="var mismatch"):
        append_cells(str(sa), str(sb_bad), _cfg())

    a2 = _adata(n=20, seed=2)
    a2.obsp["conn"] = sp.eye(20, format="csr")
    sa2 = _store(tmp_path, a2, "a2.zarr")
    sb = _store(tmp_path, _adata(n=10, seed=3), "b.zarr")
    with pytest.raises(ConversionError, match="obsp"):
        append_cells(str(sa2), str(sb), _cfg())
    append_cells(str(sa2), str(sb), _cfg(), drop_obsp=True)
    assert ad.read_zarr(str(sa2)).n_obs == 30


def test_append_refresh_expr(tmp_path):
    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr_layer(str(sa), _cfg(), fmt="csc", chunk_elems=32)
    with pytest.raises(ConversionError, match="stale"):
        append_cells(str(sa), str(sb), _cfg())
    append_cells(str(sa), str(sb), _cfg(), refresh_expr=True)
    got = ad.read_zarr(str(sa))
    np.testing.assert_allclose(
        got.layers["gexp"].toarray(),
        _expected_gexp(sp.vstack([a.X, b.X])),
        rtol=1e-5,
    )
