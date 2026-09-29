from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

# `from conftest import ...` is unsafe: pytest's default import mode gives every conftest.py
# the bare module name "conftest", and tests/ic/conftest.py collides with tests/conftest.py
# depending on collection order — so the shared helper lives in its own uniquely-named module.
from _readable import assert_anndata_readable
from annizarr._ops import add_expr, append, convert_h5ad, rechunk, sort
from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.errors import ConversionError, StorageError


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
    convert_h5ad(str(h5), output=str(out), cfg=cfg or _cfg())
    assert_anndata_readable(out)
    return out


def _expected_gexp(x, target_sum=1e4):
    x = sp.csr_matrix(x, dtype=np.float64)
    sums = np.asarray(x.sum(axis=1)).ravel()
    sf = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
    out = x.multiply(sf[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out.astype(np.float32).toarray()


def _messages(caplog):
    return [r.message for r in caplog.records]


# ── add-expr ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_add_expr_all_formats(tmp_path, fmt):
    adata = _adata()
    out = _store(tmp_path, adata)
    add_expr(str(out), fmt=fmt, chunk_elems=80, cfg=_cfg())
    assert_anndata_readable(out)
    got = ad.read_zarr(str(out))
    if fmt == "dense":
        g = zarr.open_group(str(out), mode="r")["layers/gexp"]
        assert g.chunks == (adata.n_obs, 2)
        vals = np.asarray(got.layers["gexp"])
    else:
        enc = {"csr": "csr_matrix", "csc": "csc_matrix"}[fmt]
        g = zarr.open_group(str(out), mode="r")
        assert g["layers/gexp"].attrs["encoding-type"] == enc
        vals = got.layers["gexp"].toarray()
    np.testing.assert_allclose(vals, _expected_gexp(adata.X), rtol=1e-5)
    np.testing.assert_allclose(got.X.toarray(), adata.X.toarray())

    if fmt == "csc":
        # csc's per-row factors are computed up front in the same row_step bands the csr
        # path's fused per-band transform uses (_ops/_expr.py's "bit-identical factors"
        # comment) — verify that claim directly against a csr layer on the same store,
        # not just both against the separately-computed _expected_gexp reference above.
        # toarray() already places values at their canonical column position per row, so
        # sorting is a no-op safety net here, not load-bearing.
        add_expr(str(out), fmt="csr", layer="gexp_csr_ref", chunk_elems=80, cfg=_cfg())
        csr_vals = ad.read_zarr(str(out)).layers["gexp_csr_ref"].toarray()
        assert np.array_equal(np.sort(vals, axis=1), np.sort(csr_vals, axis=1))


def test_add_expr_existing_layer(tmp_path):
    out = _store(tmp_path, _adata())
    add_expr(str(out), fmt="csc", cfg=_cfg())
    with pytest.raises(ConversionError, match="already exists"):
        add_expr(str(out), fmt="csc", cfg=_cfg())
    add_expr(str(out), fmt="dense", overwrite=True, cfg=_cfg())
    assert isinstance(zarr.open_group(str(out), mode="r")["layers/gexp"], zarr.Array)


# ── rechunk ──────────────────────────────────────────────────────────────────


def test_rechunk_sparse(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata)
    out2 = tmp_path / "rechunked.zarr"
    rechunk(str(out), output=str(out2), cfg=AppConfig(chunks=ChunkConfig(sparse_flat_chunk=16)))
    assert_anndata_readable(out2)
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
    rechunk(str(out), output=str(out2), cfg=AppConfig(chunks=ChunkConfig(x_row_chunk=8, x_col_chunk=3, cpus=2)))
    g = zarr.open_group(str(out2), mode="r")
    assert g["X"].chunks == (8, 3)
    got = ad.read_zarr(str(out2))
    np.testing.assert_allclose(np.asarray(got.X), adata.X.toarray())


@pytest.mark.parametrize("shard_factor", [1, 2])
def test_rechunk_dense_shard_factor(tmp_path, shard_factor):
    adata = _adata(n=64, v=8)
    out = _store(tmp_path, adata, cfg=_cfg(x_storage="dense"))
    out2 = tmp_path / "rechunked.zarr"
    rechunk(
        str(out),
        output=str(out2),
        cfg=AppConfig(chunks=ChunkConfig(x_row_chunk=8, x_col_chunk=4, x_shard_factor=shard_factor)),
    )
    g = zarr.open_group(str(out2), mode="r")
    assert g["X"].chunks == (8, 4)
    assert (g["X"].shards is None) == (shard_factor == 1)
    got = ad.read_zarr(str(out2))
    np.testing.assert_allclose(np.asarray(got.X), adata.X.toarray())


def test_rechunk_copies_layers(tmp_path):
    adata = _adata()
    out = _store(tmp_path, adata)
    add_expr(str(out), fmt="csc", chunk_elems=32, cfg=_cfg())
    out2 = tmp_path / "rechunked.zarr"
    rechunk(str(out), output=str(out2), cfg=AppConfig(chunks=ChunkConfig(sparse_flat_chunk=16)))
    g = zarr.open_group(str(out2), mode="r")
    assert g["layers/gexp/data"].chunks == (32,)  # non-target layer keeps its chunks
    assert g["layers/gexp/data"].shards is None  # copy-as-is preserves "unsharded" too
    got = ad.read_zarr(str(out2))
    np.testing.assert_allclose(got.layers["gexp"].toarray(), _expected_gexp(adata.X), rtol=1e-5)


# ── sort ─────────────────────────────────────────────────────────────────────


def test_sort_store(tmp_path, caplog):
    adata = _adata()
    out = _store(tmp_path, adata)
    out2 = tmp_path / "sorted.zarr"
    with caplog.at_level("INFO", logger="annizarr"):
        sort(str(out), output=str(out2), by=("cell_type",), cfg=_cfg())
    assert any("contiguous groups via backed streamed bucketing" in m for m in _messages(caplog))
    assert_anndata_readable(out2)
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
    with pytest.raises(ConversionError, match="by="):
        sort(str(out), output=str(tmp_path / "s.zarr"), by=(), cfg=_cfg())


# ── append ───────────────────────────────────────────────────────────────────


def test_append(tmp_path, caplog):
    a, b = _adata(n=40, seed=0), _adata(n=15, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())
    assert any("obsm" in m and "dropped" in m for m in _messages(caplog))
    assert_anndata_readable(sa)
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 55
    np.testing.assert_allclose(got.X.toarray(), sp.vstack([a.X, b.X]).toarray())
    assert list(got.obs_names) == list(a.obs_names) + list(b.obs_names)
    assert list(got.obs["cell_type"]) == list(a.obs["cell_type"]) + list(b.obs["cell_type"])
    assert len(got.obsm) == 0  # embeddings are invalidated, not extended


def test_append_guards(tmp_path):
    sa = _store(tmp_path, _adata(n=20, seed=0), "a.zarr")
    sb_bad = _store(tmp_path, _adata(n=10, v=5, seed=1), "bad.zarr")
    with pytest.raises(ConversionError, match="var mismatch"):
        append(str(sa), cells=str(sb_bad), cfg=_cfg())

    a2 = _adata(n=20, seed=2)
    a2.obsp["conn"] = sp.eye(20, format="csr")
    sa2 = _store(tmp_path, a2, "a2.zarr")
    sb = _store(tmp_path, _adata(n=10, seed=3), "b.zarr")
    with pytest.raises(ConversionError, match="obsp"):
        append(str(sa2), cells=str(sb), cfg=_cfg())
    append(str(sa2), cells=str(sb), drop_derived=True, cfg=_cfg())
    got = ad.read_zarr(str(sa2))
    assert got.n_obs == 30 and len(got.obsp) == 0


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_add_expr_empty_rows_and_genes(tmp_path, fmt):
    # zero-count cells at band boundaries (incl. the last row) once truncated the
    # previous row's sum; an all-zero gene exercises empty csc columns
    x = np.zeros((10, 5), dtype=np.float32)
    x[1:9, [0, 1, 3, 4]] = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
    adata = ad.AnnData(
        X=sp.csr_matrix(x),
        obs=pd.DataFrame({"cell_type": pd.Categorical(["a"] * 10)}, index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    out = _store(tmp_path, adata)
    add_expr(str(out), fmt=fmt, chunk_elems=16, cfg=_cfg())
    got = ad.read_zarr(str(out))
    vals = got.layers["gexp"]
    vals = np.asarray(vals) if fmt == "dense" else vals.toarray()
    np.testing.assert_allclose(vals, _expected_gexp(adata.X), rtol=1e-5)
    assert not vals[0].any() and not vals[-1].any() and not vals[:, 2].any()


def test_ops_on_consolidated_store(tmp_path):
    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr", cfg=_cfg(consolidate_metadata=True))
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csc", chunk_elems=32, cfg=_cfg())
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())
    add_expr(str(sa), fmt="csc", chunk_elems=32, cfg=_cfg())
    got = ad.read_zarr(str(sa))  # reads via the re-consolidated metadata
    assert got.n_obs == 42 and "gexp" in got.layers


def test_append_failed_validation_mutates_nothing(tmp_path):
    a = _adata(n=20, seed=0)
    a.obsp["conn"] = sp.eye(20, format="csr")
    sa = _store(tmp_path, a, "a.zarr")
    sb_bad = _store(tmp_path, _adata(n=10, v=5, seed=1), "bad.zarr")
    with pytest.raises(ConversionError, match="var mismatch"):
        append(str(sa), cells=str(sb_bad), drop_derived=True, cfg=_cfg())
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 20 and "conn" in got.obsp


def test_append_categorical_order_mismatch(tmp_path):
    a, b = _adata(n=20, seed=0), _adata(n=10, seed=1)
    b.obs["cell_type"] = pd.Categorical(b.obs["cell_type"], categories=["a", "b", "c"], ordered=True)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    with pytest.raises(ConversionError, match="categorical dtype mismatch"):
        append(str(sa), cells=str(sb), cfg=_cfg())


def test_append_obs_column_types(tmp_path):
    # exercises every per-column append path: categorical (from _adata), plain numeric,
    # string-array, and nullable-integer with an NA
    def build(n, seed):
        adata = _adata(n=n, seed=seed)
        rng = np.random.default_rng(seed + 100)
        adata.obs["n_genes"] = rng.integers(0, 1000, n)
        adata.obs["doublet_score"] = rng.random(n)
        # unique per row so anndata keeps it a string-array (repetitive strings
        # become categorical on write, with per-store categories)
        adata.obs["sample"] = [f"s{seed}_{i}" for i in range(n)]
        qc = pd.array(rng.integers(0, 5, n), dtype="Int64")
        qc[0] = pd.NA
        adata.obs["qc_flag"] = qc
        return adata

    a, b = build(25, 0), build(11, 1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())
    got = ad.read_zarr(str(sa))
    expected = pd.concat([a.obs, b.obs], axis=0)
    pd.testing.assert_frame_equal(got.obs, expected, check_dtype=False)


def test_append_obs_dtype_mismatch(tmp_path):
    a, b = _adata(n=20, seed=0), _adata(n=10, seed=1)
    a.obs["n_genes"] = np.arange(20, dtype=np.int64)
    b.obs["n_genes"] = np.arange(10, dtype=np.float64)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    # obs columns extend in place now, so dtypes must match exactly (no silent
    # pandas-concat unification)
    with pytest.raises(ConversionError, match="dtype mismatch"):
        append(str(sa), cells=str(sb), cfg=_cfg())


def test_append_all_duplicate_names_raises(tmp_path):
    # `b` has the identical obs index as `a` (same n, same seed) — every appended cell is
    # already present, so this looks like a re-run of the same append and is a hard error.
    sa = _store(tmp_path, _adata(n=20, seed=0), "a.zarr")
    sb = _store(tmp_path, _adata(n=20, seed=0), "b.zarr")
    with pytest.raises(ConversionError, match="already"):
        append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())


def test_append_partial_duplicate_names_warns(tmp_path, caplog):
    # only 2 of `b`'s 10 obs names collide with `a` — barcodes legitimately collide across
    # samples, so this stays a warning, not a raise.
    a, b = _adata(n=20, seed=0), _adata(n=10, seed=1)
    b.obs_names = [a.obs_names[0], a.obs_names[1], *b.obs_names[2:]]
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())
    assert any("2 duplicate" in m for m in _messages(caplog))


def test_append_cells_store_internal_duplicate_also_in_target_raises(tmp_path):
    # `b` has 2 rows sharing the SAME obs name, and that name is already in `a` — every
    # appended row is a duplicate (by target-collision, counted once each, not twice for
    # also repeating each other). Regression test for double-counting a position that is
    # both in-target and an internal repeat, which used to let n_duplicate_names overshoot
    # n_new and silently skip the "already appended" error.
    a = _adata(n=20, seed=0)
    sa = _store(tmp_path, a, "a.zarr")
    b = _adata(n=2, seed=1)
    b.obs_names = [a.obs_names[0], a.obs_names[0]]
    sb = _store(tmp_path, b, "b.zarr")
    with pytest.raises(ConversionError, match="already"):
        append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())


def test_append_three_rows_two_duplicate_names_warns(tmp_path, caplog):
    # 3 rows in `b`, only 2 of which collide with `a` — not every appended cell is already
    # present, so this stays a warning.
    a, b = _adata(n=20, seed=0), _adata(n=3, seed=1)
    b.obs_names = [a.obs_names[0], a.obs_names[1], b.obs_names[2]]
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())
    assert any("2 duplicate" in m for m in _messages(caplog))


def test_append_drop_derived(tmp_path, caplog):
    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csc", chunk_elems=32, cfg=_cfg())
    # append never touches derived elements itself: obsm/obsp/layers must be explicitly
    # dropped; layers are then re-derived with add-expr
    with pytest.raises(ConversionError, match="layers"):
        append(str(sa), cells=str(sb), cfg=_cfg())
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())
    assert any("add-expr" in m for m in _messages(caplog))
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 42 and len(got.layers) == 0 and len(got.obsm) == 0
    add_expr(str(sa), fmt="csc", chunk_elems=32, target_sum=1e6, cfg=_cfg())
    got = ad.read_zarr(str(sa))
    np.testing.assert_allclose(
        got.layers["gexp"].toarray(),
        _expected_gexp(sp.vstack([a.X, b.X]), target_sum=1e6),
        rtol=1e-5,
    )


def test_append_extend_layers(tmp_path, caplog):
    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csr", chunk_elems=32, target_sum=1e6, cfg=_cfg())
    append(str(sa), cells=str(sb), drop_derived=True, extend_layers=True, cfg=_cfg())
    assert any("extended layers ['gexp']" in m for m in _messages(caplog))
    g = zarr.open_group(str(sa), mode="r")["layers/gexp"]
    assert g.attrs["annizarr_target_sum"] == 1e6
    assert list(g.attrs["shape"]) == [42, 6]
    got = ad.read_zarr(str(sa))
    np.testing.assert_allclose(
        got.layers["gexp"].toarray(),
        _expected_gexp(sp.vstack([a.X, b.X]), target_sum=1e6),
        rtol=1e-5,
    )


def test_append_extend_layers_ineligible(tmp_path, caplog):
    # csc layers cannot extend in place (column-major); with the flag they drop as today
    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csc", chunk_elems=32, cfg=_cfg())
    append(str(sa), cells=str(sb), drop_derived=True, extend_layers=True, cfg=_cfg())
    assert any("dropped" in m and "gexp" in m for m in _messages(caplog))
    assert "gexp" not in ad.read_zarr(str(sa)).layers


def test_append_extend_layers_sparsity_mismatch(tmp_path, caplog):
    # a marked layer whose indptr no longer matches X falls back to being dropped
    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csr", chunk_elems=32, cfg=_cfg())
    ip_arr = zarr.open_group(str(sa), mode="r+")["layers/gexp/indptr"]
    ip_arr[1] = int(ip_arr[1]) + 1
    append(str(sa), cells=str(sb), drop_derived=True, extend_layers=True, cfg=_cfg())
    assert any("sparsity differs from X" in m for m in _messages(caplog))
    assert "gexp" not in ad.read_zarr(str(sa)).layers


def test_plan_append_reports_extendable_and_drops(tmp_path):
    from annizarr._ops import plan_append

    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csr", chunk_elems=32, cfg=_cfg())

    plan = plan_append(str(sa), cells=str(sb))
    assert plan.n_new == 12
    assert plan.extendable_layers == ("gexp",)
    assert plan.drop_layers == ()
    assert plan.drop_obsm == ("X_umap",)
    assert plan.n_duplicate_names == 0

    assert plan.drops() == ("obsm/X_umap", "layers/gexp")
    assert plan.drops(extend_layers=True) == ("obsm/X_umap",)  # gexp extended, not dropped


def test_gexp_legacy_zarrsmith_target_sum_key_still_recognised(tmp_path):
    """A gexp layer written by pre-merge zarrsmith (attr key zarrsmith_target_sum, not
    annizarr_target_sum) is still eligible for append --extend-layers and re-derivable."""
    from annizarr._ops import plan_append
    from annizarr._ops._expr import introspect_gexp

    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csr", chunk_elems=32, target_sum=1e6, cfg=_cfg())

    g = zarr.open_group(str(sa), mode="r+")["layers/gexp"]
    g.attrs["zarrsmith_target_sum"] = g.attrs.pop("annizarr_target_sum")

    plan = plan_append(str(sa), cells=str(sb))
    assert plan.extendable_layers == ("gexp",)  # recognised despite the legacy attr key

    fmt, chunk_elems, target_sum = introspect_gexp(zarr.open_group(str(sa), mode="r")["layers/gexp"])
    assert (fmt, chunk_elems, target_sum) == ("csr", 32, 1e6)


def test_add_expr_multiband(tmp_path, monkeypatch):
    # tiny band budget → many column bands + multiple row batches, exercising the
    # bucket cursors and band-edge math the default 256 MB budget never hits in tests
    monkeypatch.setattr("annizarr._layout.BATCH_BYTES", 600)
    adata = _adata(n=1200, v=12, seed=4)
    for fmt in ("csc", "dense"):
        out = _store(tmp_path, adata, f"mb-{fmt}.zarr")
        add_expr(str(out), fmt=fmt, chunk_elems=80, cfg=_cfg())
        got = ad.read_zarr(str(out))
        vals = got.layers["gexp"]
        vals = np.asarray(vals) if fmt == "dense" else vals.toarray()
        np.testing.assert_allclose(vals, _expected_gexp(adata.X), rtol=1e-5)


def test_lifecycle_plain(tmp_path, caplog):
    a, b = _adata(n=40, seed=0), _adata(n=15, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")

    sorted1 = tmp_path / "sorted1.zarr"
    sort(str(sa), output=str(sorted1), by=("cell_type",), cfg=_cfg())
    append(str(sorted1), cells=str(sb), drop_derived=True, cfg=_cfg())
    add_expr(str(sorted1), fmt="csc", chunk_elems=32, cfg=_cfg())
    sorted2 = tmp_path / "sorted2.zarr"
    sort(str(sorted1), output=str(sorted2), by=("cell_type",), cfg=_cfg())
    assert any("re-derived" in m for m in _messages(caplog))

    got = ad.read_zarr(str(sorted2))
    assert got.n_obs == 55 and "gexp" in got.layers
    codes = got.obs["cell_type"].cat.codes.to_numpy()
    assert (np.diff(codes) >= 0).all()
    orig = {n: r for src in (a, b) for n, r in zip(src.obs_names, src.X.toarray(), strict=True)}
    for i, n in enumerate(got.obs_names):
        np.testing.assert_allclose(got.X[i].toarray().ravel(), orig[n])
    np.testing.assert_allclose(got.layers["gexp"].toarray(), _expected_gexp(got.X), rtol=1e-5)


def test_lifecycle_icechunk(tmp_path):
    pytest.importorskip("icechunk")
    from anndata.io import read_elem, sparse_dataset

    from annizarr._storage import open_input_group

    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    cfg_ic = AppConfig(
        io=IOConfig(backend="icechunk"),
        chunks=ChunkConfig(x_row_chunk=16, x_col_chunk=4, sparse_flat_chunk=64),
    )
    h5 = tmp_path / "a.h5ad"
    a.write_h5ad(h5)
    sa = tmp_path / "a.icechunk"
    convert_h5ad(str(h5), output=str(sa), cfg=cfg_ic)
    sb = _store(tmp_path, b, "b.zarr")
    append(str(sa), cells=str(sb), drop_derived=True, cfg=_cfg())  # icechunk target auto-detected
    add_expr(str(sa), fmt="csc", chunk_elems=32, cfg=cfg_ic)

    sorted_ic = tmp_path / "sorted.icechunk"
    sort(str(sa), output=str(sorted_ic), by=("cell_type",), cfg=cfg_ic)  # icechunk input auto-detected

    root = open_input_group(str(sorted_ic))
    assert_anndata_readable(root)  # icechunk-backed store; not a plain zarr path, so pass the Group
    obs = read_elem(root["obs"])
    assert len(obs) == 42
    assert (np.diff(obs["cell_type"].cat.codes.to_numpy()) >= 0).all()
    x = sparse_dataset(root["X"])[:]
    orig = {n: r for src in (a, b) for n, r in zip(src.obs_names, src.X.toarray(), strict=True)}
    for i, n in enumerate(obs.index):
        np.testing.assert_allclose(x[i].toarray().ravel(), orig[n])
    gexp = sparse_dataset(root["layers/gexp"])[:]
    np.testing.assert_allclose(gexp.toarray(), _expected_gexp(x), rtol=1e-5)


def test_sort_store_output_exists(tmp_path):
    # StorageError, not ConversionError: sort's target-exists check now goes through the
    # shared check_output_target (item 6) instead of an ad-hoc raise in _sorting.
    out = _store(tmp_path, _adata())
    out2 = tmp_path / "sorted.zarr"
    out2.mkdir()
    with pytest.raises(StorageError, match="already exists"):
        sort(str(out), output=str(out2), by=("cell_type",), cfg=_cfg())


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_store_ops(tmp_path):
    from annizarr._cli import main as run

    out = _store(tmp_path, _adata())
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text("[chunks]\nsparse_flat_chunk = 32\n")
    assert run(["add-expr", str(out), "--chunk-elems", "32", "--config", str(cfg_file)]) == 0
    assert "gexp" in ad.read_zarr(str(out)).layers
    assert run(["add-expr", str(tmp_path / "missing.zarr")]) == 1


def test_cli_append_extend_layers(tmp_path):
    from annizarr._cli import main as run

    a, b = _adata(n=30, seed=0), _adata(n=12, seed=1)
    sa = _store(tmp_path, a, "a.zarr")
    sb = _store(tmp_path, b, "b.zarr")
    add_expr(str(sa), fmt="csr", chunk_elems=32, cfg=_cfg())
    assert run(["append", str(sa), str(sb), "--extend-layers", "--drop-derived"]) == 0
    got = ad.read_zarr(str(sa))
    np.testing.assert_allclose(got.layers["gexp"].toarray(), _expected_gexp(sp.vstack([a.X, b.X])), rtol=1e-5)
