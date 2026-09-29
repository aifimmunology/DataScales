"""Tests for the icechunk backend (Feature A) and sort/partition (Feature B)."""

from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from annizarr._config import _validate_config
from annizarr._ops import concat, convert_h5ad
from annizarr._storage import open_input_group
from annizarr.config import (
    AppConfig,
    ChunkConfig,
    ConcatConfig,
    GroupingConfig,
    IOConfig,
    ValidationConfig,
)
from annizarr.errors import ConversionError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks(**kw) -> ChunkConfig:
    base = dict(x_row_chunk=2, x_col_chunk=2, sparse_flat_chunk=2048)
    base.update(kw)
    return ChunkConfig(**base)


def _messages(caplog):
    return [r.message for r in caplog.records]


def _labelled_h5ad(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Write a 6-cell CSR h5ad. X[:,0] is a unique 1-based id per cell so a subset's
    membership can be checked. Returns (ids, cell_type, demographic)."""
    cell_type = ["B", "A", "A", "B", "A", "B"]
    demographic = ["y", "x", "y", "x", "x", "y"]
    n = len(cell_type)
    dense = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        dense[i, 0] = i + 1  # unique id
        dense[i, 1] = (i + 1) * 10.0
    X = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"cell_type": cell_type, "demographic": demographic},
        index=[str(i) for i in range(n)],
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["coords"] = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    adata.write_h5ad(path)
    return np.arange(1, n + 1), cell_type, demographic


def _id_set(X) -> set[int]:
    """Recover the unique-id set from a subset's X[:,0]."""
    col0 = X[:, 0]
    dense = col0.todense() if sp.issparse(col0) else col0
    return {round(v) for v in np.asarray(dense).ravel()}


def _self_serve_subset(g, **keys):
    """Read rows matching ``keys`` from a sorted store using ONLY stock anndata/zarr — no
    annizarr, no annizarr index. Because the store is physically sorted by the keys, the
    matching rows form contiguous span(s); we find them by masking the (sorted) obs column(s)
    and splitting the matched row indices into contiguous runs. Returns (X, obs)."""
    from anndata.io import read_elem, sparse_dataset

    obs = read_elem(g["obs"])
    mask = np.ones(len(obs), dtype=bool)
    for k, v in keys.items():
        mask &= obs[k].to_numpy() == v
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        spans = []
    else:
        cut = np.flatnonzero(np.diff(rows) > 1)  # boundaries between contiguous runs
        starts = np.concatenate([rows[:1], rows[cut + 1]])
        ends = np.concatenate([rows[cut], rows[-1:]]) + 1
        spans = list(zip(starts.tolist(), ends.tolist(), strict=True))
    x_ds = sparse_dataset(g["X"])
    parts = [x_ds[s:e] for s, e in spans]
    X = parts[0] if len(parts) == 1 else sp.vstack(parts, format="csr")
    return X, obs.iloc[rows]


# ---------------------------------------------------------------------------
# Feature A — icechunk backend
# ---------------------------------------------------------------------------


def test_icechunk_roundtrip_eager_and_op_result_snapshot_ids(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    from annizarr._ops import add_expr

    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "repo.icechunk"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, backend="icechunk"),
        chunks=_chunks(),
        validation=ValidationConfig(),
    )
    ic_result = convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg)
    assert isinstance(ic_result.snapshot_id, str) and ic_result.snapshot_id

    # Reopen through an icechunk read-only session and check X round-trips.
    g = open_input_group(str(out), branch="main")
    from anndata.io import read_elem, sparse_dataset

    X = sparse_dataset(g["X"])[:]
    assert sp.isspmatrix_csr(X)
    assert np.array_equal(np.asarray(X[:, 0].todense()).ravel(), np.arange(1, 7))
    assert g.attrs["encoding-type"] == "anndata"
    assert list(read_elem(g["obs"])["cell_type"]) == ["B", "A", "A", "B", "A", "B"]

    # icechunk never supports backed input
    cfg_backed = replace(cfg, io=replace(cfg.io, backed=True))
    with pytest.raises(ConversionError, match="does not support --backed"):
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / "repo2.icechunk"), cfg=cfg_backed)

    # a plain-zarr OpResult carries no snapshot_id; an icechunk one always does, and each op
    # gets its own distinct snapshot
    plain_cfg = AppConfig(io=IOConfig(overwrite=True), chunks=_chunks(), validation=ValidationConfig())
    plain_result = convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / "plain.zarr"), cfg=plain_cfg)
    assert plain_result.snapshot_id is None

    expr_result = add_expr(str(out), cfg=cfg)
    assert isinstance(expr_result.snapshot_id, str) and expr_result.snapshot_id
    assert expr_result.snapshot_id != ic_result.snapshot_id


# ---------------------------------------------------------------------------
# Feature B — sort/partition (self-serve subset reads with stock anndata/zarr)
# ---------------------------------------------------------------------------


def _sorted_cfg(backend: str = "zarr") -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, backend=backend),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type", "demographic")),
    )


def test_sort_writes_valid_anndata_and_self_serve_reads(tmp_path: Path) -> None:
    """Convert-time --sort-by physically sorts rows (no annizarr-specific index written);
    self-serve subset reads then use ONLY stock anndata/zarr. A whole-primary-key block
    (cell_type="A") and a compound sub-range (cell_type="A", demographic="x") both span
    contiguous run(s) — but a non-leading key alone (demographic="x") cuts across
    primary-key blocks into several NON-adjacent spans, a genuinely different
    span-reconstruction case worth checking separately."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=_sorted_cfg())

    # Still a valid anndata store (stock ad.read_zarr works), rows sorted by the keys.
    adata = ad.read_zarr(str(out))
    ct = list(adata.obs["cell_type"])
    assert ct == sorted(ct)  # primary key non-decreasing
    # Permutation = original ids order after sort: A/x(2,5), A/y(3), B/x(4), B/y(1,6)
    assert list(np.asarray(adata.X[:, 0].todense()).ravel().astype(int)) == [2, 5, 3, 4, 1, 6]
    # obsm is reordered consistently with the row permutation (coords col0 was 0,2,4,6,8,10).
    assert list(adata.obsm["coords"][:, 0].astype(int)) == [2, 8, 4, 6, 0, 10]
    # NO annizarr-specific index is written — the store is a plain sorted AnnData.
    assert "zarrsmith_sort_index" not in adata.uns

    g = open_input_group(str(out))
    X, obs = _self_serve_subset(g, cell_type="A")  # whole A block: ids 2,3,5
    assert _id_set(X) == {2, 3, 5}
    assert set(obs["cell_type"]) == {"A"}

    Xsub, _ = _self_serve_subset(g, cell_type="A", demographic="x")  # sub-range: ids 2,5
    assert _id_set(Xsub) == {2, 5}

    # demographic="x" cuts across cell types A and B (non-adjacent spans): ids 2,5 (A/x) + 4 (B/x)
    Xcross, _ = _self_serve_subset(g, demographic="x")
    assert _id_set(Xcross) == {2, 4, 5}


def test_sort_dense_writes_contiguous_ranges_and_rejects_unsupported_x_storage(tmp_path: Path) -> None:
    """Dense X supports --sort-by: rows are physically sorted so each key tuple is a
    contiguous run derivable from the sorted obs and read directly via X[start:end]
    (stock zarr, no annizarr, no index). --sort-by also requires x_storage='csr' or 'dense'
    in general, and further requires 'csr' specifically when combined with --backed
    (streamed bucketing is csr-only; dense sort still works eagerly, as above)."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted_dense.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="dense"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type", "demographic")),
    )
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg)

    # X is a plain dense zarr array, rows sorted by the keys, still a valid anndata store.
    g = open_input_group(str(out))
    import zarr
    from anndata.io import read_elem

    assert isinstance(g["X"], zarr.Array)
    adata = ad.read_zarr(str(out))
    ct = list(adata.obs["cell_type"])
    assert ct == sorted(ct)
    assert list(np.asarray(adata.X[:, 0]).ravel().astype(int)) == [2, 5, 3, 4, 1, 6]

    # Each contiguous run of equal (cell_type, demographic) in the sorted obs is a block,
    # derivable from obs alone and readable directly from the dense X with a single slice.
    obs_full = read_elem(g["obs"])
    keys = list(zip(obs_full["cell_type"].astype(str), obs_full["demographic"].astype(str), strict=True))
    s = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[s]:
            block = g["X"][s:i]
            assert block.shape == (i - s, adata.n_vars)
            assert (obs_full["cell_type"].to_numpy()[s:i] == keys[s][0]).all()
            s = i

    cfg_csc = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csc"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type",)),
    )
    with pytest.raises(ConversionError, match="requires x_storage='csr' or 'dense'"):
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / "o1.zarr"), cfg=cfg_csc)

    cfg_backed_dense = AppConfig(
        io=IOConfig(overwrite=True, backed=True, x_storage="dense"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type",)),
    )
    with pytest.raises(ConversionError, match="csr"):
        convert_h5ad(str(tmp_path / "in.h5ad"), output=str(tmp_path / "o2.zarr"), cfg=cfg_backed_dense)


def _backed_sorted_cfg() -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, backed=True, x_storage="csr"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type", "demographic")),
    )


def test_sort_backed_streamed_matches_eager(tmp_path: Path) -> None:
    """--backed --sort-by (streamed bucketing, Option C) yields the SAME sorted csr
    store as the eager path — same row order, X values, and reordered obsm — without ever
    materialising X in full."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out_backed = tmp_path / "backed.zarr"
    out_eager = tmp_path / "eager.zarr"

    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out_eager), cfg=_sorted_cfg())
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out_backed), cfg=_backed_sorted_cfg())

    a_backed = ad.read_zarr(str(out_backed))
    a_eager = ad.read_zarr(str(out_eager))
    # A/x(2,5), A/y(3), B/x(4), B/y(1,6): the same permutation the eager test asserts.
    assert list(np.asarray(a_backed.X[:, 0].todense()).ravel().astype(int)) == [2, 5, 3, 4, 1, 6]
    assert np.array_equal(np.asarray(a_backed.X.todense()), np.asarray(a_eager.X.todense()))
    assert list(a_backed.obs["cell_type"]) == list(a_eager.obs["cell_type"])
    assert np.array_equal(a_backed.obsm["coords"], a_eager.obsm["coords"])
    assert "zarrsmith_sort_index" not in a_backed.uns

    # Self-serve subset reads (stock anndata/zarr) work on the backed-sorted store too.
    g = open_input_group(str(out_backed))
    assert _id_set(_self_serve_subset(g, cell_type="A")[0]) == {2, 3, 5}
    assert _id_set(_self_serve_subset(g, demographic="x")[0]) == {2, 4, 5}


# ---------------------------------------------------------------------------
# Features compose: sorted store written through icechunk
# ---------------------------------------------------------------------------


def test_backed_sort_default_commit_message_names_sort_columns(tmp_path: Path) -> None:
    """_write_sorted_backed's default commit message names the sort columns (item 9), so a
    sorted convert is distinguishable in `ic log`. Exercised directly against an icechunk
    cfg, since --backed + backend=icechunk is otherwise rejected upstream (icechunk never
    supports backed input) before it would reach this code path."""
    pytest.importorskip("icechunk")
    from annizarr._ic import Repo
    from annizarr._sorting import _write_sorted_backed

    _labelled_h5ad(tmp_path / "in.h5ad")
    adata = ad.read_h5ad(tmp_path / "in.h5ad", backed="r")
    out = tmp_path / "repo.icechunk"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, backed=True, x_storage="csr", backend="icechunk"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type", "demographic")),
    )
    try:
        _write_sorted_backed(adata, out, cfg)
    finally:
        adata.file.close()

    repo = Repo(str(out))
    assert repo.log(branch="main")[0].message == "annizarr convert (sorted by cell_type,demographic) → repo.icechunk"


def test_sort_through_icechunk_and_read(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "repo.icechunk"
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=_sorted_cfg(backend="icechunk"))

    # Read subsets from the icechunk-backed sorted store with stock anndata/zarr.
    g = open_input_group(str(out), branch="main")
    X_a, _ = _self_serve_subset(g, cell_type="A")
    assert _id_set(X_a) == {2, 3, 5}
    X_x, _ = _self_serve_subset(g, demographic="x")
    assert _id_set(X_x) == {2, 4, 5}


# ---------------------------------------------------------------------------
# Dense X sharding (--x-shard-factor)
# ---------------------------------------------------------------------------


def _dense_cfg(shard_factor: int = 1, **chunk_kw) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, x_storage="dense"),
        chunks=_chunks(x_shard_factor=shard_factor, **chunk_kw),
        validation=ValidationConfig(),
    )


def _x_object_count(store_dir: Path) -> int:
    """Count stored chunk/shard objects under the X array (not metadata)."""
    xdir = store_dir / "X"
    return sum(1 for p in xdir.rglob("*") if p.is_file() and p.name != "zarr.json")


def _expected_dense(in_h5ad: Path) -> np.ndarray:
    a = ad.read_h5ad(str(in_h5ad))
    return np.asarray(a.X.todense() if sp.issparse(a.X) else a.X)


def test_dense_sharding_metadata_roundtrip_and_object_count(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    sharded = tmp_path / "sharded.zarr"
    plain = tmp_path / "plain.zarr"
    # 6x3 X, chunks 2x2, factor 2 -> shard 4x4 (capped at the array's chunk extent).
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(sharded), cfg=_dense_cfg(shard_factor=2))
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(plain), cfg=_dense_cfg(shard_factor=1))

    g = open_input_group(str(sharded))
    xa = g["X"]
    assert xa.chunks == (2, 2)  # inner chunk = read granularity, unchanged
    assert xa.shards == (4, 4)  # shard = chunk * factor
    assert xa.attrs["encoding-type"] == "array"

    # Same logical layout (chunks 2x2), but shards pack many inner chunks per object.
    assert _x_object_count(sharded) < _x_object_count(plain)

    # Still a valid, byte-identical anndata store, and identical data between the two.
    expected = _expected_dense(tmp_path / "in.h5ad")
    assert np.array_equal(np.asarray(ad.read_zarr(str(sharded)).X), expected)
    assert np.array_equal(np.asarray(ad.read_zarr(str(plain)).X), expected)


def test_sharding_backed_dense_parallel_roundtrip(tmp_path: Path) -> None:
    """Backed input + cpus>1 fans densify-bands across processes; each band must write
    whole shards (no read-modify-write, no inter-worker shard sharing)."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "backed_sharded.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="dense", backed=True),
        chunks=_chunks(x_shard_factor=2, cpus=2),
        validation=ValidationConfig(),
    )
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg)

    xa = open_input_group(str(out))["X"]
    assert xa.chunks == (2, 2) and xa.shards == (4, 4)
    assert np.array_equal(np.asarray(ad.read_zarr(str(out)).X), _expected_dense(tmp_path / "in.h5ad"))


def test_sharding_ignored_for_sparse(tmp_path: Path, caplog) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sparse.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=_chunks(x_shard_factor=4),
        validation=ValidationConfig(),
    )
    convert_h5ad(str(tmp_path / "in.h5ad"), output=str(out), cfg=cfg)

    # Sparse X is a group of 1-D arrays; none of them are sharded.
    g = open_input_group(str(out))
    for name in ("data", "indices", "indptr"):
        assert g["X"][name].shards is None
    assert any("only applies to dense X" in m for m in _messages(caplog))


def test_shard_factor_below_one_rejected() -> None:
    cfg = AppConfig(chunks=ChunkConfig(x_shard_factor=0))
    with pytest.raises(ValueError, match="x_shard_factor must be >= 1"):
        _validate_config(cfg)


# ---------------------------------------------------------------------------
# Parallel (cpus>1) write correctness — guards the da.store lock choice
# ---------------------------------------------------------------------------


def _rand_h5ad(path: Path, n_obs: int, n_vars: int, seed: int) -> np.ndarray:
    """Write a CSR h5ad with seeded ~30%-dense values; returns the dense X."""
    rng = np.random.default_rng(seed)
    dense = (
        (rng.random((n_obs, n_vars), dtype=np.float32) < 0.3) * rng.random((n_obs, n_vars), dtype=np.float32)
    ).astype(np.float32)
    obs = pd.DataFrame({"batch": ["b"] * n_obs}, index=[f"{seed}_{i}" for i in range(n_obs)])
    ad.AnnData(X=sp.csr_matrix(dense), obs=obs).write_h5ad(path)
    return dense


def _read_X(out: Path) -> np.ndarray:
    X = ad.read_zarr(str(out)).X
    return np.asarray(X.todense() if sp.issparse(X) else X)


def test_parallel_write_roundtrip_inmem_and_concat_seam(tmp_path: Path) -> None:
    """In-memory conversion at cpus>1 (threaded da.store) must round-trip bit-exact for both
    dense and csr -- rows span several row-chunks so multiple chunks are written concurrently,
    the path that carries lock=False (a lock/alignment regression would corrupt the output).
    concat at cpus>1 writes each file at a misaligned row offset, so the file-seam chunk is
    read-modify-written; that path MUST keep the da.store lock (row counts here are
    deliberately not multiples of the row chunk) -- with lock=False concurrent RMW corrupts X
    (200 + 300 rows at row_chunk 64 puts the second file at a misaligned offset, 200 % 64 != 0;
    200 cols/chunk make the seam-chunk RMW window wide enough that lock=False corrupts
    reliably, verified 6/6, so this is a real guard, not a coin flip)."""
    for x_storage in ("dense", "csr"):
        src = _rand_h5ad(tmp_path / f"in_{x_storage}.h5ad", n_obs=300, n_vars=200, seed=1)
        out = tmp_path / f"out_{x_storage}.zarr"
        cfg = AppConfig(
            io=IOConfig(overwrite=True, x_storage=x_storage),
            chunks=ChunkConfig(x_row_chunk=64, x_col_chunk=200, sparse_flat_chunk=500, cpus=4),
            validation=ValidationConfig(),
        )
        convert_h5ad(str(tmp_path / f"in_{x_storage}.h5ad"), output=str(out), cfg=cfg)
        assert np.array_equal(_read_X(out), src)

    for x_storage in ("dense", "csr"):
        a = _rand_h5ad(tmp_path / f"a_{x_storage}.h5ad", n_obs=200, n_vars=500, seed=1)
        b = _rand_h5ad(tmp_path / f"b_{x_storage}.h5ad", n_obs=300, n_vars=500, seed=2)
        out = tmp_path / f"concat_{x_storage}.zarr"
        cfg = AppConfig(
            io=IOConfig(overwrite=True, x_storage=x_storage),
            chunks=ChunkConfig(x_row_chunk=64, x_col_chunk=500, sparse_flat_chunk=500, cpus=4),
            validation=ValidationConfig(),
        )
        concat([str(tmp_path / f"a_{x_storage}.h5ad"), str(tmp_path / f"b_{x_storage}.h5ad")], output=str(out), cfg=cfg)
        assert np.array_equal(_read_X(out), np.vstack([a, b]))


def test_concat_backed_csc_input_above_eager_max_bytes_raises(tmp_path: Path) -> None:
    """concat's ensure_csr refuses a backed CSC input whose on-disk X is bigger than
    io.eager_max_bytes (item 5) instead of silently loading it whole into memory; the error
    hints at converting that input to csr first."""
    _rand_h5ad(tmp_path / "a.h5ad", n_obs=50, n_vars=40, seed=1)
    _rand_h5ad(tmp_path / "b.h5ad", n_obs=50, n_vars=40, seed=2)
    b_adata = ad.read_h5ad(tmp_path / "b.h5ad")
    b_adata.X = sp.csc_matrix(b_adata.X)  # b's X is CSC on disk; a stays CSR
    b_adata.write_h5ad(tmp_path / "b.h5ad")

    out = tmp_path / "out.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr", backed=True, eager_max_bytes=64),
        chunks=ChunkConfig(x_row_chunk=16, x_col_chunk=40, sparse_flat_chunk=500),
        validation=ValidationConfig(),
    )
    with pytest.raises(ConversionError, match="annizarr convert --x-storage csr"):
        concat([str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], output=str(out), cfg=cfg)


# ---------------------------------------------------------------------------
# concat obs-column selection (ConcatConfig.obs_columns)
# ---------------------------------------------------------------------------


def _obs_cols_h5ad(path: Path, n: int, cats: list[str], extra: str) -> None:
    """CSR h5ad whose obs has a categorical `cell_type` (categories=`cats`), a shared
    `donor`, and one file-unique `extra` column. Two files may carry the SAME or DIFFERENT
    category sets — the latter is what the categorical-mismatch guard rejects."""
    dense = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical([cats[i % len(cats)] for i in range(n)], categories=cats),
            "donor": [f"d{i % 2}" for i in range(n)],
            extra: np.arange(n),
        },
        index=[f"{extra}_{i}" for i in range(n)],
    )
    ad.AnnData(X=sp.csr_matrix(dense), obs=obs).write_h5ad(path)


def _plain_obs_h5ad(path: Path, n: int, extra: str, score: np.ndarray) -> None:
    """CSR h5ad with non-categorical obs: string `cell_type`/`donor`, a numeric `score`
    (dtype set by the caller, to exercise int+float coercion) and a file-unique `extra`."""
    dense = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    obs = pd.DataFrame(
        {
            "cell_type": [f"ct{i % 2}" for i in range(n)],
            "donor": [f"d{i % 2}" for i in range(n)],
            "score": score,
            extra: np.arange(n),
        },
        index=[f"{extra}_{i}" for i in range(n)],
    )
    ad.AnnData(X=sp.csr_matrix(dense), obs=obs).write_h5ad(path)


def test_concat_obs_columns_projects_and_warns(tmp_path: Path, caplog) -> None:
    """obs_columns projects obs to exactly the named cols (in order), drops the rest (warns),
    and warns on a harmless numeric coercion (int+float -> float)."""
    _plain_obs_h5ad(tmp_path / "a.h5ad", 6, "qc_a", np.arange(6, dtype="int64"))
    _plain_obs_h5ad(tmp_path / "b.h5ad", 4, "qc_b", np.arange(4, dtype="float64"))
    out = tmp_path / "out.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        concat=ConcatConfig(obs_columns=("cell_type", "donor", "score")),
    )
    concat([str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], output=str(out), cfg=cfg)
    res = ad.read_zarr(str(out))
    assert list(res.obs.columns) == ["cell_type", "donor", "score"]  # projected + ordered
    assert res.n_obs == 10
    msgs = _messages(caplog)
    assert any("qc_a" in m for m in msgs) and any("qc_b" in m for m in msgs)  # extras dropped
    assert any("score" in m and "coerced" in m for m in msgs)  # int + float -> float


def test_concat_obs_columns_missing_or_categorical_mismatch_raises(tmp_path: Path) -> None:
    # a requested obs column absent from any input is a hard error
    _obs_cols_h5ad(tmp_path / "a.h5ad", 6, ["Tcell", "Bcell"], "qc_a")
    _obs_cols_h5ad(tmp_path / "b.h5ad", 4, ["Tcell", "Bcell"], "qc_b")
    cfg_missing = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        concat=ConcatConfig(obs_columns=("cell_type", "qc_a")),  # qc_a exists only in a.h5ad
    )
    with pytest.raises(ConversionError, match="obs columns not found"):
        concat([str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], output=str(tmp_path / "o1.zarr"), cfg=cfg_missing)

    # a selected categorical column with differing category sets across inputs is a hard
    # error too -- concat would degrade it to a (slow) string array, so we refuse
    _obs_cols_h5ad(tmp_path / "c.h5ad", 6, ["Tcell", "Bcell"], "qc_c")
    _obs_cols_h5ad(tmp_path / "d.h5ad", 4, ["Bcell", "NK"], "qc_d")  # different category set
    cfg_mismatch = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        concat=ConcatConfig(obs_columns=("cell_type", "donor")),
    )
    with pytest.raises(ConversionError, match="mismatched categorical categories"):
        concat([str(tmp_path / "c.h5ad"), str(tmp_path / "d.h5ad")], output=str(tmp_path / "o2.zarr"), cfg=cfg_mismatch)


def test_concat_obs_columns_categorical_match_ok(tmp_path: Path) -> None:
    """A selected categorical column with IDENTICAL category sets concatenates fine and
    stays categorical (codes+categories) in the output."""
    _obs_cols_h5ad(tmp_path / "a.h5ad", 6, ["Tcell", "Bcell", "NK"], "qc_a")
    _obs_cols_h5ad(tmp_path / "b.h5ad", 4, ["Tcell", "Bcell", "NK"], "qc_b")  # same categories
    out = tmp_path / "out.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        concat=ConcatConfig(obs_columns=("cell_type", "donor")),
    )
    concat([str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], output=str(out), cfg=cfg)
    res = ad.read_zarr(str(out))
    assert list(res.obs.columns) == ["cell_type", "donor"]
    assert res.n_obs == 10
    assert isinstance(res.obs["cell_type"].dtype, pd.CategoricalDtype)  # categorical preserved


def test_concat_default_strict_rejects_mismatched_obs(tmp_path: Path) -> None:
    """With no obs_columns, differing obs schemas still abort (unchanged default)."""
    _obs_cols_h5ad(tmp_path / "a.h5ad", 6, ["Tcell", "Bcell"], "qc_a")
    _obs_cols_h5ad(tmp_path / "b.h5ad", 4, ["Tcell", "Bcell"], "qc_b")  # different extra col name
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=_chunks(),
        validation=ValidationConfig(),
    )
    with pytest.raises(ConversionError, match="obs schema mismatch"):
        concat([str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], output=str(tmp_path / "o.zarr"), cfg=cfg)
