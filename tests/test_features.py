"""Tests for the icechunk backend (Feature A) and sort/partition (Feature B)."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from datascale.config import (
    AppConfig,
    ChunkConfig,
    GroupingConfig,
    IOConfig,
    ValidationConfig,
)
from datascale.converter import (
    ConversionError,
    convert_h5ad_to_zarr,
    convert_h5ads_to_zarr,
)
from datascale.config import _validate_config
from datascale.storage import open_input_group


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(**kw) -> ChunkConfig:
    base = dict(x_row_chunk=2, x_col_chunk=2, sparse_flat_chunk=2048)
    base.update(kw)
    return ChunkConfig(**base)


def _labelled_h5ad(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Write a 6-cell CSR h5ad. X[:,0] is a unique 1-based id per cell so a subset's
    membership can be checked. Returns (ids, cell_type, demographic)."""
    cell_type = ["B", "A", "A", "B", "A", "B"]
    demographic = ["y", "x", "y", "x", "x", "y"]
    n = len(cell_type)
    dense = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        dense[i, 0] = i + 1          # unique id
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
    return set(int(round(v)) for v in np.asarray(dense).ravel())


def _self_serve_subset(g, **keys):
    """Read rows matching ``keys`` from a sorted store using ONLY stock anndata/zarr
    (no datascale) — proves the store is self-describing. Returns (X, obs)."""
    from anndata.io import read_elem, sparse_dataset

    ranges = read_elem(g["uns"]["datascale_sort_index"]["ranges"])
    for k, v in keys.items():
        ranges = ranges[ranges[k] == v]
    spans = sorted((int(r["start"]), int(r["end"])) for _, r in ranges.iterrows())
    rows = np.concatenate([np.arange(s, e) for s, e in spans])
    x_ds = sparse_dataset(g["X"])
    parts = [x_ds[s:e] for s, e in spans]
    X = parts[0] if len(parts) == 1 else sp.vstack(parts, format="csr")
    obs = read_elem(g["obs"]).iloc[rows]
    return X, obs


# ---------------------------------------------------------------------------
# Feature A — icechunk backend
# ---------------------------------------------------------------------------

def test_icechunk_roundtrip_eager(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "repo.icechunk"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, backend="icechunk"),
        chunks=_chunks(),
        validation=ValidationConfig(),
    )
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), cfg)

    # Reopen through an icechunk read-only session and check X round-trips.
    g = open_input_group(str(out), icechunk=True, branch="main")
    from anndata.io import read_elem, sparse_dataset

    X = sparse_dataset(g["X"])[:]
    assert sp.isspmatrix_csr(X)
    assert np.array_equal(np.asarray(X[:, 0].todense()).ravel(), np.arange(1, 7))
    assert g.attrs["encoding-type"] == "anndata"
    assert list(read_elem(g["obs"])["cell_type"]) == ["B", "A", "A", "B", "A", "B"]


def test_icechunk_rejects_backed(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    cfg = AppConfig(
        io=IOConfig(overwrite=True, backend="icechunk", backed=True),
        chunks=_chunks(),
        validation=ValidationConfig(),
    )
    with pytest.raises(ConversionError, match="does not support --backed"):
        convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(tmp_path / "repo.icechunk"), cfg)


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


def test_sort_writes_valid_anndata_and_index(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg())

    # Still a valid anndata store (stock ad.read_zarr works), rows sorted by the keys.
    adata = ad.read_zarr(str(out))
    ct = list(adata.obs["cell_type"])
    assert ct == sorted(ct)  # primary key non-decreasing
    # Permutation = original ids order after sort: A/x(2,5), A/y(3), B/x(4), B/y(1,6)
    assert list(np.asarray(adata.X[:, 0].todense()).ravel().astype(int)) == [2, 5, 3, 4, 1, 6]

    # Index lives under uns/datascale_sort_index and round-trips with anndata.
    idx = adata.uns["datascale_sort_index"]
    ranges = idx["ranges"]
    assert set(ranges.columns) >= {"cell_type", "demographic", "start", "end"}
    ax = ranges[(ranges.cell_type == "A") & (ranges.demographic == "x")].iloc[0]
    assert (int(ax["start"]), int(ax["end"])) == (0, 2)
    assert np.array_equal(np.asarray(idx["obs_order"]), [1, 4, 2, 3, 0, 5])
    # obsm is reordered consistently with the row permutation (coords col0 was 0,2,4,6,8,10).
    assert list(adata.obsm["coords"][:, 0].astype(int)) == [2, 8, 4, 6, 0, 10]


def test_sorted_store_self_serve_contiguous_block(tmp_path: Path) -> None:
    """A whole-primary-key block spans contiguous range(s): read it with stock anndata/zarr."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg())

    g = open_input_group(str(out))
    X, obs = _self_serve_subset(g, cell_type="A")           # whole A block: ids 2,3,5
    assert _id_set(X) == {2, 3, 5}
    assert set(obs["cell_type"]) == {"A"}

    Xsub, _ = _self_serve_subset(g, cell_type="A", demographic="x")  # sub-range: ids 2,5
    assert _id_set(Xsub) == {2, 5}


def test_sorted_store_self_serve_crosscut(tmp_path: Path) -> None:
    """A non-leading key cuts across primary-key blocks into several non-adjacent spans;
    gather them with stock anndata/zarr via the sort_index."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg())

    g = open_input_group(str(out))
    # demographic="x" cuts across cell types A and B (non-adjacent spans): ids 2,5 (A/x) + 4 (B/x)
    X, _ = _self_serve_subset(g, demographic="x")
    assert _id_set(X) == {2, 4, 5}


def test_sort_dense_writes_contiguous_ranges(tmp_path: Path) -> None:
    """Dense X supports --sort-by: rows are physically sorted and the sort_index ranges
    line up with X[start:end] (read directly from the dense X with stock zarr, no datascale)."""
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted_dense.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="dense"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type", "demographic")),
    )
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), cfg)

    # X is a plain dense zarr array, rows sorted by the keys, still a valid anndata store.
    g = open_input_group(str(out))
    from anndata.io import read_elem
    import zarr

    assert isinstance(g["X"], zarr.Array)
    adata = ad.read_zarr(str(out))
    ct = list(adata.obs["cell_type"])
    assert ct == sorted(ct)
    assert list(np.asarray(adata.X[:, 0]).ravel().astype(int)) == [2, 5, 3, 4, 1, 6]

    # Each sort_index range is a contiguous block whose obs rows share the key tuple,
    # readable directly from the dense X with a single slice.
    ranges = read_elem(g["uns"]["datascale_sort_index"]["ranges"])
    obs_full = read_elem(g["obs"])
    for _, row in ranges.iterrows():
        s, e = int(row["start"]), int(row["end"])
        block = g["X"][s:e]
        assert block.shape == (e - s, adata.n_vars)
        assert (obs_full["cell_type"].to_numpy()[s:e] == row["cell_type"]).all()


def test_sort_rejects_sparse_csc(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="sparse-csc"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type",)),
    )
    with pytest.raises(ConversionError, match="requires x_storage='sparse-csr' or 'dense'"):
        convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(tmp_path / "o.zarr"), cfg)


# ---------------------------------------------------------------------------
# Features compose: sorted store written through icechunk
# ---------------------------------------------------------------------------

def test_sort_through_icechunk_and_read(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "repo.icechunk"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg(backend="icechunk"))

    # Read subsets from the icechunk-backed sorted store with stock anndata/zarr.
    g = open_input_group(str(out), icechunk=True, branch="main")
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


def test_dense_sharding_metadata_and_roundtrip(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sharded.zarr"
    # 6x3 X, chunks 2x2, factor 2 -> shard 4x4 (capped at the array's chunk extent).
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _dense_cfg(shard_factor=2))

    g = open_input_group(str(out))
    xa = g["X"]
    assert xa.chunks == (2, 2)          # inner chunk = read granularity, unchanged
    assert xa.shards == (4, 4)          # shard = chunk * factor
    assert xa.attrs["encoding-type"] == "array"

    # Still a valid, byte-identical anndata store.
    got = np.asarray(ad.read_zarr(str(out)).X)
    assert np.array_equal(got, _expected_dense(tmp_path / "in.h5ad"))


def test_sharding_cuts_object_count(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    sharded = tmp_path / "sharded.zarr"
    plain = tmp_path / "plain.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(sharded), _dense_cfg(shard_factor=2))
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(plain), _dense_cfg(shard_factor=1))

    # Same logical layout (chunks 2x2), but shards pack many inner chunks per object.
    assert _x_object_count(sharded) < _x_object_count(plain)
    # ...and the data is identical between the two.
    assert np.array_equal(
        np.asarray(ad.read_zarr(str(sharded)).X),
        np.asarray(ad.read_zarr(str(plain)).X),
    )


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
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), cfg)

    xa = open_input_group(str(out))["X"]
    assert xa.chunks == (2, 2) and xa.shards == (4, 4)
    assert np.array_equal(
        np.asarray(ad.read_zarr(str(out)).X), _expected_dense(tmp_path / "in.h5ad")
    )


def test_sharding_ignored_for_sparse(tmp_path: Path, capsys) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sparse.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="sparse-csr"),
        chunks=_chunks(x_shard_factor=4),
        validation=ValidationConfig(),
    )
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), cfg)

    # Sparse X is a group of 1-D arrays; none of them are sharded.
    g = open_input_group(str(out))
    for name in ("data", "indices", "indptr"):
        assert g["X"][name].shards is None
    assert "only applies to dense X" in capsys.readouterr().err


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
    dense = ((rng.random((n_obs, n_vars), dtype=np.float32) < 0.3)
             * rng.random((n_obs, n_vars), dtype=np.float32)).astype(np.float32)
    obs = pd.DataFrame({"batch": ["b"] * n_obs}, index=[f"{seed}_{i}" for i in range(n_obs)])
    ad.AnnData(X=sp.csr_matrix(dense), obs=obs).write_h5ad(path)
    return dense


def _read_X(out: Path) -> np.ndarray:
    X = ad.read_zarr(str(out)).X
    return np.asarray(X.todense() if sp.issparse(X) else X)


@pytest.mark.parametrize("x_storage", ["dense", "sparse-csr"])
def test_inmem_parallel_write_roundtrip(tmp_path: Path, x_storage: str) -> None:
    """In-memory conversion at cpus>1 (threaded da.store) must round-trip bit-exact.
    Rows span several row-chunks so multiple chunks are written concurrently — this is the
    path that carries lock=False, so a lock/alignment regression would corrupt the output."""
    src = _rand_h5ad(tmp_path / "in.h5ad", n_obs=300, n_vars=200, seed=1)
    out = tmp_path / "out.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=64, x_col_chunk=200, sparse_flat_chunk=500, cpus=4),
        validation=ValidationConfig(),
    )
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), cfg)
    assert np.array_equal(_read_X(out), src)


@pytest.mark.parametrize("x_storage", ["dense", "sparse-csr"])
def test_concat_parallel_write_roundtrip(tmp_path: Path, x_storage: str) -> None:
    """concat-h5ads at cpus>1 writes each file at a misaligned row offset, so the file-seam
    chunk is read-modify-written. That path MUST keep the da.store lock (row counts here are
    deliberately not multiples of the row chunk) — with lock=False concurrent RMW corrupts X."""
    # 200 + 300 rows at row_chunk 64 puts the second file at a misaligned offset (200 % 64 != 0);
    # 200 cols/chunk make the seam-chunk RMW window wide enough that lock=False corrupts
    # reliably (verified 6/6), so this is a real guard, not a coin flip.
    a = _rand_h5ad(tmp_path / "a.h5ad", n_obs=200, n_vars=500, seed=1)
    b = _rand_h5ad(tmp_path / "b.h5ad", n_obs=300, n_vars=500, seed=2)
    out = tmp_path / "out.zarr"
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=64, x_col_chunk=500, sparse_flat_chunk=500, cpus=4),
        validation=ValidationConfig(),
    )
    convert_h5ads_to_zarr(
        [str(tmp_path / "a.h5ad"), str(tmp_path / "b.h5ad")], str(out), cfg
    )
    assert np.array_equal(_read_X(out), np.vstack([a, b]))
