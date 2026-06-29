"""Tests for the icechunk backend (Feature A) and sort/partition + reader (Feature B)."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from datascale import open_sorted
from datascale.config import (
    AppConfig,
    ChunkConfig,
    GroupingConfig,
    IOConfig,
    ValidationConfig,
)
from datascale.converter import ConversionError, convert_h5ad_to_zarr
from datascale.config import _validate_config
from datascale.reader import QueryError
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


def _ids(adata: ad.AnnData) -> set[int]:
    """Recover the unique-id set from a subset's X[:,0]."""
    col0 = adata.X[:, 0]
    return set(int(round(v)) for v in np.asarray(col0.todense()).ravel())


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
# Feature B — sort/partition + reader
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


def test_reader_select_contiguous_block(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg())

    store = open_sorted(str(out))
    a = store.select(cell_type="A")           # whole A block: ids 2,3,5
    assert _ids(a) == {2, 3, 5}
    assert set(a.obs["cell_type"]) == {"A"}
    assert "coords" in a.obsm and a.obsm["coords"].shape == (3, 2)

    sub = store.select(cell_type="A", demographic="x")  # sub-range: ids 2,5
    assert _ids(sub) == {2, 5}


def test_reader_select_crosscut_gathers_spans(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg())

    store = open_sorted(str(out))
    # demographic="x" cuts across cell types A and B (non-adjacent spans): ids 2,5 (A/x) + 4 (B/x)
    x = store.select(demographic="x")
    assert _ids(x) == {2, 4, 5}


def test_reader_unknown_key_and_no_match(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "sorted.zarr"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg())
    store = open_sorted(str(out))
    with pytest.raises(QueryError, match="Unknown sort key"):
        store.select(nope="A")
    with pytest.raises(QueryError, match="No rows match"):
        store.select(cell_type="ZZZ")


def test_open_sorted_on_unsorted_store_errors(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "plain.zarr"
    cfg = AppConfig(io=IOConfig(overwrite=True), chunks=_chunks(), validation=ValidationConfig())
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), cfg)
    with pytest.raises(QueryError, match="datascale_sort_index"):
        open_sorted(str(out))


def test_sort_requires_sparse_csr(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    cfg = AppConfig(
        io=IOConfig(overwrite=True, x_storage="dense"),
        chunks=_chunks(),
        validation=ValidationConfig(),
        grouping=GroupingConfig(enabled=True, sort_by=("cell_type",)),
    )
    with pytest.raises(ConversionError, match="requires x_storage='sparse-csr'"):
        convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(tmp_path / "o.zarr"), cfg)


# ---------------------------------------------------------------------------
# Features compose: sorted store written through icechunk
# ---------------------------------------------------------------------------

def test_sort_through_icechunk_and_read(tmp_path: Path) -> None:
    _labelled_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "repo.icechunk"
    convert_h5ad_to_zarr(str(tmp_path / "in.h5ad"), str(out), _sorted_cfg(backend="icechunk"))

    store = open_sorted(str(out), icechunk=True, branch="main")
    assert _ids(store.select(cell_type="A")) == {2, 3, 5}
    assert _ids(store.select(demographic="x")) == {2, 4, 5}


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
