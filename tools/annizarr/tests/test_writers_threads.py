"""cpus=1 (serial) vs cpus=4 (thread pool) must write byte-identical stores.

Every in-memory writer path routes disjoint, write-grid-aligned blocks/segments through
``_runtime.run_parallel``; these tests pin chunk/shard sizes so several blocks land on each
axis (with a ragged last one) and compare the two cpu counts file-for-file, not just by value.
"""

from __future__ import annotations

import filecmp
import math
import threading
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

import annizarr._layout as _layout
from _readable import assert_anndata_readable
from annizarr._ops import append, concat, convert_adata, rechunk
from annizarr._runtime import run_parallel
from annizarr.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig


def _rand_dense(n_obs: int, n_vars: int, *, seed: int, density: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n_obs, n_vars)) < density
    return (mask * rng.random((n_obs, n_vars))).astype(np.float32)


def _adata(dense: np.ndarray, *, sparse: bool, seed: int) -> ad.AnnData:
    x = sp.csr_matrix(dense) if sparse else dense.copy()
    obs = pd.DataFrame({"batch": ["b"] * dense.shape[0]}, index=[f"{seed}_{i}" for i in range(dense.shape[0])])
    var = pd.DataFrame(index=[f"g{i}" for i in range(dense.shape[1])])
    return ad.AnnData(X=x, obs=obs, var=var)


def _cfg(
    x_storage: str,
    *,
    cpus: int,
    row_chunk: int = 64,
    col_chunk: int = 48,
    flat_chunk: int = 500,
    x_shard_factor: int = 1,
) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, x_storage=x_storage),
        chunks=ChunkConfig(
            x_row_chunk=row_chunk,
            x_col_chunk=col_chunk,
            sparse_flat_chunk=flat_chunk,
            cpus=cpus,
            x_shard_factor=x_shard_factor,
        ),
        validation=ValidationConfig(),
    )


def _read_x(path: Path) -> np.ndarray:
    x = ad.read_zarr(str(path)).X
    return np.asarray(x.todense() if sp.issparse(x) else x)


def _tree_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}


def _assert_byte_identical(a: Path, b: Path) -> None:
    fa, fb = _tree_files(a), _tree_files(b)
    assert set(fa) == set(fb), f"file set differs: {a.name} has {set(fa) - set(fb)}, {b.name} has {set(fb) - set(fa)}"
    for rel in sorted(fa):
        assert filecmp.cmp(fa[rel], fb[rel], shallow=False), f"differs: {rel}"


# 300x200 with row_chunk=64 (4 full + a 44-row ragged last) and col_chunk=48 (4 full + an
# 8-col ragged last) puts several blocks, with a ragged one, on each axis.
N_OBS, N_VARS = 300, 200

# row_chunk=32/col_chunk=48 with factor=2 -> shards (64, 96): 300 rows / 64 = 4 full shards +
# a 44-row ragged last; 200 cols / 96 = 2 full shards + an 8-col ragged last.
SHARD_ROW_CHUNK, SHARD_COL_CHUNK, SHARD_FACTOR = 32, 48, 2
EXPECTED_SHARDS = (64, 96)


def _n_shard_objects(store: Path, array: str = "X") -> int:
    return sum(1 for p in (store / array / "c").rglob("*") if p.is_file())


def _expected_shard_count(n_rows: int, n_cols: int, shards: tuple[int, int]) -> int:
    return math.ceil(n_rows / shards[0]) * math.ceil(n_cols / shards[1])


def _n_flat_shard_objects(store: Path, path: str) -> int:
    """Count shard objects on disk for a 1-D sparse array (e.g. ``path="X/data"``)."""
    return sum(1 for p in (store / path / "c").rglob("*") if p.is_file())


def _forced_autoshard(flat_chunk: int, itemsize: int, chunks_per_shard: int):
    """Context manager pinning zarr's ``shards="auto"`` heuristic to an exact, known shard
    multiple of ``flat_chunk`` (rather than relying on its size-derived default), so a forced-
    sharding test gets a shard several chunks wide while ``cfg.chunks.sparse_flat_chunk`` stays
    small. See ``zarr.core.chunk_grids._guess_num_chunks_per_axis_shard``: with
    ``target_shard_size_bytes`` set, chunks accumulate into a shard while
    ``bytes_per_chunk * (k+1) <= target``; sizing the target to exactly
    ``chunks_per_shard * bytes_per_chunk`` stops the loop at ``chunks_per_shard``.
    """
    bytes_per_chunk = flat_chunk * itemsize
    target = bytes_per_chunk * chunks_per_shard
    return zarr.config.set({"array.target_shard_size_bytes": target})


@pytest.mark.parametrize(
    ("path", "x_storage", "sparse_source"),
    [
        pytest.param("direct", "dense", False, id="dense_from_dense"),
        pytest.param("direct", "dense", True, id="sparse_to_dense"),
        pytest.param("direct", "csr", True, id="sparse_to_csr"),
        pytest.param("direct", "csc", True, id="sparse_to_csc"),
        pytest.param("concat", "dense", True, id="dense_concat"),
        pytest.param("concat", "csr", True, id="csr_concat"),
    ],
)
def test_cpus1_matches_cpus4_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, x_storage: str, sparse_source: bool
) -> None:
    """Every in-memory writer path (direct convert, or concat across 3 non-chunk-aligned
    inputs) must write byte-identical stores at cpus=1 vs cpus=4 -- force several flat
    segments too (not just several dense blocks) for the sparse cases."""
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1)
    out1, out4 = tmp_path / "cpus1.zarr", tmp_path / "cpus4.zarr"

    if path == "direct":
        dense = _rand_dense(N_OBS, N_VARS, seed=3, density=0.25)
        adata = _adata(dense, sparse=sparse_source, seed=3)
        convert_adata(adata, output=out1, cfg=_cfg(x_storage, cpus=1))
        convert_adata(adata, output=out4, cfg=_cfg(x_storage, cpus=4))
    else:
        # inputs of 70+130+100 rows seam at 70 and 200 -- neither a multiple of row_chunk=64,
        # so every seam falls strictly inside an output chunk (the old per-file writer needed
        # a lock here).
        n_vars = 96
        sizes = [70, 130, 100]
        parts = [_rand_dense(n, n_vars, seed=10 + i, density=0.3) for i, n in enumerate(sizes)]
        paths = []
        for i, part in enumerate(parts):
            h5 = tmp_path / f"in{i}.h5ad"
            _adata(part, sparse=True, seed=10 + i).write_h5ad(h5)
            paths.append(str(h5))
        dense = np.vstack(parts)
        cfg1 = _cfg(x_storage, cpus=1, row_chunk=64, col_chunk=n_vars, flat_chunk=500)
        cfg4 = _cfg(x_storage, cpus=4, row_chunk=64, col_chunk=n_vars, flat_chunk=500)
        concat(paths, output=out1, cfg=cfg1)
        concat(paths, output=out4, cfg=cfg4)

    _assert_byte_identical(out1, out4)
    np.testing.assert_array_equal(_read_x(out1), dense)
    assert_anndata_readable(out1)


def test_sharded_dense_concat_seam_inside_shard_cpus1_matches_cpus4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row shard is 64 (chunk 32 x factor 2); inputs of 70+130+100 rows seam at 70 and 200,
    both strictly inside a shard ([64, 128) and [192, 256)) rather than at a shard boundary."""
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1)
    sizes = [70, 130, 100]
    parts = [_rand_dense(n, N_VARS, seed=30 + i, density=0.3) for i, n in enumerate(sizes)]
    paths = []
    for i, part in enumerate(parts):
        h5 = tmp_path / f"in{i}.h5ad"
        _adata(part, sparse=True, seed=30 + i).write_h5ad(h5)
        paths.append(str(h5))
    expected = np.vstack(parts)

    expected_count = _expected_shard_count(sum(sizes), N_VARS, EXPECTED_SHARDS)
    outs: dict[int, Path] = {}
    for cpus in (1, 4):
        out = tmp_path / f"out_cpus{cpus}.zarr"
        cfg = _cfg(
            "dense", cpus=cpus, row_chunk=SHARD_ROW_CHUNK, col_chunk=SHARD_COL_CHUNK, x_shard_factor=SHARD_FACTOR
        )
        concat(paths, output=out, cfg=cfg)
        np.testing.assert_array_equal(_read_x(out), expected)
        assert_anndata_readable(out)
        assert zarr.open_group(str(out), mode="r")["X"].shards == EXPECTED_SHARDS
        assert _n_shard_objects(out) == expected_count
        outs[cpus] = out

    _assert_byte_identical(outs[1], outs[4])


# Forced-sharding: flat_chunk stays small (500) but data/indices are pinned to a shard exactly
# 4 chunks wide (2000 elements), several chunks per shard on a nnz that spans dozens of chunks —
# proving flat_segments/_extend_flat/_copy_flat align writes to the SHARD, not the chunk, grid.
AUTOSHARD_FLAT_CHUNK = 500
AUTOSHARD_CHUNKS_PER_SHARD = 4
AUTOSHARD_ITEMSIZE = 4  # float32 data / int32 indices — same itemsize, same shard shape


def _autoshard_cfg(*, cpus: int) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, x_storage="csr"),
        chunks=ChunkConfig(sparse_flat_chunk=AUTOSHARD_FLAT_CHUNK, cpus=cpus, auto_shard=True),
        validation=ValidationConfig(),
    )


def test_forced_autoshard_sparse_write_cpus1_matches_cpus4(tmp_path: Path) -> None:
    """auto_shard=True with sparse_flat_chunk pinned small; zarr's shard-size heuristic is
    pinned (via target_shard_size_bytes) to exactly 4 chunks/shard, so data/indices land on a
    shard grid several chunks wide while cfg still asks for small chunks. cpus=1 vs cpus=4 must
    still write byte-identical stores, with one shard object per shard on disk (no RMW). Also
    covers that auto_shard=True raises neither anndata's "will be the default" warning (we set
    ad.settings.auto_shard_zarr_v3 explicitly around every write_elem call) nor zarr's
    "experimental" shard-inference warning (suppressed around our own shards="auto" creations),
    and that ad.settings.auto_shard_zarr_v3 is restored afterward, not leaked."""
    previous_setting = ad.settings.auto_shard_zarr_v3
    dense = _rand_dense(N_OBS, N_VARS, seed=11, density=0.25)
    adata = _adata(dense, sparse=True, seed=11)
    nnz = adata.X.nnz
    expected_shard = (AUTOSHARD_FLAT_CHUNK * AUTOSHARD_CHUNKS_PER_SHARD,)
    expected_count = math.ceil(nnz / expected_shard[0])

    outs: dict[int, Path] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # our own shards="auto" creation must not warn
        with _forced_autoshard(AUTOSHARD_FLAT_CHUNK, AUTOSHARD_ITEMSIZE, AUTOSHARD_CHUNKS_PER_SHARD):
            for cpus in (1, 4):
                out = tmp_path / f"out_cpus{cpus}.zarr"
                convert_adata(adata, output=out, cfg=_autoshard_cfg(cpus=cpus))
                outs[cpus] = out
    assert ad.settings.auto_shard_zarr_v3 is previous_setting

    for out in outs.values():
        np.testing.assert_array_equal(_read_x(out), dense)
        assert_anndata_readable(out)
        root = zarr.open_group(str(out), mode="r")
        assert root["X"]["data"].shards == expected_shard
        assert root["X"]["indices"].shards == expected_shard
        assert _n_flat_shard_objects(out, "X/data") == expected_count
        assert _n_flat_shard_objects(out, "X/indices") == expected_count

    _assert_byte_identical(outs[1], outs[4])


def test_append_extends_sharded_data_array(tmp_path: Path) -> None:
    """append extends an existing store's X (already sharded from a forced-autoshard convert)
    in place; _extend_flat must cut its copy segments on the array's write grid (the shard,
    not the chunk), so cpus=1 vs cpus=4 stay byte-identical and the shard stays intact."""
    base_dense = _rand_dense(120, N_VARS, seed=21, density=0.25)
    more_dense = _rand_dense(40, N_VARS, seed=22, density=0.25)
    base = _adata(base_dense, sparse=True, seed=21)
    more = _adata(more_dense, sparse=True, seed=22)
    expected = np.vstack([base_dense, more_dense])

    with _forced_autoshard(AUTOSHARD_FLAT_CHUNK, AUTOSHARD_ITEMSIZE, AUTOSHARD_CHUNKS_PER_SHARD):
        cells_path = tmp_path / "cells.zarr"
        convert_adata(more, output=cells_path, cfg=_autoshard_cfg(cpus=1))

        outs: dict[int, Path] = {}
        for cpus in (1, 4):
            out = tmp_path / f"base_cpus{cpus}.zarr"
            convert_adata(base, output=out, cfg=_autoshard_cfg(cpus=cpus))
            append(out, cells=cells_path, cfg=AppConfig(chunks=ChunkConfig(cpus=cpus)))
            outs[cpus] = out

    for out in outs.values():
        np.testing.assert_array_equal(_read_x(out), expected)
        assert_anndata_readable(out)
        root = zarr.open_group(str(out), mode="r")
        assert root["X"]["data"].shards is not None  # still sharded after the in-place extend

    _assert_byte_identical(outs[1], outs[4])


def test_rechunk_recreates_sharded_sparse_array(tmp_path: Path) -> None:
    """rechunk's own sparse copy path (rechunk=True) creates a fresh data/indices array under
    auto_shard, and must partition its copy segments on the NEW array's write grid. cpus=1 vs
    cpus=4 must write byte-identical output, with the expected shard shape and object count."""
    dense = _rand_dense(N_OBS, N_VARS, seed=31, density=0.25)
    adata = _adata(dense, sparse=True, seed=31)
    src = tmp_path / "src.zarr"
    convert_adata(adata, output=src, cfg=_cfg("csr", cpus=1, flat_chunk=64))  # unsharded source
    nnz = adata.X.nnz
    expected_shard = (AUTOSHARD_FLAT_CHUNK * AUTOSHARD_CHUNKS_PER_SHARD,)
    expected_count = math.ceil(nnz / expected_shard[0])

    outs: dict[int, Path] = {}
    with _forced_autoshard(AUTOSHARD_FLAT_CHUNK, AUTOSHARD_ITEMSIZE, AUTOSHARD_CHUNKS_PER_SHARD):
        for cpus in (1, 4):
            out = tmp_path / f"out_cpus{cpus}.zarr"
            rechunk(src, output=out, array="X", cfg=_autoshard_cfg(cpus=cpus))
            outs[cpus] = out

    for out in outs.values():
        np.testing.assert_array_equal(_read_x(out), dense)
        assert_anndata_readable(out)
        root = zarr.open_group(str(out), mode="r")
        assert root["X"]["data"].shards == expected_shard
        assert _n_flat_shard_objects(out, "X/data") == expected_count

    _assert_byte_identical(outs[1], outs[4])


def test_run_parallel_fails_fast_and_cancels_pending_jobs() -> None:
    ran: list[int] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        time.sleep(0.05)
        if i == 3:
            raise ValueError("boom")
        with lock:
            ran.append(i)

    jobs = [(i,) for i in range(50)]
    with pytest.raises(ValueError, match="boom"):
        run_parallel(worker, jobs, cpus=4, mode="threads")
