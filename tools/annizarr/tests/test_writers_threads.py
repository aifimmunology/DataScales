"""cpus=1 (serial) vs cpus=4 (thread pool) must write byte-identical stores.

Every in-memory writer path routes disjoint, write-grid-aligned blocks/segments through
``_runtime.run_parallel``; these tests pin chunk/shard sizes so several blocks land on each
axis (with a ragged last one) and compare the two cpu counts file-for-file, not just by value.
"""

from __future__ import annotations

import filecmp
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import annizarr._layout as _layout
from _readable import assert_anndata_readable
from annizarr._ops import concat, convert_adata, convert_h5ad
from annizarr._runtime import progress
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


def _cfg(x_storage: str, *, cpus: int, row_chunk: int = 64, col_chunk: int = 48, flat_chunk: int = 500) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, x_storage=x_storage),
        chunks=ChunkConfig(x_row_chunk=row_chunk, x_col_chunk=col_chunk, sparse_flat_chunk=flat_chunk, cpus=cpus),
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


@pytest.mark.parametrize(
    ("x_storage", "sparse_source"),
    [
        pytest.param("dense", False, id="dense_from_dense"),
        pytest.param("dense", True, id="dense_from_sparse"),
        pytest.param("csr", True, id="sparse_to_csr"),
        pytest.param("csc", True, id="sparse_to_csc"),
    ],
)
def test_inmemory_write_cpus1_matches_cpus4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, x_storage: str, sparse_source: bool
) -> None:
    # force several flat segments too (not just several dense blocks) for the sparse cases
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1)

    dense = _rand_dense(N_OBS, N_VARS, seed=3, density=0.25)
    adata = _adata(dense, sparse=sparse_source, seed=3)

    out1 = tmp_path / "cpus1.zarr"
    out4 = tmp_path / "cpus4.zarr"
    convert_adata(adata, output=out1, cfg=_cfg(x_storage, cpus=1))
    convert_adata(adata, output=out4, cfg=_cfg(x_storage, cpus=4))

    _assert_byte_identical(out1, out4)
    np.testing.assert_array_equal(_read_x(out1), dense)
    assert_anndata_readable(out1)


@pytest.mark.parametrize("x_storage", ["dense", "csr"])
def test_concat_seam_inside_row_chunk_cpus1_matches_cpus4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, x_storage: str
) -> None:
    """Inputs of 70+130+100 rows seam at 70 and 200 — neither a multiple of row_chunk=64, so
    every seam falls strictly inside an output chunk. Values must be correct, and cpus=1 vs
    cpus=4 must write byte-identical stores (the old per-file writer needed a lock here)."""
    monkeypatch.setattr(_layout, "BATCH_BYTES", 1)
    n_vars = 96
    sizes = [70, 130, 100]
    parts = [_rand_dense(n, n_vars, seed=10 + i, density=0.3) for i, n in enumerate(sizes)]
    paths = []
    for i, part in enumerate(parts):
        h5 = tmp_path / f"in{i}.h5ad"
        _adata(part, sparse=True, seed=10 + i).write_h5ad(h5)
        paths.append(str(h5))
    expected = np.vstack(parts)

    outs: dict[int, Path] = {}
    for cpus in (1, 4):
        out = tmp_path / f"out_cpus{cpus}.zarr"
        concat(paths, output=out, cfg=_cfg(x_storage, cpus=cpus, row_chunk=64, col_chunk=n_vars, flat_chunk=500))
        np.testing.assert_array_equal(_read_x(out), expected)
        assert_anndata_readable(out)
        outs[cpus] = out

    _assert_byte_identical(outs[1], outs[4])


def test_sparse_write_nnz_not_multiple_of_flat_chunk(tmp_path: Path) -> None:
    dense = _rand_dense(137, 53, seed=42, density=0.37)
    adata = _adata(dense, sparse=True, seed=42)
    nnz = adata.X.nnz
    assert nnz >= 2
    flat_chunk = nnz - 1  # guarantees a 1-element ragged last segment

    h5 = tmp_path / "in.h5ad"
    adata.write_h5ad(h5)
    out = tmp_path / "out.zarr"
    convert_h5ad(h5, output=out, cfg=_cfg("csr", cpus=4, flat_chunk=flat_chunk))

    np.testing.assert_array_equal(_read_x(out), dense)
    assert_anndata_readable(out)


def test_progress_logs_final_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="annizarr._runtime")
    tick = progress(5, "Writing test")
    for _ in range(5):
        tick()
    messages = [r.message for r in caplog.records]
    assert any("5/5" in m and "done" in m for m in messages)
