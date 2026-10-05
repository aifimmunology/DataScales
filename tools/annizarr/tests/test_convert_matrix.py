from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

import annizarr._core._layout as _layout
from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.ops import convert_h5ad

# A 60x40 float32 fixture with two fully-empty rows and one fully-empty column, so the
# (format x lazy x x_storage) grid below exercises a reader's indptr/band/flat math on real
# sparsity, not just a dense-everywhere matrix.
N_OBS, N_VARS = 60, 40


def _fixture() -> np.ndarray:
    rng = np.random.default_rng(0)
    dense = (rng.random((N_OBS, N_VARS), dtype=np.float32) < 0.3) * rng.random((N_OBS, N_VARS), dtype=np.float32)
    dense = dense.astype(np.float32)
    dense[5, :] = 0.0
    dense[41, :] = 0.0
    dense[:, 17] = 0.0
    return dense


DENSE = _fixture()

_ENCODING = {"dense": "array", "csr": "csr_matrix", "csc": "csc_matrix"}


def _write_input(path: Path, fmt: str) -> None:
    if fmt == "dense":
        x: Any = DENSE.copy()
    elif fmt == "csr":
        x = sp.csr_matrix(DENSE)
    else:
        x = sp.csc_matrix(DENSE)
    ad.AnnData(X=x).write_h5ad(path)


def _slice_count(key: Any, shape: tuple[int, ...]) -> int:
    if not isinstance(key, tuple):
        key = (key,)
    count = 1
    for axis, k in enumerate(key):
        dim = shape[axis] if axis < len(shape) else 1
        if isinstance(k, slice):
            count *= len(range(*k.indices(dim)))
        elif isinstance(k, int | np.integer):
            count *= 1
        else:
            count *= dim  # fancy/boolean indexing: count conservatively as the whole axis
    return count


class _GetitemRecorder:
    """Patches h5py.Dataset.__getitem__ to record every (basename, requested count, full
    count) for the X-bearing datasets ('X' dense, or a CSR/CSC group's 'data'/'indices'),
    so a lazy conversion can be checked for genuine banding rather than one whole-array read.
    'indptr' is excluded: reading it once, in full, is expected and cheap."""

    def __init__(self) -> None:
        self.records: list[tuple[str, int, int]] = []

    def __call__(self, dataset: h5py.Dataset, key: Any, _orig: Any) -> Any:
        name = dataset.name.rsplit("/", 1)[-1]
        if name in ("X", "data", "indices"):
            total = 1
            for d in dataset.shape:
                total *= d
            self.records.append((name, _slice_count(key, dataset.shape), total))
        return _orig(dataset, key)


@pytest.fixture
def _getitem_recorder(monkeypatch: pytest.MonkeyPatch) -> _GetitemRecorder:
    recorder = _GetitemRecorder()
    orig = h5py.Dataset.__getitem__

    def patched(self: h5py.Dataset, key: Any) -> Any:
        return recorder(self, key, orig)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", patched)
    return recorder


@pytest.mark.parametrize("input_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("x_storage", ["dense", "csr", "csc"])
def test_convert_matrix_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _getitem_recorder: _GetitemRecorder,
    input_format: str,
    lazy: bool,
    x_storage: str,
) -> None:
    monkeypatch.setattr(_layout, "BATCH_BYTES", 512)  # forces real banding on this small fixture

    input_h5 = tmp_path / "in.h5ad"
    _write_input(input_h5, input_format)
    out = tmp_path / "out.zarr"
    # cpus=1 pinned: a lazy conversion fans out across worker processes by default (item 2),
    # and the h5py read recorder below only patches __getitem__ in THIS process.
    cfg = AppConfig(
        io=IOConfig(x_storage=x_storage, lazy=lazy),
        chunks=ChunkConfig(x_row_chunk=7, x_col_chunk=6, sparse_flat_chunk=11, cpus=1),
    )
    convert_h5ad(str(input_h5), output=str(out), cfg=cfg)

    got = ad.read_zarr(str(out))
    got_dense = np.asarray(got.X.todense() if sp.issparse(got.X) else got.X)
    np.testing.assert_array_equal(got_dense, DENSE)

    root = zarr.open_group(str(out), mode="r")
    assert root["X"].attrs["encoding-type"] == _ENCODING[x_storage]
    assert root["X"].attrs["encoding-version"]

    if lazy:
        by_name: dict[str, list[tuple[int, int]]] = {}
        for name, count, total in _getitem_recorder.records:
            by_name.setdefault(name, []).append((count, total))
        assert by_name, "expected at least one recorded h5py read for a lazy conversion"
        for name, calls in by_name.items():
            assert len(calls) > 1, f"expected several bounded reads of {name!r}, got {calls}"
            bounded = [count < total for count, total in calls]
            # write_transposed_sparse (the CSC writer, or as_reader's lazy-CSC-to-CSR
            # normalisation) runs whenever x_storage=="csc" or the input itself is CSC. Its
            # own nnz-counting pass is genuinely banded (several of these calls are partial,
            # proven by `any` below); its final per-band assembly pass uses a
            # `max(1_000, ...)` row-count floor (pre-existing, shared with the old bucket
            # engine, not specific to this refactor) that reads everything in one shot once
            # the input is under ~1000 rows, as this fixture deliberately is.
            uses_transpose_engine = x_storage == "csc" or input_format == "csc"
            if uses_transpose_engine:
                assert any(bounded), f"expected at least one bounded read of {name!r}, got {calls}"
            else:
                assert all(bounded), f"a single read of {name!r} covered the whole array ({calls}) — not streamed"
