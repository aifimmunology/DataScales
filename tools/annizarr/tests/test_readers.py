from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

import annizarr._core._layout as _layout
from annizarr._core._config import AppConfig
from annizarr._sources._readers import (
    ConcatReader,
    CSRH5Reader,
    CSRMemoryReader,
    CSRZarrReader,
    DenseH5Reader,
    DenseMemoryReader,
    DenseZarrReader,
)
from annizarr.errors import ConversionError

# 6x5 float32 fixture with an empty row (2) and an empty column (3), so every reader's
# nnz-counting/indptr/band logic is exercised on a non-trivial sparsity pattern.
DENSE = np.array(
    [
        [1, 0, 2, 0, 3],
        [0, 4, 0, 0, 5],
        [0, 0, 0, 0, 0],
        [6, 0, 7, 0, 0],
        [0, 8, 0, 0, 9],
        [10, 0, 0, 0, 11],
    ],
    dtype=np.float32,
)
CSR = sp.csr_matrix(DENSE)


def _dense_h5_reader(tmp_path: Path) -> DenseH5Reader:
    path = tmp_path / "dense.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=DENSE)
    return DenseH5Reader(str(path), "X")


def _csr_h5_reader(tmp_path: Path) -> CSRH5Reader:
    path = tmp_path / "in.h5ad"
    ad.AnnData(X=CSR.copy()).write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    reader = CSRH5Reader(backed.X.group.file.filename, backed.X.group.name)
    backed.file.close()
    return reader


def _dense_zarr_reader(tmp_path: Path) -> DenseZarrReader:
    group = zarr.open_group(str(tmp_path / "dense.zarr"), mode="w")
    arr = group.require_array("X", shape=DENSE.shape, dtype=DENSE.dtype, chunks=DENSE.shape)
    arr[:] = DENSE
    return DenseZarrReader(arr)


def _csr_zarr_reader(tmp_path: Path) -> CSRZarrReader:
    group = zarr.open_group(str(tmp_path / "csr.zarr"), mode="w")
    g = group.require_group("X")
    g.attrs["encoding-type"] = "csr_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = list(CSR.shape)
    g.require_array("data", shape=CSR.data.shape, dtype=CSR.data.dtype, chunks=CSR.data.shape)[:] = CSR.data
    g.require_array("indices", shape=CSR.indices.shape, dtype=CSR.indices.dtype, chunks=CSR.indices.shape)[:] = (
        CSR.indices
    )
    g.require_array("indptr", shape=CSR.indptr.shape, dtype=np.int64, chunks=CSR.indptr.shape)[:] = CSR.indptr
    return CSRZarrReader(g)


_DENSE_BUILDERS = {
    "memory": lambda tmp_path: DenseMemoryReader(DENSE.copy()),
    "h5": _dense_h5_reader,
    "zarr": _dense_zarr_reader,
}
_CSR_BUILDERS = {
    "memory": lambda tmp_path: CSRMemoryReader(CSR.copy()),
    "h5": _csr_h5_reader,
    "zarr": _csr_zarr_reader,
}


@pytest.fixture(autouse=True)
def _tiny_batch_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    # forces dense_block/flat/indptr to actually band/segment on this tiny fixture, instead
    # of covering it in one shot under the real 256 MiB default.
    monkeypatch.setattr(_layout, "BATCH_BYTES", 24)


@pytest.mark.parametrize("kind", ["memory", "h5", "zarr"])
def test_dense_reader_contract(tmp_path: Path, kind: str) -> None:
    reader = _DENSE_BUILDERS[kind](tmp_path)
    assert reader.shape == DENSE.shape
    assert reader.dtype == DENSE.dtype
    assert reader.lazy == (kind != "memory")

    np.testing.assert_array_equal(reader.dense_block(1, 4, 1, 4), DENSE[1:4, 1:4])
    np.testing.assert_array_equal(reader.csr_rows(1, 4).toarray(), DENSE[1:4])
    np.testing.assert_array_equal(np.diff(reader.indptr), np.count_nonzero(DENSE, axis=1))

    s0, s1 = 3, 8
    data, idx = reader.flat(s0, s1)
    np.testing.assert_array_equal(data, CSR.data[s0:s1])
    np.testing.assert_array_equal(idx, CSR.indices[s0:s1])


@pytest.mark.parametrize("kind", ["memory", "h5", "zarr"])
def test_csr_reader_contract(tmp_path: Path, kind: str) -> None:
    reader = _CSR_BUILDERS[kind](tmp_path)
    assert reader.shape == CSR.shape
    assert reader.dtype == CSR.dtype
    assert reader.lazy == (kind != "memory")

    np.testing.assert_array_equal(reader.dense_block(1, 4, 1, 4), DENSE[1:4, 1:4])
    np.testing.assert_array_equal(reader.csr_rows(1, 4).toarray(), DENSE[1:4])
    np.testing.assert_array_equal(reader.indptr, CSR.indptr)

    s0, s1 = 3, 8
    data, idx = reader.flat(s0, s1)
    np.testing.assert_array_equal(data, CSR.data[s0:s1])
    np.testing.assert_array_equal(idx, CSR.indices[s0:s1])


def test_concat_reader_stitches_across_a_seam_inside_a_band_and_a_flat_segment(tmp_path: Path) -> None:
    top = CSRMemoryReader(sp.csr_matrix(DENSE[:3]))
    bottom = CSRZarrReader(_csr_group_for(DENSE[3:], tmp_path))
    reader = ConcatReader([top, bottom])

    assert reader.shape == DENSE.shape
    np.testing.assert_array_equal(np.diff(reader.indptr), np.count_nonzero(DENSE, axis=1))

    # the seam (row 3) falls inside this row band
    np.testing.assert_array_equal(reader.dense_block(1, 5, 0, 5), DENSE[1:5])
    np.testing.assert_array_equal(reader.csr_rows(1, 5).toarray(), DENSE[1:5])

    # the seam's nnz offset (5, from CSR.indptr[3]) falls inside this flat segment
    s0, s1 = 3, 8
    data, idx = reader.flat(s0, s1)
    np.testing.assert_array_equal(data, CSR.data[s0:s1])
    np.testing.assert_array_equal(idx, CSR.indices[s0:s1])

    reader.close()


def _csr_group_for(sub_dense: np.ndarray, tmp_path: Path) -> zarr.Group:
    sub = sp.csr_matrix(sub_dense)
    group = zarr.open_group(str(tmp_path / "bottom.zarr"), mode="w")
    g = group.require_group("X")
    g.attrs["encoding-type"] = "csr_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = list(sub.shape)
    g.require_array("data", shape=sub.data.shape, dtype=sub.data.dtype, chunks=sub.data.shape or (1,))[:] = sub.data
    g.require_array("indices", shape=sub.indices.shape, dtype=sub.indices.dtype, chunks=sub.indices.shape or (1,))[
        :
    ] = sub.indices
    g.require_array("indptr", shape=sub.indptr.shape, dtype=np.int64, chunks=sub.indptr.shape)[:] = sub.indptr
    return g


def test_csr_band_tiles_slice_columns_before_densifying(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, int]] = []
    real = sp.csr_matrix.toarray

    def recording(self: sp.csr_matrix, *args: object, **kwargs: object) -> np.ndarray:
        seen.append(self.shape)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", recording)
    band = CSRMemoryReader(CSR.copy()).dense_band(0, 6)
    np.testing.assert_array_equal(np.hstack([band.tile(0, 2), band.tile(2, 5)]), DENSE)
    assert seen == [(6, 2), (6, 3)]


def test_dense_h5_reader_grain_reflects_source_chunking(tmp_path: Path) -> None:
    chunked = tmp_path / "chunked.h5"
    with h5py.File(chunked, "w") as f:
        f.create_dataset("X", data=DENSE, chunks=(2, DENSE.shape[1]))
    reader = DenseH5Reader(str(chunked), "X")
    assert reader.grain == 2
    reader.close()

    unchunked = tmp_path / "unchunked.h5"
    with h5py.File(unchunked, "w") as f:
        f.create_dataset("X", data=DENSE)  # contiguous by default (no chunks=)
    reader2 = DenseH5Reader(str(unchunked), "X")
    assert reader2.grain is None
    reader2.close()


def _manual_csr_h5(path: Path, *, chunks: tuple[int, ...] | None) -> None:
    with h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = list(CSR.shape)
        g.create_dataset("data", data=CSR.data, chunks=chunks)
        g.create_dataset("indices", data=CSR.indices, chunks=chunks)
        g.create_dataset("indptr", data=CSR.indptr)


def test_csr_h5_reader_grain_reflects_source_chunking(tmp_path: Path) -> None:
    chunked = tmp_path / "chunked.h5"
    _manual_csr_h5(chunked, chunks=(3,))
    reader = CSRH5Reader(str(chunked), "X")
    assert reader.grain == 3
    reader.close()

    unchunked = tmp_path / "unchunked.h5"
    _manual_csr_h5(unchunked, chunks=None)
    reader2 = CSRH5Reader(str(unchunked), "X")
    assert reader2.grain is None
    reader2.close()


def test_as_reader_rejects_non_matrix_inputs() -> None:
    from annizarr._sources._readers import as_reader

    cfg = AppConfig()
    with pytest.raises(ConversionError, match="1-D"):
        as_reader(np.zeros(5, dtype=np.float32), cfg=cfg)
    with pytest.raises(ConversionError, match="coo"):
        as_reader(sp.coo_matrix(DENSE), cfg=cfg)
    with pytest.raises(ConversionError, match="masked"):
        as_reader(np.ma.masked_array(DENSE), cfg=cfg)
    assert as_reader(DENSE, cfg=AppConfig()).cpus == AppConfig().chunks.cpus


@pytest.mark.parametrize("builder", ["h5_dense", "h5_csr", "zarr_dense", "zarr_csr"])
def test_lazy_reader_survives_pickling(tmp_path: Path, builder: str) -> None:
    readers = {
        "h5_dense": _dense_h5_reader,
        "h5_csr": _csr_h5_reader,
        "zarr_dense": _dense_zarr_reader,
        "zarr_csr": _csr_zarr_reader,
    }
    reader = readers[builder](tmp_path)
    assert reader.shape  # force the handle open before pickling

    restored = pickle.loads(pickle.dumps(reader))
    np.testing.assert_array_equal(restored.dense_block(0, 3, 0, 5), DENSE[0:3])
    assert restored.shape == reader.shape
    restored.close()
    reader.close()
