from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from annizarr._core import _layout
from annizarr._core._runtime import map_parallel
from annizarr._core._zarr import get_array, shape_attr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import scipy.sparse as sp
    import zarr
    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CSRH5Reader",
    "CSRMemoryReader",
    "CSRZarrReader",
    "ConcatReader",
    "DenseBand",
    "DenseH5Reader",
    "DenseMemoryReader",
    "DenseZarrReader",
    "Reader",
    "as_reader",
]


class DenseBand(Protocol):
    def tile(self, c0: int, c1: int) -> NDArray[Any]: ...


class _ArrayBand:
    __slots__ = ("_rows",)

    def __init__(self, rows: Any) -> None:
        self._rows = rows

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._rows[:, c0:c1])


class _TiledBand:
    # each tile read straight from the lazy array: used when a whole row band would exceed BATCH_BYTES
    __slots__ = ("_arr", "_r0", "_r1")

    def __init__(self, arr: Any, r0: int, r1: int) -> None:
        self._arr = arr
        self._r0 = r0
        self._r1 = r1

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._arr[self._r0 : self._r1, c0:c1])


class _CSRBand:
    __slots__ = ("_rows",)

    def __init__(self, rows: sp.csr_matrix) -> None:
        self._rows = rows

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        sub = self._rows if (c0 == 0 and c1 == self._rows.shape[1]) else self._rows[:, c0:c1]
        return np.asarray(sub.toarray())


class _StitchedBand:
    __slots__ = ("_dtype", "_n_rows", "_parts")

    def __init__(self, parts: list[tuple[int, int, DenseBand]], n_rows: int, dtype: np.dtype[Any]) -> None:
        self._parts = parts
        self._n_rows = n_rows
        self._dtype = dtype

    def tile(self, c0: int, c1: int) -> NDArray[Any]:
        out = np.empty((self._n_rows, c1 - c0), dtype=self._dtype)
        for lo, hi, band in self._parts:
            out[lo:hi] = band.tile(c0, c1)
        return out


class Reader(ABC):
    def __init__(self) -> None:
        self._indptr_cache: NDArray[np.int64] | None = None
        self.cpus = 1

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]: ...

    @property
    @abstractmethod
    def lazy(self) -> bool: ...

    @property
    @abstractmethod
    def thread_safe(self) -> bool: ...

    @property
    def grain(self) -> int | None:
        # the source's own chunk size along the read axis: rows of one source chunk for a
        # dense reader, flat elements of one source chunk of `data` for a CSR-like reader.
        # None when the source isn't chunked (memory readers, a concat of mixed sources).
        return None

    @property
    def indptr(self) -> NDArray[np.int64]:
        if self._indptr_cache is None:
            self._indptr_cache = self._compute_indptr()
        return self._indptr_cache

    @abstractmethod
    def _compute_indptr(self) -> NDArray[np.int64]: ...

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    @abstractmethod
    def _row_bytes(self) -> int: ...

    @property
    def rows_per_batch(self) -> int:
        step = max(1, _layout.BATCH_BYTES // max(1, self._row_bytes()))
        grain = self.grain
        if not grain:
            return step
        if step <= grain:
            return grain
        return (step // grain) * grain

    @abstractmethod
    def dense_band(self, r0: int, r1: int) -> DenseBand: ...

    def dense_block(self, r0: int, r1: int, c0: int, c1: int) -> NDArray[Any]:
        return self.dense_band(r0, r1).tile(c0, c1)

    @abstractmethod
    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix: ...

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        # default path (dense readers): band the overlapping rows by rows_per_batch, convert
        # each sub-band to CSR, and copy out only the part inside [s0, s1) — never densifies
        # more than one native row band at a time.
        indptr = self.indptr
        n_rows = self.shape[0]
        r0 = max(int(np.searchsorted(indptr, s0, side="right")) - 1, 0)
        r1 = min(max(int(np.searchsorted(indptr, s1, side="left")), r0 + 1), n_rows)
        step = self.rows_per_batch

        data_parts: list[NDArray[Any]] = []
        idx_parts: list[NDArray[np.int32]] = []
        row = r0
        while row < r1:
            row_end = min(row + step, r1)
            band = self.csr_rows(row, row_end)
            band_s0 = int(indptr[row])
            lo = max(s0, band_s0) - band_s0
            hi = min(s1, int(indptr[row_end])) - band_s0
            data_parts.append(np.asarray(band.data[lo:hi]))
            idx_parts.append(np.asarray(band.indices[lo:hi], dtype=np.int32))
            row = row_end

        if not data_parts:
            return np.empty(0, dtype=self.dtype), np.empty(0, dtype=np.int32)
        return np.concatenate(data_parts), np.concatenate(idx_parts)

    def close(self) -> None:  # noqa: B027  # intentional default no-op; not every reader holds a handle
        pass


class _DenseReaderMixin(Reader):
    def _row_bytes(self) -> int:
        return self.shape[1] * self.dtype.itemsize

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        return sp.csr_matrix(self.dense_block(r0, r1, 0, self.shape[1]))

    def _count_band(self, r0: int, r1: int) -> NDArray[np.int64]:
        band = self.dense_block(r0, r1, 0, self.shape[1])
        return np.asarray(np.count_nonzero(band, axis=1), dtype=np.int64)

    def _compute_indptr(self) -> NDArray[np.int64]:
        n_rows, n_cols = self.shape
        step = max(1, _layout.BATCH_BYTES // max(1, n_cols * self.dtype.itemsize))
        bands = _layout.band_plan(n_rows, step)
        if not bands:
            return np.zeros(1, dtype=np.int64)
        jobs = [(r0, r1) for r0, r1 in bands]
        mode: Literal["threads", "processes"] = "threads" if self.thread_safe else "processes"
        counts = map_parallel(self._count_band, jobs, self.cpus, mode=mode)
        return np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.int64)

    def _lazy_band(self, arr: Any, r0: int, r1: int) -> DenseBand:
        if (r1 - r0) * self._row_bytes() <= _layout.BATCH_BYTES:
            return _ArrayBand(np.asarray(arr[r0:r1]))
        return _TiledBand(arr, r0, r1)


class _CSRReaderMixin(Reader):
    def _row_bytes(self) -> int:
        n_rows = self.shape[0]
        return max(1, self.nnz // max(1, n_rows)) * (self.dtype.itemsize + 4)

    @abstractmethod
    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix: ...

    @abstractmethod
    def _flat_data(self) -> Any: ...

    @abstractmethod
    def _flat_indices(self) -> Any: ...

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return _CSRBand(self._fetch_csr_band(r0, r1))

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._fetch_csr_band(r0, r1)

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        data = self._flat_data()
        indices = self._flat_indices()
        return np.asarray(data[s0:s1]), np.asarray(indices[s0:s1], dtype=np.int32)


class DenseMemoryReader(_DenseReaderMixin):
    def __init__(self, arr: NDArray[Any]) -> None:
        super().__init__()
        self._arr = arr

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._arr.shape[0]), int(self._arr.shape[1]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._arr.dtype

    @property
    def lazy(self) -> bool:
        return False

    @property
    def thread_safe(self) -> bool:
        return True

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return _ArrayBand(self._arr[r0:r1])


class CSRMemoryReader(_CSRReaderMixin):
    def __init__(self, mat: sp.csr_matrix) -> None:
        super().__init__()
        self._mat = mat

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._mat.shape[0]), int(self._mat.shape[1]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._mat.dtype)  # type: ignore[no-any-return]  # scipy-stubs: csr_matrix.dtype is Any

    @property
    def lazy(self) -> bool:
        return False

    @property
    def thread_safe(self) -> bool:
        return True

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._mat.indptr, dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._mat[r0:r1]

    def _flat_data(self) -> NDArray[Any]:
        return np.asarray(self._mat.data)

    def _flat_indices(self) -> NDArray[Any]:
        return np.asarray(self._mat.indices)


def _clear_h5_state(state: dict[str, Any]) -> dict[str, Any]:
    for key in ("_file", "_dataset_obj"):
        if key in state:
            state[key] = None
    return state


class DenseH5Reader(_DenseReaderMixin):
    def __init__(self, path: str, dataset: str) -> None:
        super().__init__()
        self._path = path
        self._dataset_name = dataset
        self._file: Any = None
        self._shape_val: tuple[int, int] | None = None
        self._dtype_val: np.dtype[Any] | None = None

    def _ds(self) -> Any:
        if self._file is None:
            import h5py

            self._file = h5py.File(self._path, "r")
        return self._file[self._dataset_name]

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape_val is None:
            self._shape_val = (int(self._ds().shape[0]), int(self._ds().shape[1]))
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._dtype_val is None:
            self._dtype_val = self._ds().dtype
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return False  # h5py is not thread-safe

    @property
    def grain(self) -> int | None:
        chunks = self._ds().chunks
        return int(chunks[0]) if chunks else None

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return self._lazy_band(self._ds(), r0, r1)

    def dense_block(self, r0: int, r1: int, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._ds()[r0:r1, c0:c1])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict[str, Any]:
        return _clear_h5_state(self.__dict__.copy())

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class CSRH5Reader(_CSRReaderMixin):
    def __init__(self, path: str, group: str) -> None:
        super().__init__()
        self._path = path
        self._group_name = group
        self._file: Any = None
        self._dataset_obj: Any = None
        self._shape_val: tuple[int, int] | None = None
        self._dtype_val: np.dtype[Any] | None = None

    def _dataset(self) -> Any:
        if self._dataset_obj is None:
            import h5py
            from anndata.io import sparse_dataset

            self._file = h5py.File(self._path, "r")
            self._dataset_obj = sparse_dataset(self._file[self._group_name])
        return self._dataset_obj

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape_val is None:
            shape = self._dataset().shape
            self._shape_val = (int(shape[0]), int(shape[1]))
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._dtype_val is None:
            self._dtype_val = self._dataset().dtype
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return False  # h5py is not thread-safe

    @property
    def grain(self) -> int | None:
        chunks = self._flat_data().chunks
        return int(chunks[0]) if chunks else None

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._dataset().group["indptr"][:], dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        return self._dataset()[r0:r1]

    def _flat_data(self) -> Any:
        return self._dataset().group["data"]

    def _flat_indices(self) -> Any:
        return self._dataset().group["indices"]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset_obj = None

    def __getstate__(self) -> dict[str, Any]:
        return _clear_h5_state(self.__dict__.copy())

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class DenseZarrReader(_DenseReaderMixin):
    def __init__(self, arr: zarr.Array[Any]) -> None:
        super().__init__()
        self._arr = arr

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._arr.shape[0]), int(self._arr.shape[1]))

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._arr.dtype

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return True  # zarr is thread-safe

    @property
    def grain(self) -> int | None:
        return int(self._arr.chunks[0])

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        return self._lazy_band(self._arr, r0, r1)

    def dense_block(self, r0: int, r1: int, c0: int, c1: int) -> NDArray[Any]:
        return np.asarray(self._arr[r0:r1, c0:c1])


class CSRZarrReader(_CSRReaderMixin):
    def __init__(self, group: zarr.Group, *, _tmp_dir: Any = None) -> None:
        super().__init__()
        self._group = group
        self._shape_val = shape_attr(group)
        self._dtype_val = get_array(group, "data").dtype
        self._tmp_dir = _tmp_dir

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return True  # zarr is thread-safe

    @property
    def grain(self) -> int | None:
        return int(get_array(self._group, "data").chunks[0])

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(get_array(self._group, "indptr")[:], dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        indptr = self.indptr
        s0, s1 = int(indptr[r0]), int(indptr[r1])
        data = np.asarray(get_array(self._group, "data")[s0:s1])
        indices = np.asarray(get_array(self._group, "indices")[s0:s1])
        sub_indptr = (indptr[r0 : r1 + 1] - s0).astype(np.int64)
        return sp.csr_matrix((data, indices, sub_indptr), shape=(r1 - r0, self._shape_val[1]))

    def _flat_data(self) -> Any:
        return get_array(self._group, "data")

    def _flat_indices(self) -> Any:
        return get_array(self._group, "indices")

    def close(self) -> None:
        if self._tmp_dir is not None:
            import shutil

            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def __getstate__(self) -> dict[str, Any]:
        # a pickled (worker-process) copy doesn't own the temp dir: only the original,
        # parent-process reader deletes it (once, after every job has finished) — closing a
        # worker's own copy must not race-delete a directory sibling jobs still need.
        state = self.__dict__.copy()
        state["_tmp_dir"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class _SwappedCSRView(_CSRReaderMixin):
    # a raw CSC group's own (data, indices, indptr) reinterpreted as CSR over the transposed
    # shape (scipy's own `.T` trick, applied lazily): feeds a CSC source into
    # write_transposed_sparse without ever densifying or copying it. Internal to as_reader's
    # CSC normalisation; never part of the public reader surface.
    def __init__(self, group: Any, true_shape: tuple[int, int], dtype: np.dtype[Any]) -> None:
        import zarr

        super().__init__()
        self._group = group
        self._shape_val = (true_shape[1], true_shape[0])
        self._dtype_val = dtype
        self._thread_safe = isinstance(group, zarr.Group)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape_val

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return True

    @property
    def thread_safe(self) -> bool:
        return self._thread_safe

    def _compute_indptr(self) -> NDArray[np.int64]:
        return np.asarray(self._group["indptr"][:], dtype=np.int64)

    def _fetch_csr_band(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        indptr = self.indptr
        s0, s1 = int(indptr[r0]), int(indptr[r1])
        data = np.asarray(self._group["data"][s0:s1])
        indices = np.asarray(self._group["indices"][s0:s1])
        sub_indptr = (indptr[r0 : r1 + 1] - s0).astype(np.int64)
        return sp.csr_matrix((data, indices, sub_indptr), shape=(r1 - r0, self._shape_val[1]))

    def _flat_data(self) -> Any:
        return self._group["data"]

    def _flat_indices(self) -> Any:
        return self._group["indices"]


class ConcatReader(Reader):
    def __init__(self, readers: Sequence[Reader]) -> None:
        super().__init__()
        if not readers:
            raise ConversionError("ConcatReader requires at least one reader.")
        n_cols = readers[0].shape[1]
        dtype = readers[0].dtype
        for r in readers[1:]:
            if r.shape[1] != n_cols:
                raise ConversionError(f"ConcatReader: column count mismatch ({r.shape[1]} vs {n_cols}).")
            if r.dtype != dtype:
                raise ConversionError(f"ConcatReader: dtype mismatch ({r.dtype} vs {dtype}).")
        self._readers = tuple(readers)
        self._n_cols = n_cols
        self._dtype_val = dtype
        self._row_offsets = tuple(int(v) for v in np.cumsum([0, *(r.shape[0] for r in readers)]))

    @property
    def shape(self) -> tuple[int, int]:
        return (self._row_offsets[-1], self._n_cols)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype_val

    @property
    def lazy(self) -> bool:
        return any(r.lazy for r in self._readers)

    @property
    def thread_safe(self) -> bool:
        return all(r.thread_safe for r in self._readers)

    def _row_bytes(self) -> int:
        return max(r._row_bytes() for r in self._readers)

    def _compute_indptr(self) -> NDArray[np.int64]:
        chunks: list[NDArray[np.int64]] = []
        nnz_offset = 0
        for r in self._readers:
            ip = r.indptr.astype(np.int64)
            chunks.append(ip[:-1] + nnz_offset)
            nnz_offset += int(ip[-1])
        chunks.append(np.array([nnz_offset], dtype=np.int64))
        return np.concatenate(chunks)

    def dense_band(self, r0: int, r1: int) -> DenseBand:
        parts: list[tuple[int, int, DenseBand]] = []
        for reader, off_lo, off_hi in zip(self._readers, self._row_offsets[:-1], self._row_offsets[1:], strict=True):
            lo, hi = max(r0, off_lo), min(r1, off_hi)
            if lo >= hi:
                continue
            parts.append((lo - r0, hi - r0, reader.dense_band(lo - off_lo, hi - off_lo)))
        return _StitchedBand(parts, r1 - r0, self._dtype_val)

    def csr_rows(self, r0: int, r1: int) -> sp.csr_matrix:
        import scipy.sparse as sp

        parts = []
        for reader, off_lo, off_hi in zip(self._readers, self._row_offsets[:-1], self._row_offsets[1:], strict=True):
            lo, hi = max(r0, off_lo), min(r1, off_hi)
            if lo >= hi:
                continue
            parts.append(reader.csr_rows(lo - off_lo, hi - off_lo))
        if len(parts) == 1:
            return parts[0]
        return sp.vstack(parts, format="csr")

    def flat(self, s0: int, s1: int) -> tuple[NDArray[Any], NDArray[np.int32]]:
        data_buf = np.empty(s1 - s0, dtype=self._dtype_val)
        idx_buf = np.empty(s1 - s0, dtype=np.int32)
        nnz_offset = 0
        for reader in self._readers:
            off_lo, off_hi = nnz_offset, nnz_offset + reader.nnz
            lo, hi = max(s0, off_lo), min(s1, off_hi)
            if lo < hi:
                d, idx = reader.flat(lo - off_lo, hi - off_lo)
                data_buf[lo - s0 : hi - s0] = d
                idx_buf[lo - s0 : hi - s0] = idx
            nnz_offset = off_hi
        return data_buf, idx_buf

    def close(self) -> None:
        for r in self._readers:
            r.close()


def _is_anndata_sparse_dataset(matrix: Any) -> bool:
    fmt = getattr(matrix, "format", None)
    return fmt in ("csr", "csc") and hasattr(matrix, "group")


def _transpose_csc_to_csr(
    group: Any, true_shape: tuple[int, int], dtype: np.dtype[Any], cfg: AppConfig, tmp_dir: str | None
) -> Reader:
    import shutil
    import tempfile
    from pathlib import Path

    import zarr

    from annizarr._writers._sparse import write_transposed_sparse

    scratch = Path(tempfile.mkdtemp(prefix="annizarr_csc2csr_", dir=tmp_dir))
    try:
        tmp_root = zarr.open_group(str(scratch), mode="w")
        csr_group = write_transposed_sparse(tmp_root, "X", _SwappedCSRView(group, true_shape, dtype), cfg, target="csr")
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    logger.info("lazy CSC input streamed to a temporary CSR store (never materialised in memory).")
    return CSRZarrReader(csr_group, _tmp_dir=scratch)


def as_reader(matrix: Any, *, cfg: AppConfig, tmp_dir: str | None = None) -> Reader:
    # CSC never reaches a writer: in-memory CSC converts to CSR here, a lazy CSC streams through
    # write_transposed_sparse into a temp CSR store under tmp_dir (the output's directory).
    reader = _reader_for(matrix, cfg, tmp_dir)
    reader.cpus = cfg.chunks.cpus
    return reader


def _reader_for(matrix: Any, cfg: AppConfig, tmp_dir: str | None) -> Reader:
    import h5py
    import numpy as np
    import scipy.sparse as sp
    import zarr

    if isinstance(matrix, np.ma.MaskedArray):
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {type(matrix).__name__} (masked)")

    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {matrix.ndim}-D ndarray")
        return DenseMemoryReader(matrix)

    if sp.issparse(matrix):
        fmt = matrix.format
        if fmt == "csr":
            return CSRMemoryReader(matrix)
        if fmt == "csc":
            logger.info("adata.X was CSC and converted to CSR in memory.")
            return CSRMemoryReader(matrix.tocsr())
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {fmt}")

    if isinstance(matrix, h5py.Dataset):
        return DenseH5Reader(matrix.file.filename, matrix.name)

    if isinstance(matrix, zarr.Array):
        return DenseZarrReader(matrix)

    if _is_anndata_sparse_dataset(matrix):
        if matrix.format == "csr":
            if isinstance(matrix.group, zarr.Group):
                return CSRZarrReader(matrix.group)
            return CSRH5Reader(matrix.group.file.filename, matrix.group.name)
        return _transpose_csc_to_csr(matrix.group, matrix.shape, matrix.dtype, cfg, tmp_dir)

    if isinstance(matrix, zarr.Group):
        enc = matrix.attrs.get("encoding-type")
        if enc == "csr_matrix":
            return CSRZarrReader(matrix)
        if enc == "csc_matrix":
            return _transpose_csc_to_csr(matrix, shape_attr(matrix), get_array(matrix, "data").dtype, cfg, tmp_dir)
        raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: zarr group ({enc!r})")

    got = getattr(matrix, "format", None) or type(matrix).__name__
    raise ConversionError(f"adata.X must be CSR, CSC, or a 2-D dense array. Got: {got}")
